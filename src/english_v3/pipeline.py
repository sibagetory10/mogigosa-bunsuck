"""English Engine v3 orchestration pipeline: single, batch, explanation, and follow-up chat."""

from __future__ import annotations

import concurrent.futures
import hashlib
from typing import Any, Callable

from google.genai import types

from src.config import FLASH_LITE_MODEL, resolve_routed_model_name
from src.english_v3.assembler import AssemblyError, assemble_question
from src.english_v3.blind_validator import run_blind_solver_gate
from src.english_v3.preprocessor import (
    IndexedPassage,
    build_indexed_passage,
    check_type_feasibility,
    needs_passage_restoration,
    restore_passage_if_needed,
)
from src.english_v3.routing import (
    ExecutionTier,
    classify_target_family,
    get_type_routing_rule,
    resolve_active_routing_profile,
    resolve_execution_tier,
)
from src.english_v3.rules_gate import validate_assembled_question
from src.english_v3.schemas import (
    ChoiceAbcSpec,
    GistSpec,
    InsertionSpec,
    IrrelevantSpec,
    OrderSpec,
    SpanInferenceSpec,
    SummarySpec,
    Underline5Spec,
)
from src.english_v3.transport import (
    build_v3_client,
    call_structured_v3,
    temporary_explicit_cache,
)
from src.gemini.usage import _emit_usage_from_response, submit_with_usage_context
from src.models.concept import RewrittenQuestion
from src.models.question import QuestionRecord
from src.storage.english_pool_store import EnglishPoolStoreMixin

_needs_english_passage_restoration = needs_passage_restoration


def prepare_clean_english_source_passage(
    api_key: str,
    model_name: str,
    source_passage: str,
    base_question: QuestionRecord | None = None,
    source_pdf_bytes: bytes | None = None,
) -> str:
    """Clean and conditionally restore an English source passage before transformation."""
    return restore_passage_if_needed(
        api_key=api_key,
        model_name=model_name,
        raw_passage=source_passage,
        base_question=base_question,
        source_pdf_bytes=source_pdf_bytes,
    )


def resolve_schema_for_type(target_type: str, vocab_style: str = "") -> type[Any]:
    """Return the Pydantic delta schema class for the requested target type."""
    family = classify_target_family(target_type, vocab_style=vocab_style)
    if family == "gist":
        return GistSpec
    if family == "span_inference":
        return SpanInferenceSpec
    if family == "underline_5":
        return Underline5Spec
    if family == "choice_abc":
        return ChoiceAbcSpec
    if family == "irrelevant":
        return IrrelevantSpec
    if family == "order":
        return OrderSpec
    if family == "insertion":
        return InsertionSpec
    if family == "summary":
        return SummarySpec
    return GistSpec


def build_generation_prompt(
    indexed: IndexedPassage,
    *,
    target_type: str,
    difficulty: str,
    blank_unit: str,
    vocab_style: str,
    previous_attempt: RewrittenQuestion | None = None,
    failure_reasons: list[str] | None = None,
) -> str:
    """Build a concise coordinate-only prompt tailored to the target CSAT English question type."""
    t = (target_type or "").strip()
    family = classify_target_family(t, vocab_style=vocab_style)
    level_note = (
        "Target difficulty: 고3 평가원/수능 킬러~준킬러 수준 (abstract paraphrasing, subtle logical traps)."
        if "고3" in (difficulty or "")
        else "Target difficulty: 고1·고2 학평 수준 (clear logical cues, high-frequency academic vocabulary)."
    )

    type_directive = ""
    if family == "gist":
        lang = "Korean ('ko')" if "요지" in t else "English ('en')"
        type_directive = (
            f"Task: Create a '{t}' item.\n"
            f"- Set `choice_language` to {lang}.\n"
            "- All 5 choices must have balanced length and matching grammatical form."
        )
    elif family == "span_inference":
        if "함축" in t:
            type_directive = (
                "Task: Create an Implied Meaning ('함축적 의미 추론') item.\n"
                "- Pick a figurative, idiomatic, or context-loaded phrase (`exact_target_span`, 3~10 words) "
                "inside `sentences[target_sentence_idx]`.\n"
                "- Write `correct_choice` in English explaining what that phrase implies in context, "
                "plus 4 symmetrical English distractors."
            )
        else:
            type_directive = (
                f"Task: Create a Blank Inference ('빈칸 추론', target unit: {blank_unit}) item.\n"
                "- Select a logically crucial span (`exact_target_span`, without sentence-final punctuation) "
                f"matching the requested unit '{blank_unit}'.\n"
                "- CRITICAL: `correct_choice` MUST paraphrase `exact_target_span` using different words/syntax "
                "(NO verbatim copying!), and must fit grammatically into the blank without repeating adjacent function words."
            )
    elif family == "underline_5":
        if "어법" in t:
            type_directive = (
                "Task: Create a Grammar Error Identification ('어법상 틀린 것 ①~⑤') item.\n"
                "- Pick 5 distinct, non-overlapping grammatical spans (1~4 words each) across the passage.\n"
                "- Mark EXACTLY ONE item with `is_answer_target=true` and provide an unambiguously ungrammatical "
                "`mutated_span` (e.g., verb agreement, participle vs finite verb, relative pronoun vs adverb, voice, parallelism). "
                "Leave `mutated_span=''` for the 4 `is_answer_target=false` items."
            )
        else:
            type_directive = (
                "Task: Create a Contextual Vocabulary ('문맥상 어휘 적절성 밑줄형 ①~⑤') item.\n"
                "- Pick 5 distinct, non-overlapping content words (verbs, adjectives, adverbs, abstract nouns).\n"
                "- Mark EXACTLY ONE item with `is_answer_target=true` and replace it with a contextually opposite/inappropriate "
                "word of the exact same part of speech and inflection in `mutated_span`."
            )
    elif family == "choice_abc":
        type_directive = (
            "Task: Create a 3-slot Contextual Vocabulary ('문맥상 어휘 적절성 선택형 (A)(B)(C)') item.\n"
            "- Pick 3 distinct content words in ascending order of appearance.\n"
            "- For each slot, `exact_span` is the original word and `foil_span` is a contextually opposite word "
            "of identical part of speech and inflection."
        )
    elif family == "irrelevant":
        type_directive = (
            "Task: Create an Irrelevant Sentence ('전체 흐름과 무관한 문장') item.\n"
            "- Pick `insert_after_sentence_idx` (0 to 4) and write ONE natural English `intruder_sentence` (12~26 words) "
            "that reuses 1~2 surface topic nouns from the passage while making an off-topic claim that breaks paragraph unity."
        )
    elif family == "order":
        type_directive = (
            "Task: Create a Paragraph Order ('글의 순서 (A)-(B)-(C)') item.\n"
            "- Choose `given_end_sentence_idx`, `second_chunk_start_idx`, and `third_chunk_start_idx` "
            "such that `0 <= given_end < second_start < third_start < N`.\n"
            "- Ensure each split boundary has clear cohesive clues (pronouns, demonstratives, connectors)."
        )
    elif family == "insertion":
        type_directive = (
            "Task: Create a Sentence Insertion ('주어진 문장 삽입') item.\n"
            "- Pick `extracted_sentence_idx` (>= 1) containing explicit cohesive anchors "
            "(e.g., However, Instead, Therefore, This, Such, For example) so there is ONLY ONE logical gap."
        )
    elif family == "summary":
        type_directive = (
            "Task: Create a Summary Completion ('요약문 완성 (A), (B)') item.\n"
            "- Write a 1~2 sentence English `summary_sentence` containing literal `(A)` and `(B)` placeholders.\n"
            "- Provide `correct_pair` (word_a, word_b) and 4 plausible `distractor_pairs` of matching parts of speech."
        )

    extra_blocks: list[str] = []
    if previous_attempt is not None:
        extra_blocks.append(
            "[REGENERATION NOTICE]\n"
            "Generate a FRESH variation with a DIFFERENT target focus/angle/distractors from this previous version:\n"
            f"{previous_attempt.rewritten_stem[:400]}"
        )
    if failure_reasons:
        reasons_txt = "\n".join(f"- {r}" for r in failure_reasons)
        extra_blocks.append(
            "[PREVIOUS ATTEMPT FAILED QUALITY AUDIT — FIX ALL ISSUES BELOW]\n"
            f"{reasons_txt}"
        )

    extra_str = ("\n\n" + "\n\n".join(extra_blocks)) if extra_blocks else ""
    return (
        f"{level_note}\n"
        f"{type_directive}\n\n"
        f"[Indexed Source Passage (N={len(indexed.sentences)} sentences)]\n"
        f"{indexed.indexed_prompt_block()}"
        f"{extra_str}"
    )


def _seed_from_inputs(clean_text: str, target_type: str, attempt: int, extra: str = "") -> int:
    raw = f"{clean_text}||{target_type}||{attempt}||{extra}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def generate_english_transformed_question(
    api_key: str,
    model_name: str,
    source_passage: str,
    target_type: str,
    difficulty: str = "고3 평가원·수능 수준",
    blank_unit: str = "구 (3~6단어)",
    vocab_style: str = "밑줄형 (①~⑤)",
    source_label: str = "",
    previous_attempt: RewrittenQuestion | None = None,
    use_precision_mode: bool = False,
    storage: Any = None,
    use_cache: bool = True,
    username: str = "",
    base_question: QuestionRecord | None = None,
    source_pdf_bytes: bytes | None = None,
    *,
    _service_tier: ExecutionTier = "standard",
    _cached_content_name: str | None = None,
    _routing_profile: str | None = None,
) -> RewrittenQuestion:
    """Clean-slate v3 CSAT English question generator with Python hard rules & Blind Solver."""
    _ = (model_name, source_label, use_precision_mode)
    clean_raw = prepare_clean_english_source_passage(
        api_key=api_key,
        model_name=FLASH_LITE_MODEL,
        source_passage=source_passage,
        base_question=base_question,
        source_pdf_bytes=source_pdf_bytes,
    )
    indexed = build_indexed_passage(clean_raw)
    family = classify_target_family(target_type, vocab_style=vocab_style)
    feasibility_err = check_type_feasibility(indexed, family)
    if feasibility_err:
        raise ValueError(feasibility_err)

    cache_key = ""
    if storage is not None and hasattr(storage, "get_cached_english_question"):
        cache_key = EnglishPoolStoreMixin.build_english_pool_key(
            source_passage=indexed.clean_text,
            target_type=target_type,
            difficulty=difficulty,
            blank_unit=blank_unit,
            vocab_style=vocab_style,
        )
        if use_cache and previous_attempt is None:
            cached_rw = storage.get_cached_english_question(
                cache_key,
                exclude_stem="",
                username=username,
            )
            if cached_rw is not None:
                return cached_rw

    eff_profile = resolve_active_routing_profile(_routing_profile)
    allow_escalation = eff_profile != "cost_saver_lite"
    rule = get_type_routing_rule(target_type, vocab_style=vocab_style, profile_name=eff_profile)
    schema_cls = resolve_schema_for_type(target_type, vocab_style=vocab_style)
    client = build_v3_client(api_key)

    failure_reasons: list[str] = []
    last_error = "변형문항 품질 기준을 충족하지 못했습니다."

    for attempt in range(3):
        route = rule.retry_2_generate if attempt >= 2 else rule.generate
        # Explicit cache is tied to the default model; skip cached_content if escalated to another model
        eff_cache_name = _cached_content_name if route.model == FLASH_LITE_MODEL else None

        prompt = build_generation_prompt(
            indexed,
            target_type=target_type,
            difficulty=difficulty,
            blank_unit=blank_unit,
            vocab_style=vocab_style,
            previous_attempt=previous_attempt,
            failure_reasons=failure_reasons or None,
        )
        try:
            spec = call_structured_v3(
                client,
                route=route,
                user_prompt=prompt,
                schema_cls=schema_cls,
                service_tier=_service_tier,
                cached_content_name=eff_cache_name,
                allow_flash_escalation=allow_escalation,
            )
            seed_val = _seed_from_inputs(
                indexed.clean_text,
                target_type,
                attempt,
                extra=(previous_attempt.rewritten_stem[:40] if previous_attempt else ""),
            )
            note = (
                f"v3({eff_profile}) | gen={route.model}:{route.thinking_level} | "
                f"tier={_service_tier} | attempt={attempt + 1}"
            )
            rw, meta = assemble_question(
                indexed,
                spec,
                target_type=target_type,
                blank_unit=blank_unit,
                vocab_style=vocab_style,
                seed_val=seed_val,
                self_note=note,
            )
            gate1_violations = validate_assembled_question(indexed, spec, rw, meta)
            if gate1_violations:
                failure_reasons = gate1_violations
                last_error = " / ".join(gate1_violations)
                continue

            if rule.require_blind_solver and rule.blind_solver is not None:
                bs_cache = _cached_content_name if rule.blind_solver.model == FLASH_LITE_MODEL else None
                bs_ok, bs_reason = run_blind_solver_gate(
                    client,
                    rw=rw,
                    route=rule.blind_solver,
                    service_tier=_service_tier,
                    cached_content_name=bs_cache,
                    allow_flash_escalation=allow_escalation,
                )
                if not bs_ok:
                    failure_reasons = [bs_reason]
                    last_error = bs_reason
                    continue

            if storage is not None and cache_key and hasattr(storage, "save_cached_english_question"):
                storage.save_cached_english_question(cache_key, rw)
            return rw

        except AssemblyError as a_exc:
            msg = str(a_exc)
            failure_reasons = [msg]
            last_error = msg
        except Exception as exc:
            msg = str(exc)
            # Fatal auth/quota errors should surface immediately
            low = msg.lower()
            if any(k in low for k in ("401", "403", "api key not valid", "api_key_invalid")):
                raise
            failure_reasons = [msg]
            last_error = msg

    raise ValueError(f"품질 검수 3회 시도 후 보류됨: {last_error}")


def generate_batch_questions_v3(
    api_key: str,
    jobs: list[dict[str, Any]],
    *,
    storage: Any = None,
    username: str = "",
    max_workers: int = 4,
    progress_callback: Callable[[int, int], None] | None = None,
    routing_profile: str | None = None,
) -> tuple[dict[int, RewrittenQuestion], dict[int, str]]:
    """Run multiple English transform jobs with cache pre-check, Flex routing, and short-TTL Explicit Cache."""
    total_jobs = len(jobs)
    if total_jobs == 0:
        return {}, {}

    results_by_idx: dict[int, RewrittenQuestion] = {}
    errors_by_idx: dict[int, str] = {}
    uncached_indices: list[int] = []
    cleaned_passages: dict[int, str] = {}

    for idx, job in enumerate(jobs):
        raw_p = str(job.get("passage_text") or "").strip()
        base_q = job.get("base_question")
        clean_p = prepare_clean_english_source_passage(
            api_key=api_key,
            model_name=FLASH_LITE_MODEL,
            source_passage=raw_p,
            base_question=base_q,
        )
        cleaned_passages[idx] = clean_p
        indexed = build_indexed_passage(clean_p)

        if storage is not None and hasattr(storage, "get_cached_english_question"):
            ck = EnglishPoolStoreMixin.build_english_pool_key(
                source_passage=indexed.clean_text,
                target_type=job["target_type"],
                difficulty=job.get("difficulty", "고3 평가원·수능 수준"),
                blank_unit=job.get("blank_unit", "구 (3~6단어)"),
                vocab_style=job.get("vocab_style", "밑줄형 (①~⑤)"),
            )
            cached = storage.get_cached_english_question(ck, username=username)
            if cached is not None:
                results_by_idx[idx] = cached
                if progress_callback:
                    progress_callback(len(results_by_idx) + len(errors_by_idx), total_jobs)
                continue

        uncached_indices.append(idx)

    if not uncached_indices:
        return results_by_idx, errors_by_idx

    service_tier, use_explicit_cache = resolve_execution_tier(len(uncached_indices))
    eff_profile = resolve_active_routing_profile(routing_profile)
    client = build_v3_client(api_key)

    with temporary_explicit_cache(
        client,
        model_name=FLASH_LITE_MODEL,
        enabled=use_explicit_cache,
        ttl_seconds=600,
    ) as cached_name:
        n_workers = max(1, min(len(uncached_indices), int(max_workers)))
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=n_workers)
        try:
            future_map = {
                submit_with_usage_context(
                    executor,
                    generate_english_transformed_question,
                    api_key=api_key,
                    model_name=FLASH_LITE_MODEL,
                    source_passage=cleaned_passages[idx],
                    target_type=jobs[idx]["target_type"],
                    difficulty=jobs[idx].get("difficulty", "고3 평가원·수능 수준"),
                    blank_unit=jobs[idx].get("blank_unit", "구 (3~6단어)"),
                    vocab_style=jobs[idx].get("vocab_style", "밑줄형 (①~⑤)"),
                    source_label=jobs[idx].get("source_label", ""),
                    storage=storage,
                    use_cache=False,
                    username=username,
                    base_question=None,
                    _service_tier=service_tier,
                    _cached_content_name=cached_name,
                    _routing_profile=eff_profile,
                ): idx
                for idx in uncached_indices
            }
            batch_timeout_s = max(65.0, float(len(uncached_indices)) * 22.0)
            try:
                for fut in concurrent.futures.as_completed(future_map, timeout=batch_timeout_s):
                    idx = future_map[fut]
                    try:
                        results_by_idx[idx] = fut.result(timeout=5)
                    except Exception as exc:
                        errors_by_idx[idx] = str(exc)
                    if progress_callback:
                        progress_callback(len(results_by_idx) + len(errors_by_idx), total_jobs)
            except concurrent.futures.TimeoutError:
                for fut, idx in future_map.items():
                    if idx not in results_by_idx and idx not in errors_by_idx:
                        fut.cancel()
                        errors_by_idx[idx] = "응답 시간 초과로 중단되었습니다. 해당 문항만 다시 시도해 주세요."
                        if progress_callback:
                            progress_callback(len(results_by_idx) + len(errors_by_idx), total_jobs)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    return results_by_idx, errors_by_idx


def generate_english_explanation(
    api_key: str,
    model_name: str,
    question: QuestionRecord,
    rewritten: RewrittenQuestion | None,
    user_answer: str = "",
    source_reference: str = "",
) -> str:
    """On-demand detailed Korean CSAT English syntax/logic/distractor explanation."""
    _ = (model_name, source_reference)
    client = build_v3_client(api_key)
    eff_model = resolve_routed_model_name(FLASH_LITE_MODEL)
    stem = rewritten.rewritten_stem if rewritten is not None else (question.passage_text or question.question_text)
    choices = rewritten.rewritten_choices if rewritten is not None else []
    ans = rewritten.preserved_answer if rewritten is not None else question.answer
    choices_block = "\n".join(f"{i}. {c}" for i, c in enumerate(choices, start=1))

    prompt = (
        "아래 수능 영어 변형문제에 대해 수험생이 완벽히 이해할 수 있도록 한국어로 상세 해설을 작성하세요.\n"
        "구성:\n"
        "1. **핵심 논지 및 글의 흐름**\n"
        "2. **정답 도출 근거** (핵심 문장·구문 해석 포함)\n"
        "3. **①~⑤ 선지별 정오답 분석**\n"
        "4. **핵심 어휘·구문 정리**\n"
        + (f"\n(참고: 학생 제출 답안은 {user_answer}번입니다. 오답이라면 왜 틀렸는지 1줄 짚어 주세요.)\n" if user_answer else "")
        + f"\n[정답]: {ans}번\n[문제]\n{stem}\n\n[선지]\n{choices_block}"
    )
    resp = client.models.generate_content(
        model=eff_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.25,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.LOW,
                include_thoughts=False,
            ),
        ),
    )
    _emit_usage_from_response(resp, eff_model, service_tier="standard")
    return (getattr(resp, "text", "") or "").strip() or "해설을 생성하지 못했습니다."


def chat_followup_with_english(
    api_key: str,
    model_name: str,
    question: QuestionRecord,
    rewritten: RewrittenQuestion | None,
    chat_history: list[dict[str, str]],
    user_query: str,
) -> str:
    """1:1 follow-up Q&A grounded in the transformed CSAT English question."""
    _ = model_name
    client = build_v3_client(api_key)
    eff_model = resolve_routed_model_name(FLASH_LITE_MODEL)
    stem = rewritten.rewritten_stem if rewritten is not None else (question.passage_text or question.question_text)
    choices = rewritten.rewritten_choices if rewritten is not None else []
    ans = rewritten.preserved_answer if rewritten is not None else question.answer
    history_txt = "\n".join(
        f"{'학생' if m.get('role') == 'user' else '선생님'}: {m.get('content', '')}"
        for m in (chat_history or [])[-6:]
    )
    prompt = (
        "당신은 수능 영어 1타 강사입니다. 아래 문항의 지문, 선지, 정답만을 근거로 학생의 질문에 친절하고 정확하게 답하세요.\n\n"
        f"[문항]\n{stem}\n[선지]\n" + "\n".join(f"{i}. {c}" for i, c in enumerate(choices, start=1))
        + f"\n[정답]: {ans}번\n"
        + (f"\n[이전 대화]\n{history_txt}\n" if history_txt else "")
        + f"\n[학생 질문]: {user_query}"
    )
    resp = client.models.generate_content(
        model=eff_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.LOW,
                include_thoughts=False,
            ),
        ),
    )
    _emit_usage_from_response(resp, eff_model, service_tier="standard")
    return (getattr(resp, "text", "") or "").strip() or "답변을 생성하지 못했습니다."
