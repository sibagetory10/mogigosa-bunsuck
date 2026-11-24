"""English Engine v3 orchestration pipeline: single, batch, explanation, and follow-up chat."""

from __future__ import annotations

import concurrent.futures
import hashlib
from typing import Any, Callable

from google.genai import types

from src.config import FLASH_LITE_MODEL, resolve_routed_model_name
from src.english_v3.assembler import AssemblyError, _format_explanation, assemble_question
from src.english_v3.blind_validator import run_blind_solver_gate
from src.english_v3.discourse import rank_insertion_targets, rank_order_splits
from src.english_v3.preprocessor import (
    IndexedPassage,
    build_indexed_passage,
    check_type_feasibility,
    extract_passage_key_vocab,
    needs_passage_restoration,
    restore_passage_if_needed,
)
from src.english_v3.routing import (
    ExecutionTier,
    StageRoute,
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
    tried_order_splits: set[tuple[int, int, int]] | None = None,
    tried_insertion_idxs: set[int] | None = None,
) -> str:
    """Build a token-lean coordinate/delta prompt tailored only to the requested target type."""
    _ = (tried_order_splits, tried_insertion_idxs)
    t = (target_type or "").strip()
    family = classify_target_family(t, vocab_style=vocab_style)
    lvl = "고3 평가원·수능" if "고3" in (difficulty or "") else "고1·고2 학평"

    type_directive = ""
    if family == "gist":
        if "요지" in t:
            fmt = "Korean declarative sentences ('~다.')"
        elif "제목" in t:
            fmt = "concise English title phrases"
        else:
            fmt = "concise English noun phrases"
        type_directive = (
            f"Type: {t} ({lvl}). Write `correct_choice` and 4 `distractors` as {fmt} with balanced length/syntax."
        )
    elif family == "span_inference":
        if "함축" in t:
            type_directive = (
                f"Type: 함축적 의미 추론 ({lvl}). Pick figurative/context-loaded `exact_target_span` (3~10 words) "
                "in `sentences[target_sentence_idx]`; write English `correct_choice` + 4 symmetrical English `distractors`."
            )
        else:
            type_directive = (
                f"Type: 빈칸 추론 ({lvl}, unit={blank_unit}). Pick pivotal `exact_target_span` (no final punctuation); "
                "`correct_choice` MUST paraphrase with different words/syntax (NO verbatim copy, NO duplicate adjacent function word) "
                "+ 4 symmetrical English `distractors`."
            )
    elif family == "underline_5":
        if "어법" in t:
            type_directive = (
                f"Type: 어법상 틀린 것 ①~⑤ ({lvl}). Pick 5 non-overlapping grammar spans (1~4 words) in `items` (0..4); "
                "set `wrong_item_index` (0..4) and unambiguously ungrammatical `mutated_span` (agreement, participle vs finite verb, "
                "relative pronoun/adverb, voice, parallelism)."
            )
        else:
            type_directive = (
                f"Type: 문맥상 어휘 밑줄형 ①~⑤ ({lvl}). Pick 5 distinct content words in `items` (0..4); "
                "set `wrong_item_index` (0..4) and contextually opposite `mutated_span` with identical POS & inflection."
            )
    elif family == "choice_abc":
        type_directive = (
            f"Type: 문맥상 어휘 (A)(B)(C) ({lvl}). Pick 3 content words in ascending order; "
            "`exact_span`=original word, `foil_span`=contextually opposite word of identical POS & inflection."
        )
    elif family == "irrelevant":
        type_directive = (
            f"Type: 무관한 문장 ({lvl}). Pick `insert_after_sentence_idx` (0..4) and write 1 English `intruder_sentence` "
            "(12~24 words) reusing 1~2 surface topic nouns while making an off-topic claim."
        )
    elif family == "summary":
        type_directive = (
            f"Type: 요약문 완성 (A),(B) ({lvl}). Write 1~2 sentence English `summary_sentence` containing `(A)` and `(B)`; "
            "provide `correct_pair` and 4 plausible `distractor_pairs` of matching POS."
        )
    elif family == "order":
        n_s = len(indexed.sentences)
        type_directive = f"Type: 글의 순서 ({lvl}). Output 1 <= first < second < third <= {n_s - 1}."
    elif family == "insertion":
        n_s = len(indexed.sentences)
        type_directive = f"Type: 문장 삽입 ({lvl}). Output 1 <= extracted_sentence_idx <= {max(1, n_s - 2)}."

    extra_blocks: list[str] = []
    if previous_attempt is not None:
        extra_blocks.append(
            "[REGENERATE DIFFERENT TARGET/CHOICES]\n"
            f"{previous_attempt.rewritten_stem[:280]}"
        )
    if failure_reasons:
        reasons_txt = "\n".join(f"- {r}" for r in failure_reasons)
        extra_blocks.append(f"[FIX AUDIT ERRORS]\n{reasons_txt}")

    extra_str = ("\n\n" + "\n\n".join(extra_blocks)) if extra_blocks else ""
    return (
        f"{type_directive}\n"
        "Keep `explanation_ko` to 1~2 concise Korean sentences.\n\n"
        f"[Passage (N={len(indexed.sentences)})]\n"
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
    hint_vocab: list[str] | None = None,
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
    vocab_list = extract_passage_key_vocab(
        indexed.clean_text,
        hint_vocab=(hint_vocab or (getattr(base_question, "key_vocabulary", None) if base_question is not None else None)),
    )

    failure_reasons: list[str] = []
    last_error = "변형문항 품질 기준을 충족하지 못했습니다."
    tried_order_splits: set[tuple[int, int, int]] = set()
    tried_insertion_idxs: set[int] = set()
    is_zero_gen_family = family in ("order", "insertion")
    max_attempts = 2

    for attempt in range(max_attempts):
        route = rule.retry_2_generate if attempt >= 1 else rule.generate
        eff_cache_name = _cached_content_name if route.model == FLASH_LITE_MODEL else None

        coord_tag = ""
        try:
            if family == "order":
                cands = rank_order_splits(indexed, exclude=tried_order_splits)
                if not cands:
                    last_error = "유효한 단락 분할 후보가 부족합니다."
                    break
                pick_idx = 1 if (attempt == 0 and previous_attempt is not None and len(cands) >= 2) else 0
                best_t = cands[pick_idx][0]
                tried_order_splits.add(best_t)
                coord_tag = f"분할 {best_t}"
                spec: Any = OrderSpec(
                    first_chunk_start_idx=best_t[0],
                    second_chunk_start_idx=best_t[1],
                    third_chunk_start_idx=best_t[2],
                )
            elif family == "insertion":
                cands_i = rank_insertion_targets(indexed, exclude=tried_insertion_idxs)
                if not cands_i:
                    last_error = "유효한 문장 삽입 후보가 부족합니다."
                    break
                pick_idx = 1 if (attempt == 0 and previous_attempt is not None and len(cands_i) >= 2) else 0
                best_i = cands_i[pick_idx][0]
                tried_insertion_idxs.add(best_i)
                coord_tag = f"추출 S{best_i}"
                spec = InsertionSpec(extracted_sentence_idx=best_i)
            else:
                prompt = build_generation_prompt(
                    indexed,
                    target_type=target_type,
                    difficulty=difficulty,
                    blank_unit=blank_unit,
                    vocab_style=vocab_style,
                    previous_attempt=previous_attempt,
                    failure_reasons=failure_reasons or None,
                )
                spec = call_structured_v3(
                    client,
                    route=route,
                    user_prompt=prompt,
                    schema_cls=schema_cls,
                    service_tier=_service_tier,
                    cached_content_name=eff_cache_name,
                    allow_flash_escalation=allow_escalation,
                    max_output_tokens=600,
                )

            seed_val = _seed_from_inputs(
                indexed.clean_text,
                target_type,
                attempt,
                extra=(previous_attempt.rewritten_stem[:40] if previous_attempt else ""),
            )
            gen_tag = "py-discourse:0tok" if is_zero_gen_family else f"{route.model}:{route.thinking_level}"
            note = (
                f"v3({eff_profile}) | gen={gen_tag} | "
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
                key_vocab=vocab_list,
            )
            gate1_violations = validate_assembled_question(indexed, spec, rw, meta)
            if gate1_violations:
                prefix = f"[{coord_tag} 탈락] " if coord_tag else ""
                failure_reasons = [f"{prefix}{v}" for v in gate1_violations]
                last_error = " / ".join(gate1_violations)
                continue

            if rule.require_blind_solver and rule.blind_solver is not None:
                solver_route = (
                    StageRoute(rule.blind_solver.model, "medium", 0.10)
                    if (is_zero_gen_family and attempt >= 1)
                    else rule.blind_solver
                )
                bs_cache = _cached_content_name if solver_route.model == FLASH_LITE_MODEL else None
                bs_ok, bs_note = run_blind_solver_gate(
                    client,
                    rw=rw,
                    route=solver_route,
                    service_tier=_service_tier,
                    cached_content_name=bs_cache,
                    allow_flash_escalation=allow_escalation,
                    need_rationale_ko=is_zero_gen_family,
                )
                if not bs_ok:
                    prefix = f"[{coord_tag} 불합격] " if coord_tag else ""
                    failure_reasons = [f"{prefix}{bs_note}"]
                    last_error = bs_note
                    continue
                if is_zero_gen_family and bs_note.strip():
                    rw = rw.model_copy(
                        update={"pdf_grounded_explanation": _format_explanation(rw.preserved_answer, bs_note)}
                    )

            if storage is not None and cache_key and hasattr(storage, "save_cached_english_question"):
                storage.save_cached_english_question(cache_key, rw)
            return rw

        except AssemblyError as a_exc:
            msg = str(a_exc)
            failure_reasons = [msg]
            last_error = msg
        except Exception as exc:
            msg = str(exc)
            low = msg.lower()
            if any(k in low for k in ("401", "403", "api key not valid", "api_key_invalid")):
                raise
            failure_reasons = [msg]
            last_error = msg

    raise ValueError(f"품질 검수 {max_attempts}회 시도 후 보류됨: {last_error}")


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
    job_vocabs: dict[int, list[str]] = {}

    for idx, job in enumerate(jobs):
        raw_p = str(job.get("passage_text") or "").strip()
        base_q = job.get("base_question")
        job_vocabs[idx] = list(
            job.get("key_vocabulary")
            or (getattr(base_q, "key_vocabulary", None) if base_q is not None else None)
            or []
        )
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
                    hint_vocab=job_vocabs.get(idx) or None,
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
            max_output_tokens=900,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL,
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
            max_output_tokens=900,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL,
                include_thoughts=False,
            ),
        ),
    )
    _emit_usage_from_response(resp, eff_model, service_tier="standard")
    return (getattr(resp, "text", "") or "").strip() or "답변을 생성하지 못했습니다."
