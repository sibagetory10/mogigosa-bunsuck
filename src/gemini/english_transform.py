from __future__ import annotations

import hashlib
import re
import threading
from typing import Any, Callable, Sequence

from google.genai import types

from src.config import EXAM_TAGGING_MODEL, FLASH_LITE_MODEL
from src.gemini.core import (
    DEFAULT_API_TIMEOUT_SECONDS,
    _CIRCLED_DIGIT_TO_NUM,
    _build_client,
    _generate_content_with_fallback,
    _generate_structured_with_fallback,
    _is_flash_lite_model,
    _normalize_extracted_answer,
    _parse_structured,
)
from src.gemini.english_validators import (
    _AI_META_MONOLOGUE_RE,
    _DANGLING_TO_END_RE,
    _assemble_english_slim_stem,
    _assemble_rewritten_question_from_spec,
    _build_english_transform_rule,
    _detect_foreign_script_contamination,
    _ensure_korean_instruction_prefix,
    _has_exam_structure_markers,
    _has_incomplete_english_blank,
    _is_irrelevant_answer_from_source,
    _needs_english_passage_restoration,
    _prepare_indexed_sentences_for_transform,
    _preprocess_english_source_passage,
    _rebuild_insertion_stem_from_source,
    _reconcile_english_insertion_answer,
    _reconcile_english_order_answer,
    _reconcile_english_passage_with_question_text,
    _reconcile_english_summary_answer,
    _resolve_english_transform_schema_and_rule,
    _split_english_sentences,
    _validate_english_summary_diagram_output,
    _validate_english_transformed_output,
)
from src.models import (
    CURRENT_ENGLISH_ANALYSIS_VERSION,
    EnglishPassageAnalysisOutput,
    EnglishPassageSummaryDiagramOutput,
    EnglishSentenceAnalysisItem,
    KoreanVisualDiagram,
    QuestionRecord,
    RewrittenQuestion,
)
from src.text_formatter import (
    format_explanation_text,
    normalize_english_exam_text,
    normalize_english_korean_translation,
    strip_english_embedded_choices,
)

_RESTORED_PASSAGE_CACHE: dict[str, str] = {}
_RESTORED_PASSAGE_LOCK = threading.RLock()


def _restore_incomplete_source_passage(
    api_key: str,
    model_name: str,
    passage: str,
    orig_question_type: str = "",
    answer: str = "",
    choices: Sequence[str] | None = None,
    explanation: str = "",
) -> str:
    """빈칸·어법·어휘·무관한 문장·삽입·순서 등 기출 변형 지문을 원래의 완전하고 문법적으로 올바른 영어 지문으로 복원한다 (해시 캐싱)."""
    if not api_key or not _needs_english_passage_restoration(passage, orig_question_type):
        return passage
    has_circled_markers = bool(re.search(r"[①②③④⑤]", passage))
    is_irrelevant_q = any(kw in (orig_question_type or "") for kw in ("무관", "흐름"))
    if is_irrelevant_q and not has_circled_markers and not _has_incomplete_english_blank(passage):
        # 이미 ①~⑤ 마커와 무관한 문장이 제거된 지문에서 정상 문장을 다시 지우는 사고 방지
        return _preprocess_english_source_passage(passage)

    choices_str = " | ".join(str(c).strip() for c in (choices or []) if str(c).strip())
    hash_src = f"{passage}\n||{orig_question_type}||{answer}||{choices_str}"
    p_hash = hashlib.sha256(hash_src.encode("utf-8")).hexdigest()[:24]
    with _RESTORED_PASSAGE_LOCK:
        cached = _RESTORED_PASSAGE_CACHE.get(p_hash)
        if cached:
            return cached
    try:
        client = _build_client(api_key, timeout_seconds=45)
        meta_lines: list[str] = []
        if (orig_question_type or "").strip():
            meta_lines.append(f"- Original Question Type: {orig_question_type.strip()}")
        if (answer or "").strip():
            meta_lines.append(f"- Official Correct Answer: {answer.strip()}")
        if choices_str:
            meta_lines.append(f"- Original Choices: {choices_str[:700]}")
        if (explanation or "").strip():
            meta_lines.append(f"- Official Explanation (reference): {explanation.strip()[:1000]}")
        meta_block = ("\n[Exam Metadata]\n" + "\n".join(meta_lines) + "\n") if meta_lines else ""

        prompt = (
            "Below is an English reading passage extracted from a Korean CSAT/mock exam question. "
            "Because it was used as an exam question, it may contain one of the following intentional modifications or extraction artifacts:\n"
            "1) Blank-filling: a missing phrase/clause (e.g. '____', '[   ]', an abrupt gap, or an incomplete sentence).\n"
            "2) Grammar (어법): ONE grammatically incorrect word/expression (e.g. 'making' instead of 'makes', wrong verb/relative pronoun/tense) OR an '(A)/(B)/(C)' bracketed choice.\n"
            "3) Vocabulary (어휘): ONE contextually inappropriate/antonym word OR an '(A)/(B)/(C)' bracketed choice.\n"
            "4) Irrelevant sentence (무관한 문장): ONE off-topic sentence marked with ①~⑤.\n"
            "5) Sentence insertion / Ordering (문장 삽입 / 글의 순서): insertion markers ①~⑤ or shuffled paragraph markers (A)/(B)/(C).\n\n"
            "Your task: Restore ONLY the defective/missing/incorrect/misplaced part using the exam metadata (answer/choices/explanation) and context so the passage is a 100% grammatically sound, logically coherent original English text.\n"
            "- Remove exam markers like ①, ②, ③, ④, ⑤ or (A)/(B)/(C) labels if they are question markers.\n"
            "- For grammar/vocabulary questions, fix ONLY the single erroneous/antonym expression to the correct form/word.\n"
            "- For irrelevant-sentence questions, remove ONLY the SINGLE off-topic sentence matching the official answer number (①~⑤) and keep EVERY OTHER sentence 100% intact. If ①~⑤ markers are NOT present, do NOT delete any sentence.\n"
            "- STRICT ANTI-MUTATION RULE: Keep ALL other words, idioms, parenthetical remarks like '(as commonly supposed)', punctuation, and sentence structures 100% VERBATIM.\n"
            "- NEVER overwrite unusual source phrasing with famous quotes or prior knowledge (e.g., if the source says 'does it make it sound?', keep 'it sound', do NOT change it to 'a sound'; keep 'at one and the same time', do NOT change it to 'at once').\n"
            "- Output ONLY the restored English passage with no commentary or markdown.\n"
            f"{meta_block}\n"
            f"[Passage]\n{passage}"
        )
        resp = _generate_content_with_fallback(
            client=client,
            model_name=model_name or EXAM_TAGGING_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=2048,
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.MINIMAL,
                    include_thoughts=False,
                ),
            ),
            fast_mode=True,
        )
        restored = _preprocess_english_source_passage((resp.text or "").strip())
        orig_sents = _split_english_sentences(_preprocess_english_source_passage(passage))
        rest_sents = _split_english_sentences(restored)
        # 무관한 문장(①~⑤ 포함)만 최대 1문장 감소를 허용하고, 그 외 유형은 정상 문장 통누락 차단
        min_allowed_sents = max(1, len(orig_sents) - (1 if (is_irrelevant_q and has_circled_markers) else 0))
        if (
            restored
            and len(restored) >= int(len(passage) * 0.65)
            and not _has_incomplete_english_blank(restored)
            and (not orig_sents or len(rest_sents) >= min_allowed_sents)
        ):
            with _RESTORED_PASSAGE_LOCK:
                if len(_RESTORED_PASSAGE_CACHE) > 512:
                    _RESTORED_PASSAGE_CACHE.clear()
                _RESTORED_PASSAGE_CACHE[p_hash] = restored
            return restored
    except Exception:
        pass
    return _preprocess_english_source_passage(passage)


def prepare_clean_english_source_passage(
    *,
    api_key: str = "",
    model_name: str = "",
    source_passage: str = "",
    base_question: QuestionRecord | None = None,
    question_text: str = "",
    source_pdf_bytes: bytes | None = None,
    q_num: int = 0,
    passage_group_id: str = "",
    orig_question_type: str = "",
    orig_answer: str = "",
    orig_choices: Sequence[str] | None = None,
    orig_explanation: str = "",
) -> str:
    """
    변형 문제 출제 전 원문 영어 지문을 정밀 정화·복원한다 (Source Sanitizer v2):
    1) 기출 DB 레코드(`base_question`)나 원본 `question_text` / `source_pdf_bytes`가 있으면
       `extract_english_passage_from_problem_source` 및 `_reconcile_english_passage_with_question_text`를 통해
       글의 순서 재배열, 문장 삽입 원위치 복원, 무관한 문장 제거, 빈칸 정답 채움, 요약문 분리를 결정론적으로 수행한다.
    2) 여전히 미완성 빈칸(`____`), 선택형 네모(`[a / b]`), 기출 마커가 남아 있는 경우에만
       원래 문항의 메타데이터(유형·정답·선지·해설)를 모두 주입해 `_restore_incomplete_source_passage`로 복원한다.
    """
    raw_passage = (source_passage or (getattr(base_question, "passage_text", "") if base_question is not None else "") or "").strip()
    raw_q_text = (question_text or (getattr(base_question, "question_text", "") if base_question is not None else "") or "").strip()
    eff_q_num = int(q_num or (getattr(base_question, "q_num", 0) if base_question is not None else 0) or 0)
    eff_gid = (passage_group_id or (getattr(base_question, "passage_group_id", "") if base_question is not None else "") or "").strip()
    eff_orig_type = (
        orig_question_type
        or (getattr(base_question, "english_question_type", "") if base_question is not None else "")
        or (getattr(base_question, "unit_name", "") if base_question is not None else "")
        or ""
    ).strip()
    eff_ans = (orig_answer or (getattr(base_question, "answer", "") if base_question is not None else "") or "").strip()
    eff_exp = (orig_explanation or (getattr(base_question, "official_explanation", "") if base_question is not None else "") or "").strip()

    candidate = raw_passage

    # Fast-path: 이미 80자 이상이고 미복원 빈칸/선택형/기출 마커가 전혀 없는 깨끗한 지문이면
    # 무거운 PDF 파싱(PyMuPDF)을 건너뛰어 워커 스레드 병목·데드락을 원천 방지한다.
    if len(candidate) >= 80 and not _needs_english_passage_restoration(candidate, eff_orig_type):
        return _preprocess_english_source_passage(candidate)

    # 1. question_text가 있거나 원문에 기출 구조 마커/빈칸이 남아 있으면 결정론적 복원기 우선 실행
    if raw_q_text:
        try:
            reconciled, _ = _reconcile_english_passage_with_question_text(
                passage_text=candidate,
                question_text=raw_q_text,
                orig_question_type=eff_orig_type,
                answer=eff_ans,
            )
            if reconciled:
                candidate = reconciled
        except Exception:
            pass

    if source_pdf_bytes or raw_q_text or _needs_english_passage_restoration(candidate, eff_orig_type):
        try:
            from src.pdf_parser.english_source_extractor import (
                _reconstruct_english_from_raw_qtext,
                extract_english_passage_from_problem_source,
            )

            extracted, _ = extract_english_passage_from_problem_source(
                passage_text=candidate,
                question_text=raw_q_text or candidate,
                source_pdf_bytes=source_pdf_bytes,
                q_num=eff_q_num,
                passage_group_id=eff_gid,
                orig_question_type=eff_orig_type,
                answer=eff_ans,
            )
            if extracted and len(extracted) >= 60:
                candidate = extracted
            elif not raw_q_text and eff_ans and _needs_english_passage_restoration(candidate, eff_orig_type):
                rebuilt = _reconstruct_english_from_raw_qtext(
                    candidate,
                    q_num=eff_q_num,
                    passage_group_id=eff_gid,
                    orig_question_type=eff_orig_type,
                    answer=eff_ans,
                )
                if rebuilt and len(rebuilt) >= 60:
                    candidate = rebuilt
        except Exception:
            pass

    # 2. 아직도 미복원 빈칸/선택형/기출 마커가 남아 있으면 메타데이터 포함 AI 복원 수행
    if api_key and _needs_english_passage_restoration(candidate, eff_orig_type):
        eff_model = (model_name or "").strip() or EXAM_TAGGING_MODEL
        candidate = _restore_incomplete_source_passage(
            api_key=api_key,
            model_name=eff_model,
            passage=candidate,
            orig_question_type=eff_orig_type,
            answer=eff_ans,
            choices=orig_choices,
            explanation=eff_exp,
        )

    return _preprocess_english_source_passage(candidate)


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
    storage: Any | None = None,
    use_cache: bool = True,
    username: str = "",
    *,
    base_question: QuestionRecord | None = None,
    question_text: str = "",
    source_pdf_bytes: bytes | None = None,
    q_num: int = 0,
    passage_group_id: str = "",
    orig_question_type: str = "",
    orig_answer: str = "",
    orig_choices: Sequence[str] | None = None,
    orig_explanation: str = "",
) -> RewrittenQuestion:
    """
    원문 영어 독해 지문을 지정된 수능 독해 유형으로 변형·출제한다 (Transform Engine v2):
    1) 원문 지문 정밀 정화·복원(`prepare_clean_english_source_passage`)
    2) 문장 인덱싱(`[S0]`~`[Sn-1]`) 및 짧은 복합문 절 경계 안전 분할
    3) 유형군별 전용 Pydantic 스키마로 Gemini 구조화 호출
    4) 100% 파이썬 결정론적 조립(`_assemble_rewritten_question_from_spec`) 및 사후 검증
    5) 검증 통과 문항만 v2 풀(Pool)에 캐싱하며, 최종 실패 시 불량 문항 반환 차단
    """
    eff_model = (model_name or "").strip() or EXAM_TAGGING_MODEL
    clean_passage = prepare_clean_english_source_passage(
        api_key=api_key,
        model_name=eff_model,
        source_passage=source_passage,
        base_question=base_question,
        question_text=question_text,
        source_pdf_bytes=source_pdf_bytes,
        q_num=q_num,
        passage_group_id=passage_group_id,
        orig_question_type=orig_question_type,
        orig_answer=orig_answer,
        orig_choices=orig_choices,
        orig_explanation=orig_explanation,
    )
    if not clean_passage:
        raise ValueError("변형할 영어 지문(source_passage)이 비어 있습니다.")

    canonical_type, schema_cls, type_rule_block = _resolve_english_transform_schema_and_rule(
        target_type=target_type,
        blank_unit=blank_unit,
        vocab_style=vocab_style,
    )

    cache_key = ""
    if storage is not None and hasattr(storage, "build_english_pool_key"):
        cache_key = storage.build_english_pool_key(
            source_passage=clean_passage,
            target_type=canonical_type,
            difficulty=difficulty,
            blank_unit=blank_unit,
            vocab_style=vocab_style,
        )
        if use_cache and previous_attempt is None and hasattr(storage, "get_cached_english_question"):
            cached_rw = storage.get_cached_english_question(cache_key, username=username)
            if cached_rw is not None:
                return cached_rw

    norm_passage, sent_list, sent_spans = _prepare_indexed_sentences_for_transform(
        clean_passage,
        canonical_type=canonical_type,
    )
    n_sents = len(sent_list)
    if "순서" in canonical_type and n_sents < 4:
        raise ValueError(
            f"글의 순서 유형은 최소 4문장(또는 4절) 이상인 지문에서만 출제할 수 있습니다 (현재 지문: {n_sents}문장)."
        )
    if "무관" in canonical_type and n_sents < 5:
        raise ValueError(
            f"전체 흐름과 무관한 문장 유형은 최소 5문장(또는 5절) 이상인 지문에서만 출제할 수 있습니다 (현재 지문: {n_sents}문장)."
        )
    if "삽입" in canonical_type and n_sents < 6:
        raise ValueError(
            f"주어진 문장 삽입 유형은 최소 6문장(또는 6절) 이상인 지문에서만 출제할 수 있습니다 (현재 지문: {n_sents}문장)."
        )

    target_thinking_level = (
        types.ThinkingLevel.LOW
        if (use_precision_mode and not _is_flash_lite_model(eff_model))
        else types.ThinkingLevel.MINIMAL
    )

    client = _build_client(api_key, timeout_seconds=25)
    system_inst = (
        "대한민국 수능·평가원 영어 출제위원으로서 지정된 **[목표 유형 1개]**의 전용 JSON 스키마를 정확히 생성하세요.\n\n"
        f"[유형: {canonical_type} | 난이도: {difficulty} | 총 문장 수: {n_sents}문장 (S0 ~ S{max(0, n_sents - 1)})]\n"
        f"{type_rule_block}\n\n"
        "[공통 필수 규칙]\n"
        "1. 원문 각 문장 앞에 붙은 `[S0]`, `[S1]`, ... 번호가 `sentence_index`(0부터 시작하는 정수)입니다.\n"
        "2. 원문 구절(`target_span`, `original_span`, `original_word`)을 적을 때는 `[S0]` 같은 인덱스 라벨을 제외하고 해당 문장 안에 실제로 존재하는 영어 구절만 철자 그대로 적으세요.\n"
        "3. `explanation`은 한국어 2문장(공백 포함 120자) 이내로 정답 도출 근거만 명쾌하게 작성하고, AI 내부 사고 독백('Wait, let me...', '수정 필요' 등)은 절대 출력하지 마세요.\n"
        "4. `key_terms`는 지문 핵심 어휘 2개만 `'영단어 → 한국어 뜻'` 형식으로 기록하세요."
    )

    prev_feedback_block = ""
    if previous_attempt is not None:
        prev_feedback_block = (
            "\n[이전 생성본 (재변환 요청 — 이전과 다른 문장/구절/어휘 포인트를 선택하여 새롭게 출제)]\n"
            f"- 이전 지문/발문 일부: {previous_attempt.rewritten_stem[:450]}\n"
            f"- 이전 정답: {previous_attempt.preserved_answer} / 이전 치환: {', '.join(previous_attempt.replaced_terms_map or [])}\n"
        )

    indexed_passage_block = "\n".join(f"[S{i}] {s}" for i, s in enumerate(sent_list))
    base_prompt = (
        f"[출처]: {source_label or '영어 기출 독해 지문'}\n"
        f"[목표 수능 유형]: {canonical_type} | [난이도]: {difficulty}\n"
        f"{prev_feedback_block}\n"
        f"[문장 번호가 부여된 원문 영어 지문 (총 {n_sents}문장: S0 ~ S{max(0, n_sents - 1)})]\n"
        f"{indexed_passage_block}\n\n"
        "위 원문 영어 지문을 바탕으로 목표 유형의 JSON 스키마를 엄격히 준수하여 출제 명세를 생성하세요."
    )

    prompt = base_prompt
    rw: RewrittenQuestion | None = None
    issues: list[str] = []
    last_error_msg = ""
    base_out_tokens = 900

    for attempt in range(2):
        out_tokens = base_out_tokens if attempt == 0 else 1200
        try:
            parsed_spec = _generate_structured_with_fallback(
                client=client,
                model_name=eff_model,
                contents=prompt,
                schema_cls=schema_cls,
                config=types.GenerateContentConfig(
                    system_instruction=system_inst,
                    response_mime_type="application/json",
                    response_schema=schema_cls,
                    temperature=0.35 if (previous_attempt is not None or attempt > 0) else 0.2,
                    max_output_tokens=out_tokens,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=target_thinking_level,
                        include_thoughts=False,
                    ),
                ),
                fast_mode=not use_precision_mode,
                max_parse_retries=1,
            )
        except Exception as api_exc:
            last_error_msg = str(api_exc)
            if attempt < 1:
                continue
            raise

        try:
            if isinstance(parsed_spec, RewrittenQuestion):
                # 하위 호환: 테스트 목(Mock) 등에서 RewrittenQuestion을 직접 반환한 경우
                cand_rw = parsed_spec
                if not cand_rw.rewritten_choices and any(
                    k in canonical_type for k in ("어법", "무관", "삽입", "밑줄")
                ):
                    cand_rw.rewritten_choices = ["①", "②", "③", "④", "⑤"]
                cleaned_choices: list[str] = []
                for ch in cand_rw.rewritten_choices or []:
                    c_str = (ch or "").strip()
                    if len(c_str) > 1 and c_str[0] in _CIRCLED_DIGIT_TO_NUM:
                        c_str = c_str[1:].strip()
                    cleaned_choices.append(c_str)
                cand_rw.rewritten_choices = cleaned_choices[:5]
                cand_rw.preserved_answer = _normalize_extracted_answer(cand_rw.preserved_answer)
                _assemble_english_slim_stem(cand_rw, canonical_type, norm_passage)
                if "순서" in canonical_type:
                    _reconcile_english_order_answer(cand_rw)
                elif "삽입" in canonical_type:
                    if not _rebuild_insertion_stem_from_source(cand_rw, norm_passage):
                        _reconcile_english_insertion_answer(cand_rw)
                elif "요약" in canonical_type:
                    _reconcile_english_summary_answer(cand_rw)
            else:
                cand_rw = _assemble_rewritten_question_from_spec(
                    spec=parsed_spec,
                    canonical_type=canonical_type,
                    clean_passage=norm_passage,
                    sent_list=sent_list,
                    sent_spans=sent_spans,
                    blank_unit=blank_unit,
                    vocab_style=vocab_style,
                    attempt_seed=attempt + (10 if previous_attempt is not None else 0),
                )
        except Exception as asm_exc:
            last_error_msg = str(asm_exc)
            issues = [f"조립 오류: {last_error_msg}"]
            if attempt < 1:
                prompt = (
                    base_prompt
                    + "\n\n[⚠️ 직전 출력 조립 오류 — 아래 사항을 반드시 수정하여 다시 생성하세요]\n"
                    + f"- {last_error_msg}\n"
                    + "- 원문에 실제로 존재하는 영어 구절(`original_span` / `target_span` / `original_word`)과 정확한 문장 번호(`sentence_index`)만 지정하세요."
                )
                continue
            break

        norm_stem = normalize_english_exam_text(cand_rw.rewritten_stem)
        if any(k in canonical_type for k in ("어법", "어휘")) and "<u" in norm_stem.lower():
            cand_rw.rewritten_stem = norm_stem
        else:
            cand_rw.rewritten_stem = strip_english_embedded_choices(norm_stem)
        if "빈칸" in canonical_type:
            cand_rw.rewritten_stem = re.sub(r"[ \t]*\(\s*[①②③④⑤]\s*\)[ \t]*", " ", cand_rw.rewritten_stem)
            cand_rw.rewritten_stem = re.sub(r"[ \t]*[①②③④⑤][ \t]*", " ", cand_rw.rewritten_stem)
            cand_rw.rewritten_stem = re.sub(r"[ \t]{2,}", " ", cand_rw.rewritten_stem).strip()
        cand_rw.rewritten_stem = _ensure_korean_instruction_prefix(cand_rw.rewritten_stem, canonical_type, vocab_style)
        cand_rw.pdf_grounded_explanation = format_explanation_text(cand_rw.pdf_grounded_explanation)
        cand_rw.preserved_answer = _normalize_extracted_answer(cand_rw.preserved_answer) or "1"
        cand_rw.target_question_type = canonical_type
        cand_rw.subject_domain = "영어"

        rw = cand_rw
        issues = _validate_english_transformed_output(
            rw,
            canonical_type,
            vocab_style,
            source_passage=norm_passage,
        )
        if not issues:
            break
        if attempt < 1:
            prompt = (
                base_prompt
                + "\n\n[⚠️ 직전 출력 검증 오류 — 반드시 아래 누락/오류 사항을 모두 수정하여 다시 생성하세요]\n"
                + "\n".join(f"- {iss}" for iss in issues)
            )

    if rw is None or issues:
        err_detail = "; ".join(issues) if issues else (last_error_msg or "알 수 없는 조립/검증 오류")
        raise ValueError(f"[{canonical_type}] 영어 변형 문항 생성·검증에 실패했습니다: {err_detail}")

    if (
        cache_key
        and storage is not None
        and hasattr(storage, "save_cached_english_question")
    ):
        storage.save_cached_english_question(cache_key, rw, max_per_key=3)
    return rw


def generate_english_explanation(
    api_key: str,
    model_name: str,
    question: QuestionRecord,
    rewritten: RewrittenQuestion | None,
    user_answer: str = "",
    source_reference: str = "",
) -> str:
    """모드 B에서 영어(원문 또는 수능형 변형) 문항에 대해 구문·논리·오답 분석 맞춤 해설을 생성한다."""
    client = _build_client(api_key, timeout_seconds=45)
    stem_txt = rewritten.rewritten_stem if rewritten is not None else (question.passage_text or question.question_text)
    choices_txt = (
        "\n".join(f"{idx + 1}) {c}" for idx, c in enumerate(rewritten.rewritten_choices))
        if (rewritten is not None and rewritten.rewritten_choices)
        else ""
    )
    truth_ans = rewritten.preserved_answer if rewritten is not None else question.answer
    q_type = (
        getattr(rewritten, "target_question_type", "")
        if rewritten is not None
        else (getattr(question, "english_question_type", "") or question.unit_name)
    )
    base_exp = rewritten.pdf_grounded_explanation if rewritten is not None else question.official_explanation

    system_inst = (
        "당신은 수능 영어 1등급 전문 해설 강사입니다.\n"
        "학생이 방금 푼 영어 문항에 대해 다음 구조의 가독성 높은 마크다운 해설을 작성하세요:\n"
        "1. **핵심 논지 & 구문 독해**: 지문의 주제와 정답 단서가 되는 핵심 영어 문장 직독직해·구문 분석\n"
        "2. **정답 도출 근거**: 본문의 어떤 단서/논리/어법 규칙 때문에 정답이 도출되는지 명확히 설명\n"
        "3. **①~⑤ 선지별 정오답 분석**: 학생이 선택한 답안이 오답이면 왜 헷갈렸는지 짚어주고, 각 번호를 하나씩 분석\n"
        "4. **필수 암기 어휘·표현**: 지문 핵심 어휘 5개 정리"
    )
    prompt = (
        f"[출처]: {source_reference or f'{question.exam_name} {question.q_num}번'}\n"
        f"[유형]: {q_type}\n"
        f"[정답]: {truth_ans}번 | [학생 선택 답안]: {user_answer or '미입력'}\n\n"
        f"[문항 본문]\n{stem_txt}\n\n"
        + (f"[선택지]\n{choices_txt}\n\n" if choices_txt else "")
        + (f"[기존 요약 해설 참고]\n{base_exp}\n\n" if base_exp else "")
        + "위 영어 문항에 대한 상세 맞춤 해설을 마크다운으로 작성하세요."
    )
    resp = _generate_content_with_fallback(
        client=client,
        model_name=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            temperature=0.2,
        ),
        fast_mode=True,
    )
    return format_explanation_text(getattr(resp, "text", "") or base_exp or "", for_markdown=True)


def chat_followup_with_english(
    api_key: str,
    model_name: str,
    question: QuestionRecord,
    rewritten: RewrittenQuestion | None,
    chat_history: list[dict[str, str]],
    user_query: str,
) -> str:
    """모드 B에서 영어 문항에 대한 학생의 추가 질문(구문, 문법, 어휘, 오답 이유 등)에 1:1 답변한다."""
    client = _build_client(api_key, timeout_seconds=45)
    stem_txt = rewritten.rewritten_stem if rewritten is not None else (question.passage_text or question.question_text)
    truth_ans = rewritten.preserved_answer if rewritten is not None else question.answer
    base_exp = rewritten.pdf_grounded_explanation if rewritten is not None else question.official_explanation
    history_lines = [
        f"{'학생' if m.get('role') == 'user' else '튜터'}: {m.get('content', '')}"
        for m in (chat_history or [])[-6:]
    ]
    history_block = "\n".join(history_lines)

    system_inst = (
        "당신은 친절하고 명쾌한 수능 영어 1:1 튜터입니다.\n"
        "아래 영어 지문과 문제, 해설을 바탕으로 학생의 추가 질문(문장 구조, 어법 원리, 어휘 뉘앙스, 선지 오답 이유 등)에 정확하고 이해하기 쉽게 답변하세요."
    )
    prompt = (
        f"[문항 본문]\n{stem_txt}\n\n"
        f"[정답]: {truth_ans}\n"
        f"[해설 참고]\n{base_exp}\n\n"
        + (f"[이전 대화]\n{history_block}\n\n" if history_block else "")
        + f"[학생의 질문]\n{user_query}\n"
    )
    resp = _generate_content_with_fallback(
        client=client,
        model_name=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            temperature=0.25,
        ),
        fast_mode=True,
    )
    return format_explanation_text(getattr(resp, "text", "") or "", for_markdown=True)


_ENGLISH_DIAGRAM_GUIDE = (
    "영어 지문의 핵심 논리 구조와 개념 관계를 한눈에 이해할 수 있도록 아래 12종 중 가장 적합한 `diagram_type`을 고르세요 "
    "(모두 한국어로 작성하며, 단순 박스 나열 `flow`만 반복하지 말고 지문 성격에 가장 직관적인 형식을 선택하세요):\n"
    "  ① `matrix` (항목별 비교 매트릭스표): 두 이론·관점·실험군·대상의 기준별 차이를 비교할 때 사용. "
    "`nodes`에 비교 대상 2~3개 제목을 넣고, `matrix_rows`에 기준(`criterion`)과 대상별 값(`values`) 3~5행을 채우세요.\n"
    "  ② `graph` (X-Y 비례/함수 그래프): 두 변수 간의 비례·반비례·임계점 관계를 설명할 때 사용. "
    "`graph_spec`(`x_label`, `y_label`, `curves`, `highlight_point`)을 채우고 `nodes`에 1~2개 해석을 넣으세요.\n"
    "  ③ `anatomy` (구성 요소·구조 모식도): 시스템·장치·인지 과정·현상의 구성 요소와 역할을 설명할 때 사용. "
    "`center_label`에 중심 개념명을, `nodes`에 3~4개 구성 요소를 넣으세요.\n"
    "  ④ `cycle` (원형 순환·피드백 루프): 상호작용, 악순환/선순환, 반복 과정에 사용. `center_label`과 `nodes` 3~4개, 짧은 `relation_labels`(6자 이내)를 넣으세요.\n"
    "  ⑤ `timeline` (시대·단계 변천 연표): 연구 발전, 역사적 변화, 시간 흐름에 사용. `nodes`에 시간순 3~4개 단계를 넣으세요.\n"
    "  ⑥ `venn` (공통점·차이점 벤다이어그램): 두 개념의 교집합(`center_label`)과 고유 특성을 구분할 때 사용.\n"
    "  ⑦ `scale` (대립·균형 저울도): 두 상충 주장·장단점·트레이드오프를 비교할 때 사용 (`nodes` 2개, `scale_tilt`: balanced/left/right).\n"
    "  ⑧ `fishbone` (피쉬본 어골도): 여러 복합 원인/요인(`nodes` 2~4개)이 하나의 결과/현상(`center_label`)으로 모이는 다발적 인과 분석에 사용.\n"
    "  ⑨ `quadrant` (2×2 사분면 분류): 두 개의 직교 기준(`quadrant_x_label`, `quadrant_y_label`)으로 4가지 유형(`nodes` 4개)을 분류할 때 사용.\n"
    "  ⑩ `decision_tree` (조건분기 결정 트리): 특정 조건(`center_label`)에 따라 `예 (O)` / `아니오 (X)`로 결과가 갈리는 논리 구조에 사용.\n"
    "  ⑪ `pyramid` (위계·계층 피라미드): 상하 위계나 단계별 중요도가 쌓이는 구조(`nodes` 최상위→최하위 3~4개)에 사용.\n"
    "  ⑫ `flow` (원리·논리 흐름도): 통념 제시 → 반박 → 실험/근거 → 결론처럼 순차적 인과·논증 흐름에 사용 (`nodes` 2~4개, 짧은 `relation_labels` 6자 이내).\n"
    "공통 규칙: 각 카드 `details`는 1~3개의 짧은 한국어 핵심 구절(26자 이내), `takeaway`는 1줄 한국어 핵심 정리.\n"
    "⚠️ **도식 근거 엄수(환각·외국어 혼입 금지)**: 지문 원문에 실제로 등장하는 개념·예시·용어만 그대로 사용하세요. "
    "원문에 없는 해부학·과학·학술 개념이나 상위 범주를 임의로 창작·변형하지 말고, 일본어·중국어 등 비한글 외국 문자 혼입 없이 순수 한국어로만 작성하세요."
)


def generate_english_passage_analysis(
    api_key: str,
    model_name: str,
    passage_text: str,
    source_label: str = "",
    orig_question_type: str = "",
    passage_topic: str = "",
    answer: str = "",
    choices: Sequence[str] | None = None,
    explanation: str = "",
    question_text: str = "",
    *,
    source_pdf_bytes: bytes | None = None,
    answer_pdf_bytes: bytes | None = None,
    q_num: int = 0,
    passage_group_id: str = "",
    direct_korean_translation: str = "",
    include_diagrams: bool = True,
) -> EnglishPassageAnalysisOutput:
    """
    영어 독해 지문 분석지를 생성한다 (v5):
    - [3. 문장별 정밀 구문 분석] 및 [공식 각주]: AI를 일절 사용하지 않고 저장된 문제지 PDF / DB 원문과
      해설지 PDF([해석] 섹션)에서 순수 Python/PyMuPDF로 결정론적 추출·정렬한다.
    - [1. 핵심 주제 · 논리 전개 흐름] 및 [2. 구조 도식]: Gemini API로 순수 한국어 요약·도식만 생성한다.
    - 기출 포인트 박스는 생성하지 않는다.
    """
    from src.pdf_parser.english_source_extractor import (
        build_english_sentence_items,
        extract_english_passage_from_problem_source,
        extract_english_question_translation_from_pdf,
    )

    _ = choices  # 하위 호환 인자 유지

    # 1단계 (AI 0%): 문제지 PDF / DB 원문에서 영어 원문·각주 추출, 해설지 PDF에서 한국어 해석 추출
    clean_passage, footnotes = extract_english_passage_from_problem_source(
        passage_text=passage_text,
        question_text=question_text,
        source_pdf_bytes=source_pdf_bytes,
        q_num=q_num,
        passage_group_id=passage_group_id,
        orig_question_type=orig_question_type,
        answer=answer,
    )
    if not clean_passage:
        raise ValueError("분석할 영어 지문 원문(passage_text)이 비어 있습니다.")

    direct_kor = (direct_korean_translation or "").strip()
    if direct_kor:
        kor_translation = direct_kor
        translation_source = "direct_input"
    elif answer_pdf_bytes and q_num > 0:
        kor_translation = extract_english_question_translation_from_pdf(
            answer_pdf_bytes=answer_pdf_bytes,
            q_num=q_num,
            passage_group_id=passage_group_id,
            orig_question_type=orig_question_type,
            answer=answer,
        )
        translation_source = "answer_pdf" if kor_translation else "none"
    else:
        kor_translation = ""
        translation_source = "none"

    sentence_items = build_english_sentence_items(clean_passage, kor_translation)

    # 2단계 (AI): 제목, 핵심 주제, 논리 전개 흐름, 구조 도식만 생성 (문장별 구문/번역 생성 금지)
    eff_model = (model_name or "").strip() or EXAM_TAGGING_MODEL
    client = _build_client(api_key, timeout_seconds=25)

    diagram_inst = (
        "4. `visual_diagram`: 지문의 논리·비교·인과·순환 구조를 한눈에 파악할 수 있는 도식 그림(`KoreanVisualDiagram`)을 반드시 순수 한국어로 작성.\n"
        f"{_ENGLISH_DIAGRAM_GUIDE}\n"
        if include_diagrams
        else "4. `visual_diagram`: 이번 요청에서는 도식을 생성하지 않으므로 반드시 `null`로 두세요.\n"
    )
    system_inst = (
        "당신은 대한민국 수능·내신 영어 1등급 전문 강사입니다.\n"
        "주어진 영어 독해 지문을 바탕으로, 학생이 글의 핵심 구조와 논리 흐름을 한눈에 파악할 수 있도록 "
        "`EnglishPassageSummaryDiagramOutput` JSON만 작성하세요.\n\n"
        "[출력 필드 규칙 — 순수 한국어 · 원문 근거 엄수]\n"
        "1. `title`: 지문의 핵심 소재와 주제를 담은 간결한 제목 (순수 한국어 중심 또는 영문 핵심구(한국어)).\n"
        "2. `core_theme`: 지문 전체의 핵심 주제·요지를 한국어 1~2문장으로 명쾌하게 요약.\n"
        "3. `structure_flow`: 글 전체의 논리 전개 흐름을 `도입 : ...\\n전개 : ...\\n심화/반전 : ...\\n결론 : ...` 형식의 단계별 줄바꿈으로 작성.\n"
        f"{diagram_inst}"
        "5. 문장별 구문독해(`/`·`[]`·`()`), 문장별 한국어 해석, 기출 포인트는 시스템이 별도로 처리하므로 절대 생성하지 마세요.\n"
        "6. 일본어·중국어 한자 등 비한글 외국 문자를 한국어 출력 필드 어디에도 절대 섞지 마세요."
    )

    exp_hint = f"\n[공식 해설 참고]\n{explanation.strip()[:1000]}" if (explanation or "").strip() else ""
    fn_hint = (
        "\n[공식 각주 참고]\n" + "\n".join(f"- {fn}" for fn in footnotes)
        if footnotes
        else ""
    )
    base_prompt = (
        f"[지문 출처] {source_label or '-'} (원래 유형: {orig_question_type or '-'})\n"
        f"[참고 주제] {passage_topic or '-'}"
        f"{exp_hint}{fn_hint}\n\n"
        f"[영어 지문 원문]\n{clean_passage}\n\n"
        "위 영어 지문의 핵심 제목, 핵심 주제, 논리 전개 흐름"
        + (" 및 구조 도식" if include_diagrams else "")
        + "을 JSON으로 작성하세요."
    )

    prompt = base_prompt
    summary_out: EnglishPassageSummaryDiagramOutput | None = None
    for attempt in range(2):
        summary_out = _generate_structured_with_fallback(
            client=client,
            model_name=eff_model,
            contents=prompt,
            schema_cls=EnglishPassageSummaryDiagramOutput,
            config=types.GenerateContentConfig(
                system_instruction=system_inst,
                response_mime_type="application/json",
                response_schema=EnglishPassageSummaryDiagramOutput,
                temperature=0.15 if attempt == 0 else 0.25,
                max_output_tokens=4096,
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.MINIMAL,
                    include_thoughts=False,
                ),
            ),
            fast_mode=True,
            max_parse_retries=2,
        )
        issues = _validate_english_summary_diagram_output(
            summary_out,
            require_diagram=include_diagrams,
        )
        if not issues:
            break
        if attempt == 0:
            prompt = (
                base_prompt
                + "\n\n[⚠️ 직전 출력에서 오류가 감지되었습니다 — 아래 항목을 수정하여 다시 생성하세요]\n"
                + "\n".join(f"- {iss}" for iss in issues)
            )

    assert summary_out is not None
    return EnglishPassageAnalysisOutput(
        title=(summary_out.title or "").strip() or (passage_topic or source_label or "영어 지문 분석"),
        core_theme=(summary_out.core_theme or "").strip(),
        structure_flow=(summary_out.structure_flow or "").strip(),
        visual_diagram=summary_out.visual_diagram if include_diagrams else None,
        sentence_analyses=sentence_items,
        exam_point_title="",
        exam_point_note="",
        footnotes=footnotes,
        translation_source=translation_source,
        is_source_restored=True,
        analysis_version=CURRENT_ENGLISH_ANALYSIS_VERSION,
    )


def generate_english_visual_diagram(
    *,
    api_key: str,
    model_name: str,
    passage_text: str,
    title: str = "",
    core_theme: str = "",
) -> KoreanVisualDiagram:
    """기존에 분석된 영어 지문에 대해 이해를 돕는 도식 그림 스펙만 보완 생성한다."""
    clean_passage = _preprocess_english_source_passage(passage_text)
    if not clean_passage:
        raise ValueError("도식을 생성할 영어 지문 원문이 비어 있습니다.")

    eff_model = (model_name or "").strip() or EXAM_TAGGING_MODEL
    client = _build_client(api_key, timeout_seconds=60)
    system_inst = (
        "당신은 수능 영어 일타 강사입니다. 주어진 영어 지문 원문만을 100% 근거로, "
        "학생이 핵심 논리·비교·인과·구조를 그림으로 한눈에 이해할 수 있는 도식 명세(`KoreanVisualDiagram`) JSON을 순수 한국어로 작성하세요.\n"
        "원문에 없는 해부학·과학·학술 개념을 창작하지 말고, 일본어·중국어 등 비한글 외국 문자를 절대 혼입하지 마세요.\n"
        + _ENGLISH_DIAGRAM_GUIDE
    )
    base_prompt = (
        f"[제목/주제] {title or '-'} / {core_theme or '-'}\n\n"
        f"[영어 지문 원문]\n{clean_passage[:4500]}\n\n"
        "위 영어 지문의 핵심 논리 구조를 시각화한 한국어 도식 JSON을 작성하세요."
    )
    prompt = base_prompt
    diag: KoreanVisualDiagram | None = None
    for attempt in range(2):
        diag = _generate_structured_with_fallback(
            client=client,
            model_name=eff_model,
            contents=prompt,
            schema_cls=KoreanVisualDiagram,
            config=types.GenerateContentConfig(
                system_instruction=system_inst,
                response_mime_type="application/json",
                response_schema=KoreanVisualDiagram,
                temperature=0.15 if attempt == 0 else 0.3,
                max_output_tokens=2048,
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.MINIMAL,
                    include_thoughts=False,
                ),
            ),
            fast_mode=True,
            max_parse_retries=2,
        )
        diag_blob = "\n".join(
            [
                diag.title or "",
                diag.center_label or "",
                diag.takeaway or "",
                *(diag.relation_labels or []),
                *[nd.title or "" for nd in (diag.nodes or [])],
                *[d for nd in (diag.nodes or []) for d in (nd.details or [])],
            ]
        )
        foreign_hits = _detect_foreign_script_contamination(diag_blob)
        if not foreign_hits:
            break
        if attempt == 0:
            prompt = (
                base_prompt
                + f"\n\n[⚠️ 비한글 외국 문자({', '.join(foreign_hits)})가 감지되었습니다. 순수 한국어로만 다시 작성하세요.]"
            )
    assert diag is not None
    return diag
