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
    _build_english_transform_rule,
    _detect_foreign_script_contamination,
    _ensure_korean_instruction_prefix,
    _has_exam_structure_markers,
    _has_incomplete_english_blank,
    _is_irrelevant_answer_from_source,
    _needs_english_passage_restoration,
    _preprocess_english_source_passage,
    _rebuild_insertion_stem_from_source,
    _reconcile_english_insertion_answer,
    _reconcile_english_order_answer,
    _reconcile_english_passage_with_question_text,
    _reconcile_english_summary_answer,
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
) -> RewrittenQuestion:
    """
    원문 영어 독해 지문을 지정된 수능 독해 유형(1개 유형 규칙만 동적 주입)으로 변형·출제한다.
    비변형 유형(요지·주제·제목·요약·삽입·빈칸·함축)은 지문 재출력을 생략하고 파이썬이 원문을 조립하며,
    검증 통과 문항은 storage DB 풀(Pool)에 캐싱해 동일 요청 시 0원으로 즉시 반환한다.
    """
    clean_passage = _preprocess_english_source_passage(source_passage)
    if not clean_passage:
        raise ValueError("변형할 영어 지문(source_passage)이 비어 있습니다.")

    eff_model = (model_name or "").strip() or EXAM_TAGGING_MODEL
    if _has_incomplete_english_blank(clean_passage):
        clean_passage = _restore_incomplete_source_passage(api_key, eff_model, clean_passage)
    canonical_type, type_rule_block = _build_english_transform_rule(
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

    target_thinking_level = (
        types.ThinkingLevel.LOW
        if (use_precision_mode and not _is_flash_lite_model(eff_model))
        else types.ThinkingLevel.MINIMAL
    )

    client = _build_client(api_key, timeout_seconds=90)
    system_inst = (
        "수능·평가원 영어 출제위원으로서 지정된 **[목표 유형 1개]**에 맞춰 `RewrittenQuestion` JSON 1건을 생성하세요.\n\n"
        f"[유형: {canonical_type} | 난이도: {difficulty}]\n"
        f"{type_rule_block}\n\n"
        "[출력 및 토큰 절감 규칙]\n"
        "1. `rewritten_stem`: 위 '출력 절감 규칙'이 지정된 유형은 원문 재출력 없이 지정값만 반환하고, 지문 변형 유형만 발문+지문을 반환하세요.\n"
        "2. `rewritten_choices`: 정확히 5개 선지 반환(번호 ①~⑤ 제외).\n"
        "3. `preserved_answer`: 유일한 정답 `'1'`~`'5'` 단일 숫자 문자열.\n"
        "4. `pdf_grounded_explanation`: 정답 근거와 오답 핵심만 담아 **2문장(공백 포함 120자) 이내**로 작성.\n"
        "5. `replaced_terms_map`: 핵심 어휘 2개만 `'영단어 → 뜻'` 형식으로 기록.\n"
        "6. `self_verification_note`: 20자 이내 단문."
    )

    prev_feedback_block = ""
    if previous_attempt is not None:
        prev_feedback_block = (
            "\n[이전 생성본 (재변환 요청 — 다른 포인트/오답 구성으로 새롭게 출제)]\n"
            f"- 이전 지문/발문: {previous_attempt.rewritten_stem[:500]}\n"
            f"- 이전 정답: {previous_attempt.preserved_answer}\n"
        )

    passage_for_prompt = clean_passage
    if "순서" in canonical_type:
        indexed_sents = _split_english_sentences(clean_passage)
        if len(indexed_sents) >= 4:
            passage_for_prompt = "\n".join(f"[S{i}] {s}" for i, s in enumerate(indexed_sents))

    base_prompt = (
        f"[출처]: {source_label or '영어 기출 독해 지문'}\n"
        f"[목표 수능 유형]: {canonical_type} | [난이도]: {difficulty}\n"
        f"{prev_feedback_block}\n"
        f"[원문 영어 지문]\n{passage_for_prompt}\n\n"
        "위 원문 영어 지문을 바탕으로 목표 유형의 출제 규칙만 엄격히 준수하여 변형 문제와 2줄 이내 핵심 정답 근거를 JSON으로 생성하세요."
    )

    prompt = base_prompt
    rw: RewrittenQuestion | None = None
    # 출력에는 지문 전체(+선지)가 포함되므로 지문 길이에 비례해 출력 토큰 상한을 잡는다 (JSON 중간 잘림 방지).
    base_out_tokens = min(8192, max(3072, int(len(clean_passage) / 2.0) + 1800))
    for attempt in range(3):
        out_tokens = base_out_tokens if attempt == 0 else min(12288, int(base_out_tokens * 1.6))
        try:
            parsed_rw = _generate_structured_with_fallback(
                client=client,
                model_name=eff_model,
                contents=prompt,
                schema_cls=RewrittenQuestion,
                config=types.GenerateContentConfig(
                    system_instruction=system_inst,
                    response_mime_type="application/json",
                    response_schema=RewrittenQuestion,
                    temperature=0.35 if previous_attempt is not None else 0.25,
                    max_output_tokens=out_tokens,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=target_thinking_level,
                        include_thoughts=False,
                    ),
                ),
                fast_mode=not use_precision_mode,
                max_parse_retries=2,
            )
        except Exception:
            if attempt == 0:
                continue
            if rw is not None:
                break
            raise
        rw = parsed_rw
        # 번호형 선택지(어법/무관한 문장/문장 삽입 등)에서 선택지가 비어 있으면 ①~⑤ 자동 보완
        if not rw.rewritten_choices and any(k in canonical_type for k in ("어법", "무관", "삽입", "밑줄")):
            rw.rewritten_choices = ["①", "②", "③", "④", "⑤"]
        # 선지 텍스트 앞에 중복된 ①~⑤가 붙어있고 뒤에 텍스트가 있으면 정리
        cleaned_choices: list[str] = []
        for ch in rw.rewritten_choices or []:
            c_str = (ch or "").strip()
            if len(c_str) > 1 and c_str[0] in _CIRCLED_DIGIT_TO_NUM:
                c_str = c_str[1:].strip()
            cleaned_choices.append(c_str)
        rw.rewritten_choices = cleaned_choices[:5]
        rw.preserved_answer = _normalize_extracted_answer(rw.preserved_answer)
        _assemble_english_slim_stem(rw, canonical_type, clean_passage)
        if "순서" in canonical_type:
            _reconcile_english_order_answer(rw)
        elif "삽입" in canonical_type:
            if not _rebuild_insertion_stem_from_source(rw, clean_passage):
                _reconcile_english_insertion_answer(rw)
        elif "요약" in canonical_type:
            _reconcile_english_summary_answer(rw)

        issues = _validate_english_transformed_output(
            rw,
            canonical_type,
            vocab_style,
            source_passage=clean_passage,
        )
        if not issues:
            break
        if attempt < 2:
            fallback_hint = ""
            if "빈칸" in canonical_type or "함축" in canonical_type:
                fallback_hint = "\n- 구절 치환 매칭 실패 시 이번 재시도에서는 지문 전체와 빈칸/밑줄을 직접 완성해 `rewritten_stem`에 반환하세요."
            elif "요약" in canonical_type:
                fallback_hint = "\n- `[요약문]` 섹션에는 선지 단어만 나열하지 말고 반드시 `(A) ________ ... (B) ________.` 형태의 완전한 요약문 1문장을 작성하세요."
            elif any(k in canonical_type for k in ("어법", "어휘", "무관", "순서")):
                fallback_hint = "\n- 슬림 포맷 조립 실패 시 이번 재시도에서는 원문 전체를 포함해 완성된 문제 지문을 `rewritten_stem`에 직접 반환하세요."
            prompt = (
                base_prompt
                + "\n\n[⚠️ 직전 출력 형식 오류 — 반드시 아래 누락 사항을 모두 수정하여 다시 생성하세요]\n"
                + "\n".join(f"- {iss}" for iss in issues)
                + fallback_hint
            )

    assert rw is not None
    norm_stem = normalize_english_exam_text(rw.rewritten_stem)
    if any(k in canonical_type for k in ("어법", "어휘")) and "<u" in norm_stem.lower():
        rw.rewritten_stem = norm_stem
    else:
        rw.rewritten_stem = strip_english_embedded_choices(norm_stem)
    if "빈칸" in canonical_type:
        rw.rewritten_stem = re.sub(r"[ \t]*\(\s*[①②③④⑤]\s*\)[ \t]*", " ", rw.rewritten_stem)
        rw.rewritten_stem = re.sub(r"[ \t]*[①②③④⑤][ \t]*", " ", rw.rewritten_stem)
        rw.rewritten_stem = re.sub(r"[ \t]{2,}", " ", rw.rewritten_stem).strip()
    rw.rewritten_stem = _ensure_korean_instruction_prefix(rw.rewritten_stem, canonical_type, vocab_style)
    rw.pdf_grounded_explanation = format_explanation_text(rw.pdf_grounded_explanation)
    rw.preserved_answer = _normalize_extracted_answer(rw.preserved_answer) or "1"
    if len(rw.rewritten_choices) < 5:
        default_syms = ["①", "②", "③", "④", "⑤"]
        while len(rw.rewritten_choices) < 5:
            rw.rewritten_choices.append(default_syms[len(rw.rewritten_choices)])
    rw.target_question_type = canonical_type
    rw.subject_domain = "영어"
    if (
        not issues
        and cache_key
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
    client = _build_client(api_key, timeout_seconds=90)

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
