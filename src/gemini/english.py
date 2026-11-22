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
    _apply_lang_standard_validation,
    _build_client,
    _build_lang_ncic_grounding_section,
    _generate_content_with_fallback,
    _is_flash_lite_model,
    _normalize_extracted_answer,
    _parse_structured,
    optimize_image_for_vision,
    prepare_pdf_part,
)
from src.gemini.english_validators import (
    _AI_META_MONOLOGUE_RE,
    _DANGLING_TO_END_RE,
    _assemble_english_slim_stem,
    _build_english_transform_rule,
    _ensure_korean_instruction_prefix,
    _has_incomplete_english_blank,
    _is_irrelevant_answer_from_source,
    _preprocess_english_source_passage,
    _rebuild_insertion_stem_from_source,
    _reconcile_english_insertion_answer,
    _reconcile_english_order_answer,
    _reconcile_english_summary_answer,
    _split_english_sentences,
    _validate_english_transformed_output,
)
from src.models import (
    BatchEnglishExamExtractionOutput,
    EnglishPassageAnalysisOutput,
    EnglishQuestionExtractionItem,
    EnglishSentenceAnalysisItem,
    EnglishSingleExtractionOutput,
    KoreanVisualDiagram,
    QuestionRecord,
    RewrittenQuestion,
    normalize_english_group_id,
)
from src.text_formatter import (
    format_explanation_text,
    normalize_english_exam_text,
    strip_english_embedded_choices,
)


def _infer_english_group_id(q_num: int, raw_group: str = "") -> str:
    """41~42 / 43~45 장문 세트만 그룹으로 인정하고, 그 외 번호([31~34] 등 발문 헤더)는 항상 개별 지문으로 취급한다."""
    return normalize_english_group_id(q_num, raw_group)


def analyze_english_exam_pdf_batch_with_gemini(
    api_key: str,
    model_name: str,
    prob_pdf_bytes: bytes,
    target_q_nums: list[int],
    exam_name: str,
    answer_snippets: dict[int, str] | None = None,
    answer_pdf_part: Any | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> dict[int, EnglishQuestionExtractionItem]:
    """
    영어 모의고사 문제지 PDF(및 해설지)를 일괄 분석하여
    1~17번(듣기)은 경량 메타데이터로, 18~45번(독해)은 영어 지문 원문(passage_text)·원래 유형·주제·핵심 어휘로 추출한다.
    41~42번, 43~45번 세트 문항은 공통 지문이 유실되지 않도록 그룹 내 자동 동기화한다.
    """
    ordered_q = sorted({int(q) for q in target_q_nums})
    if not ordered_q:
        return {}

    client = _build_client(api_key, timeout_seconds=120)
    if status_callback is not None:
        status_callback(f"📄 영어 모의고사 PDF({len(ordered_q)}문항) 일괄 분석 준비 중...")

    prob_part = prepare_pdf_part(
        client,
        prob_pdf_bytes,
        display_name=f"{exam_name}_english_problem.pdf",
    )

    snippets_map = answer_snippets or {}
    snippet_lines: list[str] = []
    for q in ordered_q:
        sn = (snippets_map.get(q) or "").strip()
        if sn:
            snippet_lines.append(f"[{q}번 해설 발췌] {sn[:650]}")

    ncic_section, ncic_rule = _build_lang_ncic_grounding_section("국어", exam_name)
    hint_section = (
        "\n[해설지 문항별 발췌]\n" + "\n".join(snippet_lines) + "\n"
        if snippet_lines
        else ""
    )

    ncic_section, ncic_rule = _build_lang_ncic_grounding_section("영어", exam_name)
    system_inst = (
        "당신은 대한민국 수능·평가원·교육청 영어 영역(제3교시) 기출 분석 전문가입니다.\n"
        "영어 과목은 학년 제한(min_grade)을 두지 않으며, 아래 규칙에 따라 요청된 모든 문항(1~45번)을 JSON(questions)으로 추출하세요.\n\n"
        "[추출 규칙]\n"
        "1. `english_section`: 1~17번은 반드시 `'듣기'`, 18~45번은 반드시 `'독해'`로 기록하세요.\n"
        "2. **듣기(1~17번) 경량 추출**:\n"
        "   - `passage_text`는 빈 문자열(`\"\"`)로 비워 토큰을 절약하세요.\n"
        "   - `english_question_type`에 세부 유형(예: '담화 목적', '대화 의견', '화자 관계', '그림 내용 일치', '할 일', '지불 금액', '이유', '언급되지 않은 것', '담화 내용 일치', '도표 선택', '짧은 대화 응답', '긴 대화 응답', '상황에 적절한 말', '세트 듣기')을 기록하세요.\n"
        "3. **독해(18~45번) 정밀 지문 추출**:\n"
        "   - `english_question_type`은 다음 표준 유형명 중 하나로 기록하세요: "
        "'목적', '심경·분위기', '주장', '함축적 의미 추론', '요지 파악', '주제 파악', '제목 추론', '도표·실용문', '내용 일치·불일치', "
        "'어법', '어휘', '빈칸 추론', '무관한 문장', '글의 순서', '문장 삽입', '요약문 완성', '장문 독해(41~42)', '장문 독해(43~45)'.\n"
        "   - `passage_text`: 한국어 발문(예: '다음 글의 요지로...')과 선택지(①~⑤)를 제외한 **영어 본문 지문 전체**를 빠짐없이 원문 그대로 추출하세요. "
        "(글의 순서 문항은 주어진 글과 (A)(B)(C) 본문을 자연스럽게 모두 포함하고, 문장 삽입 문항도 주어진 문장과 본문을 모두 포함하세요.)\n"
        "   - **장문 세트(41~42번, 43~45번)에만** `passage_group_id`를 `'41~42'` 또는 `'43~45'`로 표기하고 공통 장문 지문 전체(43~45번은 (A)~(D) 전체 단락)를 수록하세요. **그 외 모든 문항(예: 31~34, 36~37, 38~39 등 발문 헤더 범위 포함)은 문항마다 지문이 각각 다르므로 `passage_group_id`를 반드시 빈 문자열로 두고 해당 번호 문항의 지문만 추출하세요.**\n"
        "   - `passage_topic`: 지문의 핵심 소재·주제를 한국어 1줄로 요약하세요.\n"
        "   - `key_vocabulary`: 지문 내 주요 어휘 3~6개를 `'단어: 한국어 뜻'` 형식으로 나열하세요.\n"
        "4. `official_explanation`과 `answer`: 해설지 PDF(특히 1페이지 상단 1~45번 빠른 정답표)와 문항별 해설을 대조하여 핵심 풀이 요약과 공식 정답(`'1'`~`'5'`)을 정확히 추출하세요. "
        "**특히 41번(제목 추론)과 42번(밑줄 (a)~(e) 어휘 적절성), 그리고 43번(순서 배열)·44번(지칭 추론 (a)~(e))·45번(내용 일치) 세트 문항의 정답과 해설이 서로 뒤바뀌지 않도록 문항 번호별 발문과 정답표를 반드시 1:1 확인하세요.**"
        + ncic_rule
    )

    q_list_str = ", ".join(str(q) for q in ordered_q)
    prompt = (
        f"시험 회차: {exam_name}\n"
        f"추출 대상 문항 번호(총 {len(ordered_q)}문항): {q_list_str}\n"
        f"{ncic_section}"
        f"{hint_section}\n"
        "위 영어 문제지 PDF(및 해설지)에서 지정된 모든 문항 번호를 빠짐없이 추출하여 JSON(questions)으로 반환하세요."
    )

    contents: list[Any] = [prob_part]
    if answer_pdf_part is not None:
        contents.append(answer_pdf_part)
    contents.append(prompt)

    if status_callback is not None:
        status_callback(f"⚡ {model_name} 모델로 영어 PDF({len(ordered_q)}문항) 일괄 분석 중...")

    resp = _generate_content_with_fallback(
        client=client,
        model_name=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=BatchEnglishExamExtractionOutput,
            temperature=0.1,
            max_output_tokens=32768,
        ),
        fast_mode=True,
        status_callback=status_callback,
    )
    parsed = _parse_structured(resp, BatchEnglishExamExtractionOutput)

    out: dict[int, EnglishQuestionExtractionItem] = {}
    for item in parsed.questions:
        if item.q_num in ordered_q and item.q_num not in out:
            item.english_section = "듣기" if item.q_num <= 17 else "독해"
            item.passage_group_id = _infer_english_group_id(item.q_num, item.passage_group_id)
            item.answer = _normalize_extracted_answer(item.answer)
            if item.q_num <= 17:
                item.passage_text = ""
            _apply_lang_standard_validation(item, "영어")
            out[item.q_num] = item

    # 41~42, 43~45 세트 문항 간 공통 지문(passage_text) 및 주제 동기화
    group_best_passage: dict[str, str] = {}
    group_best_topic: dict[str, str] = {}
    for item in out.values():
        gid = item.passage_group_id
        if not gid:
            continue
        txt = (item.passage_text or "").strip()
        if len(txt) > len(group_best_passage.get(gid, "")):
            group_best_passage[gid] = txt
        top = (item.passage_topic or "").strip()
        if len(top) > len(group_best_topic.get(gid, "")):
            group_best_topic[gid] = top

    for item in out.values():
        gid = item.passage_group_id
        if gid and gid in group_best_passage and group_best_passage[gid]:
            if len((item.passage_text or "").strip()) < len(group_best_passage[gid]) * 0.7:
                item.passage_text = group_best_passage[gid]
        if gid and not (item.passage_topic or "").strip() and gid in group_best_topic:
            item.passage_topic = group_best_topic[gid]

    return out


def analyze_english_question_with_gemini(
    api_key: str,
    model_name: str,
    crop_png_bytes: bytes,
    q_num: int,
    exam_name: str,
    answer_pdf_part: Any | None = None,
    answer_text_hint: str = "",
    question_text_hint: str = "",
    shared_passage_hint: str = "",
) -> EnglishQuestionExtractionItem:
    """일괄 추출에서 누락된 단일 영어 문항을 크롭 이미지 + 추출 텍스트 + 해설로 보완 추출한다."""
    client = _build_client(api_key, timeout_seconds=DEFAULT_API_TIMEOUT_SECONDS)
    opt_png = optimize_image_for_vision(crop_png_bytes)
    img_part = types.Part.from_bytes(data=opt_png, mime_type="image/png")
    is_listening = q_num <= 17

    ncic_section, ncic_rule = _build_lang_ncic_grounding_section("영어", exam_name)
    system_inst = (
        "당신은 대한민국 수능·모의고사 영어 기출 문항 분석가입니다.\n"
        f"- 이 문항({q_num}번)은 {'듣기(1~17번)' if is_listening else '독해(18~45번)'} 문항입니다.\n"
        "- 듣기 문항이면 `english_section='듣기'`, `passage_text=''`로 두고 유형·주제·정답·해설만 추출하세요.\n"
        "- 독해 문항이면 `english_section='독해'`, 한국어 발문과 선택지 ①~⑤를 제외한 순수 영어 본문 전체를 `passage_text`에 빠짐없이 추출하고, "
        "원래 유형(`english_question_type`), 주제(`passage_topic`), 핵심 어휘(`key_vocabulary`), 정답(`answer`), 해설(`official_explanation`)을 JSON으로 반환하세요."
        + ncic_rule
    )

    hint_parts: list[str] = [f"시험 회차: {exam_name}", f"문항 번호: {q_num}번"]
    if ncic_section:
        hint_parts.append(ncic_section.strip())
    if shared_passage_hint and shared_passage_hint.strip():
        hint_parts.append(f"[세트 문항 공통 지문 텍스트]\n{shared_passage_hint.strip()[:3000]}")
    if question_text_hint and question_text_hint.strip():
        hint_parts.append(f"[문항 추출 텍스트]\n{question_text_hint.strip()[:2500]}")
    if answer_text_hint and answer_text_hint.strip():
        hint_parts.append(f"[해설지 {q_num}번 발췌]\n{answer_text_hint.strip()[:1200]}")
    hint_parts.append("위 영어 문항을 분석해 JSON 스키마로 반환하세요.")
    prompt = "\n\n".join(hint_parts)

    contents: list[Any] = [img_part]
    if answer_pdf_part is not None and not (answer_text_hint and answer_text_hint.strip()):
        contents.append(answer_pdf_part)
    contents.append(prompt)

    resp = _generate_content_with_fallback(
        client=client,
        model_name=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=EnglishSingleExtractionOutput,
            temperature=0.1,
        ),
        fast_mode=True,
    )
    single = _parse_structured(resp, EnglishSingleExtractionOutput)
    sec: Any = "듣기" if is_listening else "독해"
    grp_id = _infer_english_group_id(q_num, single.passage_group_id)
    p_text = "" if is_listening else (single.passage_text.strip() or shared_passage_hint.strip())
    _apply_lang_standard_validation(single, "영어")
    return EnglishQuestionExtractionItem(
        q_num=q_num,
        english_section=sec,
        english_question_type=single.english_question_type.strip() or ("듣기" if is_listening else "독해"),
        passage_group_id=grp_id,
        passage_text=p_text,
        passage_topic=single.passage_topic.strip(),
        key_vocabulary=list(single.key_vocabulary or []),
        official_explanation=single.official_explanation.strip(),
        answer=_normalize_extracted_answer(single.answer),
        standard_code=single.standard_code,
        standard_codes=list(single.standard_codes or []),
        needs_review=single.needs_review,
        review_notes=list(single.review_notes or []),
    )


_RESTORED_PASSAGE_CACHE: dict[str, str] = {}
_RESTORED_PASSAGE_LOCK = threading.RLock()


def _restore_incomplete_source_passage(
    api_key: str,
    model_name: str,
    passage: str,
) -> str:
    """빈칸 기출 원문에서 빠진 구절만 문맥에 맞게 복원해 완전한 영어 지문을 만든다 (해시 캐싱)."""
    if not api_key or not _has_incomplete_english_blank(passage):
        return passage
    p_hash = hashlib.sha256(passage.encode("utf-8")).hexdigest()[:24]
    with _RESTORED_PASSAGE_LOCK:
        cached = _RESTORED_PASSAGE_CACHE.get(p_hash)
        if cached:
            return cached
    try:
        client = _build_client(api_key, timeout_seconds=45)
        prompt = (
            "Below is an English reading passage extracted from a blank-filling exam question, "
            "so one phrase or clause is missing (e.g., 'there isn't much .', 'has to .', '____', or a noun phrase without a verb).\n"
            "Restore ONLY the missing word/phrase/clause naturally so every sentence is grammatically complete. "
            "Keep ALL other words 100% identical. Output ONLY the restored English passage with no commentary.\n\n"
            f"{passage}"
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
        if restored and len(restored) >= int(len(passage) * 0.8) and not _has_incomplete_english_blank(restored):
            with _RESTORED_PASSAGE_LOCK:
                if len(_RESTORED_PASSAGE_CACHE) > 512:
                    _RESTORED_PASSAGE_CACHE.clear()
                _RESTORED_PASSAGE_CACHE[p_hash] = restored
            return restored
    except Exception:
        pass
    return passage


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
        resp = _generate_content_with_fallback(
            client=client,
            model_name=eff_model,
            contents=prompt,
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
        )
        try:
            parsed_rw = _parse_structured(resp, RewrittenQuestion)
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
    "공통 규칙: 각 카드 `details`는 1~3개의 짧은 한국어 핵심 구절(26자 이내), `takeaway`는 1줄 한국어 핵심 정리."
)


def generate_english_passage_analysis(
    api_key: str,
    model_name: str,
    passage_text: str,
    source_label: str = "",
    orig_question_type: str = "",
    passage_topic: str = "",
) -> EnglishPassageAnalysisOutput:
    """
    영어 독해 지문을 내신·수능형 정밀 지문 분석지(핵심 주제, 논리 전개 흐름, 이해용 시각 도식,
    문장별 슬래시(/) 끊어읽기 + [절]/(수식어) 구문독해 + 주제문 ★ 표시 + 해석 + 구문/어법 해설)로 구조화 분석한다.
    """
    clean_passage = _preprocess_english_source_passage(passage_text)
    if not clean_passage:
        raise ValueError("분석할 영어 지문 원문(passage_text)이 비어 있습니다.")

    eff_model = (model_name or "").strip() or EXAM_TAGGING_MODEL
    if _has_incomplete_english_blank(clean_passage):
        clean_passage = _restore_incomplete_source_passage(api_key, eff_model, clean_passage)

    sents = _split_english_sentences(clean_passage)
    numbered_sents_block = (
        "\n".join(f"{idx}. {s}" for idx, s in enumerate(sents, start=1))
        if sents
        else clean_passage
    )

    client = _build_client(api_key, timeout_seconds=90)
    system_inst = (
        "당신은 대한민국 대치동·수능 영어 1등급 전문 구문독해 일타 강사입니다.\n"
        "주어진 영어 독해 지문을 바탕으로, 학생이 지문의 논리 구조와 모든 문장의 구문을 완벽히 이해할 수 있는 "
        "**정밀 지문 분석지**를 JSON(`EnglishPassageAnalysisOutput`)으로 작성하세요.\n\n"
        "[절대 준수 규칙 — 반드시 아래 4가지 핵심 파트만 정확히 작성하고 불필요한 어휘/단어장 목록이나 출제포인트 필드는 만들지 마세요]\n"
        "1. `title`: 지문의 핵심 소재와 주제를 담은 간결한 제목 (한국어 제목 또는 영문 핵심구(한국어)).\n"
        "2. `core_theme`: 지문 전체의 핵심 주제·요지를 한국어 1~2문장으로 명쾌하게 요약.\n"
        "3. `structure_flow`: 글 전체의 논리 전개 흐름을 `도입 : ...\\n전개 : ...\\n심화/반전 : ...\\n결론 : ...` 형식의 단계별 줄바꿈으로 작성.\n"
        "4. `visual_diagram`: 지문의 논리·비교·인과·순환 구조를 한눈에 파악할 수 있는 도식 그림(`KoreanVisualDiagram`)을 반드시 한국어로 작성.\n"
        f"{_ENGLISH_DIAGRAM_GUIDE}\n"
        "5. `sentence_analyses`: 지문의 1번 문장부터 마지막 문장까지 단 한 문장도 빠뜨리지 말고 순서대로(`sentence_num` 1..N) 분석하세요.\n"
        "   - `is_topic_sentence`: 지문의 핵심 주장·결론이 담긴 **가장 중요한 주제문(보통 1~2문장)**에만 `true`, 나머지는 `false`로 표시.\n"
        "   - `chunked_english`: 원문 영어 문장의 단어를 절대 생략·변경·축약하지 말고 100% 그대로 유지한 상태에서, "
        "의미 단위 끊어읽기 슬래시(` / `)와 구문 괄호를 직접 삽입하세요.\n"
        "     * **절(명사절·관계사절·부사절)**은 대괄호 `[ ... ]`로 감싸기 (예: `[that we perceive reality]`, `[which influences behavior]`, `[When people face uncertainty]`).\n"
        "     * **구 수식어(전치사구·분사구·to부정사 형용사/부사적 수식구·삽입구)**는 소괄호 `( ... )`로 감싸기 (예: `(in modern society)`, `(guided by intuition)`).\n"
        "     * **의미 단위/주어부·동사부·목적어 경계**에는 ` / ` 슬래시 넣기 (예: `The ability [to adapt (to rapid change)] / determines / [whether an organization thrives (in the long run)].`).\n"
        "     * 과도한 중첩으로 읽기가 망가지지 않도록 핵심 구문 경계 위주로 깔끔하게 끊고 묶으세요.\n"
        "   - `korean_translation`: 문맥을 살린 정확하고 자연스러운 한국어 직역·의역 조화 해석.\n"
        "   - `syntax_notes`: 문장별 문법 설명은 출력하지 않으므로 반드시 빈 배열 `[]`로 두세요."
    )

    prompt = (
        f"[지문 출처] {source_label or '-'} (원래 유형: {orig_question_type or '-'})\n"
        f"[참고 주제] {passage_topic or '-'}\n\n"
        f"[지문 원문]\n{clean_passage}\n\n"
        f"[문장 분리 참고 (총 {len(sents) if sents else 1}문장 — 번호 순서대로 누락 없이 모두 분석)]\n"
        f"{numbered_sents_block}\n\n"
        "위 영어 지문에 대한 정밀 구문·논리 분석지 JSON을 작성하세요."
    )

    resp = _generate_content_with_fallback(
        client=client,
        model_name=eff_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=EnglishPassageAnalysisOutput,
            temperature=0.2,
            max_output_tokens=8192,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL,
                include_thoughts=False,
            ),
        ),
        fast_mode=True,
    )
    out = _parse_structured(resp, EnglishPassageAnalysisOutput)
    if out.sentence_analyses:
        out.sentence_analyses = sorted(out.sentence_analyses, key=lambda x: x.sentence_num)
    return out


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
        "학생이 핵심 논리·비교·인과·구조를 그림으로 한눈에 이해할 수 있는 도식 명세(`KoreanVisualDiagram`) JSON을 한국어로 작성하세요.\n"
        + _ENGLISH_DIAGRAM_GUIDE
    )
    prompt = (
        f"[제목/주제] {title or '-'} / {core_theme or '-'}\n\n"
        f"[영어 지문 원문]\n{clean_passage[:4500]}\n\n"
        "위 영어 지문의 핵심 논리 구조를 시각화한 한국어 도식 JSON을 작성하세요."
    )
    resp = _generate_content_with_fallback(
        client=client,
        model_name=eff_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=KoreanVisualDiagram,
            temperature=0.2,
            max_output_tokens=2048,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL,
                include_thoughts=False,
            ),
        ),
        fast_mode=True,
    )
    return _parse_structured(resp, KoreanVisualDiagram)
