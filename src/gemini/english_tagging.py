from __future__ import annotations

import re
from typing import Any, Callable, Sequence

from google.genai import types

from src.config import EXAM_TAGGING_MODEL
from src.gemini.core import (
    DEFAULT_API_TIMEOUT_SECONDS,
    _apply_lang_standard_validation,
    _build_client,
    _build_lang_ncic_grounding_section,
    _generate_content_with_fallback,
    _is_flash_lite_model,
    _normalize_extracted_answer,
    _generate_structured_with_fallback,
    _parse_structured,
    optimize_image_for_vision,
    prepare_pdf_part,
)
from src.gemini.english_validators import (
    _reconcile_english_passage_with_question_text,
)
from src.models import (
    BatchEnglishExamExtractionOutput,
    EnglishQuestionExtractionItem,
    EnglishSingleExtractionOutput,
    normalize_english_group_id,
)
from src.text_formatter import normalize_english_exam_text

def _infer_english_group_id(q_num: int, raw_group: str = "") -> str:
    """41~42 / 43~45 장문 세트만 그룹으로 인정하고, 그 외 번호([31~34] 등 발문 헤더)는 항상 개별 지문으로 취급한다."""
    return normalize_english_group_id(q_num, raw_group)


_ENGLISH_PASSAGE_TAGGING_RULES = (
    "[문항 번호별 영어 지문(passage_text) 정밀 태깅 규칙 — 해설지·정답표 필수 대조 & 원문 100% 무손실 보존]\n"
    "⚠️ **절대 규칙 (환각·생략 금지)**: 아래 번호별 규칙에서 지정한 수정 지점(빈칸 1곳, 어법/어휘 오답 1곳, 무관한 문장 1개, 순서/삽입) 외의 **나머지 모든 영어 단어·전치사구·구두점은 단 한 단어도 생략·요약·의역·변형하지 말고 원문 100% 그대로 유지**하세요.\n"
    "모든 독해 문항(18~45번)은 한국어 발문과 선택지(①~⑤)를 제외한 순수 영어 본문만 `passage_text`에 담되, 아래 번호별 규칙을 반드시 따르세요:\n"
    "- **29번 (어법)**: 해설지를 분석해 정답을 찾은 뒤, **해설지가 해설한 대로만** 틀린 어법 표현 1곳을 올바른 표현으로 수정하여 태깅하세요. (밑줄/번호 ①~⑤ 마커 제거, 선택형 `(A)(B)(C)`면 정답 표현만 남김, 나머지 단어 100% 보존)\n"
    "- **30번 (어휘)**: 해설지를 참고해 정답을 찾은 뒤, **해설지가 해설한 대로만** 문맥상 부적절한 어휘 1곳을 해설지가 제시한 적절한 어휘로 수정하여 태깅하세요. (밑줄/번호 ①~⑤ 마커 제거, 나머지 단어 100% 보존)\n"
    "- **31, 32, 33, 34번 (빈칸 추론)**: **빈칸(`____`) 바로 앞과 바로 뒤에 있는 단어·전치사구(예: `at one level of analysis`)를 절대 지우거나 생략하지 마세요.** 오직 `____` 밑줄 자리 그 자체만 해설지 정답에 해당하는 보기(선지) 문자열로 1:1 치환하여 태깅하세요.\n"
    "- **35번 (무관한 문장)**: 해설지 정답을 참고해 **전체 흐름과 무관한 해당 번호 문장 1개만 제외(삭제)**하고, ①~⑤ 번호 마커를 제거하며 나머지 문장은 100% 그대로 태깅하세요.\n"
    "- **36, 37번 (글의 순서)**: 해설지 정답을 참고해 **주어진 글 + 정답 순서대로 단락을 배열**하고, `(A)`, `(B)`, `(C)` 라벨을 제거한 완전한 지문으로 태깅하세요. (단어 생략 금지)\n"
    "- **38, 39번 (문장 삽입)**: 해설지 정답을 참고해 **주어진 문장을 정답 위치(①~⑤)에 삽입**하고, 모든 ①~⑤ 위치 마커를 제거한 완전한 지문으로 태깅하세요. (단어 생략 금지)\n"
    "- **40번 (요약문 완성)**: **요약문(`[요약문]` / `￫` 아래의 `(A)...(B)...` 요약 문장 및 선지)은 지문 태깅에서 제외**하고 본문 영어 지문만 100% 그대로 태깅하세요.\n"
    "- **41~42번 (장문)**: **42번 해설지**를 참고하여 **42번 문항의 틀린 어휘 부분만** 해설지가 바꾸기 적절한 단어라고 제시한 단어로 바꿔서 태깅하세요. (41번과 42번 모두 동일하게 수정된 공통 지문을 저장, `(a)~(e)` 마커 제거, 나머지 단어 100% 보존)\n"
    "- **43~45번 (장문)**: **43번 해설지**를 분석하여 **정답 순서대로 단락을 교정·배열**해 태깅하세요. (`(A)~(D)` 단락 라벨 및 `(a)~(e)`/`①~⑤` 마커 정리, 43·44·45번 모두 동일하게 교정된 공통 지문을 저장)\n"
    "- **그 외 언급되지 않은 문항(18~28번 등)**: 발문과 선지만 제외하고 원문 100% 그대로 태깅하세요.\n"
)


_NUM_TO_CIRCLED = {"1": "①", "2": "②", "3": "③", "4": "④", "5": "⑤"}


def _extract_local_english_body_and_choices(local_text: str) -> tuple[str, dict[str, str]]:
    """로컬 추출된 영어 문항 텍스트에서 발문·각주·선지를 분리해 (영어 본문, {'1': 선지1, ...})를 반환한다."""
    raw = (local_text or "").replace("\r\n", "\n").strip()
    if not raw:
        return "", {}
    choices: dict[str, str] = {}
    # ①~⑤ 선지 추출
    c_matches = list(re.finditer(r"(?:^|\s)([①②③④⑤])\s*([^\n①②③④⑤]+)", raw))
    c_map = {"①": "1", "②": "2", "③": "3", "④": "4", "⑤": "5"}
    body_end = len(raw)
    if c_matches:
        # 선지 영역 시작점(연속 선지 첫 번째)
        first_c_idx = c_matches[0].start()
        # 본문 내 삽입/어법 ①~⑤가 아니라 하단 선택지 블록인지 확인 (최소 2개 이상 또는 줄 시작)
        if len(c_matches) >= 3:
            body_end = first_c_idx
        for cm in c_matches:
            sym = cm.group(1)
            val = re.sub(r"\s+", " ", cm.group(2)).strip()
            if sym in c_map and val:
                choices[c_map[sym]] = val

    body_raw = raw[:body_end].strip()
    body_lines: list[str] = []
    for ln in body_raw.splitlines():
        s = ln.strip()
        if not s:
            continue
        # 발문 / 배점 / 각주 제거
        if re.match(r"^(?:\d{1,2}\.\s*)?.*[가-힣].*[?:]\s*$", s):
            continue
        if re.match(r"^\[[1-4]점\]$", s):
            continue
        if re.match(r"^\*\s*[A-Za-z-]+:", s):
            continue
        body_lines.append(s)
    body = re.sub(r"\s+", " ", " ".join(body_lines)).strip()
    return body, choices


def _verify_and_fix_english_tagged_passage(
    q_num: int,
    passage_text: str,
    answer: str,
    local_question_text: str = "",
) -> str:
    """AI가 태깅한 영어 지문에서 빈칸 앞뒤 구절 누락·단어 변조·35번 정상 문장 누락·요약문 잔존을 결정론적으로 검증·복구한다."""
    p = (passage_text or "").strip()
    if not p or q_num <= 17:
        return p

    # 40번: 요약문 잔존 시 본문만 남기기
    if q_num == 40:
        p = re.split(r"\n\s*(?:\[요약문\]|￫|->)\s*", p, maxsplit=1)[0].strip()

    # 29~30(어법·어휘 본동사 누락 복구), 31~34(빈칸), 35(무관한 문장), 36~37(글의 순서), 38~39(문장 삽입): PyMuPDF 원본 텍스트와 대조해 원문 변조·문장 누락 즉시 복구
    if local_question_text and q_num in (29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39):
        inferred_type = (
            "어법"
            if q_num == 29
            else (
                "어휘"
                if q_num == 30
                else (
                    "빈칸 추론"
                    if q_num in (31, 32, 33, 34)
                    else ("무관한 문장" if q_num == 35 else ("글의 순서" if q_num in (36, 37) else "문장 삽입"))
                )
            )
        )
        reconciled, _ = _reconcile_english_passage_with_question_text(
            passage_text=p,
            question_text=local_question_text,
            orig_question_type=inferred_type,
            answer=answer,
        )
        if reconciled:
            p = reconciled

    # 31~34번 추가 안전망: 빈칸 앞뒤 3단어 및 미완성 밑줄 최종 검사
    if q_num in (31, 32, 33, 34) and local_question_text:
        local_body, choices = _extract_local_english_body_and_choices(local_question_text)
        ans_key = _normalize_extracted_answer(answer)
        ans_choice = choices.get(ans_key, "").strip()
        blank_m = re.search(r"_{3,}", local_body)
        if local_body and ans_choice and blank_m:
            before_words = re.findall(r"[A-Za-z0-9'-]+", local_body[:blank_m.start()])[-3:]
            after_words = re.findall(r"[A-Za-z0-9'-]+", local_body[blank_m.end():])[:3]
            p_norm = re.sub(r"\s+", " ", p).lower()
            before_ok = (len(before_words) < 2) or (" ".join(before_words).lower() in p_norm)
            after_ok = (len(after_words) < 2) or (" ".join(after_words).lower() in p_norm)
            has_leftover_blank = bool(re.search(r"_{3,}", p))
            if (not before_ok) or (not after_ok) or has_leftover_blank:
                filled = re.sub(r"\s*_{3,}\s*", f" {ans_choice} ", local_body, count=1)
                filled = re.sub(r"\s+([.,?!;:])", r"\1", filled)
                filled = re.sub(r"\s+", " ", filled).strip()
                if filled:
                    return filled

    return p


def analyze_english_exam_pdf_batch_with_gemini(
    api_key: str,
    model_name: str,
    prob_pdf_bytes: bytes,
    target_q_nums: list[int],
    exam_name: str,
    answer_snippets: dict[int, str] | None = None,
    answer_pdf_part: Any | None = None,
    status_callback: Callable[[str], None] | None = None,
    question_texts: dict[int, str] | None = None,
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
    q_texts_map = question_texts or {}
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
        "   - `passage_text`: 아래 [문항 번호별 영어 지문 정밀 태깅 규칙]에 따라 완성된 영어 지문을 추출하세요.\n"
        "   - **장문 세트(41~42번, 43~45번)에만** `passage_group_id`를 `'41~42'` 또는 `'43~45'`로 표기하세요. **그 외 모든 문항(예: 31~34, 36~37, 38~39 등 발문 헤더 범위 포함)은 문항마다 지문이 각각 다르므로 `passage_group_id`를 반드시 빈 문자열로 두고 해당 번호 문항의 지문만 추출하세요.**\n"
        "   - `passage_topic`: 지문의 핵심 소재·주제를 한국어 1줄로 요약하세요.\n"
        "   - `key_vocabulary`: 지문 내 주요 어휘 3~6개를 `'단어: 한국어 뜻'` 형식으로 나열하세요.\n"
        "4. `official_explanation`과 `answer`: 해설지 PDF(특히 1페이지 상단 1~45번 빠른 정답표)와 문항별 해설을 대조하여 핵심 풀이 요약과 공식 정답(`'1'`~`'5'`)을 정확히 추출하세요. "
        "**특히 41번(제목 추론)과 42번(밑줄 (a)~(e) 어휘 적절성), 그리고 43번(순서 배열)·44번(지칭 추론 (a)~(e))·45번(내용 일치) 세트 문항의 정답과 해설이 서로 뒤바뀌지 않도록 문항 번호별 발문과 정답표를 반드시 1:1 확인하세요.**\n\n"
        + _ENGLISH_PASSAGE_TAGGING_RULES
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

    parsed = _generate_structured_with_fallback(
        client=client,
        model_name=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=BatchEnglishExamExtractionOutput,
            temperature=0.0,
            max_output_tokens=32768,
        ),
        fast_mode=True,
        status_callback=status_callback,
        schema_cls=BatchEnglishExamExtractionOutput,
        max_parse_retries=2,
    )

    out: dict[int, EnglishQuestionExtractionItem] = {}
    for item in parsed.questions:
        if item.q_num in ordered_q and item.q_num not in out:
            item.english_section = "듣기" if item.q_num <= 17 else "독해"
            item.passage_group_id = _infer_english_group_id(item.q_num, item.passage_group_id)
            item.answer = _normalize_extracted_answer(item.answer)
            if item.q_num <= 17:
                item.passage_text = ""
            else:
                item.passage_text = _verify_and_fix_english_tagged_passage(
                    item.q_num,
                    item.passage_text,
                    item.answer,
                    local_question_text=q_texts_map.get(item.q_num, ""),
                )
            _apply_lang_standard_validation(item, "영어")
            out[item.q_num] = item

    # 41~42(42번 어휘 수정본 우선), 43~45(43번 순서 교정본 우선) 세트 문항 간 공통 지문(passage_text) 및 주제 동기화
    group_best_passage: dict[str, str] = {}
    group_best_topic: dict[str, str] = {}
    if 42 in out and (out[42].passage_text or "").strip():
        group_best_passage["41~42"] = (out[42].passage_text or "").strip()
    if 43 in out and (out[43].passage_text or "").strip():
        group_best_passage["43~45"] = (out[43].passage_text or "").strip()

    for item in out.values():
        gid = item.passage_group_id
        if not gid:
            continue
        txt = (item.passage_text or "").strip()
        if gid not in group_best_passage and len(txt) > len(group_best_passage.get(gid, "")):
            group_best_passage[gid] = txt
        top = (item.passage_topic or "").strip()
        if len(top) > len(group_best_topic.get(gid, "")):
            group_best_topic[gid] = top

    for item in out.values():
        gid = item.passage_group_id
        if gid and gid in group_best_passage and group_best_passage[gid]:
            if gid == "41~42" and 42 in out and (out[42].passage_text or "").strip():
                item.passage_text = group_best_passage["41~42"]
            elif gid == "43~45" and 43 in out and (out[43].passage_text or "").strip():
                item.passage_text = group_best_passage["43~45"]
            elif len((item.passage_text or "").strip()) < len(group_best_passage[gid]) * 0.7:
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
        "- 독해 문항이면 `english_section='독해'`, 아래 [문항 번호별 영어 지문 정밀 태깅 규칙]에 맞춰 순수 영어 본문 전체를 `passage_text`에 추출하고, "
        "원래 유형(`english_question_type`), 주제(`passage_topic`), 핵심 어휘(`key_vocabulary`), 정답(`answer`), 해설(`official_explanation`)을 JSON으로 반환하세요.\n\n"
        + ("" if is_listening else _ENGLISH_PASSAGE_TAGGING_RULES)
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

    single = _generate_structured_with_fallback(
        client=client,
        model_name=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=EnglishSingleExtractionOutput,
            temperature=0.0,
        ),
        fast_mode=True,
        schema_cls=EnglishSingleExtractionOutput,
        max_parse_retries=2,
    )
    sec: Any = "듣기" if is_listening else "독해"
    grp_id = _infer_english_group_id(q_num, single.passage_group_id)
    ans_norm = _normalize_extracted_answer(single.answer)
    p_text = "" if is_listening else (single.passage_text.strip() or shared_passage_hint.strip())
    if not is_listening:
        p_text = _verify_and_fix_english_tagged_passage(
            q_num,
            p_text,
            ans_norm,
            local_question_text=question_text_hint,
        )
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
