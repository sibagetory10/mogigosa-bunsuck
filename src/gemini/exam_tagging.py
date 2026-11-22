from __future__ import annotations

import re
from typing import Any, Callable, Sequence

from google.genai import types

from src.config import EXAM_TAGGING_MODEL
from src.curriculum_2022 import (
    build_tagging_grounding_context,
    validate_and_normalize_extracted_tag,
)
from src.gemini.core import (
    DEFAULT_API_TIMEOUT_SECONDS,
    MAX_VISION_IMAGE_WIDTH,
    ScannedBox,
    ScannedPageBoxes,
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
from src.models import (
    BatchExamExtractionOutput,
    BatchFilePairingOutput,
    BatchQuestionTagItem,
    ConceptBookRecord,
    ExamHeaderMetadata,
    MixedPdfClassificationOutput,
    QuestionExtractionOutput,
)


def detect_scanned_question_boxes(
    api_key: str,
    model_name: str,
    page_png_bytes: bytes,
) -> list[ScannedBox]:
    """텍스트 레이어가 없는 스캔 PDF 페이지에서 문항 좌표를 Gemini Vision으로 탐지한다."""
    client = _build_client(api_key)
    prompt = (
        "이 한국어 시험지 단일 페이지 이미지를 **왼쪽 단(xmin < 500) 위→아래** 순서로 먼저 훑고, "
        "이어서 **오른쪽 단(xmin >= 500) 위→아래** 순서로 탐색하여 이 페이지에 있는 모든 문항(1~45번 전체, 누락 없이)을 찾으세요.\n"
        "- 각 문항의 경계 상자 [ymin, xmin, ymax, xmax]를 0~1000 정규화 좌표로 반환하세요.\n"
        "- 문항 본문, 그림/표, <보기>, 선지 ①~⑤ 전체가 포함되도록 하세요.\n"
        "- 세트/공통 지문 헤더(`[1~3]`, `[16~17]` 등)가 바로 위 또는 같은 단 상단에 있으면 해당 첫 문항 상자에 포함하세요.\n"
        "- 각 단의 맨 아래 문항은 선지 ④, ⑤가 잘리지 않도록 페이지 하단 여백 위쪽(`ymax` 약 930)까지 넉넉히 잡으세요."
    )
    opt_png = optimize_image_for_vision(page_png_bytes)
    img_part = types.Part.from_bytes(data=opt_png, mime_type="image/png")
    result = _generate_structured_with_fallback(
        client=client,
        model_name=model_name,
        contents=[img_part, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ScannedPageBoxes,
            temperature=0.0,
        ),
        fast_mode=True,
        schema_cls=ScannedPageBoxes,
        max_parse_retries=2,
    )
    return result.boxes


def analyze_question_with_gemini(
    api_key: str,
    model_name: str,
    crop_png_bytes: bytes,
    q_num: int,
    subject_category: str,
    exam_name: str,
    answer_pdf_part: Any | None = None,
    answer_text_hint: str = "",
    status_callback: Callable[[str], None] | None = None,
    concept_books: Sequence[ConceptBookRecord] | None = None,
    grounding_context: str = "",
    question_text_hint: str = "",
    solve_without_answer: bool | None = None,
) -> QuestionExtractionOutput:
    """
    크롭된 문항 PNG, 해설지 정보(미첨부 시 직접 고해상도 풀이·검산), 2022 공식 기준표 및 개념서 DB를 Gemini에 전달해
    성취기준 앵커링·원리/용어 분리·정답/해설을 추출하고 후처리 검증한다.
    """
    has_answer_ref = bool((answer_text_hint and answer_text_hint.strip()) or answer_pdf_part is not None)
    is_self_solve = (not has_answer_ref) if solve_without_answer is None else bool(solve_without_answer)

    client = _build_client(
        api_key,
        timeout_seconds=120 if is_self_solve else DEFAULT_API_TIMEOUT_SECONDS,
    )
    max_img_w = 1400 if is_self_solve else MAX_VISION_IMAGE_WIDTH
    if status_callback is not None:
        status_callback(f"{q_num}번 이미지 최적화 중(≤{max_img_w}px)...")
    opt_png = optimize_image_for_vision(crop_png_bytes, max_width=max_img_w)
    img_part = types.Part.from_bytes(data=opt_png, mime_type="image/png")

    eff_grounding = grounding_context or build_tagging_grounding_context(
        subject_category=subject_category,
        exam_name=exam_name,
        concept_books=concept_books,
    )

    if is_self_solve:
        rule_5 = (
            "5) [직접 풀이·검산 및 최종 정답(`official_explanation` → `answer`)]: 해설지가 첨부되지 않았습니다. 절대 답을 찍거나 어림짐작하지 말고 다음 절차를 반드시 지키세요.\n"
            "   - 먼저 `official_explanation`에 첨부된 고해상도 문항 이미지(및 참고 텍스트) 속 모든 수치·조건·단위·그래프 축·표·`<보기>`를 순서대로 직접 풀고 검증 과정을 작성하세요.\n"
            "   - 고난도 킬러 문항 오답 방지를 위해 ① 경우의 수/구간 나누기 누락 여부, ② 등호·경계값 및 자연수·정수·정의역 조건(무연근 배제), "
            "③ `<보기>` ㄱ·ㄴ·ㄷ 각각의 반례 존재 여부, ④ 구한 해의 원본 조건 대입 역산 검산까지 마친 뒤 `official_explanation`을 완결하세요.\n"
            "   - 객관식(`①`~`⑤`)은 계산 결과 또는 참인 `<보기>` 조합을 이미지 속 실제 선지 ①~⑤와 1:1 대조하여 일치하는 선지 번호를 확인한 뒤, `answer`에는 선지 내용·수치가 아닌 선지 번호 숫자 `'1'`~`'5'` 중 하나만 기록하세요.\n"
            "   - 선지 ①~⑤가 없는 주관식/단답형만 최종 계산값/정답을 `answer`에 정확히 기록하세요."
        )
    else:
        rule_5 = (
            "5) [해설·정답(`official_explanation` → `answer`)]: 먼저 `official_explanation`에 첨부된 해설지/발췌 기준 핵심 풀이 요약을 작성한 뒤, "
            "`answer`에 공식 정답(객관식은 '1'~'5' 단일 숫자, 주관식은 정확한 값)을 기록하세요."
        )

    system_inst = (
        "당신은 대한민국 2022 개정 교육과정(수학·사회·과학·한국사) 전문 기출 문항 분석가이자 정확한 문제 풀이 전문가입니다. "
        "반드시 아래 순서와 기준표에 따라 추측(할루시네이션) 없이 정확히 태깅하세요.\n"
        "1) [과목/단원/성취기준 앵커링]: 제공된 [2022 공식 기준표 및 개념서 DB] 안에서만 가장 정확한 2022 공식 `curriculum_subject`(2015 개정 명칭 금지; 한국사의 경우 전근대·개항기는 '한국사1', 일제강점기·현대사는 '한국사2')와 `unit_name`을 정하고, "
        "기준표에 성취기준 코드가 있으면 대표 코드는 `standard_code`, 복합 문항 등 걸치는 모든 코드는 `standard_codes` 배열에 모두 적으세요(목록에 없는 코드를 지어내지 마세요).\n"
        "2) [본질 원리 추출(`core_principles`)]: 상위 과목의 포장 용어를 걷어내고, 이 문제를 실제로 풀기 위해 필요한 근본적 풀이 원리·법칙·역사적 사건/제도/인과관계를 간결한 한국어 개념구로 나열하세요.\n"
        "3) [용어 이중 분리(`advanced_terms` vs `basic_terms`)]: '그림/표/보기/값/cm/g/s' 등 일상어와 기초 단위는 제외하세요.\n"
        "   - `advanced_terms`: 고2~3 선택과목 등 상위 과정이나 심화 사료에서 도입되어 하위 학년·기초 개념서 학습자에게 낯선 전문 용어·기호·심화 역사 용어(예: '몰농도', 'mol', '한계효용', '도함수')\n"
        "   - `basic_terms`: `min_grade` 수준에서 이미 배워 당연한 교과 기본 용어(예: '속력', '질량', '이차방정식', '인권', '갑신정변', '3·1 운동')\n"
        "   - `required_terms`: 문제 지문·사료·보기에 등장하는 핵심 전문 용어 통합 목록(`advanced_terms` + 주요 `basic_terms`)\n"
        "4) [최저 학년(`min_grade`) 및 근거(`grade_evidence`)]: `core_principles`를 2022 개정 교육과정에서 처음 배우는 최저 학년(1=고1, 2=고2, 3=고3; 한국사1·한국사2는 고1 공통과목이므로 공통 교육과정 범위이면 1)과 1줄 판정 근거를 `grade_evidence`에 명시하세요.\n"
        + rule_5
    )

    hint_block = (
        f"\n[해설지 {q_num}번 발췌]\n{answer_text_hint.strip()}\n"
        if answer_text_hint and answer_text_hint.strip()
        else ""
    )
    q_text_block = (
        f"\n[문항 추출 텍스트 (참고용 · 수식/표/그림은 이미지 우선)]\n{question_text_hint.strip()[:1800]}\n"
        if question_text_hint and question_text_hint.strip()
        else ""
    )
    grounding_block = f"\n{eff_grounding}\n" if eff_grounding else ""
    prompt = (
        f"시험 회차: {exam_name}\n"
        f"대분류 과목: {subject_category}\n"
        f"문항 번호: {q_num}번\n"
        f"{grounding_block}"
        f"{q_text_block}"
        f"{hint_block}\n"
        + (
            "해설지가 없으므로 위 이미지(및 참고 텍스트)를 직접 끝까지 풀어 official_explanation을 작성·검산한 뒤, 최종 선지 번호 또는 주관식 값을 answer에 반환하세요."
            if is_self_solve
            else "위 이미지의 문항을 기준표와 해설에 대조해 분석하고 JSON 스키마에 맞게 추출하세요."
        )
    )

    contents: list[Any] = [img_part]
    if answer_pdf_part is not None and not hint_block:
        contents.append(answer_pdf_part)
    contents.append(prompt)

    if is_self_solve and not _is_flash_lite_model(model_name):
        gen_config = types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=QuestionExtractionOutput,
            temperature=0.0,
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.HIGH,
                include_thoughts=False,
            ),
        )
        use_fast_mode = False
    else:
        gen_config = types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=QuestionExtractionOutput,
            temperature=0.0 if is_self_solve else 0.1,
        )
        use_fast_mode = not is_self_solve

    parsed = _generate_structured_with_fallback(
        client=client,
        model_name=model_name,
        contents=contents,
        schema_cls=QuestionExtractionOutput,
        config=gen_config,
        fast_mode=use_fast_mode,
        status_callback=status_callback,
        max_parse_retries=2,
    )
    if status_callback is not None:
        status_callback(f"{q_num}번 응답 파싱 및 2022 기준 검증 중...")
    parsed.answer = _normalize_extracted_answer(parsed.answer)
    return validate_and_normalize_extracted_tag(
        parsed,
        subject_category=subject_category,
        concept_books=concept_books,
    )


def analyze_exam_pdf_batch_with_gemini(
    api_key: str,
    model_name: str,
    prob_pdf_bytes: bytes,
    target_q_nums: list[int],
    subject_category: str,
    exam_name: str,
    answer_snippets: dict[int, str] | None = None,
    answer_pdf_part: Any | None = None,
    status_callback: Callable[[str], None] | None = None,
    concept_books: Sequence[ConceptBookRecord] | None = None,
    grounding_context: str = "",
) -> dict[int, BatchQuestionTagItem]:
    """
    전체 모의고사 문제지 PDF(및 해설지) + 2022 공식 기준표/개념서 DB를 한 번에 Gemini에 전달해
    요청된 전체 문항 번호(target_q_nums)를 일괄 태깅하고 후처리 검증한다.
    """
    ordered_q = sorted({int(q) for q in target_q_nums})
    if not ordered_q:
        return {}

    snippets_map = answer_snippets or {}
    snippet_lines: list[str] = []
    for q in ordered_q:
        sn = (snippets_map.get(q) or "").strip()
        if sn:
            snippet_lines.append(f"[{q}번 해설 발췌] {sn[:500]}")

    has_answer_ref = bool(snippet_lines or answer_pdf_part is not None)
    use_high_solver = (not has_answer_ref) and (not _is_flash_lite_model(model_name))
    client = _build_client(api_key, timeout_seconds=180 if use_high_solver else 120)
    if status_callback is not None:
        status_callback(f"📄 전체 모의고사 PDF({len(ordered_q)}문항) 일괄 분석 준비 중...")

    prob_part = prepare_pdf_part(
        client,
        prob_pdf_bytes,
        display_name=f"{exam_name}_problem.pdf",
    )

    eff_grounding = grounding_context or build_tagging_grounding_context(
        subject_category=subject_category,
        exam_name=exam_name,
        concept_books=concept_books,
    )

    ans_rule_line = (
        "- official_explanation / answer: 해설지 기준 핵심 풀이를 official_explanation에 먼저 요약하고, "
        "answer에 공식 정답(객관식 '1'~'5' 단일 숫자, 주관식은 값)을 기록\n"
        if has_answer_ref
        else "- official_explanation / answer: 해설지가 없으므로 추측하지 말고 문항의 조건·보기(ㄱ/ㄴ/ㄷ)·선지 ①~⑤를 직접 풀어 "
        "official_explanation에 단계별 풀이·검산 과정을 먼저 작성한 뒤, 일치하는 선지 번호('1'~'5') 또는 주관식 최종 값을 answer에 기록\n"
    )

    system_inst = (
        "당신은 대한민국 2022 개정 교육과정(수학·사회·과학·한국사) 전문 기출 문항 분석가입니다. "
        "첨부된 전체 모의고사 PDF에서 요청된 모든 문항 번호를 빠짐없이 찾아, "
        "제공된 [2022 공식 기준표 및 개념서 DB]를 근거로 JSON 배열(questions)을 반환하세요.\n"
        "- q_num: 문항 번호 정수 (요청된 문항 번호를 1개도 누락하지 마세요)\n"
        "- curriculum_subject / unit_name / standard_code / standard_codes: 기준표의 2022 공식 과목명(2015 명칭 금지; 한국사는 전근대·개항기 '한국사1', 일제강점기·현대사 '한국사2')과 단원을 먼저 매핑하고, 기준표에 코드가 있으면 대표 코드는 standard_code, 복합 문항에 걸친 모든 코드는 standard_codes에 적되 목록에 없는 코드를 지어내지 마세요\n"
        "- core_principles: 상위 과목 포장 용어를 걷어내고 문제를 푸는 데 필요한 근본적 풀이 원리·법칙·역사적 사건/제도를 간결한 한국어 개념구로 나열\n"
        "- advanced_terms vs basic_terms vs required_terms: 일상어('그림','표','보기') 및 기초 측정단위('cm','g','s')는 제외하고, 상위 과정/심화 사료에서 새로 도입된 낯선 전문 용어는 advanced_terms, min_grade 시점 기본 용어는 basic_terms, 통합 핵심 용어는 required_terms에 기록\n"
        "- min_grade / grade_evidence: core_principles를 2022 개정 교육과정에서 처음 배우는 최저 학년(1=고1, 2=고2, 3=고3; 한국사1·한국사2는 고1 공통과목이므로 공통 교육과정 범위이면 1)과 1줄 근거 작성\n"
        + ans_rule_line
    )

    ncic_section, ncic_rule = _build_lang_ncic_grounding_section("국어", exam_name)
    hint_section = (
        "\n[해설지 문항별 발췌]\n" + "\n".join(snippet_lines) + "\n"
        if snippet_lines
        else ""
    )
    grounding_section = f"\n{eff_grounding}\n" if eff_grounding else ""
    q_list_str = ", ".join(str(q) for q in ordered_q)
    prompt = (
        f"시험 회차: {exam_name}\n"
        f"대분류 과목: {subject_category}\n"
        f"태깅 대상 문항 번호(총 {len(ordered_q)}문항): {q_list_str}\n"
        f"{grounding_section}"
        f"{hint_section}\n"
        "위 문제지 PDF 전체에서 지정된 모든 문항 번호를 기준표와 대조해 한 번에 분석하고 JSON 스키마(questions)로 반환하세요."
    )

    contents: list[Any] = [prob_part]
    if answer_pdf_part is not None and not snippet_lines:
        contents.append(answer_pdf_part)
    contents.append(prompt)

    if status_callback is not None:
        status_callback(f"⚡ {model_name} 모델로 전체 PDF({len(ordered_q)}문항) 2022 기준 일괄 태깅 중...")

    if use_high_solver:
        batch_config = types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=BatchExamExtractionOutput,
            temperature=0.0,
            max_output_tokens=32768,
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.HIGH,
                include_thoughts=False,
            ),
        )
    else:
        batch_config = types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=BatchExamExtractionOutput,
            temperature=0.1 if has_answer_ref else 0.0,
            max_output_tokens=32768,
        )

    parsed = _generate_structured_with_fallback(
        client=client,
        model_name=model_name,
        contents=contents,
        config=batch_config,
        fast_mode=has_answer_ref,
        status_callback=status_callback,
        schema_cls=BatchExamExtractionOutput,
        max_parse_retries=2,
    )
    out: dict[int, BatchQuestionTagItem] = {}
    for item in parsed.questions:
        if item.q_num in ordered_q and item.q_num not in out:
            item.answer = _normalize_extracted_answer(item.answer)
            validate_and_normalize_extracted_tag(
                item,
                subject_category=subject_category,
                concept_books=concept_books,
            )
            out[item.q_num] = item
    return out


def extract_exam_header_metadata(
    api_key: str,
    model_name: str,
    first_page_text: str,
    first_page_png_bytes: bytes | None = None,
    filename: str = "",
) -> ExamHeaderMetadata:
    """
    모의고사 PDF 1페이지 이미지와 텍스트를 AI로 빠르게 교차 분석해
    시행 년도, 월, 학년, 과목, 표준 회차명을 정확히 추출한다.
    """
    client = _build_client(api_key, timeout_seconds=20)

    system_inst = (
        "당신은 한국 전국연합학력평가·대학수학능력시험·평가원 모의평가·사설 모의고사 표지/1페이지 헤더 정밀 판독기입니다.\n"
        "첨부된 시험지 1페이지 이미지(상단 로고, 제목 박스, 영역 표시)와 텍스트·파일명을 종합하여 다음 5개 필드를 정확히 추출하세요:\n"
        "1) year: 4자리 시행 년도 숫자 문자열 (예: '2026', '2025').\n"
        "2) month: 시행 월 문자열 ('3월', '4월', '5월', '6월', '7월', '9월', '10월', '11월' 등. 대학수학능력시험 본시험은 '11월').\n"
        "3) grade: 학년 문자열 ('고1', '고2', '고3' 중 하나. '제1학년'/'고1'→'고1', '제2학년'/'고2'→'고2', '제3학년'/'고3'/수능/모의평가는 '고3').\n"
        "4) subject: 과목명 (예: '국어', '한국사', '통합과학', '통합사회', '물리학Ⅰ', '화학Ⅰ', '생명과학Ⅰ', '지구과학Ⅰ', '수학', '영어', '경제', '사회·문화' 등). 영역명('과학탐구 영역')과 세부 과목명(예: '물리학Ⅰ')이 같이 있으면 세부 과목명을 우선 적으세요.\n"
        "5) exam_title: 반드시 '{year}년 {month} {grade} {subject}' 형식으로 조합 (예: '2025년 9월 고1 통합과학', '2026년 3월 고3 한국사', '2026년 3월 고3 국어')."
    )

    text_excerpt = (first_page_text or "").strip()[:3000]
    fn_hint = f"[업로드 파일명]\n{filename.strip()}\n\n" if filename and filename.strip() else ""
    prompt = (
        f"{fn_hint}"
        "[시험지 1페이지 텍스트 발췌]\n"
        f"{text_excerpt if text_excerpt else '(텍스트가 거의 없는 스캔 PDF입니다. 첨부된 1페이지 이미지를 정밀 판독하세요.)'}\n\n"
        "위 1페이지 이미지 상단 헤더·로고·과목 박스를 최우선으로 읽고 시행 년도, 시행 월, 학년, 과목명, 표준 회차명(exam_title)을 JSON으로 반환하세요."
    )

    contents: list[Any] = []
    if first_page_png_bytes:
        opt_png = optimize_image_for_vision(first_page_png_bytes, max_width=1200)
        contents.append(types.Part.from_bytes(data=opt_png, mime_type="image/png"))
    contents.append(prompt)

    meta = _generate_structured_with_fallback(
        client=client,
        model_name=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=ExamHeaderMetadata,
            temperature=0.0,
        ),
        fast_mode=True,
        schema_cls=ExamHeaderMetadata,
        max_parse_retries=2,
    )

    year = meta.year.strip()
    month = meta.month.strip()
    grade = meta.grade.strip()
    subject = meta.subject.strip()
    if year and month and grade and subject:
        year_token = year if year.endswith("년") else f"{year}년"
        month_token = month if month.endswith("월") else f"{month}월"
        standardized_title = f"{year_token} {month_token} {grade} {subject}"
    else:
        standardized_title = meta.exam_title.strip()

    return ExamHeaderMetadata(
        year=year.rstrip("년"),
        month=month if (not month or month.endswith("월")) else f"{month}월",
        grade=grade,
        subject=subject,
        exam_title=standardized_title,
    )


MAX_PAIRING_VISION_IMAGES: int = 16


def _format_pairing_item_summary(
    label: str,
    item: dict[str, Any],
    max_text_chars: int = 450,
) -> str:
    name = str(item.get("name") or "").strip()
    page_count = int(item.get("page_count") or 0)
    raw_text = re.sub(r"\s+", " ", str(item.get("first_page_text") or "").strip())
    excerpt = raw_text[:max_text_chars] if raw_text else "(텍스트 없음 · 첨부 표지 이미지 참조)"
    page_info = f"{page_count}쪽" if page_count > 0 else "쪽수 미상"
    return f"- {label} | 파일명: {name} | 분량: {page_info}\n  1페이지 발췌: {excerpt}"


def _append_pairing_vision_parts(
    contents: list[Any],
    items: Sequence[dict[str, Any]],
    prefix: str,
    remaining_slots: int,
) -> int:
    """텍스트가 부족한 스캔 PDF의 1페이지 썸네일을 라벨과 함께 Vision 입력에 추가한다."""
    used = 0
    for idx, item in enumerate(items):
        if used >= remaining_slots:
            break
        raw_text = re.sub(r"\s+", "", str(item.get("first_page_text") or ""))
        png_bytes = item.get("first_page_png")
        if len(raw_text) < 40 and png_bytes:
            opt_png = optimize_image_for_vision(png_bytes, max_width=640)
            name = str(item.get("name") or "").strip()
            contents.append(f"[이미지: {prefix}{idx} - {name}]")
            contents.append(types.Part.from_bytes(data=opt_png, mime_type="image/png"))
            used += 1
    return remaining_slots - used


def match_problem_and_answer_pdfs_with_ai(
    api_key: str,
    problem_items: Sequence[dict[str, Any]],
    answer_items: Sequence[dict[str, Any]],
    model_name: str = EXAM_TAGGING_MODEL,
) -> tuple[list[int | None], dict[int, str]]:
    """
    업로드된 문제지 목록(P0..Pm-1)과 해설지 목록(A0..An-1)의 파일명·1페이지 텍스트·스캔 이미지를
    Gemini 3.5 Flash-Lite로 비교하여 각 문제지에 대응하는 해설지를 자동 매칭한다.
    """
    n_prob = len(problem_items)
    n_ans = len(answer_items)
    if n_prob == 0 or n_ans == 0:
        return [None] * n_prob, {}

    client = _build_client(api_key, timeout_seconds=45)
    system_inst = (
        "당신은 한국 중·고등학교 시험지 및 모의고사(평가원·교육청·수능·EBS·학교 기출) PDF 자동 분류·매칭 전문가입니다.\n"
        "업로드된 [문제지 목록](P0, P1, ...)과 [해설지 목록](A0, A1, ...)의 파일명, 분량, 1페이지 텍스트/이미지를 비교해 "
        "각 문제지에 가장 정확히 대응하는 해설지 인덱스(`answer_index`)를 1개 지정하세요.\n\n"
        "[핵심 판별 원칙]\n"
        "1. 파일명이 `file_1.pdf`, `다운로드.pdf`, 숫자 코드 등으로 무작위여도 1페이지 텍스트/이미지의 "
        "**시행 연도·월·학년·과목명(통합과학, 화학Ⅰ, 수학, 사회·문화 등)·시험명·1번 문항/해설 내용**을 최우선으로 비교하세요.\n"
        "2. 연도·월·학년·과목이 서로 다른 해설지는 절대 억지 매칭하지 말고 `answer_index = null`로 남기세요.\n"
        "3. 단일 과목 해설지는 1:1 매칭이 원칙입니다(같은 단일 해설지를 서로 다른 과목/시험지에 중복 배정 금지).\n"
        "4. 단, 해설지 1개가 여러 과목이 합본된 **통합 해설지**(예: '과학탐구 정답 및 해설', '사회탐구 정답 및 해설', '전과목 해설지')인 경우에만 "
        "해당 시험의 여러 과목 문제지에 같은 `answer_index`를 연결하고 `is_shared_combined_answer = true`로 표시하세요.\n"
        "5. `reason`에는 매칭 또는 미매칭 근거를 40자 이내 한국어 1줄로 적으세요. (예: '2026년 3월 고2 수학 일치', '대응 과목 해설지 없음')"
    )

    prob_lines = [
        _format_pairing_item_summary(f"P{idx}", item)
        for idx, item in enumerate(problem_items)
    ]
    ans_lines = [
        _format_pairing_item_summary(f"A{idx}", item)
        for idx, item in enumerate(answer_items)
    ]
    prompt = (
        "[문제지 목록]\n"
        + "\n".join(prob_lines)
        + "\n\n[해설지 목록]\n"
        + "\n".join(ans_lines)
        + f"\n\n총 {n_prob}개의 문제지(P0~P{n_prob - 1}) 각각에 대해 가장 적합한 해설지 인덱스(A0~A{n_ans - 1} 또는 null)를 JSON으로 반환하세요."
    )

    contents: list[Any] = []
    remaining_imgs = MAX_PAIRING_VISION_IMAGES
    remaining_imgs = _append_pairing_vision_parts(contents, problem_items, "P", remaining_imgs)
    _append_pairing_vision_parts(contents, answer_items, "A", remaining_imgs)
    contents.append(prompt)

    parsed = _generate_structured_with_fallback(
        client=client,
        model_name=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=BatchFilePairingOutput,
            temperature=0.0,
        ),
        fast_mode=True,
        schema_cls=BatchFilePairingOutput,
        max_parse_retries=2,
    )

    matches: list[int | None] = [None] * n_prob
    reasons: dict[int, str] = {}
    used_single_ans: set[int] = set()
    seen_probs: set[int] = set()

    for pair in parsed.pairs:
        p_idx = pair.problem_index
        if not (0 <= p_idx < n_prob) or p_idx in seen_probs:
            continue
        seen_probs.add(p_idx)

        reason = (pair.reason or "").strip()
        a_idx = pair.answer_index
        if a_idx is None or not (0 <= a_idx < n_ans):
            matches[p_idx] = None
            if reason:
                reasons[p_idx] = reason
            continue

        if a_idx in used_single_ans and not (pair.is_shared_combined_answer or n_ans == 1):
            matches[p_idx] = None
            reasons[p_idx] = reason or "다른 문제지에 이미 배정된 단일 해설지"
            continue

        matches[p_idx] = a_idx
        if not pair.is_shared_combined_answer:
            used_single_ans.add(a_idx)
        if reason:
            reasons[p_idx] = reason

    return matches, reasons


def classify_mixed_exam_pdfs_with_ai(
    api_key: str,
    pdf_items: Sequence[dict[str, Any]],
    model_name: str = EXAM_TAGGING_MODEL,
) -> tuple[list[int], list[int]]:
    """
    단일 업로더에 문제지와 해설지가 섞여 업로드된 경우,
    파일명과 1페이지 텍스트/이미지를 분석해 문제지('problem')와 해설지('answer') 인덱스를 자동 분리한다.
    """
    n_files = len(pdf_items)
    if n_files == 0:
        return [], []

    client = _build_client(api_key, timeout_seconds=35)
    system_inst = (
        "당신은 한국 모의고사/기출 PDF 파일이 '문제지(problem)'인지 '정답·해설지(answer)'인지 구분하는 분류기입니다.\n"
        "각 파일(F0, F1, ...)의 파일명과 1페이지 텍스트/이미지를 보고 `role`을 정확히 판별하세요.\n\n"
        "[판별 기준]\n"
        "- 'answer': 문서 상단 또는 본문에 '정답 및 해설', '정답과 해설', '빠른 정답', '문항별 해설/풀이', '정답표' 등이 중심인 해설·정답지.\n"
        "- 'problem': 실제 시험 문항(1번, 2번... 문제와 선택지 ①~⑤)이 실린 시험지·문제지.\n"
        "- 판별이 모호하거나 시험 문항과 간단한 정답표가 함께 있는 문서는 안전하게 'problem'으로 분류하세요."
    )

    item_lines = [
        _format_pairing_item_summary(f"F{idx}", item)
        for idx, item in enumerate(pdf_items)
    ]
    prompt = (
        "[업로드된 PDF 목록]\n"
        + "\n".join(item_lines)
        + f"\n\n총 {n_files}개 파일(F0~F{n_files - 1}) 각각의 role('problem' 또는 'answer')을 JSON으로 반환하세요."
    )

    contents: list[Any] = []
    _append_pairing_vision_parts(contents, pdf_items, "F", MAX_PAIRING_VISION_IMAGES)
    contents.append(prompt)

    parsed = _generate_structured_with_fallback(
        client=client,
        model_name=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=MixedPdfClassificationOutput,
            temperature=0.0,
        ),
        fast_mode=True,
        schema_cls=MixedPdfClassificationOutput,
        max_parse_retries=2,
    )

    roles: list[str] = ["problem"] * n_files
    for item in parsed.items:
        idx = item.file_index
        if 0 <= idx < n_files and item.role in ("problem", "answer"):
            roles[idx] = item.role

    prob_indices = [i for i, r in enumerate(roles) if r == "problem"]
    ans_indices = [i for i, r in enumerate(roles) if r == "answer"]
    return prob_indices, ans_indices
