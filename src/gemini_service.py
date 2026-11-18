from __future__ import annotations

import concurrent.futures
import io
import json
import random
import re
import time
from typing import Any, Callable, Sequence, TypeVar

from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel, Field

from src.config import DEFAULT_MODEL, HYDROGEN_MODEL, resolve_api_model_name
from src.curriculum_2022 import (
    build_tagging_grounding_context,
    validate_and_normalize_extracted_tag,
)
from src.models import (
    AutoConceptPageDetectionOutput,
    AutoConceptRangeItem,
    BatchExamExtractionOutput,
    BatchMatchResult,
    BatchQuestionTagItem,
    ConceptBookCoverMetadata,
    ConceptBookExtractionOutput,
    ConceptBookRecord,
    ConceptUnit,
    ConceptWhitelist,
    ExamHeaderMetadata,
    QuestionExtractionOutput,
    QuestionMatchJudgment,
    QuestionRecord,
    RewrittenQuestion,
    TocExtractionOutput,
    TocUnitItem,
)
from src.pdf_parser import extract_specific_pages_slice
from src.text_formatter import (
    find_forbidden_terms_in_text,
    format_chemical_scripts,
    normalize_exam_layout_text,
)


class ForbiddenTermRemainingError(ValueError):
    """개조 후처리 검증에서 범위 외 금지 용어가 재개조 후에도 잔존할 때 발생하는 오류."""

T = TypeVar("T")


class ScannedBox(BaseModel):
    q_num: int = Field(..., ge=1, le=50)
    ymin: int = Field(..., ge=0, le=1000)
    xmin: int = Field(..., ge=0, le=1000)
    ymax: int = Field(..., ge=0, le=1000)
    xmax: int = Field(..., ge=0, le=1000)


class ScannedPageBoxes(BaseModel):
    boxes: list[ScannedBox] = Field(default_factory=list)


DEFAULT_API_TIMEOUT_SECONDS: int = 30
MAX_VISION_IMAGE_WIDTH: int = 1200


def _build_client(api_key: str, timeout_seconds: int = DEFAULT_API_TIMEOUT_SECONDS) -> genai.Client:
    if not api_key or not api_key.strip():
        raise ValueError("API 키가 설정되지 않았습니다. 사이드바 또는 Secrets에 Gemini API 키를 입력해 주세요.")
    timeout_ms = max(5, int(timeout_seconds)) * 1000
    return genai.Client(
        api_key=api_key.strip(),
        http_options=types.HttpOptions(
            timeout=timeout_ms,
            # SDK 내부의 5회 지수 백오프 무한 대기 차단 -> 명시적 최대 2회 제어
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def build_gemini_client(api_key: str, timeout_seconds: int = DEFAULT_API_TIMEOUT_SECONDS) -> genai.Client:
    """외부 호출부에서 30초 타임아웃이 적용된 Gemini 클라이언트를 생성할 때 사용한다."""
    return _build_client(api_key, timeout_seconds=timeout_seconds)


def optimize_image_for_vision(png_bytes: bytes, max_width: int = MAX_VISION_IMAGE_WIDTH) -> bytes:
    """Gemini Vision 전송/처리 속도 최적화를 위해 가로 최대 1200px 이하로 비례 리사이징 및 압축한다."""
    if not png_bytes:
        return png_bytes
    with Image.open(io.BytesIO(png_bytes)) as img:
        if img.width <= max_width:
            if len(png_bytes) <= 450 * 1024:
                return png_bytes
            out = io.BytesIO()
            img.save(out, format="PNG", optimize=True)
            return out.getvalue()

        new_h = max(1, int(round(img.height * (max_width / img.width))))
        resized = img.resize((max_width, new_h), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        resized.save(out, format="PNG", optimize=True)
        return out.getvalue()


FALLBACK_MODEL_ORDER: tuple[str, ...] = (
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
)
RATE_LIMIT_FALLBACK_ORDER: tuple[str, ...] = (
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
)
OVERLOAD_RETRY_DELAYS: tuple[float, ...] = (2.0, 4.0, 6.0)


def build_model_fallback_chain(model_name: str) -> list[str]:
    """주 모델에서 503 등이 반복될 경우 순차 전환할 폴백 모델 체인을 구성한다."""
    primary = resolve_api_model_name(model_name) or DEFAULT_MODEL
    return [primary] + [m for m in FALLBACK_MODEL_ORDER if m != primary]


def _is_fatal_config_or_auth_error(msg: str) -> bool:
    return any(
        k in msg
        for k in (
            "400",
            "401",
            "403",
            "invalid_argument",
            "permission_denied",
            "unauthenticated",
            "api key not valid",
            "api_key_invalid",
        )
    )


def _is_rate_limit_error(msg: str) -> bool:
    return any(
        k in msg
        for k in (
            "429",
            "resource_exhausted",
            "rate limit",
            "rate_limit",
            "quota",
        )
    )


def _is_server_overload_error(msg: str) -> bool:
    return any(
        k in msg
        for k in (
            "503",
            "unavailable",
            "high demand",
            "overloaded",
        )
    )


def _is_model_not_found_error(msg: str) -> bool:
    return any(k in msg for k in ("404", "not_found", "not found"))


def _is_timeout_error(msg: str) -> bool:
    return any(k in msg for k in ("504", "deadline", "timeout"))


def _call_with_retry(fn: Callable[[], T], max_retries: int = 3) -> T:
    """503 일시 과부하는 지수 백오프(2→4→6초+지터)로 재시도하고 400/401/403/429은 즉시 반환한다."""
    max_overload_retries = min(len(OVERLOAD_RETRY_DELAYS), max(0, max_retries))
    last_err: Exception | None = None
    for attempt in range(max_overload_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_err = exc
            msg = str(exc).lower()
            if _is_fatal_config_or_auth_error(msg) or _is_rate_limit_error(msg):
                raise
            if _is_server_overload_error(msg) and attempt < max_overload_retries:
                wait_s = OVERLOAD_RETRY_DELAYS[attempt] + random.uniform(0.1, 0.7)
                time.sleep(wait_s)
                continue
            if _is_timeout_error(msg) and attempt == 0 and max_overload_retries > 0:
                time.sleep(0.5)
                continue
            raise
    assert last_err is not None
    raise last_err


def _with_thinking_config(
    config: types.GenerateContentConfig | None,
    thinking_config: types.ThinkingConfig,
) -> types.GenerateContentConfig:
    if config is None:
        return types.GenerateContentConfig(thinking_config=thinking_config)
    return config.model_copy(update={"thinking_config": thinking_config})


def _clone_config_for_model(
    config: types.GenerateContentConfig | None,
    cand_model: str,
    fast_mode: bool = False,
) -> types.GenerateContentConfig | None:
    resolved_model = resolve_api_model_name(cand_model)
    if resolved_model == HYDROGEN_MODEL:
        if config is not None and getattr(config, "thinking_config", None) is not None:
            return config
        return _with_thinking_config(
            config,
            types.ThinkingConfig(thinking_level="LOW"),
        )

    if config is None:
        return None
    if not fast_mode:
        return config
    is_lite = "lite" in resolved_model.lower()
    if is_lite:
        # lite 모델은 thinking_budget=0 전달 시 400 INVALID_ARGUMENT가 발생할 수 있어 thinking_config 미지정
        return config
    if getattr(config, "thinking_config", None) is not None:
        return config
    return _with_thinking_config(
        config,
        types.ThinkingConfig(thinking_budget=0),
    )


def _generate_content_with_fallback(
    client: genai.Client,
    model_name: str,
    contents: Any,
    config: types.GenerateContentConfig | None = None,
    fast_mode: bool = False,
    status_callback: Callable[[str], None] | None = None,
) -> Any:
    """
    - 429 RESOURCE_EXHAUSTED: 대기 없이 별도 할당량 보조 모델(gemini-3.5-flash → gemini-3.5-flash-lite)로 즉시 우회
    - 503 UNAVAILABLE/혼잡: 동일 모델 최대 3회(2s→4s→6s+지터) 재시도 후 폴백 체인 순차 전환
    - 400/401/403 설정·권한 오류: 재시도/폴백 없이 즉시 실패
    """
    attempt_models = build_model_fallback_chain(model_name)

    last_err: Exception | None = None
    idx = 0
    while idx < len(attempt_models):
        cand_model = attempt_models[idx]
        max_overload_retries = len(OVERLOAD_RETRY_DELAYS) if idx == 0 else 1
        has_next_model = idx < len(attempt_models) - 1
        next_idx = idx + 1

        for overload_attempt in range(max_overload_retries + 1):
            if status_callback is not None:
                status_callback(
                    f"Gemini API 호출 중 (모델: {cand_model}, 체인 {idx + 1}/{len(attempt_models)}, "
                    f"시도 {overload_attempt + 1}/{max_overload_retries + 1}, 타임아웃 30초)..."
                )
            try:
                eff_cfg = _clone_config_for_model(config, cand_model, fast_mode=fast_mode)
                return client.models.generate_content(
                    model=cand_model,
                    contents=contents,
                    config=eff_cfg,
                )
            except Exception as exc:
                last_err = exc
                msg = str(exc).lower()
                # thinking_config 미지원 400 오류 시 최소 thinking_budget 또는 기본 config로 복구 시도
                if "thinking" in msg and "400" in msg:
                    recovery_configs: list[types.GenerateContentConfig | None] = []
                    if cand_model == HYDROGEN_MODEL:
                        recovery_configs.append(
                            _with_thinking_config(config, types.ThinkingConfig(thinking_budget=256))
                        )
                    if config is not None or cand_model == HYDROGEN_MODEL:
                        recovery_configs.append(config)
                    for rec_cfg in recovery_configs:
                        try:
                            return client.models.generate_content(
                                model=cand_model,
                                contents=contents,
                                config=rec_cfg,
                            )
                        except Exception as exc2:
                            last_err = exc2
                            msg = str(exc2).lower()

                # 설정·인증·권한 오류(400/401/403)는 재시도 및 모델 전환 없이 즉시 중단
                if _is_fatal_config_or_auth_error(msg):
                    raise last_err

                # 429 할당량 초과: 대기 없이 별도 할당량 모델(gemini-3.5-flash → gemini-3.5-flash-lite)로 즉시 우회
                if _is_rate_limit_error(msg):
                    jump_candidates = [
                        j
                        for j in range(idx + 1, len(attempt_models))
                        if attempt_models[j] in RATE_LIMIT_FALLBACK_ORDER
                    ]
                    if jump_candidates:
                        next_idx = jump_candidates[0]
                    elif has_next_model:
                        next_idx = idx + 1
                    else:
                        raise last_err

                    if status_callback is not None:
                        status_callback(
                            f"⚡ {cand_model} 할당량 초과(429) → 대기 없이 {attempt_models[next_idx]}(으)로 즉시 우회..."
                        )
                    break

                # 503 서버 혼잡: 2s → 4s → 6s (+지터) 재시도 후 다음 모델로 전환
                if _is_server_overload_error(msg):
                    if overload_attempt < max_overload_retries:
                        base_wait = OVERLOAD_RETRY_DELAYS[
                            min(overload_attempt, len(OVERLOAD_RETRY_DELAYS) - 1)
                        ]
                        wait_s = round(base_wait + random.uniform(0.1, 0.7), 2)
                        if status_callback is not None:
                            status_callback(
                                f"⚠️ {cand_model} 503 혼잡 · {wait_s:.1f}초 후 재시도 "
                                f"({overload_attempt + 1}/{max_overload_retries})..."
                            )
                        time.sleep(wait_s)
                        continue
                    if has_next_model:
                        next_idx = idx + 1
                        if status_callback is not None:
                            status_callback(
                                f"🔀 {cand_model} 503 지속 → 보조 모델 {attempt_models[next_idx]}(으)로 전환..."
                            )
                        break
                    raise last_err

                if _is_model_not_found_error(msg) and has_next_model:
                    next_idx = idx + 1
                    if status_callback is not None:
                        status_callback(
                            f"🔀 {cand_model} 미지원(404) → 보조 모델 {attempt_models[next_idx]}(으)로 전환..."
                        )
                    break

                if _is_timeout_error(msg) and has_next_model:
                    next_idx = idx + 1
                    time.sleep(0.5)
                    break

                raise last_err

        idx = next_idx

    assert last_err is not None
    raise last_err


def _parse_structured(response: Any, schema_cls: type[T]) -> T:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, schema_cls):
        return parsed
    text = getattr(response, "text", None) or ""
    if not text.strip():
        raise RuntimeError("Gemini 응답이 비어 있습니다. 모델 이름 또는 입력 데이터를 확인해 주세요.")
    return schema_cls.model_validate_json(text)


def prepare_pdf_part(client: genai.Client, pdf_bytes: bytes, display_name: str = "document.pdf") -> Any:
    """PDF가 12MB를 넘으면 Gemini File API로 업로드하고, 작으면 인라인 Part.from_bytes를 사용한다."""
    if len(pdf_bytes) <= 12 * 1024 * 1024:
        return types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")

    uploaded = _call_with_retry(
        lambda: client.files.upload(
            file=io.BytesIO(pdf_bytes),
            config=types.UploadFileConfig(
                mime_type="application/pdf",
                display_name=display_name,
            ),
        )
    )
    # PROCESSING 상태 대기
    for _ in range(20):
        state_name = getattr(getattr(uploaded, "state", None), "name", "")
        if state_name != "PROCESSING":
            break
        time.sleep(1.0)
        uploaded = client.files.get(name=uploaded.name)
    return uploaded


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
    resp = _generate_content_with_fallback(
        client=client,
        model_name=model_name,
        contents=[img_part, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ScannedPageBoxes,
            temperature=0.0,
        ),
        fast_mode=True,
    )
    result = _parse_structured(resp, ScannedPageBoxes)
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
) -> QuestionExtractionOutput:
    """
    크롭된 문항 PNG(가로 최대 1200px 리사이즈), 해설지 정보, 2022 공식 기준표 및 개념서 DB를 Gemini에 전달해
    성취기준 앵커링·원리/용어 분리·최저 학년 근거를 추출하고 후처리 검증한다.
    """
    client = _build_client(api_key, timeout_seconds=DEFAULT_API_TIMEOUT_SECONDS)
    if status_callback is not None:
        status_callback(f"{q_num}번 이미지 최적화 중(≤1200px)...")
    opt_png = optimize_image_for_vision(crop_png_bytes)
    img_part = types.Part.from_bytes(data=opt_png, mime_type="image/png")

    eff_grounding = grounding_context or build_tagging_grounding_context(
        subject_category=subject_category,
        exam_name=exam_name,
        concept_books=concept_books,
    )

    system_inst = (
        "당신은 대한민국 2022 개정 교육과정(수학·사회·과학) 전문 기출 문항 분석가입니다. "
        "반드시 아래 순서와 기준표에 따라 추측(할루시네이션) 없이 정확히 태깅하세요.\n"
        "1) [과목/단원/성취기준 앵커링]: 제공된 [2022 공식 기준표 및 개념서 DB] 안에서만 가장 정확한 2022 공식 `curriculum_subject`(2015 개정 명칭 금지)와 `unit_name`을 정하고, "
        "기준표에 성취기준 코드가 있으면 대표 코드는 `standard_code`, 복합 문항 등 걸치는 모든 코드는 `standard_codes` 배열에 모두 적으세요(목록에 없는 코드를 지어내지 마세요).\n"
        "2) [본질 원리 추출(`core_principles`)]: 상위 과목의 포장 용어를 걷어내고, 이 문제를 실제로 풀기 위해 필요한 근본적 풀이 원리·법칙을 간결한 한국어 개념구로 나열하세요.\n"
        "3) [용어 이중 분리(`advanced_terms` vs `basic_terms`)]: '그림/표/보기/값/cm/g/s' 등 일상어와 기초 단위는 제외하세요.\n"
        "   - `advanced_terms`: 고2~3 선택과목 등 상위 과정에서 도입되어 하위 학년 학생에게는 낯선 전문 용어·기호(예: '몰농도', 'mol', '한계효용', '도함수')\n"
        "   - `basic_terms`: `min_grade` 수준에서 이미 배워 당연한 교과 기본 용어(예: '속력', '질량', '이차방정식', '인권')\n"
        "   - `required_terms`: 문제 지문·보기에 등장하는 핵심 전문 용어 통합 목록(`advanced_terms` + 주요 `basic_terms`)\n"
        "4) [최저 학년(`min_grade`) 및 근거(`grade_evidence`)]: `core_principles`를 2022 개정 교육과정에서 처음 배우는 최저 학년(1=고1, 2=고2, 3=고3)과 1줄 판정 근거를 `grade_evidence`에 명시하세요.\n"
        "5) [정답/해설]: 해설지 참고 내용이나 첨부된 해설지 PDF에서 해당 문항의 공식 정답(객관식은 '1'~'5', 주관식은 값)과 핵심 풀이 요약을 추출하세요."
    )

    hint_block = (
        f"\n[해설지 {q_num}번 발췌]\n{answer_text_hint.strip()}\n"
        if answer_text_hint and answer_text_hint.strip()
        else ""
    )
    grounding_block = f"\n{eff_grounding}\n" if eff_grounding else ""
    prompt = (
        f"시험 회차: {exam_name}\n"
        f"대분류 과목: {subject_category}\n"
        f"문항 번호: {q_num}번\n"
        f"{grounding_block}"
        f"{hint_block}\n"
        "위 이미지의 문항을 기준표에 대조해 분석하고 JSON 스키마에 맞게 추출하세요."
    )

    contents: list[Any] = [img_part]
    if answer_pdf_part is not None and not hint_block:
        contents.append(answer_pdf_part)
    contents.append(prompt)

    resp = _generate_content_with_fallback(
        client=client,
        model_name=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=QuestionExtractionOutput,
            temperature=0.1,
        ),
        fast_mode=True,
        status_callback=status_callback,
    )
    if status_callback is not None:
        status_callback(f"{q_num}번 응답 파싱 및 2022 기준 검증 중...")
    parsed = _parse_structured(resp, QuestionExtractionOutput)
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

    client = _build_client(api_key, timeout_seconds=120)
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

    system_inst = (
        "당신은 대한민국 2022 개정 교육과정(수학·사회·과학) 전문 기출 문항 분석가입니다. "
        "첨부된 전체 모의고사 PDF에서 요청된 모든 문항 번호를 빠짐없이 찾아, "
        "제공된 [2022 공식 기준표 및 개념서 DB]를 근거로 JSON 배열(questions)을 반환하세요.\n"
        "- q_num: 문항 번호 정수 (요청된 문항 번호를 1개도 누락하지 마세요)\n"
        "- curriculum_subject / unit_name / standard_code / standard_codes: 기준표의 2022 공식 과목명(2015 명칭 금지)과 단원을 먼저 매핑하고, 기준표에 코드가 있으면 대표 코드는 standard_code, 복합 문항에 걸친 모든 코드는 standard_codes에 적되 목록에 없는 코드를 지어내지 마세요\n"
        "- core_principles: 상위 과목 포장 용어를 걷어내고 문제를 푸는 데 필요한 근본적 풀이 원리·법칙을 간결한 한국어 개념구로 나열\n"
        "- advanced_terms vs basic_terms vs required_terms: 일상어('그림','표','보기') 및 기초 측정단위('cm','g','s')는 제외하고, 상위 과정에서 새로 도입된 낯선 전문 용어는 advanced_terms, min_grade 시점 기본 용어는 basic_terms, 통합 핵심 용어는 required_terms에 기록\n"
        "- min_grade / grade_evidence: core_principles를 2022 개정 교육과정에서 처음 배우는 최저 학년(1=고1, 2=고2, 3=고3)과 1줄 근거 작성\n"
        "- answer / official_explanation: 해설지 기준 공식 정답(객관식 '1'~'5', 주관식은 값)과 1~2문장 핵심 풀이 요약"
    )

    snippets_map = answer_snippets or {}
    snippet_lines: list[str] = []
    for q in ordered_q:
        sn = (snippets_map.get(q) or "").strip()
        if sn:
            snippet_lines.append(f"[{q}번 해설 발췌] {sn[:500]}")

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

    resp = _generate_content_with_fallback(
        client=client,
        model_name=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=BatchExamExtractionOutput,
            temperature=0.1,
            max_output_tokens=32768,
        ),
        fast_mode=True,
        status_callback=status_callback,
    )
    parsed = _parse_structured(resp, BatchExamExtractionOutput)
    out: dict[int, BatchQuestionTagItem] = {}
    for item in parsed.questions:
        if item.q_num in ordered_q and item.q_num not in out:
            validate_and_normalize_extracted_tag(
                item,
                subject_category=subject_category,
                concept_books=concept_books,
            )
            out[item.q_num] = item
    return out


def _extract_concept_toc_with_ai(
    client: genai.Client,
    model_name: str,
    pdf_bytes: bytes,
    page_catalog: list[dict[str, Any]],
    has_text_layer: bool,
    book_title: str = "",
    subject_and_grade: str = "",
    status_callback: Callable[[str], None] | None = None,
) -> TocExtractionOutput:
    """
    개념서·교과서 PDF 앞부분(p.1~15) 및 목차 페이지를 1차로 분석해
    전체 대단원/중단원 목록(toc_units)과 표지·구성안내·목차가 끝나는 마지막 실제 PDF 페이지(last_toc_page)를 추출한다.
    """
    total_pages = len(page_catalog)
    front_pages = set(range(1, min(total_pages, 15) + 1))
    for entry in page_catalog:
        if "목차" in (entry.get("noise_hint") or "") or "차례" in (entry.get("snippet") or "")[:60]:
            front_pages.add(entry["page"])
    target_pages = sorted(front_pages)[:18]
    if not target_pages:
        return TocExtractionOutput()

    if status_callback is not None:
        status_callback(
            f"📑 1단계: 표지·목차(차례) 경계 및 단원 분석 중 ({target_pages[0]}~{target_pages[-1]}쪽 · 모델: {model_name})..."
        )

    toc_system = (
        "당신은 한국 교과서·개념서(완자, 개념원리, 수능특강 등)의 앞부분(표지·차례)을 정밀 추출하는 AI입니다.\n"
        "제공된 PDF 앞부분에서 목차/차례 부분을 찾아 **모든 대단원과 중단원**을 순서대로 나열하고, "
        "표지·교재 구성 안내·차례(목차)가 모두 끝나고 1단원 개념 본문이 시작되기 직전의 마지막 비-개념 PDF 쪽수(`last_toc_page`)를 판별하세요.\n"
        "- last_toc_page: 반드시 책 하단 인쇄 쪽수가 아닌 `[p.X]` 또는 빨간색 `[ACTUAL_PDF_PAGE: X]` 라벨 기준의 실제 PDF 페이지 번호 정수\n"
        "- unit_name: 목차에 적힌 단원명 (예: 'Ⅰ. 물질의 규칙성과 결합 - 1. 지구와 생명체를 이루는 원소의 생성')\n"
        "- book_page_hint: 목차에 적힌 책 쪽수 (있으면 적고, 없으면 빈 문자열)"
    )

    try:
        if has_text_layer:
            lines = []
            by_page = {e["page"]: e for e in page_catalog}
            for p_num in target_pages:
                e = by_page.get(p_num, {})
                snip = (e.get("snippet") or "").replace("\n", " | ")[:450]
                lines.append(f"[ACTUAL_PDF_PAGE: {p_num}] {snip}")
            prompt = (
                f"- 교재명: {book_title or '(미입력)'} | 과목/학년: {subject_and_grade or '(미입력)'}\n\n"
                "아래 앞부분 페이지 발췌에서 표지/구성안내/목차가 끝나는 `last_toc_page`와 전체 단원 목록(`toc_units`)을 JSON으로 반환하세요.\n\n"
                + "\n".join(lines)
            )
            contents: Any = prompt
        else:
            slice_b, _ = extract_specific_pages_slice(
                pdf_bytes,
                target_pages[:15],
                stamp_actual_page=True,
            )
            pdf_part = prepare_pdf_part(client, slice_b, display_name="concept_toc_front.pdf")
            prompt = (
                f"- 교재명: {book_title or '(미입력)'} | 과목/학년: {subject_and_grade or '(미입력)'}\n"
                "각 페이지 상단에 표시된 `[ACTUAL_PDF_PAGE: 번호]`를 기준으로 표지·교재구성·목차가 끝나는 "
                "`last_toc_page`와 전체 대단원/중단원 목록(`toc_units`)을 JSON으로 반환하세요."
            )
            contents = [pdf_part, prompt]

        resp = _generate_content_with_fallback(
            client=client,
            model_name=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=toc_system,
                response_mime_type="application/json",
                response_schema=TocExtractionOutput,
                temperature=0.0,
            ),
            status_callback=status_callback,
        )
        parsed = _parse_structured(resp, TocExtractionOutput)
        parsed.toc_units = [u for u in parsed.toc_units if u.unit_name.strip()]
        return parsed
    except Exception:
        return TocExtractionOutput()


def _is_same_unit_hint(hint_a: str, hint_b: str) -> bool:
    """두 구간이 같은 단원(대단원/중단원)에 속하는지 판별해 2쪽 이내 짧은 간격 병합 여부를 결정한다."""
    a = re.sub(r"\s+", " ", (hint_a or "").strip())
    b = re.sub(r"\s+", " ", (hint_b or "").strip())
    if not a or not b:
        return True
    if a == b or a in b or b in a:
        return True

    # 대단원 기호/번호 접두어 비교 (예: 'Ⅰ.', 'Ⅱ.', '1.', '2.')
    maj_a = re.match(r"^([ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]|\d{1,2})\s*[.\-)\s]", a)
    maj_b = re.match(r"^([ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]|\d{1,2})\s*[.\-)\s]", b)
    if maj_a and maj_b:
        if maj_a.group(1) != maj_b.group(1):
            return False
        # 같은 대단원 번호이고 앞부분 주제가 비슷하면 동일 단원
        return True

    # 하이픈/대시 앞의 상위 단원명 비교
    pref_a = re.split(r"\s*[-–—:]\s*", a)[0].strip()
    pref_b = re.split(r"\s*[-–—:]\s*", b)[0].strip()
    if pref_a and pref_b and (pref_a == pref_b or pref_a in pref_b or pref_b in pref_a):
        return True

    toks_a = {t for t in re.split(r"[\s·,.\-\(\)]+", a) if len(t) >= 2}
    toks_b = {t for t in re.split(r"[\s·,.\-\(\)]+", b) if len(t) >= 2}
    return bool(toks_a & toks_b)


def _coalesce_concept_ranges(
    items: list[AutoConceptRangeItem],
    max_same_unit_gap: int = 2,
) -> list[AutoConceptRangeItem]:
    """정렬된 개념 구간 리스트에서 겹치거나, 같은 단원 내 2쪽 이내로 떨어진 짧은 간격을 하나로 병합한다."""
    if not items:
        return []
    ordered = sorted(items, key=lambda x: (x.start_page, x.end_page))
    merged: list[AutoConceptRangeItem] = [ordered[0].model_copy()]
    for cur in ordered[1:]:
        prev = merged[-1]
        gap = cur.start_page - prev.end_page - 1
        same_unit = _is_same_unit_hint(prev.unit_hint, cur.unit_hint)
        if gap <= 0 or (same_unit and gap <= max_same_unit_gap):
            prev.end_page = max(prev.end_page, cur.end_page)
            if cur.unit_hint and cur.unit_hint not in prev.unit_hint and len(prev.unit_hint) < 40:
                pass
            if gap > 0:
                prev.reason = (prev.reason or "개념 설명") + f" (단원 내 {gap}쪽 간격 병합)"
        else:
            merged.append(cur.model_copy())
    return merged


def _run_chunk_with_backoff(
    fn: Callable[[], AutoConceptPageDetectionOutput],
    chunk_label: str,
    max_retries: int = 2,
    status_callback: Callable[[str], None] | None = None,
) -> AutoConceptPageDetectionOutput:
    """
    429, 타임아웃, 503, JSON 파싱 실패 등 일시 오류 발생 시 지수 백오프(2s→4s)로 최소 2회 재시도한다.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            if _is_fatal_config_or_auth_error(msg) or attempt >= max_retries:
                raise
            delay = min(10.0, 2.0 * (2**attempt) + random.uniform(0.15, 0.75))
            if status_callback is not None:
                status_callback(
                    f"⚠️ {chunk_label} 일시 오류({type(exc).__name__}) → {delay:.1f}초 후 재시도 ({attempt + 1}/{max_retries})..."
                )
            time.sleep(delay)
    raise RuntimeError(str(last_exc))


def detect_concept_pages_with_ai(
    api_key: str,
    model_name: str,
    pdf_bytes: bytes,
    page_catalog: list[dict[str, Any]],
    has_text_layer: bool,
    book_title: str = "",
    subject_and_grade: str = "",
    status_callback: Callable[[str], None] | None = None,
) -> tuple[AutoConceptPageDetectionOutput, list[tuple[int, int]]]:
    """
    1) 먼저 앞부분(p.1~15)에서 목차 및 표지·목차 끝 페이지(last_toc_page)를 추출하고,
    2) 목차 단원과 본문 페이지를 `[ACTUAL_PDF_PAGE]` 기준으로 대조해
       표지·목차·대단원 종합문제·정답지를 제외한 개념 구간(탐구/개념확인퀴즈 포함)을 판별하며,
    3) 실패 청크를 누락 없이 failed_chunks에 기록하고 같은 단원 2쪽 이내 간격을 병합한다.
    """
    total_pages = len(page_catalog)
    if total_pages == 0:
        raise ValueError("PDF에 페이지가 없습니다.")

    client = _build_client(api_key, timeout_seconds=90)

    # 1단계: 앞부분 표지/목차 분석
    toc_out = _extract_concept_toc_with_ai(
        client=client,
        model_name=model_name,
        pdf_bytes=pdf_bytes,
        page_catalog=page_catalog,
        has_text_layer=has_text_layer,
        book_title=book_title,
        subject_and_grade=subject_and_grade,
        status_callback=status_callback,
    )
    toc_items = toc_out.toc_units
    toc_unit_names = [t.unit_name.strip() for t in toc_items if t.unit_name.strip()]

    # 앞부분(p.1~15) 표지·차례 마지막 페이지 컷오프 계산
    heuristic_toc_end = 0
    for entry in page_catalog[:15]:
        snip_head = (entry.get("snippet") or "")[:120]
        if entry.get("noise_hint") in ("목차/차례", "표지/안내") or any(
            k in snip_head for k in ("차례", "목차", "이 책의 구성", "구성과 특징")
        ):
            heuristic_toc_end = max(heuristic_toc_end, int(entry["page"]))

    front_cutoff = max(int(toc_out.last_toc_page or 0), heuristic_toc_end)
    if front_cutoff == 0 and total_pages > 8:
        front_cutoff = 1  # 최소 1쪽 표지는 항상 제외
    front_cutoff = min(front_cutoff, min(total_pages - 1, 20))

    toc_guide_block = ""
    if toc_items:
        toc_lines = [
            f"  {idx}. {t.unit_name.strip()}"
            + (f" (목차 쪽수: {t.book_page_hint.strip()})" if t.book_page_hint.strip() else "")
            for idx, t in enumerate(toc_items, start=1)
        ]
        toc_guide_block = (
            "[1단계 목차 분석 결과 — 본문 대조 기준표]\n"
            + "\n".join(toc_lines)
            + f"\n※ 표지·교재구성·목차는 1~{front_cutoff}쪽까지로 판단되었으므로 개념 구간은 최소 {front_cutoff + 1}쪽 이후부터 시작하세요.\n\n"
        )
        if status_callback is not None:
            status_callback(
                f"📋 목차 {len(toc_items)}개 단원 · 앞부분 제외 경계(p.1~p.{front_cutoff}) 확인 → 2단계 본문 구간 판별 중..."
            )

    system_inst = (
        "당신은 대한민국 교과서 및 개념서(완자, 개념원리, 수능특강 등) PDF 구조 판별 AI입니다.\n"
        "1단계에서 추출한 [목차 단원 기준표]와 각 페이지를 대조하여 개념 설명 및 관련 학습 구간을 빠짐없이 찾으세요.\n\n"
        "**[매우 중요 — 페이지 번호 기준]**\n"
        "- 책 하단에 인쇄된 쪽수나 첨부 파일의 순서가 아니라, 각 페이지 상단 여백에 적힌 **`[ACTUAL_PDF_PAGE: 번호]`** 를 그대로 `start_page`, `end_page`로 반환하세요.\n\n"
        "**[제외 대상 (반드시 제외)]**\n"
        "- 표지, 머리말, 교재 구성 안내, 차례(목차)\n"
        "- 대단원 종합 문제, 실전 문제, 내신 만점 테스트, 수능 실전 문제 전용 페이지 구간\n"
        "- 정답 및 해설지, 빠른 정답 찾기, 채점 기준표 페이지\n\n"
        "**[포함 대상 (파편화 방지를 위해 반드시 개념 구간에 포함)]**\n"
        "- 대단원/중단원 도입, 핵심 개념 정리, 개념 설명 본문\n"
        "- 개념 설명에 이어지거나 개념 사이에 포함된 **탐구 활동, 자료 분석, 실험 원리, 용어 정리, 개념 확인 퀴즈(개념 체크 문제)**\n\n"
        "**[출력 규칙]**\n"
        "- detected_ranges: 같은 단원의 연속 개념 구간마다 unit_hint(가급적 목차 단원명과 통일), start_page, end_page(`[ACTUAL_PDF_PAGE]` 정수), reason을 적으세요.\n"
        "- excluded_summary: 제외한 표지/목차/대단원 종합문제/정답 구간을 간단히 요약하세요."
    )

    all_items: list[AutoConceptRangeItem] = []
    excluded_notes: list[str] = []
    failed_chunks: list[str] = []

    def _normalize_chunk_items(
        raw_ranges: list[AutoConceptRangeItem],
        chunk_pages: list[int],
    ) -> list[AutoConceptRangeItem]:
        c_min, c_max = chunk_pages[0], chunk_pages[-1]
        c_len = len(chunk_pages)
        normalized: list[AutoConceptRangeItem] = []
        for it in raw_ranges:
            s_p = int(it.start_page)
            e_p = int(it.end_page)
            if s_p > e_p:
                s_p, e_p = e_p, s_p
            # AI가 ACTUAL_PDF_PAGE 대신 1..len(chunk) 로컬 번호를 반환한 경우 자동 오프셋 복원
            if c_min > c_len and 1 <= s_p <= c_len and 1 <= e_p <= c_len:
                s_p = chunk_pages[s_p - 1]
                e_p = chunk_pages[e_p - 1]
            s_p = max(c_min, min(c_max, s_p))
            e_p = max(c_min, min(c_max, e_p))
            if s_p <= e_p:
                normalized.append(
                    AutoConceptRangeItem(
                        unit_hint=it.unit_hint.strip() or f"개념 구간 p.{s_p}~p.{e_p}",
                        start_page=s_p,
                        end_page=e_p,
                        reason=it.reason.strip(),
                    )
                )
        return normalized

    if has_text_layer:
        batch_size = 80
        batches = [page_catalog[i : i + batch_size] for i in range(0, total_pages, batch_size)]
        if status_callback is not None:
            status_callback(
                f"⚡ AI 목차 대조 기반 개념 구간 병렬 판별 중 ({len(batches)}개 구간 · 전체 {total_pages}쪽)..."
            )

        def _detect_text_batch(batch: list[dict[str, Any]]) -> tuple[list[int], AutoConceptPageDetectionOutput]:
            b_pages = [int(e["page"]) for e in batch]
            b_label = f"p.{b_pages[0]}~p.{b_pages[-1]}"

            def _call() -> AutoConceptPageDetectionOutput:
                local_client = _build_client(api_key, timeout_seconds=90)
                lines = []
                for entry in batch:
                    hint_tag = f" [로컬감지:{entry['noise_hint']}]" if entry["noise_hint"] else ""
                    snip = entry["snippet"].replace("\n", " | ")[:340]
                    lines.append(
                        f"[ACTUAL_PDF_PAGE: {entry['page']}{hint_tag}] "
                        f"{snip if snip else '(텍스트 없음/이미지 페이지)'}"
                    )
                prompt = (
                    f"- 교재명: {book_title or '(미입력)'} | 과목/학년: {subject_and_grade or '(미입력)'}\n"
                    f"- 분석 대상 구간: [ACTUAL_PDF_PAGE: {b_pages[0]}] ~ [ACTUAL_PDF_PAGE: {b_pages[-1]}] (전체 {total_pages}쪽)\n\n"
                    f"{toc_guide_block}"
                    "아래 페이지별 `[ACTUAL_PDF_PAGE: 번호]`를 기준으로 개념 구간(`detected_ranges`)을 JSON으로 반환하세요.\n\n"
                    + "\n".join(lines)
                )
                resp = _generate_content_with_fallback(
                    client=local_client,
                    model_name=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_inst,
                        response_mime_type="application/json",
                        response_schema=AutoConceptPageDetectionOutput,
                        temperature=0.0,
                    ),
                    fast_mode=True,
                    status_callback=None,
                )
                return _parse_structured(resp, AutoConceptPageDetectionOutput)

            out_parsed = _run_chunk_with_backoff(_call, b_label, max_retries=2, status_callback=status_callback)
            return b_pages, out_parsed

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(2, max(1, len(batches)))) as ex:
            fut_to_pages = {ex.submit(_detect_text_batch, b): [int(e["page"]) for e in b] for b in batches}
            for fut, b_pages in fut_to_pages.items():
                try:
                    _, parsed = fut.result()
                    all_items.extend(_normalize_chunk_items(parsed.detected_ranges, b_pages))
                    if parsed.excluded_summary.strip():
                        excluded_notes.append(parsed.excluded_summary.strip())
                except Exception as exc:
                    err_label = f"p.{b_pages[0]}~p.{b_pages[-1]} ({exc})"
                    failed_chunks.append(err_label)
                    if status_callback is not None:
                        status_callback(f"❌ 구간 판별 실패: {err_label}")
    else:
        # 스캔(이미지) PDF: 20쪽씩 자른 슬라이스 각 페이지 상단에 [ACTUAL_PDF_PAGE: X] 각인 후 max_workers=2로 처리
        batch_size = 20
        page_batches = [
            [int(entry["page"]) for entry in page_catalog[i : i + batch_size]]
            for i in range(0, total_pages, batch_size)
        ]
        if status_callback is not None:
            status_callback(
                f"🖼️ 스캔 PDF AI 병렬 시각 판별 중 ({len(page_batches)}개 구간 · 동시 워커 최대 2개 · 전체 {total_pages}쪽)..."
            )

        def _detect_scan_batch(batch_pages: list[int]) -> AutoConceptPageDetectionOutput:
            b_label = f"p.{batch_pages[0]}~p.{batch_pages[-1]}"
            slice_b, _ = extract_specific_pages_slice(
                pdf_bytes,
                batch_pages,
                stamp_actual_page=True,
            )

            def _call() -> AutoConceptPageDetectionOutput:
                local_client = _build_client(api_key, timeout_seconds=120)
                pdf_part = prepare_pdf_part(
                    local_client,
                    slice_b,
                    display_name=f"scan_batch_{batch_pages[0]}_{batch_pages[-1]}.pdf",
                )
                prompt = (
                    f"- 교재명: {book_title or '(미입력)'} | 과목/학년: {subject_and_grade or '(미입력)'}\n"
                    f"- 이 슬라이스 PDF는 원본의 **`[ACTUAL_PDF_PAGE: {batch_pages[0]}]` ~ `[ACTUAL_PDF_PAGE: {batch_pages[-1]}]`** 입니다.\n\n"
                    f"{toc_guide_block}"
                    "각 페이지 상단 여백에 각인된 빨간색 라벨 **`[ACTUAL_PDF_PAGE: 번호]`** 를 보고, "
                    "반드시 그 `ACTUAL_PDF_PAGE` 번호 기준으로 `start_page`와 `end_page`를 반환하세요."
                )
                resp = _generate_content_with_fallback(
                    client=local_client,
                    model_name=model_name,
                    contents=[pdf_part, prompt],
                    config=types.GenerateContentConfig(
                        system_instruction=system_inst,
                        response_mime_type="application/json",
                        response_schema=AutoConceptPageDetectionOutput,
                        temperature=0.0,
                    ),
                    fast_mode=True,
                    status_callback=None,
                )
                return _parse_structured(resp, AutoConceptPageDetectionOutput)

            return _run_chunk_with_backoff(_call, b_label, max_retries=2, status_callback=status_callback)

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(2, max(1, len(page_batches)))) as ex:
            fut_to_pages = {ex.submit(_detect_scan_batch, bp): bp for bp in page_batches}
            for fut, bp in fut_to_pages.items():
                try:
                    parsed = fut.result()
                    all_items.extend(_normalize_chunk_items(parsed.detected_ranges, bp))
                    if parsed.excluded_summary.strip():
                        excluded_notes.append(parsed.excluded_summary.strip())
                except Exception as exc:
                    err_label = f"p.{bp[0]}~p.{bp[-1]} ({exc})"
                    failed_chunks.append(err_label)
                    if status_callback is not None:
                        status_callback(f"❌ 스캔 구간 판별 실패: {err_label}")

    # 앞부분(표지·교재구성·목차) 컷오프 적용 (p.1 ~ front_cutoff 자동 제외)
    trimmed_items: list[AutoConceptRangeItem] = []
    for it in all_items:
        s_p = max(1, min(total_pages, int(it.start_page)))
        e_p = max(1, min(total_pages, int(it.end_page)))
        if s_p > e_p:
            s_p, e_p = e_p, s_p
        if e_p <= front_cutoff:
            continue
        if s_p <= front_cutoff:
            s_p = front_cutoff + 1
        if s_p <= e_p:
            trimmed_items.append(
                AutoConceptRangeItem(
                    unit_hint=it.unit_hint,
                    start_page=s_p,
                    end_page=e_p,
                    reason=it.reason,
                )
            )

    if front_cutoff >= 1:
        excluded_notes.insert(0, f"p.1~p.{front_cutoff} 표지/목차 자동 제외")

    # 같은 단원 내 2쪽 이내 짧은 간격 병합 + 최종 연속 구간 생성
    coalesced_items = _coalesce_concept_ranges(trimmed_items, max_same_unit_gap=2)

    raw_intervals = sorted((it.start_page, it.end_page) for it in coalesced_items)
    merged_intervals: list[tuple[int, int]] = []
    for s_p, e_p in raw_intervals:
        if not merged_intervals or s_p > merged_intervals[-1][1] + 1:
            merged_intervals.append((s_p, e_p))
        else:
            merged_intervals[-1] = (merged_intervals[-1][0], max(merged_intervals[-1][1], e_p))

    merged_result = AutoConceptPageDetectionOutput(
        toc_units=toc_unit_names,
        detected_ranges=coalesced_items,
        excluded_summary=" / ".join(dict.fromkeys(excluded_notes)),
        failed_chunks=failed_chunks,
    )
    return merged_result, merged_intervals


def _merge_concept_units(units: list[ConceptUnit]) -> list[ConceptUnit]:
    """청크 분할 추출 시 동일 단원명이 나뉜 경우 핵심 개념·허용 용어·페이지 범위를 병합한다."""
    merged: list[ConceptUnit] = []
    by_name: dict[str, ConceptUnit] = {}
    for u in units:
        key = u.unit_name.strip()
        if not key:
            continue
        if key not in by_name:
            clone = ConceptUnit(
                unit_name=key,
                page_range=u.page_range.strip(),
                core_concepts=list(dict.fromkeys(c.strip() for c in u.core_concepts if c.strip())),
                allowed_terms_and_units=list(
                    dict.fromkeys(t.strip() for t in u.allowed_terms_and_units if t.strip())
                ),
            )
            by_name[key] = clone
            merged.append(clone)
        else:
            existing = by_name[key]
            if u.page_range.strip() and u.page_range.strip() not in existing.page_range:
                existing.page_range = f"{existing.page_range}, {u.page_range.strip()}".strip(", ")
            for c in u.core_concepts:
                cs = c.strip()
                if cs and cs not in existing.core_concepts:
                    existing.core_concepts.append(cs)
            for t in u.allowed_terms_and_units:
                ts = t.strip()
                if ts and ts not in existing.allowed_terms_and_units:
                    existing.allowed_terms_and_units.append(ts)
    return merged


def _has_rich_extracted_text(text: str, page_count: int = 1) -> bool:
    """추출된 텍스트에 페이지당 충분한 본문 텍스트가 있는지 판별한다 (스캔 PDF 여부 구분)."""
    if not text:
        return False
    stripped = re.sub(r"={2,}.*?={2,}|-{2,}.*?-{2,}|\(도표/이미지 중심 개념 페이지\)", "", text)
    compact_len = len(re.sub(r"\s+", "", stripped))
    return compact_len >= max(80, max(1, page_count) * 60)


def extract_concept_book_units(
    api_key: str,
    model_name: str,
    book_title: str,
    subject_and_grade: str,
    page_ranges_display: str,
    sliced_pdf_bytes: bytes,
    mapped_concept_text: str,
    original_pdf_bytes: bytes | None = None,
    kept_pages: list[int] | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> ConceptBookExtractionOutput:
    """
    지정된 페이지 범위에서 추출된 개념 텍스트(및 필요 시 슬라이스 PDF)를
    Gemini 모델(model_name)로 병렬 분석하여 단원별(대단원/중단원) 개념 + 페이지 번호 구조화 DB를 생성한다.
    텍스트가 풍부한 PDF는 텍스트 전용 경량 프롬프트 + 50쪽 청크 병렬 처리로 속도를 극대화한다.
    """
    system_inst = (
        "당신은 2022 개정 교육과정 개념서·교과서 구조화 전문가입니다.\n"
        "제공된 개념서 페이지 번호 매핑 텍스트(`=== [실제 페이지: p.X] ===`) 또는 PDF를 분석하여 "
        "단원(대단원/중단원)별로 정확히 정리하세요.\n"
        "엄격 준수 규칙:\n"
        "1. 불필요한 목차, 정답지, 문제 풀이 내용은 제외하고 순수 개념 설명만 구조화하세요.\n"
        "2. 각 단원(`units`)의 `unit_name`은 대단원/중단원명을 명확히 적으세요 (예: 'Ⅰ. 물질과 규칙성 - 1. 원소의 생성', 'Ⅲ. 변화와 다양성').\n"
        "3. `page_range`는 반드시 텍스트(`=== [실제 페이지: p.X] ===`) 또는 페이지 상단 여백의 빨간색 `[ACTUAL_PDF_PAGE: X]` 라벨에 적힌 실제 페이지 번호를 기준으로 `'p.12 ~ p.25'` 형태로 정확히 표기하세요.\n"
        "4. `core_concepts`에는 해당 단원 페이지에 실제로 설명된 핵심 원리, 법칙, 정의, 공식을 구체적으로 나열하세요.\n"
        "5. `allowed_terms_and_units`에는 해당 단원 페이지에 실제로 등장하여 학생이 사용할 수 있는 모든 교과 용어, 기호, 공식, 단위(예: mL, 온도, 이온 수 등)를 빠짐없이 나열하세요.\n"
        "6. 제공된 페이지 범위에 없는 상위 과목 개념이나 다른 단원 용어를 절대 임의로 추가하지 마세요."
    )

    def _run_chunk(
        chunk_pdf: bytes | None,
        chunk_text: str,
        chunk_label: str,
        page_cnt: int,
    ) -> ConceptBookExtractionOutput:
        local_client = _build_client(api_key, timeout_seconds=90)
        text_is_rich = _has_rich_extracted_text(chunk_text, page_cnt)
        excerpt = chunk_text[:65000] if chunk_text else "(이미지 중심 개념서 PDF)"
        prompt = (
            f"- 교재명(book_title): {book_title}\n"
            f"- 과목 및 학년(subject / grade): {subject_and_grade}\n"
            f"- 추출 대상 개념 페이지 범위: {chunk_label}\n\n"
            "아래 페이지별 추출 개념 텍스트를 정독한 뒤, "
            "단원별로 구분하여 JSON 스키마(book_title, subject, grade, units)에 맞게 반환하세요.\n\n"
            f"[페이지별 개념 텍스트]\n{excerpt}"
        )
        if text_is_rich or chunk_pdf is None:
            contents: Any = prompt
        else:
            pdf_part = prepare_pdf_part(local_client, chunk_pdf, display_name="concept_book_slice.pdf")
            contents = [pdf_part, prompt]

        resp = _generate_content_with_fallback(
            client=local_client,
            model_name=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_inst,
                response_mime_type="application/json",
                response_schema=ConceptBookExtractionOutput,
                temperature=0.0,
            ),
            fast_mode=True,
            status_callback=None,
        )
        return _parse_structured(resp, ConceptBookExtractionOutput)

    total_kept = len(kept_pages) if kept_pages else 1
    is_text_rich_book = _has_rich_extracted_text(mapped_concept_text, total_kept)
    chunk_size = 50 if is_text_rich_book else 25

    if original_pdf_bytes is not None and kept_pages and len(kept_pages) > chunk_size:
        chunk_specs: list[tuple[bytes | None, str, str, int]] = []
        total_chunks = (len(kept_pages) + chunk_size - 1) // chunk_size
        for c_idx, offset in enumerate(range(0, len(kept_pages), chunk_size), start=1):
            sub_pages = kept_pages[offset : offset + chunk_size]
            sub_label = f"p.{sub_pages[0]}~p.{sub_pages[-1]} ({c_idx}/{total_chunks} 구간)"
            sub_pdf, sub_txt = extract_specific_pages_slice(original_pdf_bytes, sub_pages)
            chunk_specs.append((None if is_text_rich_book else sub_pdf, sub_txt, sub_label, len(sub_pages)))

        if status_callback is not None:
            status_callback(
                f"⚡ 단원별 개념 구조화 병렬 분석 중 ({len(chunk_specs)}개 구간 동시 처리 · "
                f"{'텍스트 고속 모드' if is_text_rich_book else 'Vision 모드'})..."
            )

        collected_units: list[ConceptUnit] = []
        detected_grade = ""
        worker_limit = 4 if is_text_rich_book else 2
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(worker_limit, len(chunk_specs))) as ex:
            results = list(ex.map(lambda sp: _run_chunk(*sp), chunk_specs))

        for sub_out in results:
            collected_units.extend(sub_out.units)
            if not detected_grade and sub_out.grade:
                detected_grade = sub_out.grade

        merged_units = _merge_concept_units(collected_units)
        return ConceptBookExtractionOutput(
            book_title=book_title.strip(),
            subject=subject_and_grade.strip(),
            grade=detected_grade or "고1",
            units=merged_units,
        )

    parsed = _run_chunk(
        None if is_text_rich_book else sliced_pdf_bytes,
        mapped_concept_text,
        page_ranges_display,
        total_kept,
    )
    if book_title.strip():
        parsed.book_title = book_title.strip()
    if subject_and_grade.strip():
        parsed.subject = subject_and_grade.strip()
    return parsed


def extract_concept_whitelist(
    api_key: str,
    model_name: str,
    sliced_pdf_bytes: bytes,
    sliced_text: str,
    start_page: int,
    end_page: int,
    book_title: str = "",
    range_display: str = "",
) -> ConceptWhitelist:
    """업로드된 개념서·교과서 PDF 지정 페이지 범위에서 허용 개념과 허용 용어 화이트리스트를 추출한다."""
    client = _build_client(api_key, timeout_seconds=60)
    page_span = max(1, end_page - start_page + 1)
    text_is_rich = _has_rich_extracted_text(sliced_text, page_span)

    system_inst = (
        "당신은 엄격한 교육과정 범위 검증관입니다. "
        "제공된 개념서/교과서 페이지에 **실제로 설명되어 있는 내용만** 추출해야 합니다. "
        "뒤 단원이나 상위 과목에 나오는 개념·용어를 절대 추측하거나 추가하지 마세요.\n"
        "1. pdf_concepts: 이 페이지 범위에서 실제로 설명·정의·예시된 개념, 원리, 법칙, 계산 방법을 빠짐없이 나열하세요.\n"
        "2. pdf_vocabulary: 이 페이지 범위 본문, 그림, 표, 공식, 용어 사전 박스에 실제로 등장하는 "
        "모든 교과 용어, 기호, 단위를 나열하세요.\n"
        "3. units: 이 페이지 범위에 포함된 단원명(unit_name), 페이지 범위(page_range, 예: 'p.12 ~ p.25'), "
        "핵심 개념(core_concepts), 허용 용어·단위(allowed_terms_and_units)를 단원별로 함께 정리하세요."
    )

    effective_range = range_display.strip() or f"p.{start_page} ~ p.{end_page}"
    excerpt = sliced_text[:30000] if sliced_text else "(이미지 중심 PDF)"
    prompt = (
        f"교재명: {book_title or '업로드 개념서 PDF'}\n"
        f"대상 PDF 페이지 범위: {effective_range}\n"
        "제공된 개념서 페이지 내용을 정독한 뒤, "
        "이 페이지 범위 안에 실제로 존재하는 개념·원리(pdf_concepts), 허용 용어(pdf_vocabulary), 그리고 단원별 정보(units)를 추출하세요.\n\n"
        f"[추출 텍스트]\n{excerpt}"
    )

    if text_is_rich:
        contents: Any = prompt
    else:
        pdf_part = prepare_pdf_part(client, sliced_pdf_bytes, display_name="concept_slice.pdf")
        contents = [pdf_part, prompt]

    resp = _generate_content_with_fallback(
        client=client,
        model_name=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=ConceptWhitelist,
            temperature=0.0,
        ),
        fast_mode=True,
    )
    wl = _parse_structured(resp, ConceptWhitelist)
    if book_title.strip():
        wl.book_title = book_title.strip()
    if not wl.units:
        wl.units = [
            ConceptUnit(
                unit_name=wl.subject_guess or "지정 개념 범위",
                page_range=effective_range,
                core_concepts=list(wl.pdf_concepts),
                allowed_terms_and_units=list(wl.pdf_vocabulary),
            )
        ]
    return wl


def match_questions_to_whitelist(
    api_key: str,
    model_name: str,
    candidates: list[QuestionRecord],
    whitelist: ConceptWhitelist,
) -> list[QuestionMatchJudgment]:
    """후보 기출 문항들의 단원과 core_principles가 사용자가 선택한 개념서 단원·pdf_concepts 범위만으로 해결 가능한지 엄격히 판정한다."""
    if not candidates:
        return []

    client = _build_client(api_key)
    selected_unit_names = [u.unit_name for u in whitelist.units if u.unit_name]
    system_inst = (
        "당신은 기출 문항과 개념서 PDF 범위의 정합성을 판정하는 매우 엄격한 교육과정 심사관입니다.\n"
        "억지 매칭·억지 개조를 차단하기 위해 아래 규칙을 반드시 지키세요:\n"
        "1. is_solvable_by_pdf_only (엄격 일치 필수):\n"
        "   - 후보 문항의 **단원(unit_name)**과 **본질 원리(core_principles)**, 그리고 **실제 풀이(official_explanation)**가 "
        "사용자가 선택한 개념서 단원 목록(`whitelist.units`)의 핵심 주제 및 `pdf_concepts`와 **직접 일치**하고, "
        "해당 범위(+ 초중등 기초 연산·독해)만으로 **100% 완전히 해결 가능할 때만 True**로 판정하세요.\n"
        "   - **반드시 False로 제외해야 하는 경우:**\n"
        "     (a) 선택한 단원과 무관한 다른 단원 문항 (예: '산, 염기와 중화 반응' 단원을 선택했는데 '동적 평형', '가역 반응', '물 분자의 이온화' 단원 문항이 섞인 경우)\n"
        "     (b) 개념서 범위를 벗어나는 상위 실험 기법·계산 원리 (예: 고1 통합과학 범위를 벗어난 '중화 적정 실험', '피펫 분취 및 희석 계산', '이상기체 상태방정식' 등)를 말장난식 용어 치환으로 우회해야만 풀리는 문항\n"
        "     (c) 개념서에 없는 상위 원리·공식이 하나라도 필수인 문항\n"
        "2. out_of_scope_terms: 문항의 required_terms 중 pdf_vocabulary에 없는 상위/범위 외 전문 용어·단위(`M`, `몰농도`, `mol`, `몰수`, `분자량`, `화학식량`, `엔탈피` 등)를 빠짐없이 적으세요.\n"
        "3. needs_rewriting: is_solvable_by_pdf_only가 True이고 out_of_scope_terms가 1개 이상이면 True, 용어까지 모두 개념서 범위 안이면 False입니다.\n"
        "4. matched_unit_name 및 matched_page_range: is_solvable_by_pdf_only가 True인 경우, "
        "제공된 [개념서 단원별 상세 목록] 중 이 문항의 핵심 원리가 수록된 가장 정확한 단원명(`unit_name`)과 페이지 범위(`page_range`, 예: 'p.184~189')를 그대로 적으세요."
    )

    units_block = ""
    if whitelist.units:
        unit_lines = [
            f"- 단원명: {u.unit_name} | 페이지: {u.page_range} | 핵심 개념: {u.core_concepts[:15]} | 허용 용어: {u.allowed_terms_and_units[:20]}"
            for u in whitelist.units
        ]
        units_block = "[개념서 단원별 상세 목록 — 이 단원 외 타 단원 문항은 모두 제외]\n" + "\n".join(unit_lines) + "\n\n"

    all_judgments: list[QuestionMatchJudgment] = []
    batch_size = 25

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        cand_lines = [
            json.dumps(
                {
                    "id": q.id,
                    "subj": f"{q.subject_category}/{q.curriculum_subject}",
                    "unit": q.unit_name,
                    "gr": q.min_grade,
                    "prin": q.core_principles,
                    "terms": q.required_terms,
                    "exp": q.official_explanation,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for q in batch
        ]

        prompt = (
            f"[참고 교재명] {whitelist.book_title or '업로드 개념서 PDF'}\n"
            f"[선택된 단원명 목록] {selected_unit_names or [whitelist.subject_guess]}\n"
            f"[개념서 범위 요약]\n{whitelist.pdf_summary}\n\n"
            f"{units_block}"
            f"[허용 개념·원리 (pdf_concepts)]\n{whitelist.pdf_concepts}\n\n"
            f"[허용 용어 목록 (pdf_vocabulary)]\n{whitelist.pdf_vocabulary}\n\n"
            "[평가할 기출 문항 목록 (JSONL)]\n" + "\n".join(cand_lines)
        )

        resp = _generate_content_with_fallback(
            client=client,
            model_name=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_inst,
                response_mime_type="application/json",
                response_schema=BatchMatchResult,
                temperature=0.0,
            ),
        )
        batch_res = _parse_structured(resp, BatchMatchResult)
        all_judgments.extend(batch_res.judgments)

    return all_judgments


def rewrite_question_for_pdf(
    api_key: str,
    model_name: str,
    question: QuestionRecord,
    crop_png_bytes: bytes | None,
    whitelist: ConceptWhitelist,
    out_of_scope_terms: list[str],
    sliced_text: str = "",
    previous_attempt: RewrittenQuestion | None = None,
    extra_feedback: str = "",
) -> RewrittenQuestion:
    """
    풀이 원리는 PDF 범위 안이지만 상위 과목 용어(out_of_scope_terms)가 포함된 문항을
    업로드된 PDF 허용 용어만 사용해 개조하고, 금지 용어 잔존 여부를 후처리 정규식으로 검증한다.
    잔존 시 누락 용어를 명시해 1회 자동 재개조하며, 재개조 후에도 잔존 시 ForbiddenTermRemainingError를 발생시킨다.
    """
    client = _build_client(api_key)

    system_inst = (
        "당신은 고난도 교육과정 기반 문항 리라이터(PDF-Grounded Rewriter)입니다.\n"
        "다음 절대 원칙을 반드시 지키세요:\n"
        "1. 업로드된 개념서 PDF에 있는 개념(pdf_concepts)과 용어(pdf_vocabulary), 그리고 일반 일상 한국어만 사용하세요.\n"
        "2. 범위 외 상위 용어 목록(out_of_scope_terms) 및 개념서에 없는 상위 단위/용어(`M`, `몰 농도`, `몰농도`, `mol`, `몰수`, `분자량`, `화학식량`, `엔탈피` 등)는 개조된 지문, <자료>, <보기>, 선지 어디에도 절대 남기지 마세요.\n"
        "3. **핵심 원리 억지 동어반복 치환 금지**: 개념서 범위를 벗어나는 실험/원리 개념 자체를 동어반복 문장으로 어색하게 풀어쓰지 말고, 개념서 허용 용어와 구체적 수치·입자 수 비율로 논리가 온전히 성립되게 표현하세요.\n"
        "4. **자연스러운 국문 문장 구조 유지**: 용어 치환 후 문장 끝이 어색하게 끊기지 않도록(예: `'질량: x g x는?'` 금지 → `'질량을 x g이라 할 때, x는?'`) 조사와 어미를 매끄러운 시험지 문체로 다듬으세요.\n"
        "5. **구조화 줄바꿈**: `<자료>`와 `<보기>`는 반드시 독립된 줄로 분리하고, `<보기>` 내 `ㄱ.`, `ㄴ.`, `ㄷ.` 항목은 각각 별도의 줄에 적으세요.\n"
        "6. **그림·그래프 시각 자료 안내(`visual_term_note`)**: 원본 그림·표·그래프의 축이나 범례에 상위 단위(예: `mol`, `M`)가 그려져 있다면, `visual_term_note`에 예: `그림 속 'mol'은 본문의 '입자 수'와 같은 의미로 해석합니다`처럼 학생용 안내 문구를 작성하세요(시각 자료에 상위 용어가 없으면 빈 문자열 `\"\"`).\n"
        "7. 문제의 수학적·논리적 구조, 수치 조건, 정답의 참/거짓 판정 로직은 원본과 **100% 동일하게** 유지하며 `preserved_answer`도 원본 정답과 동일해야 합니다.\n"
        "8. `rewritten_choices`는 객관식이면 5개 항목 리스트(선지 번호 ①~⑤ 제외하고 내용만), 주관식이면 빈 리스트 []로 반환하세요."
    )

    q_text = (getattr(question, "question_text", "") or "").strip()
    has_visual_cue = any(
        kw in q_text
        for kw in ("그림", "그래프", "표", "도표", "모식도", "사진", "지도", "회로", "실험 장치")
    )
    need_image = bool(crop_png_bytes) and (len(q_text) < 40 or has_visual_cue)

    context_excerpt = sliced_text[:6000] if sliced_text else ""
    q_text_block = f"[원본 문항 텍스트]\n{q_text}\n\n" if q_text else ""

    def _invoke_rewrite(correction_note: str = "") -> RewrittenQuestion:
        prev_block = ""
        if previous_attempt is not None:
            prev_block = (
                "\n[직전 개조 결과 참고 (개선·재변환 필요)]\n"
                f"- 직전 지문: {previous_attempt.rewritten_stem}\n"
                f"- 직전 선지: {previous_attempt.rewritten_choices}\n"
            )
        feedback_block = f"\n[재변환 강화 지시]\n{correction_note}\n" if correction_note else ""

        prompt = (
            f"[원본 문항 정보]\n"
            f"- ID: {question.id} ({question.exam_name} {question.q_num}번)\n"
            f"- 본질 원리: {question.core_principles}\n"
            f"- 원본 정답: {question.answer}\n"
            f"- 원본 공식 해설 요약: {question.official_explanation}\n"
            f"- 반드시 제거·치환해야 할 범위 외 상위 용어: {out_of_scope_terms}\n\n"
            f"{q_text_block}"
            f"{prev_block}"
            f"{feedback_block}"
            f"[업로드된 PDF 허용 개념 (pdf_concepts)]\n{whitelist.pdf_concepts}\n\n"
            f"[업로드된 PDF 허용 용어 (pdf_vocabulary)]\n{whitelist.pdf_vocabulary}\n\n"
            f"[업로드된 PDF 본문 참고 발췌]\n{context_excerpt}\n\n"
            "위 원본 문항을 보고, 범위 외 상위 용어를 PDF 내부 표현으로 완전히 치환한 '변형 문제 카드'와 'PDF 맞춤 해설'을 작성하세요."
        )

        contents: list[Any] = []
        if need_image and crop_png_bytes:
            contents.append(types.Part.from_bytes(data=optimize_image_for_vision(crop_png_bytes), mime_type="image/png"))
        contents.append(prompt)

        resp = _generate_content_with_fallback(
            client=client,
            model_name=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_inst,
                response_mime_type="application/json",
                response_schema=RewrittenQuestion,
                temperature=0.2,
            ),
            fast_mode=True,
        )
        rw = _parse_structured(resp, RewrittenQuestion)
        rw.preserved_answer = question.answer
        rw.rewritten_stem = normalize_exam_layout_text(rw.rewritten_stem)
        rw.rewritten_choices = [format_chemical_scripts(c.strip()) for c in (rw.rewritten_choices or []) if c.strip()]
        rw.pdf_grounded_explanation = format_chemical_scripts(rw.pdf_grounded_explanation)
        return rw

    initial_note = extra_feedback.strip()
    if previous_attempt is not None:
        prev_residuals = find_forbidden_terms_in_text(
            [previous_attempt.rewritten_stem, *previous_attempt.rewritten_choices],
            out_of_scope_terms=out_of_scope_terms,
            allowed_vocab=whitelist.pdf_vocabulary,
        )
        if prev_residuals:
            add_msg = (
                f"이전 변환 결과에 아직 상위 용어 {prev_residuals}가 남아 있습니다. "
                "개념서 허용 용어와 구체적 수치 환산으로 100% 완전히 치환하고, "
                "어색한 조사·문장 끊김도 자연스러운 국문으로 다시 다듬으세요."
            )
            initial_note = f"{initial_note}\n{add_msg}".strip()
        elif not initial_note:
            initial_note = "이전 변환 결과의 문장을 더 자연스러운 국문 시험지 문체로 다듬고, 개념서 허용 용어만 사용해 완전히 재변환하세요."

    rewritten = _invoke_rewrite(correction_note=initial_note)
    residual_terms = find_forbidden_terms_in_text(
        [rewritten.rewritten_stem, *rewritten.rewritten_choices],
        out_of_scope_terms=out_of_scope_terms,
        allowed_vocab=whitelist.pdf_vocabulary,
    )

    if residual_terms:
        retry_note = (
            f"후처리 검증 결과 개조 지문/선지에 범위 외 금지 용어 {residual_terms}가 여전히 남아 있습니다! "
            "해당 용어나 단위를 발문, <자료>, <보기>, 선지에서 단 한 글자도 남기지 말고 "
            "개념서 허용 용어(pdf_vocabulary) 및 구체적 수치/입자 수 표현으로 100% 치환해 다시 작성하세요."
        )
        rewritten = _invoke_rewrite(correction_note=retry_note)
        residual_after_retry = find_forbidden_terms_in_text(
            [rewritten.rewritten_stem, *rewritten.rewritten_choices],
            out_of_scope_terms=out_of_scope_terms,
            allowed_vocab=whitelist.pdf_vocabulary,
        )
        if residual_after_retry:
            raise ForbiddenTermRemainingError(
                f"재개조 후에도 범위 외 금지 용어({', '.join(residual_after_retry)})가 잔존하여 문항을 제외했습니다."
            )

    return rewritten


def generate_pdf_grounded_explanation(
    api_key: str,
    model_name: str,
    question: QuestionRecord,
    crop_png_bytes: bytes | None,
    whitelist: ConceptWhitelist,
    user_answer: str,
    rewritten: RewrittenQuestion | None = None,
    sliced_text: str = "",
    concept_reference: str = "",
) -> str:
    """모드 B: 문항 이미지 + PDF 범위 + 사용자 오답을 바탕으로 PDF 용어만 사용한 AI 자동 해설을 생성한다."""
    client = _build_client(api_key)

    system_inst = (
        "당신은 학생 맞춤형 1:1 AI 튜터입니다.\n"
        "반드시 **선택된 개념서/교과서 범위의 개념(pdf_concepts)과 허용 용어(pdf_vocabulary)만** 사용하여 설명하세요. "
        "개념서 범위를 벗어나는 상위 교육과정 용어(예: 학생이 아직 안 배운 전공 용어)는 절대 사용하지 마세요.\n"
        "학생이 고른 답과 정답을 비교해 왜 맞았는지 또는 어디서 헷갈렸는지 짚어주고, 단계별 풀이를 친절하고 명료하게 제시하세요."
    )

    rewritten_block = ""
    if rewritten is not None:
        choices_str = "\n".join(
            f"{idx + 1}) {c}" for idx, c in enumerate(rewritten.rewritten_choices)
        )
        rewritten_block = (
            f"\n[PDF 맞춤 개조 문항]\n{rewritten.rewritten_stem}\n{choices_str}\n"
        )

    ref_line = f"[참고 개념서 단원 및 페이지] {concept_reference}\n" if concept_reference else ""
    prompt = (
        f"[문항 정보] {question.exam_name} {question.q_num}번 ({question.curriculum_subject})\n"
        f"[정답] {question.answer}\n"
        f"[학생이 입력한 답] {user_answer or '(미입력)'}\n"
        f"{ref_line}"
        f"{rewritten_block}\n"
        f"[공식 해설 참고] {question.official_explanation}\n\n"
        f"[반드시 이 범위 내 개념·용어만 사용]\n"
        f"- 허용 개념: {whitelist.pdf_concepts}\n"
        f"- 허용 용어: {whitelist.pdf_vocabulary}\n"
        f"- 개념서 본문 일부: {sliced_text[:4000]}\n\n"
        "학생에게 보여줄 단계별 AI 자동 해설을 마크다운으로 작성해 주세요."
    )

    contents: list[Any] = []
    if crop_png_bytes:
        contents.append(types.Part.from_bytes(data=optimize_image_for_vision(crop_png_bytes), mime_type="image/png"))
    contents.append(prompt)

    resp = _generate_content_with_fallback(
        client=client,
        model_name=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            temperature=0.2,
        ),
        fast_mode=True,
    )
    out_md = (resp.text or "").strip()
    if concept_reference and concept_reference not in out_md:
        out_md = f"{out_md}\n\n---\n**{concept_reference}**"
    return out_md


def chat_followup_with_pdf(
    api_key: str,
    model_name: str,
    question: QuestionRecord,
    crop_png_bytes: bytes | None,
    whitelist: ConceptWhitelist,
    chat_history: list[dict[str, str]],
    user_question: str,
    rewritten: RewrittenQuestion | None = None,
    sliced_text: str = "",
) -> str:
    """모드 B: 문항별 추가 질문 채팅에 대해 PDF 개념/용어 범위 내에서만 답변한다."""
    client = _build_client(api_key)

    system_inst = (
        "당신은 업로드된 개념서 PDF 범위 안에서만 답변하는 친절한 학습 튜터입니다. "
        "반드시 아래 허용 개념(pdf_concepts)과 허용 용어(pdf_vocabulary)만 사용해 학생의 추가 질문에 답하세요. "
        "PDF에 없는 상위 개념을 묻는 질문이라면 '현재 업로드하신 개념서 범위 밖의 용어이므로 개념서 내 표현으로 풀어서 설명드리면~'처럼 "
        "PDF 범위 안의 표현으로 치환해 안내하세요."
    )

    rewritten_block = ""
    if rewritten is not None:
        choices_str = "\n".join(
            f"{idx + 1}) {c}" for idx, c in enumerate(rewritten.rewritten_choices)
        )
        rewritten_block = f"\n[개조 문제]\n{rewritten.rewritten_stem}\n{choices_str}\n"

    history_text = "\n".join(
        f"{'학생' if m['role'] == 'user' else 'AI 튜터'}: {m['content']}"
        for m in chat_history[-8:]
    )

    prompt = (
        f"[문항] {question.exam_name} {question.q_num}번 | 정답: {question.answer}\n"
        f"{rewritten_block}\n"
        f"[허용 개념] {whitelist.pdf_concepts}\n"
        f"[허용 용어] {whitelist.pdf_vocabulary}\n"
        f"[개념서 참고] {sliced_text[:3000]}\n\n"
        f"[이전 대화]\n{history_text}\n\n"
        f"[학생의 새 질문]\n{user_question}"
    )

    contents: list[Any] = []
    if crop_png_bytes:
        contents.append(types.Part.from_bytes(data=optimize_image_for_vision(crop_png_bytes), mime_type="image/png"))
    contents.append(prompt)

    resp = _generate_content_with_fallback(
        client=client,
        model_name=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            temperature=0.25,
        ),
        fast_mode=True,
    )
    return (resp.text or "").strip()


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
        "4) subject: 과목명 (예: '통합과학', '통합사회', '물리학Ⅰ', '화학Ⅰ', '생명과학Ⅰ', '지구과학Ⅰ', '수학', '경제', '사회·문화' 등). 영역명('과학탐구 영역')과 세부 과목명(예: '물리학Ⅰ')이 같이 있으면 세부 과목명을 우선 적으세요.\n"
        "5) exam_title: 반드시 '{year}년 {month} {grade} {subject}' 형식으로 조합 (예: '2025년 9월 고1 통합과학')."
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

    resp = _generate_content_with_fallback(
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
    )
    meta = _parse_structured(resp, ExamHeaderMetadata)

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


def extract_concept_book_cover_metadata(
    api_key: str,
    model_name: str,
    first_page_text: str,
    first_page_png_bytes: bytes | None = None,
    filename: str = "",
) -> ConceptBookCoverMetadata:
    """
    개념서·교과서 PDF 1페이지(표지) 이미지와 텍스트·파일명을 분석해
    교재명(book_title)과 과목 및 학년(subject_and_grade)을 자동 추출한다.
    """
    client = _build_client(api_key, timeout_seconds=20)
    system_inst = (
        "당신은 한국 고등학교 교과서·개념서(완자, 개념원리, 마플, 수능특강, 자이스토리 등) 표지 판독기입니다.\n"
        "첨부된 표지(1페이지) 이미지, 텍스트, 파일명을 종합하여 다음 2개 필드를 정확히 추출하세요:\n"
        "1) book_title: 교재명 또는 출판사+교과서명 (예: '완자 통합과학', '개념원리 공통수학1', '비상 통합사회1')\n"
        "2) subject_and_grade: 학년과 과목명 (예: '고1 통합과학', '고1 공통수학1', '고2 물리학Ⅰ', '고3 화학Ⅱ')"
    )
    text_excerpt = (first_page_text or "").strip()[:2500]
    fn_hint = f"[업로드 파일명]\n{filename.strip()}\n\n" if filename and filename.strip() else ""
    prompt = (
        f"{fn_hint}"
        "[표지/1페이지 텍스트 발췌]\n"
        f"{text_excerpt if text_excerpt else '(텍스트가 거의 없는 스캔 표지입니다. 첨부된 표지 이미지를 확인하세요.)'}\n\n"
        "위 개념서·교과서의 교재명(book_title)과 과목 및 학년(subject_and_grade)을 JSON으로 반환하세요."
    )
    contents: list[Any] = []
    if first_page_png_bytes:
        opt_png = optimize_image_for_vision(first_page_png_bytes, max_width=1200)
        contents.append(types.Part.from_bytes(data=opt_png, mime_type="image/png"))
    contents.append(prompt)

    resp = _generate_content_with_fallback(
        client=client,
        model_name=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=ConceptBookCoverMetadata,
            temperature=0.0,
        ),
        fast_mode=True,
    )
    meta = _parse_structured(resp, ConceptBookCoverMetadata)
    return ConceptBookCoverMetadata(
        book_title=meta.book_title.strip(),
        subject_and_grade=meta.subject_and_grade.strip(),
    )
