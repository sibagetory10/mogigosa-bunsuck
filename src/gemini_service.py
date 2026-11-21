from __future__ import annotations

import concurrent.futures
import contextlib
import contextvars
import hashlib
import io
import json
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence, TypeVar

from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel, Field

from src.config import (
    DEFAULT_MODEL,
    EXAM_TAGGING_MODEL,
    FLASH_LITE_CACHED_INPUT_USD_PER_1M,
    FLASH_LITE_INPUT_USD_PER_1M,
    FLASH_LITE_MODEL,
    FLASH_LITE_OUTPUT_USD_PER_1M,
    MODEL_TOKEN_PRICING_USD_PER_1M,
    resolve_api_model_name,
)
from src.curriculum_2022 import (
    build_tagging_grounding_context,
    has_ncic_standards_for_category,
    validate_and_normalize_extracted_tag,
    validate_standards_only,
)


def _build_lang_ncic_grounding_section(subject_category: str, exam_name: str, max_standards: int = 25) -> tuple[str, str]:
    """국어/영어에 NCIC 성취기준이 등록된 경우에만 프롬프트용 기준표 섹션과 규칙 문구를 반환한다."""
    if not has_ncic_standards_for_category(subject_category):
        return "", ""
    ctx = build_tagging_grounding_context(
        subject_category=subject_category,
        exam_name=exam_name,
        max_standards=max_standards,
    )
    if not ctx:
        return "", ""
    rule = (
        "\n- `standard_code` / `standard_codes`: 아래 제공된 [2022 개정 교육과정 공식 기준표]의 성취기준 코드 중에서만 가장 부합하는 대표 코드 1개(`standard_code`)와 관련 코드 목록(`standard_codes`)을 매핑하세요(목록에 없는 코드를 지어내지 마세요)."
    )
    return f"\n{ctx}\n", rule


def _apply_lang_standard_validation(item: Any, subject_category: str) -> None:
    """국어/영어 추출 결과의 성취기준 코드를 정제하고 등록된 NCIC 목록 기준으로 검증한다."""
    std_code, cleaned_codes, needs_review, review_notes = validate_standards_only(
        getattr(item, "standard_code", "") or "",
        list(getattr(item, "standard_codes", []) or []),
        subject_category=subject_category,
    )
    item.standard_code = std_code
    item.standard_codes = cleaned_codes
    if needs_review:
        item.needs_review = True
        existing_notes = list(getattr(item, "review_notes", []) or [])
        item.review_notes = list(dict.fromkeys([*existing_notes, *review_notes]))
from src.models import (
    AutoConceptPageDetectionOutput,
    AutoConceptRangeItem,
    BatchEnglishExamExtractionOutput,
    BatchExamExtractionOutput,
    BatchFilePairingOutput,
    BatchKoreanExamExtractionOutput,
    BatchMatchResult,
    BatchQuestionTagItem,
    ConceptBookCoverMetadata,
    ConceptBookExtractionOutput,
    ConceptBookRecord,
    ConceptUnit,
    ConceptWhitelist,
    EnglishQuestionExtractionItem,
    EnglishSingleExtractionOutput,
    ExamHeaderMetadata,
    KoreanPassageAnalysisOutput,
    KoreanVisualDiagram,
    KoreanQuestionExtractionItem,
    KoreanSetReconstructionOutput,
    KoreanSingleExtractionOutput,
    normalize_english_group_id,
    normalize_korean_group_id,
    MixedPdfClassificationOutput,
    QuestionExtractionOutput,
    QuestionMatchJudgment,
    QuestionRecord,
    RewrittenQuestion,
    TocExtractionOutput,
)
from src.pdf_parser import extract_specific_pages_slice
from src.text_formatter import (
    find_forbidden_terms_in_text,
    format_chemical_scripts,
    format_explanation_text,
    normalize_english_exam_text,
    normalize_exam_layout_text,
    strip_english_embedded_choices,
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
    FLASH_LITE_MODEL,
)
RATE_LIMIT_FALLBACK_ORDER: tuple[str, ...] = (
    FLASH_LITE_MODEL,
)
OVERLOAD_RETRY_DELAYS: tuple[float, ...] = (2.0, 4.0, 6.0)


@dataclass
class TokenUsageRecord:
    """단일 Gemini API 호출에서 실제 소비된 토큰 수와 USD 비용."""

    model_name: str
    prompt_tokens: int = 0  # 일반 입력 토큰(캐시 제외)
    cached_tokens: int = 0  # 캐시 히트 입력 토큰
    output_tokens: int = 0  # 출력 + 사고(thoughts) 토큰
    total_tokens: int = 0
    cost_usd: float = 0.0


_ACTIVE_USAGE_RECORDER: contextvars.ContextVar[Callable[[TokenUsageRecord], None] | None] = (
    contextvars.ContextVar("mogigosa_active_usage_recorder", default=None)
)


def calculate_token_cost_usd(
    prompt_tokens: int,
    cached_tokens: int,
    output_tokens: int,
    model_name: str = FLASH_LITE_MODEL,
) -> float:
    """실제 토큰 수(일반 입력 / 캐시 입력 / 출력+사고)를 모델 단가(USD)로 계산한다."""
    resolved = resolve_api_model_name(model_name)
    rates = MODEL_TOKEN_PRICING_USD_PER_1M.get(
        resolved,
        {
            "input": FLASH_LITE_INPUT_USD_PER_1M,
            "cached_input": FLASH_LITE_CACHED_INPUT_USD_PER_1M,
            "output": FLASH_LITE_OUTPUT_USD_PER_1M,
        },
    )
    non_cached_in = max(0, int(prompt_tokens or 0))
    cached_in = max(0, int(cached_tokens or 0))
    out_tok = max(0, int(output_tokens or 0))
    usd = (
        (non_cached_in * rates["input"])
        + (cached_in * rates["cached_input"])
        + (out_tok * rates["output"])
    ) / 1_000_000.0
    return max(0.0, usd)


def extract_usage_from_response(response: Any, model_name: str = FLASH_LITE_MODEL) -> TokenUsageRecord:
    """Gemini 응답의 usage_metadata에서 실제 토큰 수와 USD 예상 비용을 추출한다."""
    resolved = resolve_api_model_name(model_name)
    um = getattr(response, "usage_metadata", None)
    if um is None:
        return TokenUsageRecord(model_name=resolved)

    raw_prompt = int(getattr(um, "prompt_token_count", 0) or 0)
    cached = int(getattr(um, "cached_content_token_count", 0) or 0)
    non_cached_prompt = max(0, raw_prompt - cached)
    candidates = int(getattr(um, "candidates_token_count", 0) or 0)
    thoughts = int(getattr(um, "thoughts_token_count", 0) or 0)
    output_total = max(0, candidates + thoughts)
    reported_total = int(getattr(um, "total_token_count", 0) or 0)
    total_tokens = max(reported_total, raw_prompt + output_total)

    cost_usd = calculate_token_cost_usd(
        prompt_tokens=non_cached_prompt,
        cached_tokens=cached,
        output_tokens=output_total,
        model_name=resolved,
    )
    return TokenUsageRecord(
        model_name=resolved,
        prompt_tokens=non_cached_prompt,
        cached_tokens=cached,
        output_tokens=output_total,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
    )


def has_active_usage_tracker() -> bool:
    """현재 실행 컨텍스트에 활성 토큰 추적 콜백이 바인딩되어 있는지 반환한다."""
    return _ACTIVE_USAGE_RECORDER.get() is not None


@contextlib.contextmanager
def bind_usage_tracker(recorder: Callable[[TokenUsageRecord], None] | None) -> Iterator[None]:
    """현재 실행 컨텍스트의 모든 Gemini API 호출에서 실제 토큰 사용량을 수집할 콜백을 바인딩한다."""
    token = _ACTIVE_USAGE_RECORDER.set(recorder)
    try:
        yield
    finally:
        _ACTIVE_USAGE_RECORDER.reset(token)


def submit_with_usage_context(
    executor: concurrent.futures.ThreadPoolExecutor,
    fn: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> concurrent.futures.Future[T]:
    """ThreadPoolExecutor 워커 스레드에도 현재 토큰 추적 컨텍스트(ContextVar)를 그대로 전파해 실행한다."""
    ctx = contextvars.copy_context()
    return executor.submit(ctx.run, fn, *args, **kwargs)


def _emit_usage_from_response(response: Any, model_name: str) -> None:
    recorder = _ACTIVE_USAGE_RECORDER.get()
    if recorder is None:
        return
    try:
        usage = extract_usage_from_response(response, model_name=model_name)
        if usage.total_tokens > 0 or usage.prompt_tokens > 0 or usage.output_tokens > 0:
            recorder(usage)
    except Exception:
        pass


def build_model_fallback_chain(model_name: str) -> list[str]:
    """전 기능 gemini-3.5-flash-lite 단일 모델 체인을 반환한다."""
    _ = model_name
    return [FLASH_LITE_MODEL]


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


def _is_flash_lite_model(model_name: str) -> bool:
    low = (resolve_api_model_name(model_name) or "").lower()
    return "flash-lite" in low or low.endswith("-lite")


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
    if _is_flash_lite_model(resolved_model):
        # 3.5 Flash-Lite 호출 시 fast_mode·config 여부와 무관하게 항상 최소 토큰(MINIMAL 사고 + 저해상도 미디어) 강제 적용
        base_cfg = config if config is not None else types.GenerateContentConfig()
        return base_cfg.model_copy(
            update={
                "thinking_config": types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.MINIMAL,
                    include_thoughts=False,
                ),
                "media_resolution": types.MediaResolution.MEDIA_RESOLUTION_LOW,
            }
        )

    if config is None:
        return None
    if not fast_mode:
        return config
    if getattr(config, "thinking_config", None) is not None:
        return config
    return _with_thinking_config(
        config,
        types.ThinkingConfig(thinking_budget=0),
    )


def _build_400_recovery_configs(
    config: types.GenerateContentConfig | None,
    eff_cfg: types.GenerateContentConfig | None,
    cand_model: str,
    err_msg: str,
) -> list[types.GenerateContentConfig | None]:
    """thinking_config 또는 media_resolution 관련 400 오류 시 최소 토큰 옵션을 유지하며 순차 복구 후보를 생성한다."""
    is_bad_req = "400" in err_msg or "invalid_argument" in err_msg
    is_media_err = "media_resolution" in err_msg or "resolution" in err_msg
    is_thinking_err = "thinking" in err_msg
    if not is_bad_req or not (is_media_err or is_thinking_err):
        return []

    recovery_configs: list[types.GenerateContentConfig | None] = []
    if _is_flash_lite_model(cand_model) and eff_cfg is not None:
        if is_media_err:
            recovery_configs.append(eff_cfg.model_copy(update={"media_resolution": None}))
        if is_thinking_err:
            recovery_configs.append(
                eff_cfg.model_copy(
                    update={"thinking_config": types.ThinkingConfig(thinking_budget=0)}
                )
            )
            recovery_configs.append(eff_cfg.model_copy(update={"thinking_config": None}))
        recovery_configs.append(config)
        return recovery_configs

    if config is not None or eff_cfg is not None:
        recovery_configs.append(config)
    return recovery_configs


_CIRCLED_DIGIT_TO_NUM: dict[str, str] = {
    "①": "1",
    "②": "2",
    "③": "3",
    "④": "4",
    "⑤": "5",
}
_ANSWER_PREFIX_RE = re.compile(r"^\s*(?:정답|답)\s*[:：]?\s*")
_CHOICE_NUM_ONLY_RE = re.compile(r"^\(?([1-5])\)?\s*번?$")
_CHOICE_NUM_PREFIX_RE = re.compile(r"^\(?([1-5])\)?\s*번(?:\s|\(|$)")


def _normalize_extracted_answer(raw_answer: str) -> str:
    """객관식 원문자(①~⑤)나 '정답: 3번' 표기를 '1'~'5'로 정규화하고 주관식 값은 그대로 보존한다."""
    s = _ANSWER_PREFIX_RE.sub("", (raw_answer or "").strip()).strip()
    if not s:
        return ""
    if s[0] in _CIRCLED_DIGIT_TO_NUM:
        return _CIRCLED_DIGIT_TO_NUM[s[0]]
    m = _CHOICE_NUM_ONLY_RE.match(s) or _CHOICE_NUM_PREFIX_RE.match(s)
    if m:
        return m.group(1)
    return s


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
            eff_cfg = _clone_config_for_model(config, cand_model, fast_mode=fast_mode)
            try:
                resp = client.models.generate_content(
                    model=cand_model,
                    contents=contents,
                    config=eff_cfg,
                )
                _emit_usage_from_response(resp, cand_model)
                return resp
            except Exception as exc:
                last_err = exc
                msg = str(exc).lower()
                # thinking_config / media_resolution 미지원 400 오류 시 단계별 안전 복구 시도
                recovery_configs = _build_400_recovery_configs(config, eff_cfg, cand_model, msg)
                for rec_cfg in recovery_configs:
                    try:
                        rec_resp = client.models.generate_content(
                            model=cand_model,
                            contents=contents,
                            config=rec_cfg,
                        )
                        _emit_usage_from_response(rec_resp, cand_model)
                        return rec_resp
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

    resp = _generate_content_with_fallback(
        client=client,
        model_name=model_name,
        contents=contents,
        config=gen_config,
        fast_mode=use_fast_mode,
        status_callback=status_callback,
    )
    if status_callback is not None:
        status_callback(f"{q_num}번 응답 파싱 및 2022 기준 검증 중...")
    parsed = _parse_structured(resp, QuestionExtractionOutput)
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

    resp = _generate_content_with_fallback(
        client=client,
        model_name=model_name,
        contents=contents,
        config=batch_config,
        fast_mode=has_answer_ref,
        status_callback=status_callback,
    )
    parsed = _parse_structured(resp, BatchExamExtractionOutput)
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
            fut_to_pages = {
                submit_with_usage_context(ex, _detect_text_batch, b): [int(e["page"]) for e in b]
                for b in batches
            }
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
            fut_to_pages = {
                submit_with_usage_context(ex, _detect_scan_batch, bp): bp for bp in page_batches
            }
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
            futs = [submit_with_usage_context(ex, _run_chunk, *sp) for sp in chunk_specs]
            results = [f.result() for f in futs]

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
        "5. **구조화 줄바꿈 (필수)**: 한 문단으로 뭉치지 않도록 도입 발문 / `<자료>`·`<조건>`·`<실험 과정>` / 질문 문장 / `<보기>` / `(단, ...)` 사이는 빈 줄(`\\n\\n`)로 띄우고, 자료 안의 각 항목(`◦`, `(가)`, `(나)`)과 `<보기>`의 `ㄱ.`, `ㄴ.`, `ㄷ.` 및 `pdf_grounded_explanation`의 풀이 단계(`1.`, `2.`, `ㄱ.`, `ㄴ.`)는 반드시 각각 줄바꿈(`\\n`)해 별도의 줄에 적으세요.\n"
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
        rw.pdf_grounded_explanation = format_explanation_text(rw.pdf_grounded_explanation, for_markdown=False)
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

    resp = _generate_content_with_fallback(
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
    )
    parsed = _parse_structured(resp, BatchFilePairingOutput)

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

    resp = _generate_content_with_fallback(
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
    )
    parsed = _parse_structured(resp, MixedPdfClassificationOutput)

    roles: list[str] = ["problem"] * n_files
    for item in parsed.items:
        idx = item.file_index
        if 0 <= idx < n_files and item.role in ("problem", "answer"):
            roles[idx] = item.role

    prob_indices = [i for i, r in enumerate(roles) if r == "problem"]
    ans_indices = [i for i, r in enumerate(roles) if r == "answer"]
    return prob_indices, ans_indices


# ===========================================================================
# 영어(English) 과목 전용 기출 추출 & 수능 독해 11개 유형 변형 엔진
# ===========================================================================
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


def _build_english_transform_rule(
    target_type: str,
    blank_unit: str = "구 (3~6단어)",
    vocab_style: str = "밑줄형 (①~⑤)",
) -> tuple[str, str]:
    """입력 토큰 절감을 위해 지정된 target_type 1개 유형의 핵심 규칙만 압축해 반환한다."""
    t = (target_type or "").strip()

    if "함축" in t:
        return (
            "함축적 의미 추론",
            "- 발문: `밑줄 친 부분이 다음 글에서 의미하는 바로 가장 적절한 것은?`\n"
            "- 출력 절감 규칙: `rewritten_stem`에 지문 전체를 쓰지 말고, 원문에서 비유·핵심 의미를 담은 **유일한 연속 구절 1개(target_phrase)**만 원문 철자 그대로 반환하세요(시스템이 `<u>...</u>` 자동 치환).\n"
            "- 선지: 밑줄 표현의 문맥상 함축 의미를 담은 **영어 선지 5개**(번호 제외).",
        )
    if "요지" in t:
        return (
            "요지 파악",
            "- 발문: `다음 글의 요지로 가장 적절한 것은?`\n"
            "- 출력 절감 규칙: 지문은 원문 그대로 쓰이므로 `rewritten_stem`은 **빈 문자열 `\"\"`**로 반환하세요.\n"
            "- 선지: 평가원 표준 **한국어 완결 문장 선지 5개**(번호 제외).",
        )
    if "주제" in t:
        return (
            "주제 파악",
            "- 발문: `다음 글의 주제로 가장 적절한 것은?`\n"
            "- 출력 절감 규칙: 지문은 원문 그대로 쓰이므로 `rewritten_stem`은 **빈 문자열 `\"\"`**로 반환하세요.\n"
            "- 선지: 평가원 표준 **영어 명사구 선지 5개**(번호 제외).",
        )
    if "제목" in t:
        return (
            "제목 추론",
            "- 발문: `다음 글의 제목으로 가장 적절한 것은?`\n"
            "- 출력 절감 규칙: 지문은 원문 그대로 쓰이므로 `rewritten_stem`은 **빈 문자열 `\"\"`**로 반환하세요.\n"
            "- 선지: 평가원 표준 **영어 제목 선지 5개**(번호 제외, 첫 글자 대문자 타이틀 형식).",
        )
    if "어법" in t:
        return (
            "어법상 틀린 것 (밑줄 ①~⑤)",
            "- 발문: `다음 글의 밑줄 친 부분 중, 어법상 틀린 것은?`\n"
            "- 출력 절감 규칙: `rewritten_stem`에 지문 전체를 쓰지 말고, 원문 등장 순서대로 문법 포인트 5곳을 아래 5줄 형식으로만 반환하세요(시스템이 원문에 `① <u>...</u>`~`⑤ <u>...</u>`를 자동 조립).\n"
            "  `1: 원문구절1`\n  `2: 원문구절2`\n  `3: 원문구절3 -> 틀린어법구절3`\n  `4: 원문구절4`\n  `5: 원문구절5`\n"
            "- 규칙: 5개 구절은 원문에서 유일하게 찾을 수 있도록 1~3단어로 적고, 정확히 1줄(정답)에만 `-> 틀린어법`을 적어 실제 비문(수일치·분사 능/수동·관계사/접속사 what/that·병렬·대동사 do/be 중 1가지)으로 변형. 나머지 4줄은 화살표 없이 원문 그대로 유지.\n"
            "- 금지: 원문 정상 표현을 틀렸다고 우기기, 해설 자체 모순('타당하다/적절하다'), 의미만 반대인 반의어 교체, 단순 시제·진행형(`is occurring`↔`occurs`, `signaled`↔`signals`) 변형 절대 금지.\n"
            "- 선지: `[\"①\", \"②\", \"③\", \"④\", \"⑤\"]` 5개.",
        )
    if "어휘" in t:
        if "선택형" in vocab_style or "(A)" in vocab_style:
            return (
                "문맥상 어휘 적절성 (선택형 (A)(B)(C))",
                "- 발문: `(A), (B), (C)의 각 네모 안에서 문맥에 맞는 낱말로 가장 적절한 것은?`\n"
                "- 규칙: 기존 본문 내 핵심 어휘 3곳을 `(A) [w1 / w2]`, `(B) [w3 / w4]`, `(C) [w5 / w6]`(원문 단어와 반의어 쌍)으로 표시. 지문 끝에 새 문장 추가·어색한 연어(`severe encouragement`)·어법 변형(`makes`↔`making`) 절대 금지. 반어법·풍자·부정어(`not`/`never`/`hardly`) 절의 단어는 교체 대상에서 제외.\n"
                "- 선지: `\"(A) w1 — (B) w3 — (C) w5\"` 형식 5개.",
            )
        return (
            "문맥상 어휘 적절성 (밑줄 ①~⑤)",
            "- 발문: `다음 글의 밑줄 친 부분 중, 문맥상 낱말의 쓰임이 적절하지 않은 것은?`\n"
            "- 출력 절감 규칙: `rewritten_stem`에 지문 전체를 쓰지 말고, 원문 등장 순서대로 핵심 어휘 5곳을 아래 5줄 형식으로만 반환하세요(시스템이 원문에 `① <u>...</u>`~`⑤ <u>...</u>`를 자동 조립).\n"
            "  `1: 원문단어1`\n  `2: 원문단어2 -> 반의어2`\n  `3: 원문단어3`\n  `4: 원문단어4`\n  `5: 원문단어5`\n"
            "- 규칙: 정확히 1줄(정답)에만 `-> 반의어`를 적어 문법 형태·품사가 100% 동일한 자연스러운 반의어로 교체. 나머지 4줄은 화살표 없이 원문 단어 그대로 유지.\n"
            "- 금지: 어법 오류(`makes`→`making` 등)·어색한 연어 생성 금지. 반어법·풍자 표현이나 부정어(`not`/`never`/`hardly`)가 걸린 절의 단어는 반의어 교체 대상에서 제외.\n"
            "- 선지: 밑줄 친 5개 어휘 리스트(정답 위치는 교체된 반의어).",
        )
    if "빈칸" in t:
        b_u = (blank_unit or "").strip()
        if b_u.startswith("절") or ("절(" in b_u or "절 (" in b_u):
            unit_desc = "주어+술어 포함 **절(6~12단어)**"
            short_u = "절"
        elif b_u.startswith("단어") or ("단어" in b_u and not b_u.startswith("구")):
            unit_desc = "핵심 **단어(1~2단어)**"
            short_u = "단어"
        else:
            unit_desc = "핵심 **구(3~6단어)**"
            short_u = "구"
        return (
            f"빈칸 추론 ({short_u})",
            "- 발문: `다음 빈칸에 들어갈 말로 가장 적절한 것은?`\n"
            f"- 출력 절감 규칙: `rewritten_stem`에 지문 전체를 쓰지 말고, 주제문/핵심 구간에서 빈칸으로 뚫을 **{unit_desc} 분량의 유일한 원문 구절 1개(target_phrase)**만 원문 철자 그대로 반환하세요(시스템이 `___________`로 자동 치환).\n"
            "- 제약: 빈칸 앞뒤 2단어와 정답 선지 시작/끝 2단어 중복 금지, 문장 내 동사 통째 삭제 금지, `(①)~(⑤)` 기호 금지.\n"
            f"- 선지: {unit_desc} 길이·품사에 맞는 **영어 선지 5개**(정답 1개 + 매력적 오답 4개).",
        )
    if "무관" in t:
        return (
            "전체 흐름과 무관한 문장",
            "- 발문: `다음 글에서 전체 흐름과 관계없는 문장은?`\n"
            "- 출력 절감 규칙: `rewritten_stem`에 지문 전체를 쓰지 말고, 아래 2줄 형식으로만 반환하세요(시스템이 원문 사이에 `①~⑤`와 새 문장을 자동 조립).\n"
            "  `INSERT_AT: 3` (2~4 중 삽입할 번호 1개)\n"
            "  `SENTENCE: <원문 키워드만 공유하고 전체 논리 흐름과 어긋나는 새 영어 문장 1개>`\n"
            "- 선지: `[\"①\", \"②\", \"③\", \"④\", \"⑤\"]` 5개.",
        )
    if "순서" in t:
        return (
            "글의 순서 ((A)-(B)-(C))",
            "- 발문: `주어진 글 다음에 이어질 글의 순서로 가장 적절한 것은?`\n"
            "- 출력 절감 규칙: 원문 각 문장에 `[S0]`, `[S1]`, ... 번호가 붙어 있습니다. `rewritten_stem`에 지문 전체를 쓰지 말고, `[주어진 글]`과 3개 단락의 시작 문장 번호 3개만 아래 1줄 형식으로 반환하세요(시스템이 원문을 자르고 오순 셔플과 정답을 자동 확정).\n"
            "  `SPLIT: s1, s2, s3` (예: `SPLIT: 1, 3, 5` → 주어진글=S0, 1번덩어리=S1~S2, 2번덩어리=S3~S4, 3번덩어리=S5~끝; `0 < s1 < s2 < s3 < 전체문장수`)\n"
            "- 선지: `[\"(A) - (C) - (B)\", \"(B) - (A) - (C)\", \"(B) - (C) - (A)\", \"(C) - (A) - (B)\", \"(C) - (B) - (A)\"]` 5개.",
        )
    if "삽입" in t:
        return (
            "주어진 문장 삽입",
            "- 발문: `글의 흐름으로 보아, 주어진 문장이 들어가기에 가장 적절한 곳은?`\n"
            "- 출력 절감 규칙: 원문이 6문장 이상이면 `rewritten_stem`에 지문 전체를 쓰지 말고 원문에서 뽑아낼 **핵심 문장 1개만 원문 그대로** 반환하세요(시스템이 `( ① )~( ⑤ )`와 정답을 자동 배치). 5문장 이하일 때만 `[주어진 문장]`과 본문 `( ① )~( ⑤ )`를 출력하세요.\n"
            "- 선지: `[\"①\", \"②\", \"③\", \"④\", \"⑤\"]` 5개.",
        )
    # 기본: 요약문 완성
    return (
        "요약문 완성 ((A), (B))",
        "- 발문: `다음 글의 내용을 한 문장으로 요약하고자 한다. 빈칸 (A), (B)에 들어갈 말로 가장 적절한 것은?`\n"
        "- 출력 절감 규칙: `rewritten_stem`에 지문 본문·`[선지]`·①~⑤를 절대 쓰지 말고, 오직 **`(A) ________`와 `(B) ________`가 포함된 영어 요약문 1문장만** 반환하세요(원문은 시스템이 자동 결합).\n"
        "- 제약: `(A)`, `(B)` 정답 어휘는 원문과 동일 단어/동일 어근 노출 금지(패러프레이징 필수).\n"
        "- 선지: `\"(A) word1 — (B) word2\"` 형식 5개.",
    )


_AI_META_MONOLOGUE_RE = re.compile(
    r"(?:wait,\s*let\s*me|let['’]?s\s+(?:rewrite|apply|check|change|fix|try|revise|make|replace|keep)|"
    r"keep\s+the\s+original\s+text|actually,\s*let['’]?s|i\s+will\s+(?:rewrite|replace|use)|"
    r"here\s+is\s+the\s+revised|note:\s*i\s+(?:changed|replaced)|"
    r"\[\s*정답[^\]]*\]|예시\s*교정|변형\s*필요|정답\s*처리|수정\s*필요|비문으로\s*만듦)",
    re.IGNORECASE,
)
_DANGLING_TO_END_RE = re.compile(
    r"\b(?:has|have|had|want|wants|wanted|able|unable|need|needs|needed|"
    r"ought|used|going|supposed|expected|likely|unlikely|tend|tends|tended|"
    r"try|tries|tried|fail|fails|failed|seem|seems|seemed|appear|appears|appeared)\s+to\s*\.",
    re.IGNORECASE,
)
_ORDER_SEQ_RE = re.compile(
    r"\(?\s*([ABC])\s*\)?\s*(?:-|→|->|–|—)\s*\(?\s*([ABC])\s*\)?\s*(?:-|→|->|–|—)\s*\(?\s*([ABC])\s*\)?"
)
_SENT_ABBREV_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Mt|vs|etc|e\.g|i\.e|U\.S|U\.K|No|Fig|Vol|p|pp)\.",
    re.IGNORECASE,
)


def _preprocess_english_source_passage(passage: str) -> str:
    """변형 출제 전 원문 지문에 남아 있는 기존 기출 기호((A)(B)(C), (①)~(⑤), <u> 등)를 제거해 순수 영문 지문으로 정규화한다."""
    txt = (passage or "").strip()
    if not txt:
        return ""
    txt = re.sub(r"</?u\b[^>]*>", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"[ \t]*\(\s*[①②③④⑤]\s*\)[ \t]*", " ", txt)
    txt = re.sub(r"[ \t]*[①②③④⑤][ \t]*", " ", txt)
    # 문장/단락 시작의 기출 순서 라벨 (A), (B), (C), (D) 제거 (문맥상 괄호 약어는 보존)
    txt = re.sub(r"(?:^|(?<=\n)|(?<=[.?!\"'”’]\s))\(\s*[ABCD]\s*\)\s*", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    txt = re.sub(r"[ \t]{2,}", " ", txt)
    return txt.strip()


_COMMON_ENG_VERBS: frozenset[str] = frozenset(
    {
        "is", "are", "was", "were", "am", "be", "been", "being",
        "has", "have", "had", "do", "does", "did",
        "can", "could", "will", "would", "shall", "should", "may", "might", "must",
        "make", "makes", "made", "take", "takes", "took", "give", "gives", "gave",
        "show", "shows", "showed", "get", "gets", "got", "go", "goes", "went",
        "come", "comes", "came", "see", "sees", "saw", "know", "knows", "knew",
        "think", "thinks", "thought", "use", "uses", "used", "find", "finds", "found",
        "tell", "tells", "told", "ask", "asks", "asked", "work", "works", "worked",
        "seem", "seems", "seemed", "feel", "feels", "felt", "try", "tries", "tried",
        "leave", "leaves", "left", "call", "calls", "called", "keep", "keeps", "kept",
        "let", "lets", "begin", "begins", "began", "help", "helps", "helped",
        "turn", "turns", "turned", "start", "starts", "started", "play", "plays", "played",
        "run", "runs", "ran", "move", "moves", "moved", "like", "likes", "liked",
        "live", "lives", "lived", "believe", "believes", "believed", "hold", "holds", "held",
        "bring", "brings", "brought", "happen", "happens", "happened", "write", "writes", "wrote",
        "provide", "provides", "provided", "sit", "sits", "sat", "stand", "stands", "stood",
        "lose", "loses", "lost", "pay", "pays", "paid", "meet", "meets", "met",
        "include", "includes", "included", "continue", "continues", "continued",
        "learn", "learns", "learned", "change", "changes", "changed", "lead", "leads", "led",
        "understand", "understands", "understood", "watch", "watches", "watched",
        "follow", "follows", "followed", "stop", "stops", "stopped", "create", "creates", "created",
        "speak", "speaks", "spoke", "read", "reads", "allow", "allows", "allowed",
        "add", "adds", "added", "spend", "spends", "spent", "grow", "grows", "grew",
        "open", "opens", "opened", "offer", "offers", "offered", "remember", "remembers",
        "consider", "considers", "considered", "appear", "appears", "appeared",
        "serve", "serves", "served", "send", "sends", "sent", "expect", "expects", "expected",
        "build", "builds", "built", "stay", "stays", "stayed", "fall", "falls", "fell",
        "reach", "reaches", "reached", "remain", "remains", "remained", "suggest", "suggests",
        "raise", "raises", "raised", "pass", "passes", "passed", "require", "requires", "required",
        "report", "reports", "reported", "decide", "decides", "decided", "mean", "means", "meant",
        "become", "becomes", "became", "cause", "causes", "caused", "enable", "enables", "enabled",
        "reduce", "reduces", "reduced", "increase", "increases", "increased", "improve", "improves",
        "affect", "affects", "affected", "depend", "depends", "depended", "tend", "tends", "tended",
        "fail", "fails", "failed", "exist", "exists", "existed", "occur", "occurs", "occurred",
        "need", "needs", "needed", "want", "wants", "wanted", "aim", "aims", "aimed",
        "reveal", "reveals", "revealed", "explain", "explains", "explained",
        "cannot", "can't", "don't", "doesn't", "didn't", "won't", "wouldn't", "isn't", "aren't",
    }
)


def _sentence_has_english_verb(sentence_without_blank: str) -> bool:
    """빈칸을 제외한 문장에 영문 동사/조동사가 최소 1개 이상 남아있는지 판별한다."""
    toks = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", (sentence_without_blank or "").lower())
    for tok in toks:
        if tok in _COMMON_ENG_VERBS:
            return True
        if len(tok) >= 5 and tok.endswith(("ed", "ize", "ise", "ify", "ate")):
            return True
        if len(tok) >= 5 and tok.endswith("ing") and tok not in ("thing", "morning", "evening", "nothing", "something", "anything", "everything", "meaning", "feeling", "during"):
            return True
    return False


def _is_order_abc_canonical(stem: str, source_passage: str) -> bool:
    """(A), (B), (C) 단락이 원문 등장 순서((A)-(B)-(C) 정순) 그대로 배열되었는지 판별한다."""
    src_sents = _split_english_sentences(source_passage)
    if len(src_sents) < 3:
        return False

    after_given = re.split(r"\[주어진\s*글\]", stem or "", maxsplit=1)[-1]
    parts = re.split(r"(?:^|\n|\s)\(([ABC])\)\s+", after_given)
    chunk_map: dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):
        lbl = parts[i]
        txt = parts[i + 1].strip()
        if lbl in ("A", "B", "C") and txt:
            chunk_map[lbl] = txt

    if len(chunk_map) != 3:
        return False

    src_tok_sets = [set(re.findall(r"[a-z0-9]+", s.lower())) for s in src_sents]
    matched_idx: dict[str, int] = {}
    for lbl in ("A", "B", "C"):
        c_sents = _split_english_sentences(chunk_map[lbl])
        first_s = c_sents[0] if c_sents else chunk_map[lbl]
        c_set = set(re.findall(r"[a-z0-9]+", first_s.lower()))
        if not c_set:
            return False
        best_i = -1
        best_sc = 0.0
        for i, s_set in enumerate(src_tok_sets):
            if not s_set:
                continue
            sc = len(c_set & s_set) / max(1, len(c_set | s_set))
            if sc > best_sc:
                best_sc = sc
                best_i = i
        if best_i == -1 or best_sc < 0.45:
            return False
        matched_idx[lbl] = best_i

    return matched_idx["A"] < matched_idx["B"] < matched_idx["C"]


def _split_english_sentences(text: str) -> list[str]:
    """약어·소수점을 보존하면서 영어 지문을 문장 단위 리스트로 분할한다."""
    cleaned_lines: list[str] = []
    for raw_ln in (text or "").splitlines():
        ln = raw_ln.strip()
        if not ln:
            continue
        if re.search(r"[가-힣]", ln) and not re.search(r"[A-Za-z]{4,}", ln):
            continue
        if ln.startswith("*") and ":" in ln:
            continue
        cleaned_lines.append(ln)
    joined = re.sub(r"\s+", " ", " ".join(cleaned_lines)).strip()
    if not joined:
        return []

    stashed: list[str] = []

    def _stash(m: re.Match[str]) -> str:
        idx = len(stashed)
        stashed.append(m.group(0))
        return f"@@SENTTOK_{idx}@@"

    masked = _SENT_ABBREV_RE.sub(_stash, joined)
    masked = re.sub(r"\b\d+\.\d+\b", _stash, masked)
    raw_sents = re.split(
        r"(?<=[.?!][\"'”’\)])\s+(?=[A-Z\"'“‘\(])|(?<=[.?!])\s+(?=[A-Z\"'“‘\(])",
        masked,
    )
    out: list[str] = []
    for s in raw_sents:
        restored = s
        for idx, orig in enumerate(stashed):
            restored = restored.replace(f"@@SENTTOK_{idx}@@", orig)
        restored = restored.strip()
        if restored:
            out.append(restored)
    return out


def _extract_given_sentence_from_insertion_stem(stem: str) -> str:
    """문장 삽입 변형 출력에서 [주어진 문장] 박스의 추출 문장만 발췌한다."""
    parts = re.split(r"\[주어진\s*문장\]", stem or "", maxsplit=1)
    if len(parts) < 2:
        return ""
    after_hdr = parts[1].strip()
    # 빈 줄 이전 또는 첫 ( ① ) 직전까지를 주어진 문장으로 간주
    block = re.split(r"\n\s*\n", after_hdr, maxsplit=1)[0].strip()
    block = re.split(r"\(\s*①\s*\)", block, maxsplit=1)[0].strip()
    sents = _split_english_sentences(block)
    return sents[0] if sents else block


def _rebuild_insertion_stem_from_source(rw: RewrittenQuestion, source_passage: str) -> bool:
    """
    원문 문장이 6문장 이상인 경우, AI가 고른 주어진 문장을 source_passage에서 찾아
    5개 문장 경계에 ( ① )~( ⑤ )를 파이썬으로 직접 배치하고 정확한 preserved_answer를 계산한다.
    """
    src_sents = _split_english_sentences(source_passage)
    if len(src_sents) < 6:
        return False

    given_cand = _extract_given_sentence_from_insertion_stem(rw.rewritten_stem or "")
    if not given_cand:
        return False

    def _norm_toks(s: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", (s or "").lower())

    g_toks = _norm_toks(given_cand)
    if len(g_toks) < 3:
        return False
    g_set = set(g_toks)

    best_idx = -1
    best_score = 0.0
    for idx_s, s in enumerate(src_sents):
        s_toks = _norm_toks(s)
        if not s_toks:
            continue
        if s_toks == g_toks:
            best_idx = idx_s
            best_score = 1.0
            break
        s_set = set(s_toks)
        jaccard = len(g_set & s_set) / max(1, len(g_set | s_set))
        if jaccard > best_score:
            best_score = jaccard
            best_idx = idx_s

    # 첫 문장(idx 0)은 통상 도입문이므로 최소 0.72 이상 일치하고 idx >= 1일 때 기계적 재조립 수행
    if best_idx < 1 or best_score < 0.72:
        return False

    given_exact = src_sents[best_idx]
    remaining = [s for i, s in enumerate(src_sents) if i != best_idx]
    if len(remaining) < 5:
        return False

    # remaining의 문장 경계 gap k: 1 <= k <= len(remaining) (k는 remaining[k-1] 뒤 위치)
    true_gap = best_idx  # match_idx >= 1 이므로 remaining[best_idx - 1] 바로 뒤
    max_gap = len(remaining) - 1 if true_gap <= len(remaining) - 1 else len(remaining)
    # 5개 연속 경계 [g_start .. g_start + 4] 안에 true_gap이 포함되도록 설정
    g_start = max(1, min(true_gap - 2, max_gap - 4))
    if true_gap < g_start:
        g_start = true_gap
    if true_gap > g_start + 4:
        g_start = true_gap - 4

    markers = ["( ① )", "( ② )", "( ③ )", "( ④ )", "( ⑤ )"]
    chunks: list[str] = []
    for pos, sent in enumerate(remaining):
        chunks.append(sent)
        gap_idx = pos + 1
        if g_start <= gap_idx <= g_start + 4:
            chunks.append(markers[gap_idx - g_start])

    main_body = " ".join(chunks).strip()
    rw.rewritten_stem = (
        "글의 흐름으로 보아, 주어진 문장이 들어가기에 가장 적절한 곳은?\n"
        "[주어진 문장]\n"
        f"{given_exact}\n\n"
        f"{main_body}"
    )
    rw.rewritten_choices = ["①", "②", "③", "④", "⑤"]
    ans_num = str((true_gap - g_start) + 1)
    rw.preserved_answer = ans_num
    # 파이썬이 확정한 정답 번호와 해설 내 위치 번호가 어긋나지 않도록 동기화
    num_to_circle = {"1": "①", "2": "②", "3": "③", "4": "④", "5": "⑤"}
    c_sym = num_to_circle.get(ans_num, "①")
    if rw.pdf_grounded_explanation:
        exp_sync = rw.pdf_grounded_explanation
        exp_sync = re.sub(
            r"(정답(?:은|:)?\s*)(?:[1-5]\s*번|[①②③④⑤](?:\s*번)?)",
            rf"\g<1>{c_sym}번",
            exp_sync,
        )
        exp_sync = re.sub(
            r"(?:[1-5]\s*번|[①②③④⑤](?:\s*번)?)\s*(위치|자리|에\s*들어가|에\s*가장\s*적절)",
            rf"{c_sym}번 \g<1>",
            exp_sync,
        )
        rw.pdf_grounded_explanation = exp_sync
    return True


def _count_english_sentences(text: str) -> int:
    """영어 지문 본문의 문장 수를 약어/소수점을 제외하고 근사 계산한다."""
    cleaned_lines: list[str] = []
    for raw_ln in (text or "").splitlines():
        ln = raw_ln.strip()
        if not ln:
            continue
        if re.search(r"[가-힣]", ln) and not re.search(r"[A-Za-z]{4,}", ln):
            continue
        if ln.startswith("*") and ":" in ln:
            continue
        cleaned_lines.append(ln)
    body = " ".join(cleaned_lines)
    body = re.sub(r"[①②③④⑤]", " ", body)
    body = _SENT_ABBREV_RE.sub("ABBR", body)
    body = re.sub(r"\b\d+\.\d+\b", "NUM", body)
    endings = re.findall(r"[.?!]+(?=\s|$|[\"'”’\)])", body)
    return len(endings) if endings else (1 if re.search(r"[A-Za-z]{3,}", body) else 0)


def _reconcile_english_insertion_answer(rw: RewrittenQuestion) -> None:
    """문장 삽입 유형에서 해설에 명시된 위치 번호(①~⑤ / N번)와 preserved_answer가 어긋나면 자동 보정한다."""
    exp = rw.pdf_grounded_explanation or ""
    if not exp:
        return
    m_exp = re.search(
        r"(?:정답(?:은|:)?|위치(?:는|가)?|삽입(?:하기에)?)\s*(?:은|는|:)?\s*(?:([1-5])\s*번|([①②③④⑤]))",
        exp,
    )
    found_num = ""
    if m_exp:
        found_num = m_exp.group(1) or _CIRCLED_DIGIT_TO_NUM.get(m_exp.group(2) or "", "")
    else:
        circled = list(dict.fromkeys(re.findall(r"[①②③④⑤]", exp)))
        if len(circled) == 1:
            found_num = _CIRCLED_DIGIT_TO_NUM.get(circled[0], "")
    if found_num in ("1", "2", "3", "4", "5") and found_num != _normalize_extracted_answer(rw.preserved_answer):
        rw.preserved_answer = found_num


def _reconcile_english_summary_answer(rw: RewrittenQuestion) -> None:
    """요약문 완성 유형에서 해설의 (A)/(B) 정답 어휘 쌍 또는 명시 번호가 preserved_answer와 어긋나면 자동 보정한다."""
    choices = rw.rewritten_choices or []
    exp = rw.pdf_grounded_explanation or ""
    if len(choices) != 5 or not exp:
        return

    # 1) 해설에 (A) wordA ... (B) wordB 쌍이 명시된 경우 선지와 대조
    m_ab = re.search(
        r"\(A\)\s*([A-Za-z][A-Za-z\-']*)[^A-Za-z]{1,20}\(B\)\s*([A-Za-z][A-Za-z\-']*)",
        exp,
        flags=re.IGNORECASE,
    )
    if m_ab:
        w_a, w_b = m_ab.group(1).lower(), m_ab.group(2).lower()
        matched_indices: list[int] = []
        for idx_c, ch in enumerate(choices):
            m_ch = re.search(
                r"\(A\)\s*([A-Za-z][A-Za-z\-']*)[^A-Za-z]{1,20}\(B\)\s*([A-Za-z][A-Za-z\-']*)",
                ch or "",
                flags=re.IGNORECASE,
            )
            if m_ch and m_ch.group(1).lower() == w_a and m_ch.group(2).lower() == w_b:
                matched_indices.append(idx_c + 1)
        if len(matched_indices) == 1:
            rw.preserved_answer = str(matched_indices[0])
            return

    # 2) 해설에 정답 번호가 명시된 경우 보정
    m_num = re.search(
        r"(?:정답(?:은|:)?)\s*(?:([1-5])\s*번|([①②③④⑤]))",
        exp,
    )
    if m_num:
        found_num = m_num.group(1) or _CIRCLED_DIGIT_TO_NUM.get(m_num.group(2) or "", "")
        if found_num in ("1", "2", "3", "4", "5"):
            rw.preserved_answer = found_num


def _reconcile_english_order_answer(rw: RewrittenQuestion) -> None:
    """글의 순서 유형에서 해설에 명시된 최종 (A)-(B)-(C) 논리 순서와 preserved_answer 선지 인덱스가 어긋나면 자동 보정한다."""
    choices = rw.rewritten_choices or []
    exp = rw.pdf_grounded_explanation or ""
    if len(choices) != 5 or not exp:
        return

    choice_tuples: list[tuple[str, str, str] | None] = []
    for ch in choices:
        letters = re.findall(r"\(([ABC])\)|\b([ABC])\b", ch or "")
        seq = [a or b for a, b in letters]
        choice_tuples.append(tuple(seq) if len(seq) == 3 and set(seq) == {"A", "B", "C"} else None)

    matches = [
        (m.start(), (m.group(1), m.group(2), m.group(3)))
        for m in _ORDER_SEQ_RE.finditer(exp)
        if set((m.group(1), m.group(2), m.group(3))) == {"A", "B", "C"}
    ]
    if not matches:
        return

    # '정답'/'순서'/'따라서' 근처의 순서 표기를 우선 선택하고, 없으면 유일/첫 매칭 사용
    target_seq: tuple[str, str, str] | None = None
    for pos, seq in matches:
        ctx = exp[max(0, pos - 24) : pos + 24]
        if any(kw in ctx for kw in ("정답", "순서", "따라서", "결론", "적절")):
            target_seq = seq
            break
    if target_seq is None:
        distinct = {seq for _, seq in matches}
        if len(distinct) == 1:
            target_seq = matches[0][1]
        else:
            return

    ans = _normalize_extracted_answer(rw.preserved_answer)
    cur_idx = int(ans) - 1 if ans in ("1", "2", "3", "4", "5") else -1
    if 0 <= cur_idx < len(choice_tuples) and choice_tuples[cur_idx] == target_seq:
        return

    for idx_c, c_seq in enumerate(choice_tuples):
        if c_seq == target_seq:
            rw.preserved_answer = str(idx_c + 1)
            return


_REPEATABLE_PREPS: frozenset[str] = frozenset(
    {"from", "to", "in", "on", "of", "with", "for", "by", "at", "into", "as", "about", "through"}
)


def _has_order_scrambled_sentences(stem: str, source_passage: str) -> bool:
    """(A), (B), (C) 단락 각각이 원문의 연속된 문장 묶음으로 구성됐는지 확인해 비연속 문장 뒤섞임을 감지한다."""
    src_sents = _split_english_sentences(source_passage)
    if len(src_sents) < 4:
        return False
    after_given = re.split(r"\[주어진\s*글\]", stem or "", maxsplit=1)[-1]
    parts = re.split(r"(?:^|\n|\s)\(([ABC])\)\s+", after_given)
    chunk_map: dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):
        lbl = parts[i]
        txt = parts[i + 1].strip()
        if lbl in ("A", "B", "C") and txt:
            chunk_map[lbl] = txt
    if len(chunk_map) != 3:
        return False

    src_sets = [set(re.findall(r"[a-z0-9]+", s.lower())) for s in src_sents]
    for lbl in ("A", "B", "C"):
        c_sents = _split_english_sentences(chunk_map[lbl])
        if len(c_sents) <= 1:
            continue
        idxs: list[int] = []
        for cs in c_sents:
            c_set = set(re.findall(r"[a-z0-9]+", cs.lower()))
            if not c_set:
                continue
            best_i, best_sc = -1, 0.0
            for i, s_set in enumerate(src_sets):
                if not s_set:
                    continue
                sc = len(c_set & s_set) / max(1, len(c_set | s_set))
                if sc > best_sc:
                    best_sc, best_i = sc, i
            if best_i != -1 and best_sc >= 0.55:
                idxs.append(best_i)
        if len(idxs) >= 2:
            for k in range(len(idxs) - 1):
                if idxs[k + 1] != idxs[k] + 1:
                    return True
    return False


def _is_irrelevant_answer_from_source(stem: str, ans_num: str, source_passage: str) -> bool:
    """무관한 문장 유형에서 정답 번호에 해당하는 문장이 원문에 이미 존재하는 문장인지 검사한다."""
    if ans_num not in ("1", "2", "3", "4", "5") or not source_passage:
        return False
    sym = {"1": "①", "2": "②", "3": "③", "4": "④", "5": "⑤"}[ans_num]
    m_sym = re.search(rf"{re.escape(sym)}\s*([^\n①②③④⑤]+)", stem or "")
    if not m_sym:
        return False
    ans_toks = set(re.findall(r"[a-z0-9]+", m_sym.group(1).lower()))
    if len(ans_toks) < 4:
        return False
    for s in _split_english_sentences(source_passage):
        s_toks = set(re.findall(r"[a-z0-9]+", s.lower()))
        if not s_toks:
            continue
        if len(ans_toks & s_toks) / max(1, len(ans_toks | s_toks)) >= 0.78:
            return True
    return False


def _ensure_korean_instruction_prefix(stem: str, canonical_type: str, vocab_style: str) -> str:
    """생성된 영어 변형 지문 첫 줄에 표준 한국어 발문이 누락된 경우 자동으로 부착한다."""
    s = (stem or "").strip()
    if not s:
        return s
    first_line = s.splitlines()[0].strip()
    if re.search(r"[가-힣]", first_line) and ("?" in first_line or "시오" in first_line or "것은" in first_line or "곳은" in first_line):
        return s

    t = canonical_type or ""
    if "함축" in t:
        inst = "밑줄 친 부분이 다음 글에서 의미하는 바로 가장 적절한 것은?"
    elif "요지" in t:
        inst = "다음 글의 요지로 가장 적절한 것은?"
    elif "주제" in t:
        inst = "다음 글의 주제로 가장 적절한 것은?"
    elif "제목" in t:
        inst = "다음 글의 제목으로 가장 적절한 것은?"
    elif "어법" in t:
        inst = "다음 글의 밑줄 친 부분 중, 어법상 틀린 것은?"
    elif "어휘" in t:
        inst = (
            "(A), (B), (C)의 각 네모 안에서 문맥에 맞는 낱말로 가장 적절한 것은?"
            if ("선택형" in t or "선택형" in vocab_style or "(A)" in vocab_style)
            else "다음 글의 밑줄 친 부분 중, 문맥상 낱말의 쓰임이 적절하지 않은 것은?"
        )
    elif "빈칸" in t:
        inst = "다음 빈칸에 들어갈 말로 가장 적절한 것은?"
    elif "무관" in t:
        inst = "다음 글에서 전체 흐름과 관계없는 문장은?"
    elif "순서" in t:
        inst = "주어진 글 다음에 이어질 글의 순서로 가장 적절한 것은?"
    elif "삽입" in t:
        inst = "글의 흐름으로 보아, 주어진 문장이 들어가기에 가장 적절한 곳은?"
    else:
        inst = "다음 글의 내용을 한 문장으로 요약하고자 한다. 빈칸 (A), (B)에 들어갈 말로 가장 적절한 것은?"
    return f"{inst}\n{s}"


def _validate_english_transformed_output(
    rw: RewrittenQuestion,
    target_type: str,
    vocab_style: str,
    source_passage: str = "",
) -> list[str]:
    """생성된 영어 변형 문항의 필수 태그·메타 독백·어휘 실제 교체·미완성 문장 여부를 검증한다."""
    issues: list[str] = []
    stem = rw.rewritten_stem or ""
    t = (target_type or "").strip()

    # 1. AI 영문/한글 독백(메타 발화) 혼입 차단
    all_text_fields = [stem, *(rw.rewritten_choices or []), rw.pdf_grounded_explanation or ""]
    for fld in all_text_fields:
        m_meta = _AI_META_MONOLOGUE_RE.search(fld or "")
        if m_meta:
            issues.append(
                f"AI 내부 사고/자기 수정 메타 문구({m_meta.group(0)!r})가 출력에 혼입되었습니다. 메타 독백 없이 순수 문제와 해설만 반환하세요."
            )
            break

    # 1-b. 영문 지문 본문(발문·섹션 헤더·각주 제외)에 한글 코멘트가 섞였는지 차단
    for ln_raw in stem.splitlines():
        ln_s = ln_raw.strip()
        if not ln_s:
            continue
        if re.search(r"가장\s*적절한\s*(?:것은|곳은)\??$", ln_s):
            continue
        if re.match(r"^\[(?:주어진\s*글|주어진\s*문장|요약문)\]$", ln_s):
            continue
        if ln_s.startswith("*"):
            continue
        if re.search(r"[가-힣]{2,}", ln_s) and re.search(r"[A-Za-z]{4,}", ln_s):
            issues.append(
                f"영문 지문 본문 안에 한글 설명/메모({ln_s[:40]!r})가 섞였습니다. 지문 본문은 순수 영어만 남기세요."
            )
            break

    # 4. 문장 끝 단어 탈락 방지 ('has to.', 'able to.' 등)
    m_to = _DANGLING_TO_END_RE.search(stem)
    if m_to:
        issues.append(
            f"지문 내에 부정사 뒤 동사원형이 탈락한 미완성 문장({m_to.group(0)!r})이 감지되었습니다. 빠진 단어를 복원해 완전한 문장으로 작성하세요."
        )

    ans = _normalize_extracted_answer(rw.preserved_answer)
    if ans not in ("1", "2", "3", "4", "5"):
        issues.append(f"정답(preserved_answer)이 '1'~'5' 중 하나여야 합니다 (현재: {rw.preserved_answer!r}).")

    if len(rw.rewritten_choices or []) != 5:
        issues.append(f"선택지(rewritten_choices)는 정확히 5개여야 합니다 (현재: {len(rw.rewritten_choices or [])}개).")

    u_matches = re.findall(r"<u\b[^>]*>(.*?)</u>", stem, flags=re.I | re.S)
    u_count = len(u_matches)
    if "어휘" in t and source_passage:
        src_cnt = _count_english_sentences(source_passage)
        gen_cnt = _count_english_sentences(stem)
        if src_cnt >= 3 and gen_cnt != src_cnt:
            issues.append(
                f"문맥상 어휘 유형은 원문 문장 수({src_cnt}개)와 동일해야 합니다 (현재 {gen_cnt}문장). "
                "지문 끝에 새 문장을 추가하거나 기존 문장을 삭제하지 말고 원문 본문 단어만 교체하세요."
            )

    if "어법" in t or ("어휘" in t and "선택형" not in vocab_style and "(A)" not in vocab_style):
        missing_markers = [mk for mk in ("①", "②", "③", "④", "⑤") if stem.count(mk) != 1]
        if u_count != 5 or missing_markers:
            issues.append(
                f"지문(rewritten_stem) 안에 `① <u>...</u>` ~ `⑤ <u>...</u>` 밑줄 5개가 모두 정확히 1개씩 포함되어야 합니다 "
                f"(현재 `<u>` {u_count}개, 누락/중복 번호: {missing_markers})."
            )
        elif "어법" in t:
            exp_txt = rw.pdf_grounded_explanation or ""
            if re.search(r"(?:의미상|의미적|문맥상\s*반대|반의어|부정어\s*(?:not|no)\s*(?:이|가)?\s*빠)", exp_txt):
                issues.append(
                    "어법 문제 해설에 어휘 뜻풀이('의미상'/'의미적'/'반의어'/'부정어 누락')가 포함되었습니다. 어휘 문제와 혼동하지 말고 명백한 문법 비문 1개만 출제하세요."
                )
            elif re.search(r"(?:과거형\s*대신\s*현재형|현재형\s*대신\s*과거형|시제\s*흐름과\s*일치)", exp_txt):
                issues.append(
                    "어법 문제에서 단순 과거/현재 시제 차이를 오류로 출제했습니다. 문맥상 가능한 시제 변경 대신 수일치·능동/수동·관계사·분사·준동사 등 명백한 통사 오류 1개만 출제하세요."
                )
            elif re.search(r"(?:타당|적절|올바르|맞)(?:하다|다|습니다|음)(?!\s*(?:않|못))", exp_txt) and not re.search(
                r"(?:적절하지\s*않|올바르지\s*않|타당하지\s*않|맞지\s*않)", exp_txt
            ):
                issues.append(
                    "어법 문제 해설이 정답 밑줄을 '적절하다/타당하다'고 모순되게 설명했습니다. 명백한 문법 오류 1개만 출제하고 왜 틀렸는지 설명하세요."
                )
        elif "어휘" in t and ans in ("1", "2", "3", "4", "5"):
            # 2. 문맥상 어휘 적절성: 해설/치환목록에 적힌 교체 단어가 실제 지문 정답 밑줄에 반영되었는지 확인
            u_words = [re.sub(r"<[^>]+>", "", w).strip() for w in u_matches]
            ans_word = u_words[int(ans) - 1]
            ans_alpha = re.sub(r"[^a-zA-Z\-']", "", ans_word).lower()
            for entry in rw.replaced_terms_map or []:
                m_map = re.search(r"([A-Za-z][A-Za-z\-']*)\s*(?:->|→)\s*([A-Za-z][A-Za-z\-']*)", entry or "")
                if m_map:
                    orig_w, new_w = m_map.group(1).lower(), m_map.group(2).lower()
                    if orig_w != new_w and ans_alpha == orig_w:
                        issues.append(
                            f"문맥상 어휘 문제에서 정답 {ans}번 밑줄({ans_word!r})이 원문 그대로 남아 있고 "
                            f"교체 어휘({new_w!r})로 바뀌지 않았습니다. 지문 본문의 정답 밑줄을 실제로 교체하세요."
                        )
                        break
            if source_passage and ans_alpha:
                # 3. 문맥상 어휘 적절성: 5개 밑줄 단어 중 정확히 1개(정답 위치)만 원문과 다른 단어인지 검증
                not_in_src_indices: list[int] = []
                for idx_u, w in enumerate(u_words[:5]):
                    cleaned_w = re.sub(r"[^a-zA-Z\-]", "", w).lower()
                    if cleaned_w and not re.search(
                        r"\b" + re.escape(cleaned_w) + r"\b",
                        source_passage,
                        flags=re.IGNORECASE,
                    ):
                        not_in_src_indices.append(idx_u)
                if not not_in_src_indices:
                    issues.append(
                        f"문맥상 어휘 문제에서 5개 밑줄 단어({u_words[:5]})가 모두 원문 그대로입니다. "
                        f"반드시 정답 {ans}번 위치의 단어를 문맥상 부적절한 반의어로 지문 본문에서 직접 교체하세요."
                    )
                elif len(not_in_src_indices) == 1 and not_in_src_indices[0] != int(ans) - 1:
                    issues.append(
                        f"문맥상 어휘 문제의 정답 번호({ans}번)와 실제 지문에서 교체된 밑줄 위치({not_in_src_indices[0] + 1}번)가 일치하지 않습니다."
                    )
    elif "함축" in t:
        if u_count < 1:
            issues.append("함축적 의미 추론 지문에는 밑줄 친 표현 `<u>...</u>`가 최소 1개 포함되어야 합니다.")
    elif "빈칸" in t:
        if "____" not in stem:
            issues.append("빈칸 추론 지문에는 `___________` 빈칸 표기가 반드시 포함되어야 합니다.")
        else:
            # 2. 빈칸 포함 문장에서 빈칸을 제외했을 때 영문 동사가 하나도 남지 않는 불완전 문장 차단
            m_blk = re.search(r"_{4,}", stem)
            if m_blk:
                pre_txt = stem[: m_blk.start()]
                post_txt = stem[m_blk.end() :]
                s_start = max(pre_txt.rfind("\n"), pre_txt.rfind(". "), pre_txt.rfind("? "), pre_txt.rfind("! "))
                sent_left = pre_txt[s_start + 1 :] if s_start != -1 else pre_txt
                m_end = re.search(r"[.?!](?=\s|$|[\"'”’\)])|\n", post_txt)
                sent_right = post_txt[: m_end.start()] if m_end else post_txt
                rem_sentence = f"{sent_left} {sent_right}".strip()
                if not _sentence_has_english_verb(rem_sentence):
                    issues.append(
                        f"빈칸이 포함된 문장({rem_sentence!r})에서 빈칸을 제외하면 영문 동사가 전혀 남지 않습니다. "
                        "문장 전체(동사 포함)를 비우지 말고 주어·동사가 유지된 상태에서 핵심 목적어/보어/구만 빈칸으로 비우세요."
                    )

            # 3. 빈칸 앞뒤 구절/전치사 중복 차단
            if m_blk and ans in ("1", "2", "3", "4", "5") and len(rw.rewritten_choices or []) >= int(ans):
                before_toks = re.findall(r"[a-zA-Z\-']+", stem[: m_blk.start()].lower())
                after_toks = re.findall(r"[a-zA-Z\-']+", stem[m_blk.end() :].lower())
                ans_toks = re.findall(r"[a-zA-Z\-']+", (rw.rewritten_choices[int(ans) - 1] or "").lower())
                if len(ans_toks) >= 2:
                    if len(before_toks) >= 2 and before_toks[-2:] == ans_toks[:2]:
                        issues.append(
                            f"빈칸 직전 구절({' '.join(before_toks[-2:])})과 정답 선지의 시작 2단어가 중복됩니다. 중복을 제거해 다시 생성하세요."
                        )
                    elif len(after_toks) >= 2 and after_toks[:2] == ans_toks[-2:]:
                        issues.append(
                            f"빈칸 직후 구절({' '.join(after_toks[:2])})과 정답 선지의 끝 2단어가 중복됩니다. 중복을 제거해 다시 생성하세요."
                        )
                if after_toks and after_toks[0] in _REPEATABLE_PREPS and after_toks[0] in ans_toks[-4:]:
                    issues.append(
                        f"빈칸 바로 뒤 전치사({after_toks[0]!r})가 정답 선지 끝부분에도 중복 포함되어 비문이 됩니다. 전치사를 겹치지 않게 다시 생성하세요."
                    )
                elif before_toks and before_toks[-1] in (_REPEATABLE_PREPS | {"a", "an", "the"}) and ans_toks and ans_toks[0] == before_toks[-1]:
                    issues.append(
                        f"빈칸 바로 앞 단어({before_toks[-1]!r})가 정답 선지 시작 단어와 중복됩니다. 중복을 제거해 다시 생성하세요."
                    )
                elif (
                    before_toks
                    and before_toks[-1] in {"that", "because", "although", "though", "since", "when", "while", "if", "unless"}
                    and ans_toks
                    and ans_toks[0] in {"without", "with", "in", "on", "at", "by", "for", "from", "through"}
                ):
                    issues.append(
                        f"접속사({before_toks[-1]!r}) 바로 뒤 빈칸에 전치사구({ans_toks[0]!r} ...)만 들어가면 비문이 됩니다. 주어+동사를 포함한 절 또는 동사구로 출제하세요."
                    )
                post_raw = stem[m_blk.end() :]
                if re.match(r"^[ \t]+[A-Z]", post_raw) and not re.search(r"[.?!]$", pre_txt.rstrip()):
                    issues.append(
                        "문장 전체를 빈칸으로 비웠으나 빈칸 뒤에 마침표(.)가 누락되어 다음 문장과 붙었습니다. 문장 종결 부호를 유지하거나 문장 일부만 빈칸 처리하세요."
                    )
        if re.search(r"\(\s*[①②③④⑤]\s*\)|[①②③④⑤]", stem):
            issues.append(
                "빈칸 추론 지문에는 문장 삽입 번호 기호 `( ① )~( ⑤ )` 또는 `①~⑤`가 포함되면 안 됩니다. 모두 제거하세요."
            )
    elif "무관" in t:
        for c_num in ("①", "②", "③", "④", "⑤"):
            if c_num not in stem:
                issues.append(f"무관한 문장 본문에 번호 `{c_num}`가 누락되었습니다.")
                break
        if source_passage:
            src_cnt = _count_english_sentences(source_passage)
            gen_cnt = _count_english_sentences(stem)
            if src_cnt >= 3 and gen_cnt != src_cnt + 1:
                issues.append(
                    f"전체 흐름과 무관한 문장은 원문 문장({src_cnt}개)을 삭제하지 않고 무관한 새 문장 1개만 삽입해 "
                    f"총 {src_cnt + 1}문장이 되어야 합니다 (현재 {gen_cnt}문장). 원문을 보존하고 새 문장 1개를 삽입하세요."
                )
            elif _is_irrelevant_answer_from_source(stem, ans, source_passage):
                issues.append(
                    f"무관한 문장의 정답 {ans}번 문장이 원문 문장 그대로입니다. 원문 문장을 오답으로 우기지 말고 흐름과 무관한 새 문장을 삽입하세요."
                )
    elif "순서" in t:
        for mk in ("(A)", "(B)", "(C)"):
            if mk not in stem:
                issues.append(f"글의 순서 지문에 `{mk}` 단락 표기가 누락되었습니다.")
        if "[주어진 글]" not in stem and "주어진 글" not in stem:
            issues.append("글의 순서 지문에 `[주어진 글]` 섹션 헤더가 누락되었습니다.")
        if source_passage:
            # 4. 글의 순서 원문 문장 보존 검증 ([주어진 글] + (A) + (B) + (C) 합계 == 원문 문장 수)
            order_clean = re.sub(r"\[주어진\s*글\]", " ", stem)
            order_clean = re.sub(r"\(\s*[ABC]\s*\)", " ", order_clean)
            src_cnt = _count_english_sentences(source_passage)
            gen_cnt = _count_english_sentences(order_clean)
            if src_cnt >= 3 and gen_cnt != src_cnt:
                issues.append(
                    f"글의 순서 문제에서 원문 문장 수({src_cnt}개)와 [주어진 글]+(A)+(B)+(C) 합계 문장 수({gen_cnt}개)가 다릅니다. "
                    "원문 문장을 누락하거나 새 문장을 임의로 추가하지 말고 원문 그대로 배치하세요."
                )
            elif _is_order_abc_canonical(stem, source_passage):
                issues.append(
                    "글의 순서 문제에서 (A), (B), (C)가 원문 등장 순서 그대로((A)-(B)-(C) 정순) 배열되었습니다. "
                    "수능 5지선다에는 (A)-(B)-(C)가 없으므로 반드시 (B)-(A)-(C), (C)-(B)-(A) 등 단락 순서를 섞어 다시 출제하세요."
                )
            elif _has_order_scrambled_sentences(stem, source_passage):
                issues.append(
                    "글의 순서 문제에서 (A), (B), (C) 단락 내부에 원문의 비연속 문장이 뒤섞여 합쳐졌습니다. 각 단락은 원문에서 연속된 문장 묶음으로만 구성하세요."
                )
    elif "삽입" in t:
        if "[주어진 문장]" not in stem and "주어진 문장" not in stem:
            issues.append("문장 삽입 지문에 `[주어진 문장]` 섹션 헤더가 누락되었습니다.")
        for c_num in ("①", "②", "③", "④", "⑤"):
            if c_num not in stem:
                issues.append(f"문장 삽입 본문에 위치 기호 `{c_num}`가 누락되었습니다.")
                break
        # ( ① )~( ⑤ ) 마커가 문장 경계(. ? ! 뒤)가 아닌 주어·동사 한가운데 삽입되었는지 검증
        ins_body = re.split(r"\[주어진\s*문장\]", stem, maxsplit=1)[-1]
        ins_body_alpha = re.findall(r"[A-Za-z]+", ins_body)
        if len(ins_body_alpha) < 25:
            issues.append(
                f"문장 삽입 본문 영문 단어가 {len(ins_body_alpha)}개뿐입니다. [주어진 문장] 외에 전체 영어 지문 본문을 빠짐없이 포함하세요."
            )
        for m_ins in re.finditer(r"\(?\s*([①②③④⑤])\s*\)", ins_body):
            before = ins_body[: m_ins.start()].rstrip()
            if before and not re.search(r"[.?!][\"'”’\)]*$", before):
                issues.append(
                    f"문장 삽입 기호 `{m_ins.group(0).strip()}`가 문장 종결 부호(. ? !) 뒤의 문장 경계가 아닌 문장 중간에 배치되었습니다. 반드시 문장과 문장 사이 경계로 옮기세요."
                )
                break
    elif "요약" in t:
        if "[요약문]" not in stem and "요약문" not in stem:
            issues.append("요약문 완성 지문에 `[요약문]` 섹션 헤더가 누락되었습니다.")
        sum_section = re.split(r"\[요약문\]", stem, maxsplit=1)[-1]
        if not re.search(r"\(A\)\s*_{2,}", sum_section) or not re.search(r"\(B\)\s*_{2,}", sum_section):
            issues.append("요약문 섹션 안에 `(A) ________`와 `(B) ________` 빈칸 표기가 모두 포함되어야 합니다.")

    return issues


def _assemble_english_slim_stem(
    rw: RewrittenQuestion,
    canonical_type: str,
    clean_passage: str,
) -> None:
    """
    지문 전체 재출력 생략 유형(요지·주제·제목·요약문·문장 삽입·빈칸·함축)에서
    AI가 반환한 경량 결과(빈 문자열 / 요약 1문장 / 추출 1문장 / target_phrase)를
    원문(clean_passage)에 직접 결합·치환해 완성된 rewritten_stem을 조립한다.
    이미 전체 지문 형태로 출력된 경우(폴백 재시도 등)는 그대로 보존한다.
    """
    raw_stem = (rw.rewritten_stem or "").strip()
    t = canonical_type or ""

    if "요지" in t:
        if len(raw_stem) < max(60, int(len(clean_passage) * 0.55)):
            rw.rewritten_stem = f"다음 글의 요지로 가장 적절한 것은?\n{clean_passage}"
        return
    if "주제" in t:
        if len(raw_stem) < max(60, int(len(clean_passage) * 0.55)):
            rw.rewritten_stem = f"다음 글의 주제로 가장 적절한 것은?\n{clean_passage}"
        return
    if "제목" in t:
        if len(raw_stem) < max(60, int(len(clean_passage) * 0.55)):
            rw.rewritten_stem = f"다음 글의 제목으로 가장 적절한 것은?\n{clean_passage}"
        return

    if "요약" in t:
        # 이미 원문+요약문 전체가 들어있으면 그대로 두고, 요약문 1문장만 반환된 경우 원문과 결합
        if len(raw_stem) < max(120, int(len(clean_passage) * 0.65)):
            s_line = strip_english_embedded_choices(normalize_english_exam_text(raw_stem))
            s_line = re.sub(r"^[^\n]*가장\s*적절한\s*것은\??\s*", "", s_line).strip()
            s_line = re.sub(r"^\[?요약문\]?\s*", "", s_line).strip()
            if s_line:
                rw.rewritten_stem = (
                    "다음 글의 내용을 한 문장으로 요약하고자 한다. 빈칸 (A), (B)에 들어갈 말로 가장 적절한 것은?\n"
                    f"{clean_passage}\n\n"
                    "[요약문]\n"
                    f"{s_line}"
                )
        return

    if "삽입" in t:
        # 6문장 이상에서 추출 문장 1개만 반환된 경우 [주어진 문장] 헤더로 감싸 기계적 마커 배치로 넘김
        if (
            len(_split_english_sentences(clean_passage)) >= 6
            and "[주어진 문장]" not in raw_stem
            and len(raw_stem) < max(80, int(len(clean_passage) * 0.55))
        ):
            ext_s = re.sub(r"^[^\n]*가장\s*적절한\s*곳은\??\s*", "", raw_stem).strip()
            if ext_s:
                rw.rewritten_stem = f"[주어진 문장]\n{ext_s}"
        return

    if "빈칸" in t or "함축" in t:
        is_full_stem = len(raw_stem) >= max(80, int(len(clean_passage) * 0.6)) and (
            ("빈칸" in t and "____" in raw_stem) or ("함축" in t and "<u" in raw_stem.lower())
        )
        if is_full_stem or not raw_stem:
            return

        phrase = re.sub(r"</?u\b[^>]*>", "", raw_stem, flags=re.IGNORECASE).strip()
        phrase = re.sub(r"^[^\n]*가장\s*적절한\s*것은\??\s*", "", phrase).strip()
        phrase = phrase.strip("`\"'“”‘’")
        if not phrase:
            return

        exact_cnt = clean_passage.count(phrase)
        matched_span: tuple[int, int, str] | None = None
        if exact_cnt == 1:
            pos = clean_passage.index(phrase)
            matched_span = (pos, pos + len(phrase), phrase)
        elif exact_cnt == 0:
            ci_matches = list(re.finditer(re.escape(phrase), clean_passage, flags=re.IGNORECASE))
            if len(ci_matches) == 1:
                m_ci = ci_matches[0]
                matched_span = (m_ci.start(), m_ci.end(), m_ci.group(0))

        if matched_span is not None:
            st_i, en_i, orig_seg = matched_span
            if "빈칸" in t:
                replaced_body = f"{clean_passage[:st_i]}___________{clean_passage[en_i:]}"
                rw.rewritten_stem = f"다음 빈칸에 들어갈 말로 가장 적절한 것은?\n{replaced_body}"
            else:
                replaced_body = f"{clean_passage[:st_i]}<u>{orig_seg}</u>{clean_passage[en_i:]}"
                rw.rewritten_stem = f"밑줄 친 부분이 다음 글에서 의미하는 바로 가장 적절한 것은?\n{replaced_body}"
        return

    if "순서" in t:
        # 이미 완성된 (A)(B)(C) 전체 지문이 반환된 경우(폴백 재시도 등)는 그대로 보존
        if "(A)" in raw_stem and "(B)" in raw_stem and "(C)" in raw_stem and len(raw_stem) >= max(100, int(len(clean_passage) * 0.6)):
            return
        m_sp = re.search(r"SPLIT\s*:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", raw_stem, flags=re.IGNORECASE)
        sents = _split_english_sentences(clean_passage)
        if m_sp and len(sents) >= 4:
            s1, s2, s3 = int(m_sp.group(1)), int(m_sp.group(2)), int(m_sp.group(3))
            if 0 < s1 < s2 < s3 < len(sents):
                intro = " ".join(sents[:s1]).strip()
                orig_blocks = [
                    " ".join(sents[s1:s2]).strip(),
                    " ".join(sents[s2:s3]).strip(),
                    " ".join(sents[s3:]).strip(),
                ]
                if all(orig_blocks):
                    # 5지선다 오순 조합: (A)-(C)-(B)=1, (B)-(A)-(C)=2, (B)-(C)-(A)=3, (C)-(A)-(B)=4, (C)-(B)-(A)=5
                    # 각 조합에서 (A), (B), (C)에 어떤 원문 덩어리(0,1,2)를 넣으면 정답이 해당 순서가 되는지 매핑
                    order_specs = [
                        ("1", "(A) - (C) - (B)", (0, 2, 1)),  # A=0, B=2, C=1 -> 원문순서 0,1,2 = (A)-(C)-(B)
                        ("2", "(B) - (A) - (C)", (1, 0, 2)),  # A=1, B=0, C=2 -> 원문순서 0,1,2 = (B)-(A)-(C)
                        ("3", "(B) - (C) - (A)", (2, 0, 1)),  # A=2, B=0, C=1 -> 원문순서 0,1,2 = (B)-(C)-(A)
                        ("4", "(C) - (A) - (B)", (1, 2, 0)),  # A=1, B=2, C=0 -> 원문순서 0,1,2 = (C)-(A)-(B)
                        ("5", "(C) - (B) - (A)", (2, 1, 0)),  # A=2, B=1, C=0 -> 원문순서 0,1,2 = (C)-(B)-(A)
                    ]
                    ans_num, ans_seq_str, perm = random.choice(order_specs)
                    block_a = orig_blocks[perm[0]]
                    block_b = orig_blocks[perm[1]]
                    block_c = orig_blocks[perm[2]]
                    rw.rewritten_stem = (
                        "주어진 글 다음에 이어질 글의 순서로 가장 적절한 것은?\n"
                        "[주어진 글]\n"
                        f"{intro}\n\n"
                        f"(A) {block_a}\n\n"
                        f"(B) {block_b}\n\n"
                        f"(C) {block_c}"
                    )
                    rw.rewritten_choices = [
                        "(A) - (C) - (B)",
                        "(B) - (A) - (C)",
                        "(B) - (C) - (A)",
                        "(C) - (A) - (B)",
                        "(C) - (B) - (A)",
                    ]
                    rw.preserved_answer = ans_num
                    if rw.pdf_grounded_explanation:
                        rw.pdf_grounded_explanation = _ORDER_SEQ_RE.sub(ans_seq_str, rw.pdf_grounded_explanation)
                    else:
                        rw.pdf_grounded_explanation = f"주어진 글 뒤에 논리적 흐름에 따라 {ans_seq_str} 순서로 이어지는 것이 자연스럽습니다."
        return

    if "무관" in t:
        if all(mk in raw_stem for mk in ("①", "②", "③", "④", "⑤")) and len(raw_stem) >= max(100, int(len(clean_passage) * 0.6)):
            return
        m_pos = re.search(r"INSERT_AT\s*:\s*([1-5])", raw_stem, flags=re.IGNORECASE)
        m_sent = re.search(r"SENTENCE\s*:\s*(.+)", raw_stem, flags=re.IGNORECASE | re.DOTALL)
        sents = _split_english_sentences(clean_passage)
        if m_sent and len(sents) >= 5:
            ins_sent = re.sub(r"\s+", " ", m_sent.group(1)).strip().strip("\"'")
            if ins_sent and not re.search(r"[.?!]$", ins_sent):
                ins_sent += "."
            if len(re.findall(r"[A-Za-z]+", ins_sent)) >= 5:
                insert_pos = int(m_pos.group(1)) if m_pos else random.choice([2, 3, 4])
                insert_pos = max(2, min(4, insert_pos))
                intro_cnt = 2 if len(sents) >= 6 else 1
                intro_part = " ".join(sents[:intro_cnt]).strip()
                body_orig = sents[intro_cnt:]
                numbered_orig = body_orig[:4]
                tail_orig = body_orig[4:]
                merged_five: list[str] = []
                orig_iter = iter(numbered_orig)
                for slot in range(1, 6):
                    if slot == insert_pos:
                        merged_five.append(ins_sent)
                    else:
                        try:
                            merged_five.append(next(orig_iter))
                        except StopIteration:
                            break
                if len(merged_five) == 5:
                    syms = ["①", "②", "③", "④", "⑤"]
                    marked_str = " ".join(f"{syms[i]} {merged_five[i]}" for i in range(5))
                    tail_str = (" " + " ".join(tail_orig)) if tail_orig else ""
                    rw.rewritten_stem = (
                        "다음 글에서 전체 흐름과 관계없는 문장은?\n"
                        f"{intro_part} {marked_str}{tail_str}".strip()
                    )
                    rw.rewritten_choices = list(syms)
                    rw.preserved_answer = str(insert_pos)
                    c_sym = syms[insert_pos - 1]
                    if rw.pdf_grounded_explanation:
                        rw.pdf_grounded_explanation = re.sub(
                            r"(?:[1-5]\s*번|[①②③④⑤](?:\s*번)?)",
                            f"{c_sym}번",
                            rw.pdf_grounded_explanation,
                            count=1,
                        )
        return

    if ("어법" in t or ("어휘" in t and "선택형" not in t and "(A)" not in t)) and not (
        "<u" in raw_stem.lower() and len(raw_stem) >= max(100, int(len(clean_passage) * 0.6))
    ):
        # 5줄 슬림 포맷 파싱: "1: orig" 또는 "3: orig -> replaced"
        spec_lines: dict[int, tuple[str, str]] = {}
        for ln in raw_stem.splitlines():
            m_ln = re.match(r"^\s*(?:([1-5])|([①②③④⑤]))\s*[:.)]\s*(.+?)\s*$", ln)
            if not m_ln:
                continue
            num_str = m_ln.group(1) or _CIRCLED_DIGIT_TO_NUM.get(m_ln.group(2) or "", "")
            if not num_str:
                continue
            idx_n = int(num_str)
            body_spec = m_ln.group(3).strip()
            if "->" in body_spec or "→" in body_spec:
                parts = re.split(r"\s*(?:->|→)\s*", body_spec, maxsplit=1)
                orig_p = parts[0].strip().strip("`\"'“”‘’")
                repl_p = parts[1].strip().strip("`\"'“”‘’")
            else:
                orig_p = body_spec.strip("`\"'“”‘’")
                repl_p = orig_p
            if orig_p:
                spec_lines[idx_n] = (orig_p, repl_p)

        if len(spec_lines) == 5:
            syms = ["①", "②", "③", "④", "⑤"]
            cursor = 0
            out_parts: list[str] = []
            display_words: list[str] = []
            changed_indices: list[int] = []
            matched_all = True
            for i in range(1, 6):
                orig_p, repl_p = spec_lines[i]
                # 커서 이후에서 단어 경계 우선 매칭, 실패 시 단순 부분 문자열 매칭
                seg_after = clean_passage[cursor:]
                m_word = re.search(rf"(?<![A-Za-z]){re.escape(orig_p)}(?![A-Za-z])", seg_after)
                if m_word is None:
                    m_word = re.search(re.escape(orig_p), seg_after, flags=re.IGNORECASE)
                if m_word is None:
                    matched_all = False
                    break
                abs_st = cursor + m_word.start()
                abs_en = cursor + m_word.end()
                out_parts.append(clean_passage[cursor:abs_st])
                shown = repl_p if repl_p != orig_p else clean_passage[abs_st:abs_en]
                out_parts.append(f"{syms[i - 1]} <u>{shown}</u>")
                display_words.append(shown)
                if repl_p.lower() != orig_p.lower():
                    changed_indices.append(i)
                cursor = abs_en

            if matched_all and len(changed_indices) == 1:
                out_parts.append(clean_passage[cursor:])
                assembled_body = "".join(out_parts)
                instr = (
                    "다음 글의 밑줄 친 부분 중, 어법상 틀린 것은?"
                    if "어법" in t
                    else "다음 글의 밑줄 친 부분 중, 문맥상 낱말의 쓰임이 적절하지 않은 것은?"
                )
                rw.rewritten_stem = f"{instr}\n{assembled_body}"
                rw.preserved_answer = str(changed_indices[0])
                if "어법" in t:
                    rw.rewritten_choices = list(syms)
                else:
                    rw.rewritten_choices = display_words


_RESTORED_PASSAGE_CACHE: dict[str, str] = {}
_RESTORED_PASSAGE_LOCK = threading.RLock()


_INCOMPLETE_BLANK_RE = re.compile(
    r"_{2,}|"
    r"[A-Za-z][ \t]+[.?!](?=\s|$|[\"'”’\)])|"
    r"\b(?:isn't\s+much|aren't\s+many|has\s+to|have\s+to|had\s+to|want\s+to|need\s+to|"
    r"able\s+to|going\s+to|ought\s+to|used\s+to|the|a|an|of|in|on|at|for|with|from|into|by|that|which)\s*[.?!]"
    r"(?=\s|$|[\"'”’\)])",
    re.IGNORECASE,
)


def _has_incomplete_english_blank(passage: str) -> bool:
    """원문 영어 지문 자체가 빈칸 기출이어서 비문/누락 문장이 있는지 판별한다."""
    if not passage:
        return False
    if _INCOMPLETE_BLANK_RE.search(passage):
        return True
    for sent in _split_english_sentences(passage):
        words = re.findall(r"[A-Za-z]+", sent)
        if len(words) >= 4 and not _sentence_has_english_verb(sent):
            return True
    return False


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


# ===========================================================================
# 국어 영역 (독서 · 문학 공통 지문 세트) 전용 추출 & 전체 문항 재구성
# ===========================================================================
def _sync_korean_group_passages(items: list[KoreanQuestionExtractionItem]) -> None:
    """동일한 passage_group_id를 공유하는 국어 세트 문항들의 지문·주제·세부 갈래를 동기화한다."""
    best_passage: dict[str, str] = {}
    best_topic: dict[str, str] = {}
    best_subgenre: dict[str, str] = {}
    for item in items:
        gid = normalize_korean_group_id(item.passage_group_id, item.q_num)
        item.passage_group_id = gid
        if not gid:
            continue
        p_txt = (item.passage_text or "").strip()
        if len(p_txt) > len(best_passage.get(gid, "")):
            best_passage[gid] = p_txt
        t_txt = (item.passage_topic or "").strip()
        if len(t_txt) > len(best_topic.get(gid, "")):
            best_topic[gid] = t_txt
        sg_txt = (item.korean_subgenre or "").strip()
        if len(sg_txt) > len(best_subgenre.get(gid, "")):
            best_subgenre[gid] = sg_txt

    for item in items:
        gid = item.passage_group_id
        if not gid:
            continue
        if gid in best_passage and best_passage[gid]:
            if len((item.passage_text or "").strip()) < len(best_passage[gid]) * 0.7:
                item.passage_text = best_passage[gid]
        if not (item.passage_topic or "").strip() and gid in best_topic:
            item.passage_topic = best_topic[gid]
        if not (item.korean_subgenre or "").strip() and gid in best_subgenre:
            item.korean_subgenre = best_subgenre[gid]


def analyze_korean_exam_pdf_batch_with_gemini(
    api_key: str,
    model_name: str,
    prob_pdf_bytes: bytes,
    target_q_nums: list[int],
    exam_name: str,
    answer_snippets: dict[int, str] | None = None,
    answer_pdf_part: Any | None = None,
    status_callback: Callable[[str], None] | None = None,
    parsed_crops: Sequence[Any] | None = None,
) -> dict[int, KoreanQuestionExtractionItem]:
    """
    국어 모의고사 문제지 PDF(및 해설지)를 일괄 분석하여
    공통 영역인 '독서'와 '문학' 지문 세트만 추출한다.
    텍스트 레이어가 있는 경우 로컬에서 이미 추출한 문항·지문 텍스트와 해설 스니펫만 경량 전송하고,
    AI가 지문 원문(passage_text)을 중복 출력하지 않도록 해 입력·출력 토큰을 대폭 절감한다.
    """
    ordered_q = sorted({int(q) for q in target_q_nums if 1 <= int(q) <= 45})
    if not ordered_q:
        return {}

    crop_by_q: dict[int, Any] = {}
    local_group_passages: dict[str, str] = {}
    for cq in (parsed_crops or []):
        qn = int(getattr(cq, "q_num", 0) or 0)
        if qn in ordered_q:
            crop_by_q[qn] = cq
            gid = normalize_korean_group_id(getattr(cq, "passage_group_id", ""), qn)
            sp = (getattr(cq, "shared_passage_text", "") or "").strip()
            if gid and len(sp) > len(local_group_passages.get(gid, "")):
                local_group_passages[gid] = sp

    # 로컬 텍스트 레이어가 충분한지 판별 (대상 문항 70% 이상에서 추출 텍스트 >= 40자)
    text_rich_count = sum(
        1
        for q in ordered_q
        if q in crop_by_q and len(re.sub(r"\s+", "", getattr(crop_by_q[q], "extracted_text", "") or "")) >= 40
    )
    use_lightweight_text_mode = bool(crop_by_q) and (text_rich_count / max(1, len(ordered_q))) >= 0.70

    snippets_map = answer_snippets or {}
    snippet_lines: list[str] = []
    for q in ordered_q:
        sn = (snippets_map.get(q) or "").strip()
        if sn:
            snippet_lines.append(f"[{q}번 해설 발췌] {sn[:500]}")

    # 스니펫이 대상 문항 80% 이상을 커버하면 무거운 해설지 PDF 바이너리 첨부 생략
    snippets_cover_mostly = (len(snippet_lines) / max(1, len(ordered_q))) >= 0.80

    client = _build_client(api_key, timeout_seconds=150)
    if status_callback is not None:
        mode_tag = "경량 텍스트 모드" if use_lightweight_text_mode else "PDF 비전 모드"
        status_callback(f"📖 국어 모의고사({len(ordered_q)}문항 · {mode_tag}) 일괄 분석 준비 중...")

    ncic_section, ncic_rule = _build_lang_ncic_grounding_section("국어", exam_name)
    hint_section = (
        "\n[해설지 문항별 발췌]\n" + "\n".join(snippet_lines) + "\n"
        if snippet_lines
        else ""
    )

    if use_lightweight_text_mode:
        # 로컬 추출 텍스트로 세트별 공통 지문 1회 + 문항별 발문·선지(공통 지문 제거) 조립
        text_blocks: list[str] = []
        emitted_groups: set[str] = set()
        for q in ordered_q:
            cq = crop_by_q.get(q)
            if cq is None:
                continue
            gid = normalize_korean_group_id(getattr(cq, "passage_group_id", ""), q)
            sp_txt = local_group_passages.get(gid, "")
            if gid and sp_txt and gid not in emitted_groups:
                emitted_groups.add(gid)
                text_blocks.append(f"=== [공통 지문 세트: {gid}번] ===\n{sp_txt}")
            raw_q_txt = (getattr(cq, "extracted_text", "") or "").strip()
            if sp_txt and sp_txt in raw_q_txt:
                raw_q_txt = raw_q_txt.replace(sp_txt, "").strip()
            text_blocks.append(f"[{q}번 문항 (세트:{gid or q})]\n{raw_q_txt[:1800]}")
        problem_text_section = "\n\n".join(text_blocks)

        passage_rule = (
            "4. `passage_text`: 지문 원문은 이미 시스템에 로컬 추출되어 있으므로, **이미 지문이 제공된 문항은 `passage_text`를 반드시 빈 문자열(`\"\"`)로 비워 출력 토큰을 절약하세요.** "
            "(단, 제공된 텍스트에 공통 지문이 전혀 누락된 문항에 한해서만 지문을 수록하세요.)\n"
        )
        contents: list[Any] = []
        if answer_pdf_part is not None and not snippets_cover_mostly:
            contents.append(answer_pdf_part)
        prompt = (
            f"시험 회차: {exam_name}\n"
            f"검토 대상 문항 번호(총 {len(ordered_q)}문항): {', '.join(str(q) for q in ordered_q)}\n\n"
            f"{ncic_section}"
            f"[문제지 추출 텍스트 (지문 및 문항)]\n{problem_text_section}\n"
            f"{hint_section}\n"
            "위 국어 문제지 텍스트와 해설을 바탕으로 각 문항의 영역(독서/문학/화작/언매), 세부 갈래, 세트 범위, 주제, 어휘, 정답, 해설을 JSON(questions)으로 반환하세요."
        )
        contents.append(prompt)
    else:
        prob_part = prepare_pdf_part(
            client,
            prob_pdf_bytes,
            display_name=f"{exam_name}_korean_problem.pdf",
        )
        passage_rule = (
            "4. `passage_text` (스캔 PDF 독서·문학 전용):\n"
            "   - 발문과 선택지(①~⑤), `<보기>`를 제외한 **순수 국어 지문 원문**을 추출하되, "
            "**반드시 각 세트([1~3], [4~9] 등)의 첫 번째 문항에만 1회 수록하고 같은 세트의 나머지 문항은 빈 문자열(`\"\"`)로 비워 중복 출력을 피하세요.**\n"
        )
        contents = [prob_part]
        if answer_pdf_part is not None and not snippets_cover_mostly:
            contents.append(answer_pdf_part)
        q_list_str = ", ".join(str(q) for q in ordered_q)
        prompt = (
            f"시험 회차: {exam_name}\n"
            f"검토 대상 문항 번호(총 {len(ordered_q)}문항): {q_list_str}\n"
            f"{ncic_section}"
            f"{hint_section}\n"
            "위 국어 문제지 PDF(및 해설지)에서 독서·문학 공통 영역 문항의 지문 세트, 주제, 어휘, 정답, 해설을 추출하여 JSON(questions)으로 반환하세요. "
            "(화법과 작문, 언어와 매체 문항은 korean_section만 표기하고 passage_text는 비워 두세요.)"
        )
        contents.append(prompt)

    system_inst = (
        "당신은 대한민국 수능·평가원·교육청 국어 영역(제1교시) 기출 분석 전문가입니다.\n"
        "요청된 전체 국어 문항(**독서 · 문학 · 화법과 작문 · 언어와 매체**, 1~45번 전체)을 빠짐없이 분석해 JSON(questions)으로 추출하세요.\n\n"
        "[추출 규칙]\n"
        "1. `korean_section`: 반드시 `'독서'`, `'문학'`, `'화법과 작문'`, `'언어와 매체'` 중 하나로 정확히 분류하세요.\n"
        "2. `korean_subgenre`: 독서는 `'독서론'`/`'인문·예술'`/`'사회·경제'`/`'과학·기술'`/`'주제 통합'`, 문학은 `'현대시'`/`'고전시가'`/`'현대소설'`/`'고전소설'`/`'극·수필'`/`'갈래 복합'`, 화작은 `'화법'`/`'작문'`/`'화법·작문 통합'`, 언매는 `'언어(문법)'`/`'매체'` 등으로 적으세요.\n"
        "3. `passage_group_id`: 시험지에 표시된 공통 지문 세트 범위(예: `'1~3'`, `'4~9'`, `'35~37'`, `'38~42'`)를 적으세요. 단독 문항이면 해당 문항 번호를 적으세요.\n"
        + passage_rule
        + "5. `passage_topic`: 지문의 핵심 주제, 문학 작품명·작가, 또는 문법·화작 핵심 소재를 1줄로 요약하세요.\n"
        "6. `key_vocabulary`: 지문·문항의 핵심 개념어·어휘·문법/문학 개념 3~5개를 나열하세요.\n"
        "7. `question_type`: 원래 문항 유형(예: `'세부 내용 일치'`, `'핵심 논지 추론'`, `'<보기> 사례 적용'`, `'표현상 특징'`, `'고쳐쓰기·자료 활용'`, `'음운·문장 구조 분석'`).\n"
        "8. `official_explanation`과 `answer`: 해설지 기준 핵심 풀이 근거 요약(1~2문장)과 공식 정답(`'1'`~`'5'`)을 정확히 추출하세요."
        + ncic_rule
    )

    if status_callback is not None:
        status_callback(f"⚡ {model_name} 모델로 국어 전 문항({len(ordered_q)}문항 · 독서/문학/화작/언매) 일괄 분석 중...")

    resp = _generate_content_with_fallback(
        client=client,
        model_name=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=BatchKoreanExamExtractionOutput,
            temperature=0.1,
            max_output_tokens=32768,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL,
                include_thoughts=False,
            ),
        ),
        fast_mode=True,
        status_callback=status_callback,
    )
    parsed = _parse_structured(resp, BatchKoreanExamExtractionOutput)

    items_list: list[KoreanQuestionExtractionItem] = []
    seen_q: set[int] = set()
    for item in parsed.questions:
        if item.q_num in ordered_q and item.q_num not in seen_q:
            item.answer = _normalize_extracted_answer(item.answer)
            cq_ref = crop_by_q.get(item.q_num)
            fallback_gid = getattr(cq_ref, "passage_group_id", "") if cq_ref is not None else ""
            item.passage_group_id = normalize_korean_group_id(item.passage_group_id or fallback_gid, item.q_num)
            if not (item.passage_text or "").strip():
                # AI 출력 토큰 절약을 위해 비워둔 경우 로컬 추출 지문(또는 단독 문항 본문) 자동 복원
                local_sp = local_group_passages.get(item.passage_group_id, "")
                if not local_sp and cq_ref is not None:
                    local_sp = (getattr(cq_ref, "shared_passage_text", "") or getattr(cq_ref, "extracted_text", "") or "").strip()
                if local_sp:
                    item.passage_text = local_sp
            _apply_lang_standard_validation(item, "국어")
            seen_q.add(item.q_num)
            items_list.append(item)

    _sync_korean_group_passages(items_list)
    return {it.q_num: it for it in items_list}


def analyze_korean_question_with_gemini(
    api_key: str,
    model_name: str,
    crop_png_bytes: bytes | None,
    q_num: int,
    exam_name: str,
    answer_pdf_part: Any | None = None,
    answer_text_hint: str = "",
    question_text_hint: str = "",
    shared_passage_hint: str = "",
) -> KoreanSingleExtractionOutput:
    """국어 단일 누락 문항 보완 추출 (독서·문학·화작·언매 전 영역 판별 및 지문·정답 추출)."""
    client = _build_client(api_key, timeout_seconds=45)
    ncic_section, ncic_rule = _build_lang_ncic_grounding_section("국어", exam_name)
    system_inst = (
        "당신은 대한민국 수능·모의고사 국어 영역 기출 분석 전문가입니다.\n"
        "제공된 문항 이미지/텍스트와 해설을 보고 영역(`독서`, `문학`, `화법과 작문`, `언어와 매체`), "
        "세부 갈래, 세트 범위(`passage_group_id`), 지문/자료 원문(`passage_text`), 주제/작품명, 핵심 어휘, 해설, 정답(`'1'`~`'5'`)을 추출하세요."
        + ncic_rule
    )
    hint_parts: list[str] = [f"시험 회차: {exam_name}", f"문항 번호: {q_num}번"]
    if ncic_section:
        hint_parts.append(ncic_section.strip())
    if shared_passage_hint.strip():
        hint_parts.append(f"[동일 세트 공통 지문 참고]\n{shared_passage_hint.strip()}")
    if question_text_hint.strip():
        hint_parts.append(f"[추출 텍스트 참고]\n{question_text_hint.strip()[:2500]}")
    if answer_text_hint.strip():
        hint_parts.append(f"[해설지 발췌]\n{answer_text_hint.strip()[:900]}")

    contents: list[Any] = []
    if crop_png_bytes:
        opt_png = optimize_image_for_vision(crop_png_bytes, max_width=1400)
        contents.append(types.Part.from_bytes(data=opt_png, mime_type="image/png"))
    if answer_pdf_part is not None and not answer_text_hint.strip():
        contents.append(answer_pdf_part)
    contents.append("\n\n".join(hint_parts))

    resp = _generate_content_with_fallback(
        client=client,
        model_name=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=KoreanSingleExtractionOutput,
            temperature=0.1,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL,
                include_thoughts=False,
            ),
        ),
        fast_mode=True,
    )
    out = _parse_structured(resp, KoreanSingleExtractionOutput)
    out.answer = _normalize_extracted_answer(out.answer)
    out.passage_group_id = normalize_korean_group_id(out.passage_group_id, q_num)
    if not (out.passage_text or "").strip():
        out.passage_text = (shared_passage_hint or question_text_hint or "").strip()
    _apply_lang_standard_validation(out, "국어")
    return out


def generate_korean_set_reconstruction(
    api_key: str,
    model_name: str,
    source_records: list[QuestionRecord],
    passage_text: str,
    korean_section: str = "독서",
    korean_subgenre: str = "",
    passage_topic: str = "",
) -> list[tuple[QuestionRecord, RewrittenQuestion]]:
    """
    국어 지문 1개 세트(원본 문항 수 N개, 보통 3~6문항)를 받아
    지문 전체에서 핵심 출제 포인트를 자유 유형으로 새로 설계하여 N개의 수능형 국어 문항 세트를 일괄 생성한다.
    반환되는 각 (QuestionRecord, RewrittenQuestion) 쌍 중 첫 문항의 rewritten_stem 상단에
    세트 공통 헤더 및 지문 원문을 배치하여 모드 A(PDF)·모드 B(웹 풀기)에서 1회만 출력되도록 구성한다.
    """
    ordered_recs = sorted(source_records, key=lambda r: r.q_num)
    if not ordered_recs:
        return []

    n_questions = len(ordered_recs)
    clean_passage = (passage_text or "").strip()
    if not clean_passage:
        raise ValueError("국어 지문 원문(passage_text)이 비어 있어 세트 문항을 재구성할 수 없습니다.")

    client = _build_client(api_key, timeout_seconds=120)
    orig_summary_lines = [
        f"- 원본 {r.q_num}번 ({r.unit_name or '기출'}): 핵심={'; '.join(r.core_principles[:2])} | 정답={r.answer}"
        for r in ordered_recs
    ]
    orig_summary_block = "\n".join(orig_summary_lines)

    if korean_section == "문학":
        type_guide = (
            "문학 영역 추천 유형을 지문 특성에 맞게 자유롭게 배분하세요: "
            "표현상·서술상 특징, 시적 화자/서술자 및 인물의 정서·태도, 구절·소재(㉠~㉤/ⓐ~ⓔ 등)의 함축적 의미, "
            "<보기> 외적 준거(비평문·창작 배경) 기반 작품 감상 적절성, (가)/(나) 작품 간 비교·공통점과 차이점."
        )
    elif korean_section == "화법과 작문":
        type_guide = (
            "화법과 작문 영역 추천 유형을 지문/대화 특성에 맞게 자유롭게 배분하세요: "
            "화자/발표자의 말하기 방식 및 전략, 청자/참여자의 반응 분석, 초고 고쳐쓰기 방안의 적절성, "
            "<보기> 자료(도표·조사 결과·메모) 활용 및 반영 여부."
        )
    elif korean_section == "언어와 매체":
        type_guide = (
            "언어와 매체 영역 추천 유형을 지문/자료 특성에 맞게 자유롭게 배분하세요: "
            "국어 문법(음운 변동, 단어 형성, 문장 성분·안은문장, 중세 국어) 원리 적용, "
            "<보기> 탐구 사례 분석, 매체 자료의 정보 전달 방식·수용자 반응·매체 언어 특성 이해."
        )
    else:
        type_guide = (
            "독서(비문학) 영역 추천 유형을 지문 특성에 맞게 자유롭게 배분하세요: "
            "글의 전개 방식·서술상 특징, 세부 내용 일치·불일치, 핵심 논지·개념 간 관계 추론, "
            "<보기>의 구체적 사례/상황에 지문 원리 적용 추론, 특정 문맥 어휘/구절(ⓐ~ⓔ, ㉠~㉤)의 문맥적 의미."
        )

    system_inst = (
        "당신은 대한민국 수능·평가원 국어 영역 최고 수준 문항 출제위원입니다.\n"
        f"주어진 [{korean_section} · {korean_subgenre or '공통'}] 지문을 바탕으로, 기존 기출 문항을 그대로 베끼지 말고 "
        f"**총 {n_questions}개의 고품질 수능 국어 세트 문항**을 자유 유형으로 새롭게 창작하세요.\n\n"
        "[출제 절대 원칙]\n"
        "1. **지문 100% 근거 원칙**: 모든 문항과 선지의 정오 판정은 반드시 아래 [공통 지문 원문]에 직접 명시되거나 논리적으로 도출 가능한 사실에만 근거해야 합니다. 지문에 없는 외부 지식을 정답 근거로 쓰지 마세요.\n"
        "2. **지문 표지 정밀 연동**: 지문 원문에 `[A]`, `(가)`, `(나)`, `㉠~㉤`, `ⓐ~ⓔ` 같은 표지가 있으면 발문이나 선지에서 해당 표지를 정확히 참조하세요. 지문에 없는 표지(예: 지문에 `㉮`가 없는데 `㉮`를 묻기)를 절대 지어내지 마세요.\n"
        f"3. **자유 유형 균형 출제 ({n_questions}문항)**: {type_guide}\n"
        "   - 최소 1문항 이상은 구체적인 `<보기>` 박스를 포함한 고난도 적용·감상 문항으로 출제하세요.\n"
        "4. **발문·<보기> 형식**: `stem`에는 각 문항의 발문(예: `윗글에 대한 이해로 가장 적절한 것은?`)과 필요한 경우 `<보기>\\n...` 박스만 작성하세요. **공통 지문 본문은 `stem`에 반복해서 넣지 마세요.**\n"
        "5. **선지 5개(`choices`) & 정답 분산**: `choices`는 선지 번호(①~⑤)를 제외한 5개 문자열 리스트로 작성하고, `answer`는 `'1'`~`'5'` 중 단일 숫자 문자열로 적되 정답 번호가 한 번호에 몰리지 않게 골고루 배치하세요.\n"
        "6. **정교한 해설(`explanation`)**: 정답 선지가 왜 지문과 일치하는지, 그리고 나머지 4개 오답 선지가 지문의 어떤 부분과 어긋나는지(인과 도치, 주체 오류, 범위 과장 등) 선지별로 명확히 설명하세요."
    )

    prompt = (
        f"[지문 정보]\n"
        f"- 시험 출처: {ordered_recs[0].exam_name}\n"
        f"- 영역/갈래: {korean_section} ({korean_subgenre or '공통'})\n"
        f"- 주제/작품: {passage_topic or ordered_recs[0].passage_topic}\n"
        f"- 생성할 문항 수: 정확히 {n_questions}문항 (slot_index 1 ~ {n_questions})\n\n"
        f"[공통 지문 원문]\n{clean_passage}\n\n"
        f"[원본 세트 구성 참고 (소재·난이도 참고용, 그대로 복제하지 말고 자유롭게 새 문항 창작)]\n{orig_summary_block}\n\n"
        f"위 지문을 바탕으로 정확히 {n_questions}개의 새로운 수능형 국어 세트 문항을 JSON(set_intro, questions)으로 생성하세요."
    )

    resp = _generate_content_with_fallback(
        client=client,
        model_name=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=KoreanSetReconstructionOutput,
            temperature=0.35,
            max_output_tokens=16384,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL,
                include_thoughts=False,
            ),
        ),
        fast_mode=True,
    )
    parsed = _parse_structured(resp, KoreanSetReconstructionOutput)
    gen_qs = sorted(parsed.questions or [], key=lambda x: x.slot_index)[:n_questions]
    if not gen_qs:
        raise ValueError("AI가 국어 세트 문항을 생성하지 못했습니다. 다시 시도해 주세요.")

    set_intro_txt = (parsed.set_intro or "").strip() or f"[1~{len(gen_qs)}] 다음 글을 읽고 물음에 답하시오."
    results: list[tuple[QuestionRecord, RewrittenQuestion]] = []
    for idx, gq in enumerate(gen_qs):
        base_rec = ordered_recs[idx] if idx < len(ordered_recs) else ordered_recs[-1]
        clean_choices = [re.sub(r"^[①②③④⑤]\s*", "", (c or "").strip()) for c in (gq.choices or []) if (c or "").strip()]
        while len(clean_choices) < 5:
            clean_choices.append("(선지 보완 필요)")
        clean_choices = clean_choices[:5]

        clean_stem = (gq.stem or "").strip()
        if idx == 0:
            full_stem = f"{set_intro_txt}\n\n{clean_passage}\n\n{clean_stem}"
        else:
            full_stem = clean_stem

        norm_ans = _normalize_extracted_answer(gq.answer)
        rw = RewrittenQuestion(
            original_question_id=base_rec.id,
            rewritten_stem=full_stem,
            rewritten_choices=clean_choices,
            replaced_terms_map=[],
            pdf_grounded_explanation=format_explanation_text(gq.explanation, for_markdown=False),
            preserved_answer=norm_ans,
            self_verification_note=f"국어 {korean_section} 지문 세트 자유 유형 재구성 완료 ({gq.question_type_label})",
            visual_term_note="",
            target_question_type=gq.question_type_label.strip() or "국어 세트 재구성",
            subject_domain="국어",
        )
        results.append((base_rec, rw))

    return results


def _strip_korean_exam_noise_lines(text: str) -> str:
    """국어 지문 원문에 섞인 시험지 쪽수('4', '12') 및 푸터('국어 영역', '12 국어') 노이즈 줄을 제거한다."""
    cleaned_lines: list[str] = []
    for raw_ln in (text or "").splitlines():
        s = raw_ln.strip()
        if re.match(r"^\[?\s*\d{1,2}\s*\]?$", s):
            continue
        if re.match(r"^(?:국어|수학|영어|한국사)\s*영역(?:\s*\d{1,2})?$", s):
            continue
        if re.match(r"^(?:\d{1,2}\s+국어|국어\s+\d{1,2})$", s):
            continue
        cleaned_lines.append(raw_ln)
    return "\n".join(cleaned_lines).strip()


_KOREAN_DIAGRAM_GUIDE = (
    "지문의 핵심 내용을 그림으로 한눈에 이해할 수 있도록 아래 12종 중 가장 적합한 `diagram_type`을 고르세요 "
    "(단순 박스 나열 `flow`만 반복하지 말고 지문 성격에 맞게 가장 직관적인 형식을 선택하세요):\n"
    "  ① `matrix` (항목별 비교 매트릭스표): 두 사상가·이론·법제·개념의 기준별 차이점을 비교할 때 사용. "
    "`nodes`에 비교 대상 2~3개 제목을 넣고, `matrix_rows`에 기준(`criterion`)과 대상별 값(`values`) 3~5행을 반드시 채우세요.\n"
    "  ② `graph` (X-Y 비례/함수 그래프): 경제(수요·공급·탄력성·비용·가격)나 과학(온도·압력·농도)에서 변수 간 비례·반비례·임계점 관계를 설명할 때 사용. "
    "`graph_spec`(`x_label`, `y_label`, `curves`[trend: up/down/u_shape/inv_u/flat], `highlight_point`)를 반드시 채우고 `nodes`에 1~2개 핵심 해석을 넣으세요.\n"
    "  ③ `anatomy` (구조·장치 모식도): 과학·기술·기계(튜링 기계, 세포, 엔진, 컴퓨터 구조 등)의 구성 요소와 역할을 설명할 때 사용. "
    "`center_label`에 중앙 장치/원리명을, `nodes`에 3~4개 부위·기능 카드를 넣으세요.\n"
    "  ④ `cycle` (원형 순환·피드백 루프): 상호작용, 피드백, 순환 과정(예: 네트워크 외부성, 생태계·경제 순환)에 사용. `center_label`과 `nodes` 3~4개, 짧은 `relation_labels`(6자 이내)를 넣으세요.\n"
    "  ⑤ `timeline` (시대·단계 변천 연표): 철학·과학사·역사적 이론 발전 순서에 사용. `nodes`에 시간순 3~4개 단계를 넣으세요.\n"
    "  ⑥ `venn` (포함·교집합 벤다이어그램): 두 개념의 공통점(`center_label`)과 고유 영역을 구분할 때 사용.\n"
    "  ⑦ `scale` (대립·균형 저울도): 두 상충 가치·법익·트레이드오프를 비교할 때 사용 (`nodes` 2개, `scale_tilt`: balanced/left/right).\n"
    "  ⑧ `fishbone` (피쉬본 어골도): 여러 복합 원인/요인(`nodes` 2~4개)이 하나의 핵심 결과/현상(`center_label`)으로 수렴하는 다발적 인과 분석에 사용.\n"
    "  ⑨ `quadrant` (2×2 사분면 분류): 두 개의 직교 기준(`quadrant_x_label`, `quadrant_y_label`)으로 4가지 유형(`nodes` 4개: 좌상, 우상, 좌하, 우하 순서)을 분류할 때 사용.\n"
    "  ⑩ `decision_tree` (조건분기 결정 트리): 법적 요건 충족 판정이나 알고리즘처럼 핵심 조건(`center_label`)에 따라 `예 (O)` / `아니오 (X)`(`relation_labels`, `nodes` 2~3개)로 갈라지는 판정 구조에 사용.\n"
    "  ⑪ `pyramid` (위계·계층 피라미드): 법규범 효력(헌법>법률>명령), 사회/생태/시스템 계층처럼 상하 위계가 쌓이는 구조(`nodes` 최상위→최하위 3~4개)에 사용.\n"
    "  ⑫ `flow` (원리·과정 흐름도): 인과·절차 단계가 순차적으로 이어지는 경우 사용 (`nodes` 2~4개, 짧은 `relation_labels` 6자 이내).\n"
    "공통 규칙: 각 카드 `details`는 1~3개의 짧은 핵심 구절(26자 이내), `takeaway`는 1줄 핵심 정리."
)


def generate_korean_passage_analysis(
    api_key: str,
    model_name: str,
    exam_name: str,
    group_id: str,
    passage_text: str,
    korean_section: str = "독서",
    korean_subgenre: str = "",
    passage_topic: str = "",
) -> KoreanPassageAnalysisOutput:
    """
    국어 지문 1개 세트를 수능·학원형 정밀 지문 분석지(핵심 주제, 구조 흐름, 문단/연별 요약,
    ㉠~㉤·ⓐ~ⓔ 표지 및 어휘 해설, 영역별 정리, 출제 포인트)로 구조화 분석한다.
    """
    clean_passage = _strip_korean_exam_noise_lines(passage_text or "")
    if not clean_passage:
        raise ValueError("분석할 국어 지문 원문(passage_text)이 비어 있습니다.")

    client = _build_client(api_key, timeout_seconds=90)
    if korean_section == "문학":
        area_guide = (
            "문학 영역이므로 화자/서술자의 위치와 정서·태도 변화, 핵심 시어·소재의 상징적 의미, "
            "표현상·서술상 특징(비유, 대비, 역설, 시점, 서술 방식 등), (가)/(나) 작품 간 비교 포인트를 정밀 정리하세요."
        )
    elif korean_section == "화법과 작문":
        area_guide = (
            "화법과 작문 영역이므로 발표/대화 참여자의 말하기 전략, 청자 반응, 초고의 전개 구조 및 고쳐쓰기·자료 활용 포인트를 정리하세요."
        )
    elif korean_section == "언어와 매체":
        area_guide = (
            "언어와 매체 영역이므로 핵심 국어 문법 개념·원리(음운, 품사, 문장, 중세국어 등) 또는 매체 언어 특성·정보 전달 방식을 정리하세요."
        )
    else:
        area_guide = (
            "독서(비문학) 영역이므로 중심 화제와 핵심 개념 간 관계(대조, 인과, 비례/반비례, 한계와 의의), "
            "문단별 논지 전개 흐름과 구체적 사례 적용 원리를 정밀 정리하세요."
        )

    system_inst = (
        "당신은 대한민국 대치동/수능 국어 영역 일타 강사 수준의 정밀 지문 분석가입니다.\n"
        f"주어진 [{korean_section} · {korean_subgenre or '공통'}] 지문 원문만을 100% 근거로 하여, "
        "학생과 강사가 바로 수업·복습에 활용할 수 있는 **수능·학원형 정밀 지문 분석지**를 JSON으로 작성하세요.\n\n"
        "[작성 규칙]\n"
        "1. `title`: 지문의 핵심 제목이나 작품명·작가를 간결하게 적으세요.\n"
        "2. `core_theme`: 지문 전체의 핵심 주제(또는 작품 주제 의식)를 1~2문장으로 명확히 요약하세요.\n"
        "3. `structure_flow`: 글 전체의 논리 흐름이나 시상/서사 전개를 `도입 : ...\\n전개 : ...\\n심화 : ...\\n결론 : ...`처럼 단계별 줄바꿈 형식으로 한눈에 보이게 정리하세요.\n"
        "4. `paragraph_summaries`: 독서는 문단별(1문단, 2문단...)로, 문학은 작품·연·장면별((가) 1연, (나) 도입...)로 빠짐없이 요약하세요.\n"
        "5. `marker_and_vocab_notes`: **절대 `㉠~㉤`, `ⓐ~ⓔ`, `[A]~[E]` 같은 표지 기호는 넣지 마세요.** "
        "오직 지문 원문에서 개념·용어의 정의가 명시된 문장(`'~~는 ~~~~이다'`, `'~~(이)란 ~~~~을 말한다'` 등)만 뽑아 정리하세요. "
        "`term`에는 정의되는 개념어(예: `'대리'`, `'현명주의'`, `'자기장'`)만 간결하게 적고, `meaning`에는 지문에 나온 정의를 `'~~는 ~~~~이다'` 형태로 적으세요. "
        "(지문에 명확한 정의 문장이 없으면 빈 리스트 `[]`로 두세요.)\n"
        f"6. `area_specific_notes`: {area_guide}\n"
        "7. `exam_checkpoints`: 실제 모의고사·내신에서 자주 출제되는 포인트와 선지 함정 주의사항 3~5개를 작성하세요.\n"
        "8. `visual_diagram`: "
        + (
            "비문학(독서) 지문이므로 **반드시** `visual_diagram`을 작성하세요.\n"
            + _KOREAN_DIAGRAM_GUIDE
            if korean_section == "독서"
            else "개념 비교나 흐름 도식이 이해에 큰 도움이 되는 지문이면 `visual_diagram`을 작성하고, 순수 문학 감상 위주라 도식이 불필요하면 `null`로 두세요.\n"
            + _KOREAN_DIAGRAM_GUIDE
        )
    )

    prompt = (
        f"[지문 정보]\n"
        f"- 출처: {exam_name} [{group_id}번]\n"
        f"- 영역/갈래: {korean_section} ({korean_subgenre or '공통'})\n"
        f"- 참고 주제/작품: {passage_topic or '-'}\n\n"
        f"[지문 원문]\n{clean_passage}\n\n"
        "위 지문을 바탕으로 정밀 지문 분석지 JSON을 작성하세요."
    )

    resp = _generate_content_with_fallback(
        client=client,
        model_name=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=KoreanPassageAnalysisOutput,
            temperature=0.2,
            max_output_tokens=8192,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL,
                include_thoughts=False,
            ),
        ),
        fast_mode=True,
    )
    return _parse_structured(resp, KoreanPassageAnalysisOutput)


def generate_korean_visual_diagram(
    *,
    api_key: str,
    model_name: str,
    passage_text: str,
    title: str = "",
    core_theme: str = "",
    korean_section: str = "독서",
) -> KoreanVisualDiagram:
    """기존에 분석된 국어 비문학 지문에 대해 이해를 돕는 도식 그림 스펙만 생성한다."""
    clean_passage = _strip_korean_exam_noise_lines(passage_text or "")
    if not clean_passage:
        raise ValueError("도식을 생성할 국어 지문 원문이 비어 있습니다.")

    client = _build_client(api_key, timeout_seconds=60)
    system_inst = (
        "당신은 수능 국어 비문학 일타 강사입니다. 지문 원문만을 100% 근거로, "
        "학생이 핵심 원리·인과·개념 비교를 그림으로 한눈에 이해할 수 있는 도식 명세(`KoreanVisualDiagram`) JSON을 작성하세요.\n"
        + _KOREAN_DIAGRAM_GUIDE
    )
    prompt = (
        f"[지문 영역] {korean_section}\n"
        f"[제목/주제] {title or '-'} / {core_theme or '-'}\n\n"
        f"[지문 원문]\n{clean_passage[:4500]}\n\n"
        "위 지문의 핵심을 시각화한 도식 JSON을 작성하세요."
    )
    resp = _generate_content_with_fallback(
        client=client,
        model_name=model_name,
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


