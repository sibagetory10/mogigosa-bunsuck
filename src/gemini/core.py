from __future__ import annotations

import io
import random
import re
import time
from typing import Any, Callable

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from src.config import (
    DEFAULT_MODEL,
    FLASH_LITE_CACHED_INPUT_USD_PER_1M,
    FLASH_LITE_INPUT_USD_PER_1M,
    FLASH_LITE_MODEL,
    FLASH_LITE_OUTPUT_USD_PER_1M,
    MODEL_TOKEN_PRICING_USD_PER_1M,
    resolve_api_model_name,
)
from src.gemini.usage import (  # noqa: F401
    _ACTIVE_USAGE_RECORDER,
    _ANSWER_PREFIX_RE,
    _CHOICE_NUM_ONLY_RE,
    _CHOICE_NUM_PREFIX_RE,
    _CIRCLED_DIGIT_TO_NUM,
    ForbiddenTermRemainingError,
    T,
    TokenUsageRecord,
    _emit_usage_from_response,
    _normalize_extracted_answer,
    bind_usage_tracker,
    calculate_token_cost_usd,
    extract_usage_from_response,
    has_active_usage_tracker,
    submit_with_usage_context,
)


def _build_lang_ncic_grounding_section(subject_category: str, exam_name: str, max_standards: int = 25) -> tuple[str, str]:
    """국어/영어에 NCIC 성취기준이 등록된 경우에만 프롬프트용 기준표 섹션과 규칙 문구를 반환한다."""
    from src.curriculum_2022 import build_tagging_grounding_context, has_ncic_standards_for_category

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
    from src.curriculum_2022 import validate_standards_only

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
    from PIL import Image

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


_JSON_STR_SPLIT_RE = re.compile(r'("(?:\\.|[^"\\])*")')
_JSON_FRAC_LOOP_RE = re.compile(r"(-?\d+)\.(\d{8})\d+")
_JSON_ZERO_LOOP_RE = re.compile(r"(?<![.\d])0{2,}(?=[,}\]\s]|$)")


def _sanitize_corrupted_json_numbers(raw_text: str) -> str:
    """JSON 문자열 리터럴 밖의 폭주한 숫자(0000... / 긴 소수부)만 안전하게 정리한다."""
    if not raw_text:
        return raw_text
    parts = _JSON_STR_SPLIT_RE.split(raw_text)
    for i in range(0, len(parts), 2):
        seg = parts[i]
        if seg:
            seg = _JSON_FRAC_LOOP_RE.sub(r"\1.\2", seg)
            seg = _JSON_ZERO_LOOP_RE.sub("0", seg)
            parts[i] = seg
    return "".join(parts)


def _parse_structured(response: Any, schema_cls: type[T]) -> T:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, schema_cls):
        return parsed
    text = getattr(response, "text", None) or ""
    if not text.strip():
        raise RuntimeError("Gemini 응답이 비어 있습니다. 모델 이름 또는 입력 데이터를 확인해 주세요.")
    try:
        return schema_cls.model_validate_json(text)
    except ValidationError as orig_err:
        sanitized = _sanitize_corrupted_json_numbers(text)
        if sanitized != text:
            try:
                return schema_cls.model_validate_json(sanitized)
            except ValidationError:
                pass
        raise orig_err


def _generate_structured_with_fallback(
    client: genai.Client,
    model_name: str,
    contents: Any,
    schema_cls: type[T],
    config: types.GenerateContentConfig | None = None,
    fast_mode: bool = False,
    status_callback: Callable[[str], None] | None = None,
    max_parse_retries: int = 2,
) -> T:
    """
    구조화 JSON 생성 + 파싱 전용 래퍼.
    Flash-Lite가 temperature=0.0/0.05에서 숫자 반복 루프(`0000...`) 또는 잘린 JSON을 낼 때
    온도를 단계적으로 올려(최대 0.7) 최대 max_parse_retries회 자동 재시도한다.
    API 호출 자체의 400/401/403/429/503 오류는 _generate_content_with_fallback이 처리/전파한다.
    """
    total_attempts = 1 + max(0, int(max_parse_retries))
    orig_temp = getattr(config, "temperature", None) if config is not None else None
    base_temp = float(orig_temp) if isinstance(orig_temp, (int, float)) else 0.1
    last_parse_err: Exception | None = None

    for attempt in range(total_attempts):
        attempt_cfg = config
        if attempt > 0:
            if attempt == 1:
                new_temp = min(0.7, max(base_temp + 0.2, 0.25))
            else:
                new_temp = min(0.7, max(base_temp + 0.35, 0.45))
            if config is not None:
                attempt_cfg = config.model_copy(update={"temperature": new_temp})
            else:
                attempt_cfg = types.GenerateContentConfig(temperature=new_temp)
            if status_callback is not None:
                status_callback(
                    f"⚠️ 구조화 JSON 파싱 오류 감지 → temperature={new_temp:.2f}로 자동 재시도 ({attempt}/{total_attempts - 1})..."
                )
            time.sleep(0.3)

        resp = _generate_content_with_fallback(
            client=client,
            model_name=model_name,
            contents=contents,
            config=attempt_cfg,
            fast_mode=fast_mode,
            status_callback=status_callback,
        )
        try:
            return _parse_structured(resp, schema_cls)
        except (ValidationError, ValueError, RuntimeError) as exc:
            last_parse_err = exc
            if attempt + 1 >= total_attempts:
                raise

    assert last_parse_err is not None
    raise last_parse_err


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
