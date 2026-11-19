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

from src.config import DEFAULT_MODEL, EXAM_TAGGING_MODEL, HYDROGEN_MODEL, resolve_api_model_name
from src.curriculum_2022 import (
    build_tagging_grounding_context,
    validate_and_normalize_extracted_tag,
)
from src.models import (
    AutoConceptPageDetectionOutput,
    AutoConceptRangeItem,
    BatchEnglishExamExtractionOutput,
    BatchExamExtractionOutput,
    BatchFilePairingOutput,
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
    normalize_english_group_id,
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
MAX_ENGLISH_TRANSFORM_JOBS: int = 20


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
    if resolved_model == HYDROGEN_MODEL:
        if config is not None and getattr(config, "thinking_config", None) is not None:
            return config
        return _with_thinking_config(
            config,
            types.ThinkingConfig(thinking_level="LOW"),
        )

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

    base_cfg = (
        eff_cfg.model_copy(update={"media_resolution": None})
        if (is_media_err and eff_cfg is not None)
        else eff_cfg
    )
    if is_media_err and not is_thinking_err and base_cfg is not None:
        recovery_configs.append(base_cfg)

    if is_thinking_err:
        cur_level = getattr(
            getattr(base_cfg if base_cfg is not None else config, "thinking_config", None),
            "thinking_level",
            None,
        )
        if cand_model == HYDROGEN_MODEL:
            t_target = base_cfg if base_cfg is not None else config
            if cur_level == types.ThinkingLevel.MINIMAL:
                recovery_configs.append(
                    _with_thinking_config(t_target, types.ThinkingConfig(thinking_budget=0))
                )
                recovery_configs.append(
                    _with_thinking_config(
                        t_target,
                        types.ThinkingConfig(
                            thinking_level=types.ThinkingLevel.LOW,
                            include_thoughts=False,
                        ),
                    )
                )
                if t_target is not None:
                    recovery_configs.append(t_target.model_copy(update={"thinking_config": None}))
            else:
                recovery_configs.append(
                    _with_thinking_config(t_target, types.ThinkingConfig(thinking_budget=256))
                )
        elif base_cfg is not None:
            if cur_level == types.ThinkingLevel.HIGH:
                recovery_configs.append(
                    base_cfg.model_copy(
                        update={
                            "thinking_config": types.ThinkingConfig(
                                thinking_level=types.ThinkingLevel.MEDIUM,
                                include_thoughts=False,
                            )
                        }
                    )
                )
            elif cur_level == types.ThinkingLevel.MINIMAL:
                recovery_configs.append(
                    base_cfg.model_copy(
                        update={"thinking_config": types.ThinkingConfig(thinking_budget=0)}
                    )
                )
                recovery_configs.append(
                    base_cfg.model_copy(
                        update={
                            "thinking_config": types.ThinkingConfig(
                                thinking_level=types.ThinkingLevel.LOW,
                                include_thoughts=False,
                            )
                        }
                    )
                )
            if cur_level != types.ThinkingLevel.MINIMAL:
                recovery_configs.append(
                    base_cfg.model_copy(
                        update={"thinking_config": types.ThinkingConfig(thinking_budget=2048)}
                    )
                )
            recovery_configs.append(base_cfg.model_copy(update={"thinking_config": None}))

    if config is not None or cand_model == HYDROGEN_MODEL or eff_cfg is not None:
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
                return client.models.generate_content(
                    model=cand_model,
                    contents=contents,
                    config=eff_cfg,
                )
            except Exception as exc:
                last_err = exc
                msg = str(exc).lower()
                # thinking_config / media_resolution 미지원 400 오류 시 단계별 안전 복구 시도
                recovery_configs = _build_400_recovery_configs(config, eff_cfg, cand_model, msg)
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
        "당신은 대한민국 2022 개정 교육과정(수학·사회·과학) 전문 기출 문항 분석가이자 정확한 문제 풀이 전문가입니다. "
        "반드시 아래 순서와 기준표에 따라 추측(할루시네이션) 없이 정확히 태깅하세요.\n"
        "1) [과목/단원/성취기준 앵커링]: 제공된 [2022 공식 기준표 및 개념서 DB] 안에서만 가장 정확한 2022 공식 `curriculum_subject`(2015 개정 명칭 금지)와 `unit_name`을 정하고, "
        "기준표에 성취기준 코드가 있으면 대표 코드는 `standard_code`, 복합 문항 등 걸치는 모든 코드는 `standard_codes` 배열에 모두 적으세요(목록에 없는 코드를 지어내지 마세요).\n"
        "2) [본질 원리 추출(`core_principles`)]: 상위 과목의 포장 용어를 걷어내고, 이 문제를 실제로 풀기 위해 필요한 근본적 풀이 원리·법칙을 간결한 한국어 개념구로 나열하세요.\n"
        "3) [용어 이중 분리(`advanced_terms` vs `basic_terms`)]: '그림/표/보기/값/cm/g/s' 등 일상어와 기초 단위는 제외하세요.\n"
        "   - `advanced_terms`: 고2~3 선택과목 등 상위 과정에서 도입되어 하위 학년 학생에게는 낯선 전문 용어·기호(예: '몰농도', 'mol', '한계효용', '도함수')\n"
        "   - `basic_terms`: `min_grade` 수준에서 이미 배워 당연한 교과 기본 용어(예: '속력', '질량', '이차방정식', '인권')\n"
        "   - `required_terms`: 문제 지문·보기에 등장하는 핵심 전문 용어 통합 목록(`advanced_terms` + 주요 `basic_terms`)\n"
        "4) [최저 학년(`min_grade`) 및 근거(`grade_evidence`)]: `core_principles`를 2022 개정 교육과정에서 처음 배우는 최저 학년(1=고1, 2=고2, 3=고3)과 1줄 판정 근거를 `grade_evidence`에 명시하세요.\n"
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
        "당신은 대한민국 2022 개정 교육과정(수학·사회·과학) 전문 기출 문항 분석가입니다. "
        "첨부된 전체 모의고사 PDF에서 요청된 모든 문항 번호를 빠짐없이 찾아, "
        "제공된 [2022 공식 기준표 및 개념서 DB]를 근거로 JSON 배열(questions)을 반환하세요.\n"
        "- q_num: 문항 번호 정수 (요청된 문항 번호를 1개도 누락하지 마세요)\n"
        "- curriculum_subject / unit_name / standard_code / standard_codes: 기준표의 2022 공식 과목명(2015 명칭 금지)과 단원을 먼저 매핑하고, 기준표에 코드가 있으면 대표 코드는 standard_code, 복합 문항에 걸친 모든 코드는 standard_codes에 적되 목록에 없는 코드를 지어내지 마세요\n"
        "- core_principles: 상위 과목 포장 용어를 걷어내고 문제를 푸는 데 필요한 근본적 풀이 원리·법칙을 간결한 한국어 개념구로 나열\n"
        "- advanced_terms vs basic_terms vs required_terms: 일상어('그림','표','보기') 및 기초 측정단위('cm','g','s')는 제외하고, 상위 과정에서 새로 도입된 낯선 전문 용어는 advanced_terms, min_grade 시점 기본 용어는 basic_terms, 통합 핵심 용어는 required_terms에 기록\n"
        "- min_grade / grade_evidence: core_principles를 2022 개정 교육과정에서 처음 배우는 최저 학년(1=고1, 2=고2, 3=고3)과 1줄 근거 작성\n"
        + ans_rule_line
    )

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
            snippet_lines.append(f"[{q}번 해설 발췌] {sn[:450]}")

    hint_section = (
        "\n[해설지 문항별 발췌]\n" + "\n".join(snippet_lines) + "\n"
        if snippet_lines
        else ""
    )

    system_inst = (
        "당신은 대한민국 수능·평가원·교육청 영어 영역(제3교시) 기출 분석 전문가입니다.\n"
        "영어 과목은 학년 제한(min_grade)이나 2022 성취기준 코드 매핑을 하지 않으며, 아래 규칙에 따라 요청된 모든 문항(1~45번)을 JSON(questions)으로 추출하세요.\n\n"
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
        "   - **장문 세트(41~42번, 43~45번)에만** `passage_group_id`를 `'41~42'` 또는 `'43~45'`로 표기하고 공통 장문 지문 전체를 수록하세요. **그 외 모든 문항(예: 31~34, 36~37, 38~39 등 발문 헤더 범위 포함)은 문항마다 지문이 각각 다르므로 `passage_group_id`를 반드시 빈 문자열로 두고 해당 번호 문항의 지문만 추출하세요.**\n"
        "   - `passage_topic`: 지문의 핵심 소재·주제를 한국어 1줄로 요약하세요.\n"
        "   - `key_vocabulary`: 지문 내 주요 어휘 3~6개를 `'단어: 한국어 뜻'` 형식으로 나열하세요.\n"
        "4. `official_explanation`과 `answer`: 해설지 기준 핵심 풀이 요약과 공식 정답(`'1'`~`'5'`)을 정확히 추출하세요."
    )

    q_list_str = ", ".join(str(q) for q in ordered_q)
    prompt = (
        f"시험 회차: {exam_name}\n"
        f"추출 대상 문항 번호(총 {len(ordered_q)}문항): {q_list_str}\n"
        f"{hint_section}\n"
        "위 영어 문제지 PDF(및 해설지)에서 지정된 모든 문항 번호를 빠짐없이 추출하여 JSON(questions)으로 반환하세요."
    )

    contents: list[Any] = [prob_part]
    if answer_pdf_part is not None and not snippet_lines:
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

    system_inst = (
        "당신은 대한민국 수능·모의고사 영어 기출 문항 분석가입니다.\n"
        f"- 이 문항({q_num}번)은 {'듣기(1~17번)' if is_listening else '독해(18~45번)'} 문항입니다.\n"
        "- 듣기 문항이면 `english_section='듣기'`, `passage_text=''`로 두고 유형·주제·정답·해설만 추출하세요.\n"
        "- 독해 문항이면 `english_section='독해'`, 한국어 발문과 선택지 ①~⑤를 제외한 순수 영어 본문 전체를 `passage_text`에 빠짐없이 추출하고, "
        "원래 유형(`english_question_type`), 주제(`passage_topic`), 핵심 어휘(`key_vocabulary`), 정답(`answer`), 해설(`official_explanation`)을 JSON으로 반환하세요."
    )

    hint_parts: list[str] = [f"시험 회차: {exam_name}", f"문항 번호: {q_num}번"]
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
    )


def _build_english_transform_rule(
    target_type: str,
    blank_unit: str = "구 (3~6단어)",
    vocab_style: str = "밑줄형 (①~⑤)",
) -> tuple[str, str]:
    """입력 토큰 절감을 위해 11개 전체 규칙 대신 지정된 target_type 1개 유형의 규칙만 동적으로 반환한다."""
    t = (target_type or "").strip()

    if "함축" in t:
        return (
            "함축적 의미 추론",
            "- 한국어 발문: `밑줄 친 부분이 다음 글에서 의미하는 바로 가장 적절한 것은?`\n"
            "- 지문 규칙: 원문 지문에서 비유적·핵심적 의미를 담은 어구 1곳에만 `<u>...</u>` 태그로 밑줄을 표시하세요. (기존 ①~⑤ 번호나 불필요한 표기는 모두 제거하고 자연스러운 본문으로 정리)\n"
            "- 선택지(`rewritten_choices`): 밑줄 친 표현이 문맥상 함축하는 의미를 나타낸 **영어 선지 5개**(선지 번호 ①~⑤는 빼고 영어 문장/구 텍스트만 5개)를 작성하세요.",
        )
    if "요지" in t:
        return (
            "요지 파악",
            "- 한국어 발문: `다음 글의 요지로 가장 적절한 것은?`\n"
            "- 지문 규칙: 원문의 기존 밑줄/빈칸/번호(①~⑤, (A)(B)(C))가 있었다면 자연스러운 완결 영어 지문으로 복원·정리하세요.\n"
            "- 선택지(`rewritten_choices`): 평가원 표준에 따라 **한국어 완결 문장 선지 5개**(번호 제외)를 작성하세요.",
        )
    if "주제" in t:
        return (
            "주제 파악",
            "- 한국어 발문: `다음 글의 주제로 가장 적절한 것은?`\n"
            "- 지문 규칙: 원문의 기존 밑줄/빈칸/번호를 제거하고 완결된 영어 본문으로 정리하세요.\n"
            "- 선택지(`rewritten_choices`): 평가원 표준에 따라 **영어 명사구 선지 5개**(번호 제외, 예: `necessity of balancing ...`)를 작성하세요.",
        )
    if "제목" in t:
        return (
            "제목 추론",
            "- 한국어 발문: `다음 글의 제목으로 가장 적절한 것은?`\n"
            "- 지문 규칙: 원문의 기존 밑줄/빈칸/번호를 제거하고 완결된 영어 본문으로 정리하세요.\n"
            "- 선택지(`rewritten_choices`): 평가원 표준에 따라 **영어 제목 선지 5개**(번호 제외, 첫 글자 대문자 타이틀 형식)를 작성하세요.",
        )
    if "어법" in t:
        return (
            "어법상 틀린 것 (밑줄 ①~⑤)",
            "- 한국어 발문: `다음 글의 밑줄 친 부분 중, 어법상 틀린 것은?`\n"
            "- 지문 규칙: 지문 전체에서 핵심 문법 포인트(수일치, 준동사/본동사, 관계사/접속사, 분사 능·수동, 대명사/대동사 등) 5곳을 골라 반드시 `① <u>word1</u>`, `② <u>word2</u>`, `③ <u>word3</u>`, `④ <u>word4</u>`, `⑤ <u>word5</u>` 형식으로 지문 안에 표시하세요.\n"
            "- **필수 검증**: 5개 중 **정확히 1개(정답 번호)**만 문맥·구조상 명백히 틀린 어법 형태로 변형하고, 나머지 4개는 원문 문법적으로 100% 올바른 형태여야 합니다(복수 정답 절대 금지).\n"
            "- 선택지(`rewritten_choices`): `[\"①\", \"②\", \"③\", \"④\", \"⑤\"]` 또는 각 번호에 밑줄 어구를 덧붙인 5개 항목(예: `[\"<u>word1</u>\", ...]`)으로 반환하세요.",
        )
    if "어휘" in t:
        if "선택형" in vocab_style or "(A)" in vocab_style:
            return (
                "문맥상 어휘 적절성 (선택형 (A)(B)(C))",
                "- 한국어 발문: `(A), (B), (C)의 각 네모 안에서 문맥에 맞는 낱말로 가장 적절한 것은?`\n"
                "- 지문 규칙: 지문 내 문맥 흐름을 결정하는 핵심 어휘 3곳을 골라 `(A) [word1 / word2]`, `(B) [word3 / word4]`, `(C) [word5 / word6]`(반의어 또는 혼동 어휘 쌍)으로 본문 안에 표시하세요.\n"
                "- 선택지(`rewritten_choices`): `\"(A) word1 — (B) word3 — (C) word5\"` 형식의 5개 영어 어휘 조합을 작성하세요.",
            )
        return (
            "문맥상 어휘 적절성 (밑줄 ①~⑤)",
            "- 한국어 발문: `다음 글의 밑줄 친 부분 중, 문맥상 낱말의 쓰임이 적절하지 않은 것은?`\n"
            "- 지문 규칙: 지문 전체에서 논리 흐름상 중요한 어휘 5곳을 골라 반드시 `① <u>word1</u>` ~ `⑤ <u>word5</u>` 형식으로 본문 안에 표시하세요.\n"
            "- **필수 검증**: 5개 중 **정확히 1개(정답 번호)**만 문맥과 정반대이거나 부적절한 어휘(반의어 등)로 바꾸고, 나머지 4개는 원문 논리에 정확히 부합해야 합니다.\n"
            "- 선택지(`rewritten_choices`): 밑줄 친 5개 어휘 리스트(예: `[\"<u>word1</u>\", ...]`)를 순서대로 반환하세요.",
        )
    if "빈칸" in t:
        b_u = (blank_unit or "").strip()
        if b_u.startswith("절") or ("절(" in b_u or "절 (" in b_u):
            unit_desc = "주어+술어 또는 접속사가 포함된 **절(6~12단어)**"
            short_u = "절"
        elif b_u.startswith("단어") or ("단어" in b_u and not b_u.startswith("구")):
            unit_desc = "핵심 추상 명사/동사 **단어(1~2단어)**"
            short_u = "단어"
        else:
            unit_desc = "핵심 의미를 압축한 **구(3~6단어)**"
            short_u = "구"
        return (
            f"빈칸 추론 ({short_u})",
            "- 한국어 발문: `다음 빈칸에 들어갈 말로 가장 적절한 것은?`\n"
            f"- 지문 규칙: 지문의 주제문 또는 핵심 추론 구간을 골라 {unit_desc} 분량을 `___________`(언더스코어 10개)으로 비우세요. (기존에 다른 빈칸이나 ①~⑤ 표기가 있었다면 원문 문맥으로 복원한 뒤 새 빈칸 1개만 남기세요.)\n"
            f"- 선택지(`rewritten_choices`): 빈칸 단위에 정확히 맞고 문법적으로 호응하는 **영어 선지 5개**({unit_desc} 길이·품사 통일, 정답 1개 + 지문 키워드를 교묘히 비튼 매력적 오답 4개)를 작성하세요.",
        )
    if "무관" in t:
        return (
            "전체 흐름과 무관한 문장",
            "- 한국어 발문: `다음 글에서 전체 흐름과 관계없는 문장은?`\n"
            "- 지문 규칙: 첫 1~2문장(주제 도입문) 뒤에 이어지는 본문 문장 5개 앞에 `①`, `②`, `③`, `④`, `⑤` 번호를 붙이세요. 그중 **정답 번호 1개 문장**만 지문의 중심 소재(단어)는 공유하되 글의 핵심 논지·방향에서 벗어난 무관한 영어 문장으로 작성(또는 교체)하세요.\n"
            "- 선택지(`rewritten_choices`): `[\"①\", \"②\", \"③\", \"④\", \"⑤\"]` 5개를 반환하세요.",
        )
    if "순서" in t:
        return (
            "글의 순서 ((A)-(B)-(C))",
            "- 한국어 발문: `주어진 글 다음에 이어질 글의 순서로 가장 적절한 것은?`\n"
            "- 지문 규칙: 도입부 1~2문장을 `[주어진 글]` 섹션으로 분리하고, 나머지 본문을 지시어(this, these, such)·연결어(However, For example, Therefore)·대명사 연결 고리가 명확한 3개 단락 `(A)`, `(B)`, `(C)`로 나눈 뒤 `(A)-(B)-(C)` 단순 순서가 아닌 평가원 표준 오순(예: `(B)-(A)-(C)`, `(C)-(A)-(B)` 등)으로 배치하세요.\n"
            "- 형식 예시:\n"
            "  `주어진 글 다음에 이어질 글의 순서로 가장 적절한 것은?`\n"
            "  `[주어진 글]`\n"
            "  `도입 영어 문장...`\n\n"
            "  `(A) 단락...`\n"
            "  `(B) 단락...`\n"
            "  `(C) 단락...`\n"
            "- 선택지(`rewritten_choices`): 반드시 `[\"(A) - (C) - (B)\", \"(B) - (A) - (C)\", \"(B) - (C) - (A)\", \"(C) - (A) - (B)\", \"(C) - (B) - (A)\"]` 5개 표준 조합을 사용하고 정답 조합이 유일하도록 구성하세요.",
        )
    if "삽입" in t:
        return (
            "주어진 문장 삽입",
            "- 한국어 발문: `글의 흐름으로 보아, 주어진 문장이 들어가기에 가장 적절한 곳은?`\n"
            "- 지문 규칙: 지문 중 논리적 전환점이나 지시어·연결어 단서가 뚜렷한 핵심 문장 1개를 빼내어 `[주어진 문장]` 섹션에 배치하고, 본문 내 5곳의 문장 경계에 `( ① )`, `( ② )`, `( ③ )`, `( ④ )`, `( ⑤ )`를 표시하세요. 정답 위치는 반드시 `①`~`⑤` 중 유일해야 합니다.\n"
            "- 형식 예시:\n"
            "  `글의 흐름으로 보아, 주어진 문장이 들어가기에 가장 적절한 곳은?`\n"
            "  `[주어진 문장]`\n"
            "  `삽입할 영어 문장...`\n\n"
            "  `본문 첫 문장... ( ① ) 다음 문장... ( ② ) ... ( ⑤ ) 마지막 문장...`\n"
            "- 선택지(`rewritten_choices`): `[\"①\", \"②\", \"③\", \"④\", \"⑤\"]` 5개를 반환하세요.",
        )
    # 기본: 요약문 완성
    return (
        "요약문 완성 ((A), (B))",
        "- 한국어 발문: `다음 글의 내용을 한 문장으로 요약하고자 한다. 빈칸 (A), (B)에 들어갈 말로 가장 적절한 것은?`\n"
        "- 지문 규칙: 원문 영어 본문 아래에 `[요약문]` 섹션을 추가하고, 글 전체의 핵심 인과·대조·요지를 한 문장으로 압축하되 핵심 어휘 2곳을 `(A) ________`와 `(B) ________`로 비운 영어 요약문을 작성하세요.\n"
        "- 형식 예시:\n"
        "  `다음 글의 내용을 한 문장으로 요약하고자 한다. 빈칸 (A), (B)에 들어갈 말로 가장 적절한 것은?`\n"
        "  `영어 본문...`\n\n"
        "  `[요약문]`\n"
        "  `... (A) ________ ... (B) ________ ...`\n"
        "- 선택지(`rewritten_choices`): `\"(A) word1 — (B) word2\"` 형식의 영어 어휘 쌍 선지 5개를 작성하세요.",
    )


def _validate_english_transformed_output(
    rw: RewrittenQuestion,
    target_type: str,
    vocab_style: str,
) -> list[str]:
    """생성된 영어 변형 문항의 필수 태그(<u>, 빈칸, 박스, (A)(B)(C), 선지 5개) 누락 여부를 검증한다."""
    issues: list[str] = []
    stem = rw.rewritten_stem or ""
    t = (target_type or "").strip()

    ans = _normalize_extracted_answer(rw.preserved_answer)
    if ans not in ("1", "2", "3", "4", "5"):
        issues.append(f"정답(preserved_answer)이 '1'~'5' 중 하나여야 합니다 (현재: {rw.preserved_answer!r}).")

    if len(rw.rewritten_choices or []) != 5:
        issues.append(f"선택지(rewritten_choices)는 정확히 5개여야 합니다 (현재: {len(rw.rewritten_choices or [])}개).")

    u_count = len(re.findall(r"<u\b[^>]*>.*?</u>", stem, flags=re.I | re.S))
    if "어법" in t or ("어휘" in t and "선택형" not in vocab_style and "(A)" not in vocab_style):
        if u_count < 5:
            issues.append(f"지문(rewritten_stem) 안에 `① <u>...</u>` ~ `⑤ <u>...</u>` 밑줄 태그 5개가 모두 포함되어야 합니다 (현재 `<u>` {u_count}개).")
    elif "함축" in t:
        if u_count < 1:
            issues.append("함축적 의미 추론 지문에는 밑줄 친 표현 `<u>...</u>`가 최소 1개 포함되어야 합니다.")
    elif "빈칸" in t:
        if "____" not in stem:
            issues.append("빈칸 추론 지문에는 `___________` 빈칸 표기가 반드시 포함되어야 합니다.")
    elif "순서" in t:
        for mk in ("(A)", "(B)", "(C)"):
            if mk not in stem:
                issues.append(f"글의 순서 지문에 `{mk}` 단락 표기가 누락되었습니다.")
        if "[주어진 글]" not in stem and "주어진 글" not in stem:
            issues.append("글의 순서 지문에 `[주어진 글]` 섹션 헤더가 누락되었습니다.")
    elif "삽입" in t:
        if "[주어진 문장]" not in stem and "주어진 문장" not in stem:
            issues.append("문장 삽입 지문에 `[주어진 문장]` 섹션 헤더가 누락되었습니다.")
        for c_num in ("①", "②", "③", "④", "⑤"):
            if c_num not in stem:
                issues.append(f"문장 삽입 본문에 위치 기호 `{c_num}`가 누락되었습니다.")
                break
    elif "요약" in t:
        if "[요약문]" not in stem and "요약문" not in stem:
            issues.append("요약문 완성 지문에 `[요약문]` 섹션 헤더가 누락되었습니다.")
        if "(A)" not in stem or "(B)" not in stem:
            issues.append("요약문에 `(A)`와 `(B)` 빈칸 표기가 모두 포함되어야 합니다.")

    return issues


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
) -> RewrittenQuestion:
    """
    원문 영어 독해 지문을 지정된 수능 독해 유형(1개 유형 규칙만 동적 주입)으로 변형·출제한다.
    Thinking 토큰 최소화(기본 MINIMAL, 정밀 모드 시 LOW) 및 2줄 이내 핵심 정답 근거만 지연 생성 전제로 출력해
    토큰 과금을 절감한 뒤 RewrittenQuestion 호환 객체로 반환한다.
    """
    clean_passage = (source_passage or "").strip()
    if not clean_passage:
        raise ValueError("변형할 영어 지문(source_passage)이 비어 있습니다.")

    eff_model = (model_name or "").strip() or EXAM_TAGGING_MODEL
    canonical_type, type_rule_block = _build_english_transform_rule(
        target_type=target_type,
        blank_unit=blank_unit,
        vocab_style=vocab_style,
    )
    target_thinking_level = (
        types.ThinkingLevel.LOW
        if (use_precision_mode and not _is_flash_lite_model(eff_model))
        else types.ThinkingLevel.MINIMAL
    )

    client = _build_client(api_key, timeout_seconds=90)
    system_inst = (
        "당신은 대한민국 수능·평가원 영어 영역 출제위원입니다.\n"
        "원문 지문의 논리와 문법성을 유지하면서 지정된 **[목표 유형 1개]**에 맞춰 `RewrittenQuestion` JSON 1건을 생성하세요.\n\n"
        f"[목표 유형: {canonical_type} | 목표 난이도: {difficulty}]\n"
        f"{type_rule_block}\n\n"
        "[출제 조판 및 출력 토큰 절감 규칙]\n"
        "1. `rewritten_stem`: 첫 줄에 한국어 발문을 적고 줄바꿈 후 지문(및 필요한 `[주어진 문장]`/`[주어진 글]`/`[요약문]` 섹션)을 배치하세요.\n"
        "2. 밑줄이 필요한 유형(어법, 어휘 밑줄형, 함축적 의미)은 반드시 HTML `<u>...</u>` 태그를 사용하세요.\n"
        "3. `rewritten_choices`: 정확히 5개 선지 리스트를 반환하세요(번호 ①~⑤ 제외, 번호형만 `['①','②','③','④','⑤']`).\n"
        "4. `preserved_answer`: 유일한 정답 번호 `'1'`, `'2'`, `'3'`, `'4'`, `'5'` 중 단일 숫자 문자열 1개만 기록하세요.\n"
        "5. `pdf_grounded_explanation`: 출력 토큰 절감을 위해 구문 분석이나 ①~⑤ 전체 오답 분석을 절대 작성하지 말고, 왜 정답 번호가 정답인지 **2줄 이내(150자 이내)의 핵심 정답 근거만** 간결하게 작성하세요. (상세 구문 및 선지별 오답 해설은 사용자가 별도 버튼 클릭 시 지연 생성됩니다.)\n"
        "6. `replaced_terms_map`: 핵심 어휘 2~3개만 `'영단어 → 뜻'` 형태로 간결히 적으세요.\n"
        "7. `self_verification_note`: 1줄(40자 이내)로만 간결히 적으세요."
    )

    prev_feedback_block = ""
    if previous_attempt is not None:
        prev_feedback_block = (
            "\n[이전 생성본 (재변환 요청 — 다른 포인트/오답 구성으로 새롭게 출제)]\n"
            f"- 이전 지문/발문: {previous_attempt.rewritten_stem[:500]}\n"
            f"- 이전 정답: {previous_attempt.preserved_answer}\n"
        )

    base_prompt = (
        f"[출처]: {source_label or '영어 기출 독해 지문'}\n"
        f"[목표 수능 유형]: {canonical_type} | [난이도]: {difficulty}\n"
        f"{prev_feedback_block}\n"
        f"[원문 영어 지문]\n{clean_passage}\n\n"
        "위 원문 영어 지문을 바탕으로 목표 유형의 출제 규칙만 엄격히 준수하여 변형 문제와 2줄 이내 핵심 정답 근거를 JSON으로 생성하세요."
    )

    prompt = base_prompt
    rw: RewrittenQuestion | None = None
    # 출력에는 지문 전체(+선지)가 포함되므로 지문 길이에 비례해 출력 토큰 상한을 잡는다 (JSON 중간 잘림 방지).
    base_out_tokens = min(8192, max(3072, int(len(clean_passage) / 2.0) + 1800))
    for attempt in range(2):
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

        issues = _validate_english_transformed_output(rw, canonical_type, vocab_style)
        if not issues:
            break
        if attempt == 0:
            prompt = (
                base_prompt
                + "\n\n[⚠️ 직전 출력 형식 오류 — 반드시 아래 누락 사항을 모두 수정하여 다시 생성하세요]\n"
                + "\n".join(f"- {iss}" for iss in issues)
            )

    assert rw is not None
    rw.rewritten_stem = normalize_english_exam_text(rw.rewritten_stem)
    rw.pdf_grounded_explanation = format_explanation_text(rw.pdf_grounded_explanation)
    rw.preserved_answer = _normalize_extracted_answer(rw.preserved_answer) or "1"
    if len(rw.rewritten_choices) < 5:
        default_syms = ["①", "②", "③", "④", "⑤"]
        while len(rw.rewritten_choices) < 5:
            rw.rewritten_choices.append(default_syms[len(rw.rewritten_choices)])
    rw.target_question_type = canonical_type
    rw.subject_domain = "영어"
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
