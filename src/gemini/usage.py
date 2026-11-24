"""Zero-heavy-dependency token usage tracking and answer normalization helpers."""

from __future__ import annotations

import concurrent.futures
import contextlib
import contextvars
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterator, TypeVar

from src.config import (
    FLASH_LITE_CACHED_INPUT_USD_PER_1M,
    FLASH_LITE_INPUT_USD_PER_1M,
    FLASH_LITE_MODEL,
    FLASH_LITE_OUTPUT_USD_PER_1M,
    MODEL_TOKEN_PRICING_USD_PER_1M,
    resolve_routed_model_name,
)


class ForbiddenTermRemainingError(ValueError):
    """개조 후처리 검증에서 범위 외 금지 용어가 재개조 후에도 잔존할 때 발생하는 오류."""


T = TypeVar("T")


@dataclass
class TokenUsageRecord:
    """단일 Gemini API 호출에서 실제 소비된 토큰 수와 USD 비용."""

    model_name: str
    prompt_tokens: int = 0  # 일반 입력 토큰(캐시 제외)
    cached_tokens: int = 0  # 캐시 히트 입력 토큰
    output_tokens: int = 0  # 출력 + 사고(thoughts) 토큰
    total_tokens: int = 0
    cost_usd: float = 0.0
    service_tier: str = "standard"


_ACTIVE_USAGE_RECORDER: contextvars.ContextVar[Callable[[TokenUsageRecord], None] | None] = (
    contextvars.ContextVar("mogigosa_active_usage_recorder", default=None)
)


def calculate_token_cost_usd(
    prompt_tokens: int,
    cached_tokens: int,
    output_tokens: int,
    model_name: str = FLASH_LITE_MODEL,
    *,
    service_tier: str = "standard",
    allow_routed_model: bool = False,
) -> float:
    """실제 토큰 수(일반 입력 / 캐시 입력 / 출력+사고)를 모델 단가 및 서비스 티어 할인(Flex/Batch 50%)으로 계산한다."""
    resolved = resolve_routed_model_name(model_name, allow_flash_escalation=allow_routed_model)
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
    if (service_tier or "").strip().lower() in ("flex", "batch"):
        usd *= 0.5
    return max(0.0, usd)


def extract_usage_from_response(
    response: Any,
    model_name: str = FLASH_LITE_MODEL,
    *,
    service_tier: str = "standard",
    allow_routed_model: bool = False,
) -> TokenUsageRecord:
    """Gemini 응답의 usage_metadata에서 실제 토큰 수와 USD 예상 비용을 추출한다."""
    resolved = resolve_routed_model_name(model_name, allow_flash_escalation=allow_routed_model)
    tier_norm = (service_tier or "standard").strip().lower() or "standard"
    um = getattr(response, "usage_metadata", None)
    if um is None:
        return TokenUsageRecord(model_name=resolved, service_tier=tier_norm)

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
        service_tier=tier_norm,
        allow_routed_model=allow_routed_model,
    )
    return TokenUsageRecord(
        model_name=resolved,
        prompt_tokens=non_cached_prompt,
        cached_tokens=cached,
        output_tokens=output_total,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        service_tier=tier_norm,
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


def _emit_usage_from_response(
    response: Any,
    model_name: str,
    *,
    service_tier: str = "standard",
    allow_routed_model: bool = False,
) -> None:
    recorder = _ACTIVE_USAGE_RECORDER.get()
    if recorder is None:
        return
    try:
        usage = extract_usage_from_response(
            response,
            model_name=model_name,
            service_tier=service_tier,
            allow_routed_model=allow_routed_model,
        )
        if usage.total_tokens > 0 or usage.prompt_tokens > 0 or usage.output_tokens > 0:
            recorder(usage)
    except Exception:
        pass


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
