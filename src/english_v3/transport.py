"""Lean Gemini SDK transport layer with thinking_level routing and Flex->Standard fallback."""

from __future__ import annotations

import contextlib
from typing import Any, Iterator, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from src.config import resolve_routed_model_name
from src.english_v3.routing import ExecutionTier, StageRoute, ThinkingLevelStr
from src.gemini.core import _sanitize_corrupted_json_numbers
from src.gemini.usage import _emit_usage_from_response

T = TypeVar("T", bound=BaseModel)

COMPACT_GEN_SYSTEM_PROMPT: str = (
    "Korean CSAT/KICE English item writer.\n"
    "1. Output JSON delta/coordinates ONLY; never rewrite passage.\n"
    "2. exact_target_span/exact_span MUST match [Si] verbatim.\n"
    "3. 4 distractors must use distinct CSAT traps (narrow/broad/opposite/distorted) with uniform grammar, "
    "casing, punctuation, and word-count symmetry vs correct_choice.\n"
    "4. Blank correct_choice MUST paraphrase exact_target_span (no verbatim copy).\n"
    "5. explanation_ko: 1~2 concise Korean sentences (<140 chars)."
)

COMPACT_SOLVER_SYSTEM_PROMPT: str = (
    "Independent Korean CSAT English solver & quality auditor. "
    "Solve strictly from passage logic, anaphora, and grammar. "
    "Flag any defensible runner-up in ambiguous_choices."
)

# Backwards-compatible aliases
STATIC_SYSTEM_PROMPT_V3: str = COMPACT_GEN_SYSTEM_PROMPT
STATIC_RUBRIC_CORPUS_V3: str = ""

_THINKING_LEVEL_MAP: dict[ThinkingLevelStr, types.ThinkingLevel] = {
    "minimal": types.ThinkingLevel.MINIMAL,
    "low": types.ThinkingLevel.LOW,
    "medium": types.ThinkingLevel.MEDIUM,
    "high": types.ThinkingLevel.HIGH,
}


def build_v3_client(api_key: str, timeout_seconds: int = 30) -> genai.Client:
    """Create a google-genai client with explicit per-request timeout."""
    if not (api_key or "").strip():
        raise ValueError("Gemini API 키가 설정되지 않았습니다.")
    return genai.Client(
        api_key=api_key.strip(),
        http_options=types.HttpOptions(
            timeout=max(5, int(timeout_seconds)) * 1000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


@contextlib.contextmanager
def temporary_explicit_cache(
    client: genai.Client,
    model_name: str,
    enabled: bool,
    ttl_seconds: int = 600,
) -> Iterator[str | None]:
    """Yield explicit cache only when static corpus exceeds Gemini's 1,024-token cache minimum."""
    _ = (client, model_name, ttl_seconds)
    if not enabled or len(COMPACT_GEN_SYSTEM_PROMPT) < 4096:
        yield None
        return
    yield None


def _should_fallback_from_flex(exc: Exception) -> bool:
    msg = str(exc).lower()
    if any(k in msg for k in ("401", "403", "api key not valid", "api_key_invalid", "unauthenticated")):
        return False
    return any(
        k in msg
        for k in (
            "429",
            "503",
            "504",
            "unavailable",
            "overloaded",
            "resource_exhausted",
            "timeout",
            "deadline",
            "service_tier",
            "flex",
        )
    )


def _parse_v3_structured(response: Any, schema_cls: type[T]) -> T:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, schema_cls):
        return parsed
    text = getattr(response, "text", None) or ""
    if not text.strip():
        raise RuntimeError("Gemini 응답이 비어 있습니다.")
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


def call_structured_v3(
    client: genai.Client,
    *,
    route: StageRoute,
    user_prompt: str,
    schema_cls: type[T],
    service_tier: ExecutionTier = "standard",
    cached_content_name: str | None = None,
    allow_flash_escalation: bool = False,
    max_parse_retries: int = 1,
    system_instruction_override: str | None = None,
    max_output_tokens: int = 750,
) -> T:
    """Execute a token-minimized structured Gemini call with thinking_level and Flex->Standard fallback."""
    eff_model = resolve_routed_model_name(route.model, allow_flash_escalation=allow_flash_escalation)
    tl_enum = _THINKING_LEVEL_MAP.get(route.thinking_level, types.ThinkingLevel.MINIMAL)
    sys_inst = system_instruction_override or COMPACT_GEN_SYSTEM_PROMPT

    last_err: Exception | None = None
    for attempt in range(1 + max(0, int(max_parse_retries))):
        temp = float(route.temperature)
        if attempt >= 1:
            temp = min(0.65, max(temp + 0.15, 0.30))

        cfg_kwargs: dict[str, Any] = {
            "temperature": temp,
            "max_output_tokens": max(128, int(max_output_tokens)),
            "response_mime_type": "application/json",
            "response_schema": schema_cls,
            "thinking_config": types.ThinkingConfig(
                thinking_level=tl_enum,
                include_thoughts=False,
            ),
        }
        if cached_content_name and not system_instruction_override:
            cfg_kwargs["cached_content"] = cached_content_name
        else:
            cfg_kwargs["system_instruction"] = sys_inst

        used_tier: ExecutionTier = "flex" if service_tier == "flex" else "standard"
        if used_tier == "flex":
            cfg_kwargs["service_tier"] = types.ServiceTier.FLEX

        cfg = types.GenerateContentConfig(**cfg_kwargs)
        resp: Any = None
        try:
            resp = client.models.generate_content(
                model=eff_model,
                contents=user_prompt,
                config=cfg,
            )
        except Exception as exc:
            if used_tier == "flex" and _should_fallback_from_flex(exc):
                used_tier = "standard"
                fallback_kwargs = dict(cfg_kwargs)
                fallback_kwargs.pop("service_tier", None)
                resp = client.models.generate_content(
                    model=eff_model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(**fallback_kwargs),
                )
            else:
                raise

        _emit_usage_from_response(
            resp,
            eff_model,
            service_tier=used_tier,
            allow_routed_model=allow_flash_escalation,
        )
        try:
            return _parse_v3_structured(resp, schema_cls)
        except (ValidationError, ValueError, RuntimeError) as parse_exc:
            last_err = parse_exc
            if attempt >= max_parse_retries:
                raise

    assert last_err is not None
    raise last_err


def build_inlined_request_dict(
    *,
    route: StageRoute,
    user_prompt: str,
    schema_cls: type[BaseModel],
    metadata: dict[str, str],
    allow_flash_escalation: bool = False,
    system_instruction_override: str | None = None,
    max_output_tokens: int = 750,
) -> dict[str, Any]:
    """Build a compact JSON-serializable InlinedRequest dict for Gemini Batch API."""
    eff_model = resolve_routed_model_name(route.model, allow_flash_escalation=allow_flash_escalation)
    sys_inst = system_instruction_override or COMPACT_GEN_SYSTEM_PROMPT
    return {
        "model": eff_model,
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}],
            }
        ],
        "metadata": metadata,
        "config": {
            "system_instruction": {"parts": [{"text": sys_inst}]},
            "temperature": route.temperature,
            "max_output_tokens": max(128, int(max_output_tokens)),
            "response_mime_type": "application/json",
            "response_schema": schema_cls.model_json_schema(),
            "thinking_config": {
                "thinking_level": route.thinking_level.upper(),
                "include_thoughts": False,
            },
        },
    }
