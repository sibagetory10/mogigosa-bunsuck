"""Gemini SDK transport layer with thinking_level routing, short-TTL Explicit Cache, and Flex fallback."""

from __future__ import annotations

import contextlib
import re
from typing import Any, Iterator, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from src.config import FLASH_LITE_MODEL, resolve_routed_model_name
from src.english_v3.routing import ExecutionTier, StageRoute, ThinkingLevelStr
from src.gemini.core import _sanitize_corrupted_json_numbers
from src.gemini.usage import _emit_usage_from_response

T = TypeVar("T", bound=BaseModel)

STATIC_SYSTEM_PROMPT_V3: str = (
    "You are a senior Korean CSAT (수능) and KICE (평가원) English item writer and quality auditor.\n"
    "CORE RULES:\n"
    "1. NEVER output or rewrite the full passage. Output ONLY exact sentence indices, verbatim target spans, "
    "and newly crafted choices/distractors/notes matching the JSON schema.\n"
    "2. Every `exact_target_span` / `exact_span` MUST be an exact, character-for-character substring of the "
    "referenced indexed sentence `[Si]` (no paraphrasing, no punctuation additions).\n"
    "3. Distractor Engineering: All 4 distractors must be plausible to a superficial reader using concepts from the "
    "passage (too narrow, too broad, opposite polarity, distorted causal relation, or irrelevant keyword overlap), "
    "with uniform syntactic form, capitalization, punctuation, and word-count symmetry against the correct choice.\n"
    "4. Blank Paraphrase Rule: For blank-filling items (`빈칸 추론`), `correct_choice` MUST express the same core "
    "meaning as `exact_target_span` using paraphrased vocabulary/syntax — NEVER copy-paste the target phrase verbatim.\n"
    "5. Single-Answer Integrity: Every item must have exactly ONE defensible answer with zero ambiguity.\n"
    "6. Korean explanations (`explanation_ko` / `cohesion_rationale_ko`) must be concise (1~2 sentences) and accurate."
)

STATIC_RUBRIC_CORPUS_V3: str = (
    "[CSAT ENGLISH ITEM WRITING RUBRIC & EDGE-CASE GUARDRAILS]\n"
    "- Gist / Title / Topic (`요지 파악`, `주제 파악`, `제목 추론`): Avoid overly specific single-example traps and "
    "overly broad truisms. Keep all 5 choices balanced in length (do not make the correct choice noticeably longest).\n"
    "- Implied Meaning (`함축적 의미 추론`): Target a figurative, idiomatic, or context-loaded phrase (`exact_target_span`) "
    "whose true meaning depends on the passage's central argument rather than literal word definitions.\n"
    "- Blank Inference (`빈칸 추론`): Target the logical crux (claim, causal mechanism, or contrast resolution). "
    "Ensure substituting `correct_choice` into the blank produces a 100% grammatical sentence without duplicate prepositions/articles.\n"
    "- Grammar (`어법상 틀린 것`): Select 5 distinct, non-overlapping grammatical structures across sentences "
    "(e.g., subject-verb agreement, participle vs. finite verb, relative pronoun vs. adverb, parallelism, voice, "
    "adverb vs. adjective, modal/subordinator). Mutate ONLY ONE (`is_answer_target=True`) into a clear, unambiguous "
    "ungrammatical form (`mutated_span`). The other 4 MUST remain 100% grammatically sound.\n"
    "- Vocabulary (`문맥상 어휘 적절성`): Select 5 (or 3 for `(A)(B)(C)`) content words (verbs, adjectives, adverbs, "
    "abstract nouns) whose polarity/direction is tightly constrained by surrounding logical markers. Avoid proper nouns.\n"
    "- Irrelevant Sentence (`전체 흐름과 무관한 문장`): Reuse 1~2 surface topic nouns from the passage, but introduce "
    "an off-topic claim (e.g., historical trivia, commercial cost, unrelated side effect) that breaks paragraph unity.\n"
    "- Paragraph Order (`글의 순서`) & Sentence Insertion (`주어진 문장 삽입`): Choose split/extraction boundaries "
    "backed by explicit cohesive devices (demonstratives `this/these/such`, definite articles, contrast `however/instead`, "
    "result `therefore/thus`, or chronological progression) so only ONE sequence is logically coherent.\n"
    "- Summary Completion (`요약문 완성`): Write a 1~2 sentence abstract summary containing literal `(A)` and `(B)` "
    "slots that synthesize the passage's main cause-effect or contrast relationship."
)

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
    """Create a short-TTL explicit context cache for batch runs (5+ items) and delete it immediately on exit."""
    if not enabled:
        yield None
        return

    cache_name: str | None = None
    try:
        eff_model = resolve_routed_model_name(model_name, allow_flash_escalation=True)
        created = client.caches.create(
            model=eff_model,
            config=types.CreateCachedContentConfig(
                system_instruction=STATIC_SYSTEM_PROMPT_V3,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=STATIC_RUBRIC_CORPUS_V3)],
                    )
                ],
                display_name="mogigosa_eng_v3_batch",
                ttl=f"{max(120, int(ttl_seconds))}s",
            ),
        )
        cache_name = getattr(created, "name", None)
    except Exception:
        # Gracefully fall back to static-prefix Implicit Caching if below min token threshold or unsupported
        cache_name = None

    try:
        yield cache_name
    finally:
        if cache_name:
            try:
                client.caches.delete(name=cache_name)
            except Exception:
                pass


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
    max_parse_retries: int = 2,
) -> T:
    """Execute a structured Gemini call with explicit thinking_level, cache support, and Flex->Standard fallback."""
    eff_model = resolve_routed_model_name(route.model, allow_flash_escalation=allow_flash_escalation)
    tl_enum = _THINKING_LEVEL_MAP.get(route.thinking_level, types.ThinkingLevel.MINIMAL)
    full_static_sys = f"{STATIC_SYSTEM_PROMPT_V3}\n\n{STATIC_RUBRIC_CORPUS_V3}"

    last_err: Exception | None = None
    for attempt in range(1 + max(0, int(max_parse_retries))):
        temp = float(route.temperature)
        if attempt == 1:
            temp = min(0.7, max(temp + 0.15, 0.30))
        elif attempt >= 2:
            temp = min(0.7, max(temp + 0.30, 0.45))

        cfg_kwargs: dict[str, Any] = {
            "temperature": temp,
            "response_mime_type": "application/json",
            "response_schema": schema_cls,
            "thinking_config": types.ThinkingConfig(
                thinking_level=tl_enum,
                include_thoughts=False,
            ),
        }
        if cached_content_name:
            cfg_kwargs["cached_content"] = cached_content_name
        else:
            cfg_kwargs["system_instruction"] = full_static_sys

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
                # If explicit cache failed under standard fallback due to tier mismatch, retry without cache
                try:
                    resp = client.models.generate_content(
                        model=eff_model,
                        contents=user_prompt,
                        config=types.GenerateContentConfig(**fallback_kwargs),
                    )
                except Exception as exc2:
                    if cached_content_name and "cached_content" in str(exc2).lower():
                        fallback_kwargs.pop("cached_content", None)
                        fallback_kwargs["system_instruction"] = full_static_sys
                        resp = client.models.generate_content(
                            model=eff_model,
                            contents=user_prompt,
                            config=types.GenerateContentConfig(**fallback_kwargs),
                        )
                    else:
                        raise
            elif cached_content_name and "cached_content" in str(exc).lower():
                no_cache_kwargs = dict(cfg_kwargs)
                no_cache_kwargs.pop("cached_content", None)
                no_cache_kwargs["system_instruction"] = full_static_sys
                resp = client.models.generate_content(
                    model=eff_model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(**no_cache_kwargs),
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
) -> dict[str, Any]:
    """Build a JSON-serializable InlinedRequest dict for Gemini Batch API."""
    eff_model = resolve_routed_model_name(route.model, allow_flash_escalation=allow_flash_escalation)
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
            "system_instruction": {
                "parts": [{"text": f"{STATIC_SYSTEM_PROMPT_V3}\n\n{STATIC_RUBRIC_CORPUS_V3}"}]
            },
            "temperature": route.temperature,
            "response_mime_type": "application/json",
            "response_schema": schema_cls.model_json_schema(),
            "thinking_config": {
                "thinking_level": route.thinking_level.upper(),
                "include_thoughts": False,
            },
        },
    }
