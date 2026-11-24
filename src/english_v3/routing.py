"""Declarative model, thinking_level, and execution-tier routing for English Engine v3."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from src.config import FLASH_3_6_MODEL, FLASH_LITE_MODEL

ThinkingLevelStr = Literal["minimal", "low", "medium", "high"]
ExecutionTier = Literal["standard", "flex", "batch"]
QuestionFamily = Literal[
    "gist",
    "span_inference",
    "underline_5",
    "choice_abc",
    "irrelevant",
    "order",
    "insertion",
    "summary",
]

DEFAULT_ROUTING_PROFILE = "cost_saver_lite"


@dataclass(frozen=True)
class StageRoute:
    """Single LLM call configuration (model + thinking_level + temperature)."""

    model: str = FLASH_LITE_MODEL
    thinking_level: ThinkingLevelStr = "minimal"
    temperature: float = 0.25


@dataclass(frozen=True)
class TypeRoutingRule:
    """Per-question-type routing rule covering generation, escalation retry, and Blind Solver."""

    family: QuestionFamily
    generate: StageRoute
    retry_2_generate: StageRoute
    require_blind_solver: bool
    blind_solver: StageRoute | None = None


def classify_target_family(target_type: str, vocab_style: str = "") -> QuestionFamily:
    """Map one of the 11 CSAT English types (+ vocab sub-style) to a canonical assembly family."""
    t = (target_type or "").strip()
    vs = (vocab_style or "").strip()
    if any(k in t for k in ("요지", "주제", "제목")):
        return "gist"
    if "함축" in t or "빈칸" in t:
        return "span_inference"
    if "어법" in t:
        return "underline_5"
    if "어휘" in t:
        if "선택형" in vs or "선택형" in t or "(A)" in vs or "(A)" in t:
            return "choice_abc"
        return "underline_5"
    if "무관" in t:
        return "irrelevant"
    if "순서" in t:
        return "order"
    if "삽입" in t:
        return "insertion"
    if "요약" in t:
        return "summary"
    return "gist"


def _build_cost_saver_rules() -> dict[str, TypeRoutingRule]:
    """100% gemini-3.5-flash-lite profile with per-type thinking_level differentiation."""
    return {
        "gist": TypeRoutingRule(
            family="gist",
            generate=StageRoute(FLASH_LITE_MODEL, "minimal", 0.25),
            retry_2_generate=StageRoute(FLASH_LITE_MODEL, "low", 0.30),
            require_blind_solver=False,
            blind_solver=None,
        ),
        "함축적 의미 추론": TypeRoutingRule(
            family="span_inference",
            generate=StageRoute(FLASH_LITE_MODEL, "low", 0.25),
            retry_2_generate=StageRoute(FLASH_LITE_MODEL, "medium", 0.30),
            require_blind_solver=True,
            blind_solver=StageRoute(FLASH_LITE_MODEL, "medium", 0.10),
        ),
        "span_inference": TypeRoutingRule(
            family="span_inference",
            generate=StageRoute(FLASH_LITE_MODEL, "medium", 0.25),
            retry_2_generate=StageRoute(FLASH_LITE_MODEL, "high", 0.30),
            require_blind_solver=True,
            blind_solver=StageRoute(FLASH_LITE_MODEL, "medium", 0.10),
        ),
        "어법": TypeRoutingRule(
            family="underline_5",
            generate=StageRoute(FLASH_LITE_MODEL, "medium", 0.20),
            retry_2_generate=StageRoute(FLASH_LITE_MODEL, "high", 0.25),
            require_blind_solver=True,
            blind_solver=StageRoute(FLASH_LITE_MODEL, "medium", 0.10),
        ),
        "underline_5": TypeRoutingRule(
            family="underline_5",
            generate=StageRoute(FLASH_LITE_MODEL, "low", 0.20),
            retry_2_generate=StageRoute(FLASH_LITE_MODEL, "medium", 0.25),
            require_blind_solver=True,
            blind_solver=StageRoute(FLASH_LITE_MODEL, "medium", 0.10),
        ),
        "choice_abc": TypeRoutingRule(
            family="choice_abc",
            generate=StageRoute(FLASH_LITE_MODEL, "low", 0.20),
            retry_2_generate=StageRoute(FLASH_LITE_MODEL, "medium", 0.25),
            require_blind_solver=True,
            blind_solver=StageRoute(FLASH_LITE_MODEL, "low", 0.10),
        ),
        "irrelevant": TypeRoutingRule(
            family="irrelevant",
            generate=StageRoute(FLASH_LITE_MODEL, "low", 0.25),
            retry_2_generate=StageRoute(FLASH_LITE_MODEL, "medium", 0.30),
            require_blind_solver=True,
            blind_solver=StageRoute(FLASH_LITE_MODEL, "low", 0.10),
        ),
        "order": TypeRoutingRule(
            family="order",
            generate=StageRoute(FLASH_LITE_MODEL, "medium", 0.20),
            retry_2_generate=StageRoute(FLASH_LITE_MODEL, "high", 0.25),
            require_blind_solver=True,
            blind_solver=StageRoute(FLASH_LITE_MODEL, "medium", 0.10),
        ),
        "insertion": TypeRoutingRule(
            family="insertion",
            generate=StageRoute(FLASH_LITE_MODEL, "medium", 0.20),
            retry_2_generate=StageRoute(FLASH_LITE_MODEL, "high", 0.25),
            require_blind_solver=True,
            blind_solver=StageRoute(FLASH_LITE_MODEL, "medium", 0.10),
        ),
        "summary": TypeRoutingRule(
            family="summary",
            generate=StageRoute(FLASH_LITE_MODEL, "low", 0.25),
            retry_2_generate=StageRoute(FLASH_LITE_MODEL, "medium", 0.30),
            require_blind_solver=True,
            blind_solver=StageRoute(FLASH_LITE_MODEL, "low", 0.10),
        ),
    }


def _build_hybrid_flash_36_rules() -> dict[str, TypeRoutingRule]:
    """Optional hybrid profile: keeps Flash-Lite as base, escalates hard retries & Blind Solver to 3.6-Flash."""
    base = _build_cost_saver_rules()
    for key in ("span_inference", "어법", "order", "insertion"):
        r = base[key]
        base[key] = TypeRoutingRule(
            family=r.family,
            generate=r.generate,
            retry_2_generate=StageRoute(FLASH_3_6_MODEL, "medium", r.retry_2_generate.temperature),
            require_blind_solver=True,
            blind_solver=StageRoute(FLASH_3_6_MODEL, "low", 0.10),
        )
    return base


ROUTING_PROFILES: dict[str, dict[str, TypeRoutingRule]] = {
    "cost_saver_lite": _build_cost_saver_rules(),
    "hybrid_flash_36": _build_hybrid_flash_36_rules(),
}


def resolve_active_routing_profile(profile_name: str | None = None) -> str:
    """Resolve active routing profile name from explicit arg or env var."""
    cand = (profile_name or os.environ.get("ENGLISH_V3_ROUTING_PROFILE", "") or "").strip()
    if cand in ROUTING_PROFILES:
        return cand
    return DEFAULT_ROUTING_PROFILE


def get_type_routing_rule(
    target_type: str,
    vocab_style: str = "",
    profile_name: str | None = None,
) -> TypeRoutingRule:
    """Return the TypeRoutingRule for a specific CSAT English target type."""
    eff_profile = resolve_active_routing_profile(profile_name)
    table = ROUTING_PROFILES[eff_profile]
    family = classify_target_family(target_type, vocab_style=vocab_style)
    t = (target_type or "").strip()
    if "함축" in t and "함축적 의미 추론" in table:
        return table["함축적 의미 추론"]
    if "어법" in t and "어법" in table:
        return table["어법"]
    return table[family]


def resolve_execution_tier(batch_size: int) -> tuple[ExecutionTier, bool]:
    """Select execution tier and whether to attempt short-TTL explicit caching.

    - 1~4 items: Standard + Implicit prefix cache (zero cache storage fee).
    - 5+ items: Flex (50% discount, auto-fallback to Standard) + short-TTL Explicit cache.
    """
    if int(batch_size) >= 5:
        return "flex", True
    return "standard", False
