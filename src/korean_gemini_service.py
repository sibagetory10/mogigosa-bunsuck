"""Backwards-compatible lazy proxy facade for Korean Gemini services."""

from __future__ import annotations

import importlib
from typing import Any

_CORE_EXPORTS = {"_build_client", "_generate_content_with_fallback", "_parse_structured"}


def __getattr__(name: str) -> Any:
    if name in _CORE_EXPORTS:
        mod = importlib.import_module("src.gemini.core")
    else:
        mod = importlib.import_module("src.gemini.korean")
    val = getattr(mod, name)
    globals()[name] = val
    return val
