"""Backwards-compatible lazy proxy facade for Gemini API services."""

from __future__ import annotations

from typing import Any

import src.gemini as _gemini_pkg

__all__ = list(_gemini_pkg.__all__)


def __getattr__(name: str) -> Any:
    return getattr(_gemini_pkg, name)


def __dir__() -> list[str]:
    return _gemini_pkg.__dir__()
