"""Lazy re-export facade for English tagging and transformation services."""

from __future__ import annotations

import importlib
from typing import Any

_TAGGING_EXPORTS = {
    "_ENGLISH_PASSAGE_TAGGING_RULES",
    "_NUM_TO_CIRCLED",
    "_extract_local_english_body_and_choices",
    "_infer_english_group_id",
    "_verify_and_fix_english_tagged_passage",
    "analyze_english_exam_pdf_batch_with_gemini",
    "analyze_english_question_with_gemini",
}

_TRANSFORM_EXPORTS = {
    "_ENGLISH_DIAGRAM_GUIDE",
    "_RESTORED_PASSAGE_CACHE",
    "_RESTORED_PASSAGE_LOCK",
    "_restore_incomplete_source_passage",
    "chat_followup_with_english",
    "generate_english_explanation",
    "generate_english_passage_analysis",
    "generate_english_transformed_question",
    "generate_english_visual_diagram",
}

__all__ = sorted(_TAGGING_EXPORTS | _TRANSFORM_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _TAGGING_EXPORTS:
        mod = importlib.import_module("src.gemini.english_tagging")
    elif name in _TRANSFORM_EXPORTS:
        mod = importlib.import_module("src.gemini.english_transform")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    val = getattr(mod, name)
    globals()[name] = val
    return val


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
