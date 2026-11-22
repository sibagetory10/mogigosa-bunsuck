"""Pillow 기반 국어·영어 지문 시각 구조도 PNG 렌더러 패키지 (PEP 562 lazy exports)."""

from __future__ import annotations

import importlib
from typing import Any

_SUBMODULE_EXPORTS: dict[str, tuple[str, ...]] = {
    "src.diagram_renderer.primitives": (
        "_BOLD_FONT_CANDIDATES",
        "_COLOR_CYCLE",
        "_PALETTES",
        "_REGULAR_FONT_CANDIDATES",
        "_draw_arrow",
        "_draw_node_card",
        "_load_font",
        "_measure_card_height",
        "_measure_line_height",
        "_wrap_text",
    ),
    "src.diagram_renderer.academic_charts": (
        "_render_anatomy",
        "_render_decision_tree",
        "_render_fishbone",
        "_render_graph",
        "_render_matrix",
        "_render_pyramid",
        "_render_quadrant",
        "_render_scale",
        "_render_timeline",
        "_render_venn",
    ),
    "src.diagram_renderer.literature_charts": (
        "_render_character_map",
        "_render_emotion_arc",
        "_render_poetic_contrast",
        "_render_space_journey",
    ),
    "src.diagram_renderer.card_layouts": (
        "_render_card_layout",
        "_render_compact_vertical_layout",
        "render_korean_diagram_png",
    ),
}

_SYMBOL_TO_MODULE: dict[str, str] = {
    sym: mod for mod, syms in _SUBMODULE_EXPORTS.items() for sym in syms
}

__all__ = sorted(_SYMBOL_TO_MODULE.keys())


def __getattr__(name: str) -> Any:
    mod_name = _SYMBOL_TO_MODULE.get(name)
    if mod_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(mod_name)
    val = getattr(mod, name)
    globals()[name] = val
    return val


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
