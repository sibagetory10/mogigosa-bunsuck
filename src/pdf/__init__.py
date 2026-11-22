"""맞춤형 시험지 · 국어/영어 분석지 · 영단어장 PDF 생성 패키지 (PEP 562 lazy exports)."""

from __future__ import annotations

import importlib
from typing import Any

_SUBMODULE_EXPORTS: dict[str, tuple[str, ...]] = {
    "src.pdf.styles": (
        "EXAM_CSS",
        "_KOREAN_ANALYSIS_CSS",
        "_VOCAB_PDF_CSS",
    ),
    "src.pdf.common": (
        "_CIRCLED_NUMS",
        "_build_korean_story_font_bundle",
        "_pick_korean_font_paths",
        "_render_story_pdf",
        "merge_korean_analysis_pdfs",
    ),
    "src.pdf.poetry_renderer": (
        "_highlight_poem_line_html",
        "_measure_rendered_line_chars",
        "_render_poetry_work_png",
        "_render_poetry_work_png_pages",
        "_wrap_text_by_width",
    ),
    "src.pdf.korean_analysis_pdf": (
        "_KOR_PASSAGE_MARKER_RE",
        "_build_korean_passage_html_block",
        "_clean_and_reflow_korean_passage",
        "_format_structure_flow_html",
        "_is_korean_poetry_item",
        "_join_korean_wrapped_line",
        "_strip_korean_passage_markers",
        "generate_korean_passage_analysis_pdf",
        "generate_korean_passage_only_pdf",
    ),
    "src.pdf.english_analysis_pdf": (
        "_build_english_landscape_css",
        "_render_single_english_landscape_page",
        "generate_english_passage_analysis_pdf",
    ),
    "src.pdf.exam_pdf": (
        "CIRCLED_NUMS",
        "_build_choices_html",
        "_build_english_stem_blocks_html",
        "_build_stem_blocks_html",
        "_draw_first_page_banner",
        "_draw_page_footer",
        "_esc_pdf",
        "_extract_short_page_ref",
        "_format_answer_badge",
        "_format_english_source_text",
        "_is_english_item_dict",
        "_measure_html_height",
        "generate_custom_exam_pdf",
    ),
    "src.pdf.vocab_pdf": (
        "_build_spelling_hint",
        "_render_vocab_story_pdf",
        "generate_vocab_wordbook_pdf",
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
