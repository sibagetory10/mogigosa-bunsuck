"""PyMuPDF 기반 시험지·개념서 PDF 파싱 및 크롭 패키지 (PEP 562 lazy exports)."""

from __future__ import annotations

import importlib
from typing import Any

_SUBMODULE_EXPORTS: dict[str, tuple[str, ...]] = {
    "src.pdf_parser.common": (
        "CroppedQuestion",
        "QUESTION_START_RE",
        "SHARED_PASSAGE_RE",
        "crop_normalized_box",
        "extract_first_page_for_metadata",
        "extract_pdf_preview_for_pairing",
        "get_page_text_lengths",
        "get_pdf_page_count",
        "render_pdf_page_png",
    ),
    "src.pdf_parser.metadata_and_pairing": (
        "_ANS_LINE_MARKER_RE",
        "_ANS_ONLY_HINT_RE",
        "_ANS_RANGE_MARKER_RE",
        "_FILENAME_ROLE_SUFFIX_RE",
        "_FILENAME_ROLE_TOKENS",
        "_GRADE_TOKEN_RE",
        "_LOCAL_SUBJECT_CANDIDATES",
        "_MONTH_TOKEN_RE",
        "_PROB_ONLY_HINT_RE",
        "_YEAR_TOKEN_RE",
        "_score_prob_ans_pair",
        "_tokenize_exam_filename",
        "_trim_answer_snippet_preserving_explanation",
        "extract_answer_snippets_from_pdf",
        "extract_exam_metadata_fast",
        "is_likely_answer_filename",
        "normalize_exam_filename_stem",
        "pair_problem_and_answer_files",
        "parse_exam_header_locally",
    ),
    "src.pdf_parser.concept_slicer": (
        "_ANSWER_HEADER_RE",
        "_EXERCISE_HEADER_RE",
        "_INLINE_NOISE_LINE_RE",
        "_PAGE_RANGE_TOKEN_RE",
        "_PROB_NUMBERED_LINE_RE",
        "_TOC_DOT_LINE_RE",
        "_TOC_HEADER_RE",
        "_clean_and_classify_concept_page",
        "_stamp_actual_pdf_page_badge",
        "extract_concept_pages_by_ranges",
        "extract_pdf_page_catalog",
        "extract_pdf_slice",
        "extract_specific_pages_slice",
        "parse_page_ranges_spec",
    ),
    "src.pdf_parser.exam_cropper": (
        "_PDF_ENDASH_ALNUM_RE",
        "_PDF_HYPHEN_CHARS_RE",
        "_collect_content_boxes_in_column",
        "_detect_columns",
        "_extract_lines_in_column",
        "_has_vector_hyphen_between",
        "_is_footer_or_header_noise",
        "_normalize_pdf_hyphens",
        "_stitch_vertical_pngs",
        "parse_exam_pdf",
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
