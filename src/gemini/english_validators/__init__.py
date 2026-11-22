"""영어 수능형 변형 및 지문 분석지 결정론적 검증·보정 패키지 (PEP 562 lazy exports)."""

from __future__ import annotations

import importlib
from typing import Any

_SUBMODULE_EXPORTS: dict[str, tuple[str, ...]] = {
    "src.gemini.english_validators.transform_helpers": (
        "_AI_META_MONOLOGUE_RE",
        "_COMMON_ENG_VERBS",
        "_DANGLING_TO_END_RE",
        "_DETERMINER_SET",
        "_NUM_TO_CIRCLED_SYM",
        "_ORDER_SEQ_RE",
        "_PDF_ENDASH_ALNUM_RE",
        "_PDF_HYPHEN_CHARS_RE",
        "_REPEATABLE_PREPS",
        "_SENT_ABBREV_RE",
        "_assemble_rewritten_question_from_spec",
        "_build_english_transform_rule",
        "_count_english_sentences",
        "_dedup_and_trim_blank_choice_overlap",
        "_ensure_korean_instruction_prefix",
        "_expand_compound_sentences_for_structure",
        "_extract_given_sentence_from_insertion_stem",
        "_has_order_scrambled_sentences",
        "_is_irrelevant_answer_from_source",
        "_is_order_abc_canonical",
        "_locate_span_in_indexed_passage",
        "_normalize_pdf_hyphens",
        "_prepare_indexed_sentences_for_transform",
        "_preprocess_english_source_passage",
        "_rebuild_insertion_stem_from_source",
        "_reconcile_english_insertion_answer",
        "_reconcile_english_order_answer",
        "_reconcile_english_summary_answer",
        "_resolve_english_transform_schema_and_rule",
        "_sentence_has_english_verb",
        "_split_english_sentences",
        "_strip_stray_exam_qnum_noise",
        "_sync_explanation_answer_number",
        "_trim_blank_span_to_preserve_verb",
    ),
    "src.gemini.english_validators.transform_validate": (
        "_DANGLING_VERB_END_RE",
        "_EXAM_STRUCTURE_MARKER_RE",
        "_INCOMPLETE_BLANK_RE",
        "_NON_VERB_S_WORDS",
        "_RESTORE_TARGET_TYPE_KEYWORDS",
        "_assemble_english_slim_stem",
        "_has_exam_structure_markers",
        "_has_incomplete_english_blank",
        "_needs_english_passage_restoration",
        "_validate_english_transformed_output",
    ),
    "src.gemini.english_validators.exam_context": (
        "_CIRCLED_NUM_LIST",
        "_DROPPED_HYPHEN_COMPOUNDS",
        "_ENG_FOOTNOTE_EXTRACT_RE",
        "_FOOTNOTE_TAIL_CUT_RE",
        "_HYPHEN_SUFFIX_LEVEL_RE",
        "_NUM_TO_CIRCLED_MAP",
        "_extract_english_exam_context",
        "_extract_english_footnotes",
        "_extract_english_word_tokens",
        "_infer_blank_phrase_by_diff",
        "_is_clean_english_reconstruction",
        "_preserve_hyphens_from_reference",
        "_reconcile_english_passage_with_question_text",
        "_reconcile_words_against_question_body",
        "_repair_pdf_dropped_hyphens",
        "_resolve_english_exam_point_title",
        "_split_available_circled_choices",
        "_split_numbered_five_markers",
        "_strip_english_question_header_and_footnotes",
    ),
    "src.gemini.english_validators.analysis_validate": (
        "_DISALLOWED_CURLY_RE",
        "_FOREIGN_SCRIPT_RE",
        "_FULLWIDTH_BRACKET_TRANS",
        "_PARALLEL_SUBJ_BRACKET_RE",
        "_UNEXPECTED_CJK_HANJA_RE",
        "_WEAK_TOPIC_SENT_RE",
        "_WH_PAREN_CLAUSE_RE",
        "_balance_added_markup_chars",
        "_check_chunked_english_fidelity",
        "_collect_all_analysis_korean_blobs",
        "_detect_foreign_script_contamination",
        "_enforce_english_analysis_token_guard",
        "_has_balanced_syntax_brackets",
        "_normalize_syntax_bracket_consistency",
        "_project_chunk_markup_onto_source",
        "_repair_english_analysis_output",
        "_sanitize_disallowed_syntax_braces",
        "_strip_spurious_quotes_against_source",
        "_validate_english_analysis_output",
        "_validate_english_summary_diagram_output",
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
