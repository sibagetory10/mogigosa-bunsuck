"""Gemini API service subpackage organized by educational domain."""

from __future__ import annotations

import importlib
from typing import Any

_SUBMODULE_EXPORTS: dict[str, tuple[str, ...]] = {
    "src.gemini.usage": (
        "_ACTIVE_USAGE_RECORDER",
        "_ANSWER_PREFIX_RE",
        "_CHOICE_NUM_ONLY_RE",
        "_CHOICE_NUM_PREFIX_RE",
        "_CIRCLED_DIGIT_TO_NUM",
        "ForbiddenTermRemainingError",
        "T",
        "TokenUsageRecord",
        "_emit_usage_from_response",
        "_normalize_extracted_answer",
        "bind_usage_tracker",
        "calculate_token_cost_usd",
        "extract_usage_from_response",
        "has_active_usage_tracker",
        "submit_with_usage_context",
    ),
    "src.gemini.core": (
        "DEFAULT_API_TIMEOUT_SECONDS",
        "FALLBACK_MODEL_ORDER",
        "MAX_VISION_IMAGE_WIDTH",
        "OVERLOAD_RETRY_DELAYS",
        "RATE_LIMIT_FALLBACK_ORDER",
        "ScannedBox",
        "ScannedPageBoxes",
        "_apply_lang_standard_validation",
        "_build_400_recovery_configs",
        "_build_client",
        "_build_lang_ncic_grounding_section",
        "_call_with_retry",
        "_clone_config_for_model",
        "_generate_content_with_fallback",
        "_generate_structured_with_fallback",
        "_is_fatal_config_or_auth_error",
        "_is_flash_lite_model",
        "_is_model_not_found_error",
        "_is_rate_limit_error",
        "_is_server_overload_error",
        "_is_timeout_error",
        "_parse_structured",
        "_sanitize_corrupted_json_numbers",
        "_with_thinking_config",
        "build_gemini_client",
        "build_model_fallback_chain",
        "optimize_image_for_vision",
        "prepare_pdf_part",
    ),
    "src.gemini.exam_tagging": (
        "MAX_PAIRING_VISION_IMAGES",
        "_append_pairing_vision_parts",
        "_format_pairing_item_summary",
        "analyze_exam_pdf_batch_with_gemini",
        "analyze_question_with_gemini",
        "classify_mixed_exam_pdfs_with_ai",
        "detect_scanned_question_boxes",
        "extract_exam_header_metadata",
        "match_problem_and_answer_pdfs_with_ai",
    ),
    "src.gemini.concept_book": (
        "_coalesce_concept_ranges",
        "_extract_concept_toc_with_ai",
        "_has_rich_extracted_text",
        "_is_same_unit_hint",
        "_merge_concept_units",
        "_run_chunk_with_backoff",
        "detect_concept_pages_with_ai",
        "extract_concept_book_cover_metadata",
        "extract_concept_book_units",
        "extract_concept_whitelist",
        "match_questions_to_whitelist",
    ),
    "src.gemini.rewrite": (
        "chat_followup_with_pdf",
        "generate_pdf_grounded_explanation",
        "rewrite_question_for_pdf",
    ),
    "src.gemini.english_validators": (
        "_AI_META_MONOLOGUE_RE",
        "_COMMON_ENG_VERBS",
        "_DANGLING_TO_END_RE",
        "_EXAM_STRUCTURE_MARKER_RE",
        "_FOREIGN_SCRIPT_RE",
        "_INCOMPLETE_BLANK_RE",
        "_NUM_TO_CIRCLED_SYM",
        "_ORDER_SEQ_RE",
        "_REPEATABLE_PREPS",
        "_SENT_ABBREV_RE",
        "_UNEXPECTED_CJK_HANJA_RE",
        "_assemble_english_slim_stem",
        "_assemble_rewritten_question_from_spec",
        "_build_english_transform_rule",
        "_check_chunked_english_fidelity",
        "_count_english_sentences",
        "_dedup_and_trim_blank_choice_overlap",
        "_detect_foreign_script_contamination",
        "_enforce_english_analysis_token_guard",
        "_ensure_korean_instruction_prefix",
        "_expand_compound_sentences_for_structure",
        "_extract_english_exam_context",
        "_extract_english_footnotes",
        "_extract_english_word_tokens",
        "_extract_given_sentence_from_insertion_stem",
        "_has_exam_structure_markers",
        "_has_incomplete_english_blank",
        "_has_order_scrambled_sentences",
        "_infer_blank_phrase_by_diff",
        "_is_irrelevant_answer_from_source",
        "_is_order_abc_canonical",
        "_locate_span_in_indexed_passage",
        "_needs_english_passage_restoration",
        "_normalize_syntax_bracket_consistency",
        "_prepare_indexed_sentences_for_transform",
        "_preprocess_english_source_passage",
        "_project_chunk_markup_onto_source",
        "_rebuild_insertion_stem_from_source",
        "_reconcile_english_insertion_answer",
        "_reconcile_english_order_answer",
        "_reconcile_english_passage_with_question_text",
        "_reconcile_english_summary_answer",
        "_repair_english_analysis_output",
        "_resolve_english_exam_point_title",
        "_resolve_english_transform_schema_and_rule",
        "_sentence_has_english_verb",
        "_split_english_sentences",
        "_strip_spurious_quotes_against_source",
        "_sync_explanation_answer_number",
        "_trim_blank_span_to_preserve_verb",
        "_validate_english_analysis_output",
        "_validate_english_summary_diagram_output",
        "_validate_english_transformed_output",
    ),
    "src.gemini.english_tagging": (
        "_ENGLISH_PASSAGE_TAGGING_RULES",
        "_NUM_TO_CIRCLED",
        "_extract_local_english_body_and_choices",
        "_infer_english_group_id",
        "_verify_and_fix_english_tagged_passage",
        "analyze_english_exam_pdf_batch_with_gemini",
        "analyze_english_question_with_gemini",
    ),
    "src.gemini.english_transform": (
        "_ENGLISH_DIAGRAM_GUIDE",
        "_RESTORED_PASSAGE_CACHE",
        "_RESTORED_PASSAGE_LOCK",
        "_restore_incomplete_source_passage",
        "chat_followup_with_english",
        "generate_english_explanation",
        "generate_english_passage_analysis",
        "generate_english_transformed_question",
        "generate_english_visual_diagram",
        "prepare_clean_english_source_passage",
    ),
    "src.gemini.korean": (
        "_KOREAN_DIAGRAM_GUIDE",
        "_KOREAN_LIT_DIAGRAM_GUIDE",
        "_is_korean_poetry_genre",
        "_sanitize_korean_analysis_output",
        "_strip_korean_exam_noise_lines",
        "_sync_korean_group_passages",
        "analyze_korean_exam_pdf_batch_with_gemini",
        "analyze_korean_question_with_gemini",
        "fetch_korean_literature_web_brief",
        "generate_korean_passage_analysis",
        "generate_korean_set_reconstruction",
        "generate_korean_visual_diagram",
        "proofread_korean_passage_analysis",
    ),
    "src.gemini.dictionary": (
        "lookup_english_word_with_gemini",
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
