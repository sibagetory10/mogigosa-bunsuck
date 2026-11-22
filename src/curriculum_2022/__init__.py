"""2022 개정 교육과정 기준표 · 성취기준 · 그라운딩 검증 패키지 (PEP 562 lazy exports)."""

from __future__ import annotations

import importlib
from typing import Any

_SUBMODULE_EXPORTS: dict[str, tuple[str, ...]] = {
    "src.curriculum_2022.catalog": (
        "CURRICULUM_CATEGORIES",
        "_CATALOG_CACHE",
        "_DATA_PATH",
        "_MOE_SEED_PATH",
        "_RUNTIME_STANDARDS_PATH",
        "_canon_subject_key",
        "_load_curriculum_2022_catalog_readonly",
        "_load_runtime_standards",
        "_save_runtime_standards",
        "_stat_sig",
        "clear_ncic_standards_for_category",
        "get_catalog_summary_stats",
        "get_official_subjects_for_category",
        "has_ncic_standards_for_category",
        "load_curriculum_2022_catalog",
        "merge_ncic_standards_to_catalog",
        "normalize_subject_2022",
        "save_curriculum_2022_catalog",
    ),
    "src.curriculum_2022.importers": (
        "_MOE_2022_33_REFERER",
        "_MOE_2022_33_ZIP_URL",
        "_NCIC_STD_LINE_RE",
        "_download_moe_zip",
        "_extract_hwp_text",
        "_extract_moe_zip_items",
        "_infer_category_from_std",
        "_parse_curriculum_text_lines",
        "_read_ole2_streams",
        "fetch_and_sync_moe_2022_curriculum",
        "parse_ncic_curriculum_pdf",
    ),
    "src.curriculum_2022.grounding": (
        "_EVERYDAY_TERM_STOPWORDS",
        "_STD_CODE_ONLY_RE",
        "_clean_term_list",
        "_collect_lower_grade_allowed_vocab",
        "_infer_target_subject_from_exam_name",
        "build_tagging_grounding_context",
        "validate_and_normalize_extracted_tag",
        "validate_standards_only",
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
