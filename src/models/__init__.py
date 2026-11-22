"""Pydantic v2 schemas and domain helpers (PEP 562 lazy exports)."""

from __future__ import annotations

import importlib
from typing import Any

_SUBMODULE_EXPORTS: dict[str, tuple[str, ...]] = {
    "src.models.user": (
        "LEGACY_TIER_MAP",
        "TierType",
        "UserRecord",
    ),
    "src.models.vocab": (
        "EnglishDictEntry",
        "EnglishDictSense",
        "UserVocabItem",
        "normalize_english_dict_key",
    ),
    "src.models.question": (
        "BatchExamExtractionOutput",
        "BatchFilePairingOutput",
        "BatchQuestionTagItem",
        "ENGLISH_DEFAULT_EXCLUDED_Q_NUMS",
        "ENGLISH_SHARED_PASSAGE_SETS",
        "ENGLISH_TRANSFORM_TYPES",
        "ExamHeaderMetadata",
        "MixedPdfClassificationOutput",
        "MixedPdfRoleItem",
        "ProblemAnswerPairItem",
        "QuestionExtractionOutput",
        "QuestionRecord",
        "SubjectCategory",
        "_coerce_grade_int",
        "infer_subject_category",
        "normalize_english_group_id",
        "normalize_exam_title_year",
        "normalize_exam_year",
        "normalize_korean_group_id",
    ),
    "src.models.concept": (
        "AutoConceptPageDetectionOutput",
        "AutoConceptRangeItem",
        "BatchMatchResult",
        "ConceptBookCoverMetadata",
        "ConceptBookExtractionOutput",
        "ConceptBookRecord",
        "ConceptUnit",
        "ConceptWhitelist",
        "QuestionMatchJudgment",
        "RewrittenQuestion",
        "TocExtractionOutput",
        "TocUnitItem",
        "build_whitelist_from_concept_units",
        "resolve_question_concept_reference",
    ),
    "src.models.korean": (
        "BatchKoreanExamExtractionOutput",
        "KoreanCharacterEdge",
        "KoreanDiagramNode",
        "KoreanEmotionPoint",
        "KoreanGraphCurve",
        "KoreanGraphSpec",
        "KoreanMarkerVocabItem",
        "KoreanMatrixRow",
        "KoreanParagraphSummary",
        "KoreanPassageAnalysisOutput",
        "KoreanPoetryLineNote",
        "KoreanPoetryMarginBracket",
        "KoreanPoetryMark",
        "KoreanPoetryWorkAnnotation",
        "KoreanQuestionExtractionItem",
        "KoreanSetQuestionOutput",
        "KoreanSetReconstructionOutput",
        "KoreanSingleExtractionOutput",
        "KoreanVisualDiagram",
    ),
    "src.models.english": (
        "BatchEnglishExamExtractionOutput",
        "CURRENT_ENGLISH_ANALYSIS_VERSION",
        "EnglishPassageAnalysisOutput",
        "EnglishPassageSummaryDiagramOutput",
        "EnglishQuestionExtractionItem",
        "EnglishSentenceAnalysisItem",
        "EnglishSingleExtractionOutput",
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
