"""로컬 영구 디스크(./exam_db, ./concept_db) 전용 저장소 패키지."""

from __future__ import annotations

from src.storage.analysis_pool_store import AnalysisPoolStoreMixin
from src.storage.base import (
    _GLOBAL_CACHE_LOCK,
    _GLOBAL_JSON_CACHE,
    _GLOBAL_STORAGE_STATS_CACHE,
    STORAGE_STATS_TTL_SECONDS,
    StorageBase,
    _atomic_write_bytes,
    _atomic_write_text,
    _dir_json_stat_sig,
    _file_stat_sig,
    _safe_segment,
)
from src.storage.english_pool_store import EnglishPoolStoreMixin
from src.storage.exam_store import ExamStoreMixin
from src.storage.user_exams_store import UserExamsStoreMixin
from src.storage.vocab_store import VocabStoreMixin


class StorageManager(
    ExamStoreMixin,
    UserExamsStoreMixin,
    EnglishPoolStoreMixin,
    AnalysisPoolStoreMixin,
    VocabStoreMixin,
    StorageBase,
):
    """오라클 서버 로컬 영구 디스크(./exam_db, ./concept_db) 전용 저장소 관리자."""


__all__ = [
    "STORAGE_STATS_TTL_SECONDS",
    "StorageManager",
    "_GLOBAL_CACHE_LOCK",
    "_GLOBAL_JSON_CACHE",
    "_GLOBAL_STORAGE_STATS_CACHE",
    "_atomic_write_bytes",
    "_atomic_write_text",
    "_dir_json_stat_sig",
    "_file_stat_sig",
    "_safe_segment",
]
