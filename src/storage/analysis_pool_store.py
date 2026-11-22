from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.storage.base import (
    _GLOBAL_CACHE_LOCK,
    _GLOBAL_JSON_CACHE,
    _GLOBAL_STORAGE_STATS_CACHE,
    _atomic_write_bytes,
    _atomic_write_text,
    _file_stat_sig,
)

if TYPE_CHECKING:
    from src.models import EnglishPassageAnalysisOutput, KoreanPassageAnalysisOutput


class AnalysisPoolStoreMixin:
    @staticmethod
    def build_korean_analysis_key(
        passage_text: str,
        group_id: str = "",
        korean_section: str = "",
    ) -> str:
        """국어 지문 원문·세트 번호·영역 조합으로 분석지 캐시 키를 생성한다."""
        norm_p = re.sub(r"\s+", " ", (passage_text or "").strip())
        raw_key = f"{norm_p}||{(group_id or '').strip()}||{(korean_section or '').strip()}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:24]

    def _load_korean_analysis_pool_unlocked(self) -> dict[str, dict[str, Any]]:
        if not self.korean_analysis_pool_path.exists():
            return {}
        sig = _file_stat_sig(self.korean_analysis_pool_path)
        if sig is None or sig[1] == 0:
            return {}
        cache_key = f"kor_pool::{self.korean_analysis_pool_path.resolve()}"
        with _GLOBAL_CACHE_LOCK:
            cached = _GLOBAL_JSON_CACHE.get(cache_key)
            if cached is not None and cached[0] == sig:
                return {k: dict(v) for k, v in cached[1].items()}
        try:
            data = json.loads(self.korean_analysis_pool_path.read_text(encoding="utf-8"))
            clean = {
                str(k): dict(v)
                for k, v in (data if isinstance(data, dict) else {}).items()
                if isinstance(v, dict)
            }
            with _GLOBAL_CACHE_LOCK:
                _GLOBAL_JSON_CACHE[cache_key] = (sig, clean)
            return {k: dict(v) for k, v in clean.items()}
        except Exception:
            return {}

    def _save_korean_analysis_pool_unlocked(self, pool: dict[str, dict[str, Any]]) -> None:
        raw = json.dumps(pool, ensure_ascii=False, indent=2)
        _atomic_write_text(self.korean_analysis_pool_path, raw, encoding="utf-8")
        new_sig = _file_stat_sig(self.korean_analysis_pool_path)
        if new_sig is not None:
            with _GLOBAL_CACHE_LOCK:
                _GLOBAL_JSON_CACHE[f"kor_pool::{self.korean_analysis_pool_path.resolve()}"] = (
                    new_sig,
                    {str(k): dict(v) for k, v in pool.items() if isinstance(v, dict)},
                )

    def load_korean_analysis_pool(self) -> dict[str, dict[str, Any]]:
        """저장된 국어 지문 분석지 풀(Pool) 전체를 안전하게 조회한다."""
        with self._lock:
            pool = self._load_korean_analysis_pool_unlocked()
            return {k: dict(v) for k, v in pool.items() if isinstance(v, dict)}

    def get_cached_korean_analysis(self, cache_key: str) -> KoreanPassageAnalysisOutput | None:
        """국어 분석지 풀에서 캐시 히트 시 KoreanPassageAnalysisOutput을 즉시 반환한다."""
        if not cache_key:
            return None
        with self._lock:
            pool = self._load_korean_analysis_pool_unlocked()
            entry = pool.get(cache_key)
            if not isinstance(entry, dict):
                return None
            raw_analysis = entry.get("analysis")
            if not isinstance(raw_analysis, dict):
                return None
            from src.models.korean import KoreanPassageAnalysisOutput

            try:
                return KoreanPassageAnalysisOutput.model_validate(raw_analysis)
            except Exception:
                return None

    def save_cached_korean_analysis(
        self,
        cache_key: str,
        analysis: KoreanPassageAnalysisOutput,
        *,
        exam_name: str = "",
        group_id: str = "",
        korean_section: str = "",
        korean_subgenre: str = "",
    ) -> None:
        """AI로 신규 생성한 국어 지문 분석지를 서버 풀에 영구 저장한다."""
        if not cache_key or analysis is None:
            return
        with self._lock:
            pool = self._load_korean_analysis_pool_unlocked()
            pool[cache_key] = {
                "exam_name": (exam_name or "").strip(),
                "group_id": (group_id or "").strip(),
                "section": (korean_section or "").strip(),
                "subgenre": (korean_subgenre or "").strip(),
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "analysis": analysis.model_dump(),
            }
            self._save_korean_analysis_pool_unlocked(pool)

    @staticmethod
    def _safe_kor_cache_key(cache_key: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_-]+", "_", (cache_key or "").strip()) or "empty"

    def get_korean_analysis_pdf_path(self, cache_key: str) -> Path:
        """지문별 국어 분석지(원문 제외) PDF 파일 경로를 반환한다."""
        return self.korean_analysis_pdfs_root / f"{self._safe_kor_cache_key(cache_key)}_analysis_v5.pdf"

    def get_korean_passage_pdf_path(self, cache_key: str) -> Path:
        """지문별 국어 '원문 지문 전체' 전용 PDF 파일 경로를 반환한다."""
        return self.korean_analysis_pdfs_root / f"{self._safe_kor_cache_key(cache_key)}_passage_v5.pdf"
    def _write_kor_pdf_bytes(self, path: Path, pdf_bytes: bytes) -> None:
        with self._lock:
            self.korean_analysis_pdfs_root.mkdir(parents=True, exist_ok=True)
            _atomic_write_bytes(path, pdf_bytes)
            with _GLOBAL_CACHE_LOCK:
                _GLOBAL_STORAGE_STATS_CACHE.clear()

    def _unlink_kor_pdfs_for_key(self, cache_key: str) -> bool:
        safe_key = self._safe_kor_cache_key(cache_key)
        removed_any = False
        for pdf_p in (
            self.get_korean_analysis_pdf_path(cache_key),
            self.get_korean_passage_pdf_path(cache_key),
            self.korean_analysis_pdfs_root / f"{safe_key}_v4.pdf",
            self.korean_analysis_pdfs_root / f"{safe_key}_v3.pdf",
            self.korean_analysis_pdfs_root / f"{safe_key}_v2.pdf",
            self.korean_analysis_pdfs_root / f"{safe_key}.pdf",
        ):
            if pdf_p.exists():
                pdf_p.unlink(missing_ok=True)
                removed_any = True
        return removed_any

    def has_korean_analysis_pdf(self, cache_key: str) -> bool:
        """지문별 국어 분석지(원문 제외) PDF가 서버 DB에 저장되어 있는지 반환한다."""
        return bool(cache_key) and self._has_nonempty_file(self.get_korean_analysis_pdf_path(cache_key))

    def has_korean_passage_pdf(self, cache_key: str) -> bool:
        """지문별 국어 원문 전용 PDF가 서버 DB에 저장되어 있는지 반환한다."""
        return bool(cache_key) and self._has_nonempty_file(self.get_korean_passage_pdf_path(cache_key))

    def get_korean_analysis_pdf_bytes(self, cache_key: str) -> bytes | None:
        """서버 DB에 저장된 지문별 국어 분석지(원문 제외) PDF 바이트를 반환한다."""
        if not cache_key:
            return None
        return self._read_nonempty_bytes(self.get_korean_analysis_pdf_path(cache_key))

    def get_korean_passage_pdf_bytes(self, cache_key: str) -> bytes | None:
        """서버 DB에 저장된 지문별 국어 원문 전용 PDF 바이트를 반환한다."""
        if not cache_key:
            return None
        return self._read_nonempty_bytes(self.get_korean_passage_pdf_path(cache_key))

    def save_korean_analysis_pdf_bytes(self, cache_key: str, pdf_bytes: bytes) -> None:
        """지문별 국어 분석지(원문 제외) PDF 바이트를 서버 DB에 영구 저장한다."""
        if not cache_key or not pdf_bytes:
            return
        with self._lock:
            self._write_kor_pdf_bytes(self.get_korean_analysis_pdf_path(cache_key), pdf_bytes)
            safe_key = self._safe_kor_cache_key(cache_key)
            for old_name in (f"{safe_key}.pdf", f"{safe_key}_v2.pdf", f"{safe_key}_v3.pdf", f"{safe_key}_v4.pdf"):
                legacy_p = self.korean_analysis_pdfs_root / old_name
                if legacy_p.exists():
                    legacy_p.unlink(missing_ok=True)

    def save_korean_passage_pdf_bytes(self, cache_key: str, pdf_bytes: bytes) -> None:
        """지문별 국어 원문 전용 PDF 바이트를 서버 DB에 영구 저장한다."""
        if not cache_key or not pdf_bytes:
            return
        self._write_kor_pdf_bytes(self.get_korean_passage_pdf_path(cache_key), pdf_bytes)

    def delete_korean_analysis_entry(self, cache_key: str) -> int:
        """국어 지문 분석지 캐시 및 저장된 지문 분석/원문 PDF 1건을 삭제한다."""
        if not cache_key:
            return 0
        with self._lock:
            pool = self._load_korean_analysis_pool_unlocked()
            removed = 0
            if cache_key in pool:
                pool.pop(cache_key, None)
                self._save_korean_analysis_pool_unlocked(pool)
                removed = 1
            if self._unlink_kor_pdfs_for_key(cache_key):
                removed = 1
            return removed

    def clear_korean_analysis_pool(self) -> int:
        """국어 지문 분석지 캐시 및 저장된 지문별 PDF 전체를 비운다."""
        with self._lock:
            pool = self._load_korean_analysis_pool_unlocked()
            removed = len(pool)
            self._save_korean_analysis_pool_unlocked({})
            if self.korean_analysis_pdfs_root.exists():
                for child in self.korean_analysis_pdfs_root.iterdir():
                    if child.is_file():
                        child.unlink(missing_ok=True)
            with _GLOBAL_CACHE_LOCK:
                _GLOBAL_STORAGE_STATS_CACHE.clear()
            return removed

    def delete_korean_analyses_by_exam(self, exam_name: str) -> int:
        """특정 회차가 삭제될 때 해당 회차의 국어 분석지 캐시 및 지문별 PDF도 함께 삭제한다."""
        target_exam = (exam_name or "").strip()
        if not target_exam:
            return 0
        with self._lock:
            pool = self._load_korean_analysis_pool_unlocked()
            to_del = [
                k
                for k, v in pool.items()
                if isinstance(v, dict) and str(v.get("exam_name") or "").strip() == target_exam
            ]
            if not to_del:
                return 0
            for k in to_del:
                pool.pop(k, None)
                self._unlink_kor_pdfs_for_key(k)
            self._save_korean_analysis_pool_unlocked(pool)
            return len(to_del)

    def rename_korean_analyses_exam(self, old_exam_name: str, new_exam_name: str) -> int:
        """회차명 변경 시 국어 분석지 캐시의 exam_name 메타데이터를 동기화한다."""
        old_n = (old_exam_name or "").strip()
        new_n = re.sub(r"\s+", " ", (new_exam_name or "").strip())
        if not old_n or not new_n or old_n == new_n:
            return 0
        with self._lock:
            pool = self._load_korean_analysis_pool_unlocked()
            changed = 0
            for k, v in pool.items():
                if isinstance(v, dict) and str(v.get("exam_name") or "").strip() == old_n:
                    v["exam_name"] = new_n
                    changed += 1
            if changed > 0:
                self._save_korean_analysis_pool_unlocked(pool)
            return changed

    # -----------------------------------------------------------------------
    # 영어 정밀 지문 분석지 캐시 풀(Pool) 영구 저장 / 조회 / 관리
    # -----------------------------------------------------------------------
    @staticmethod
    def build_english_analysis_key(
        passage_text: str,
        q_label: str = "",
    ) -> str:
        """영어 지문 원문 정규화 텍스트로 분석지 캐시 키를 생성한다."""
        norm_p = re.sub(r"\s+", " ", (passage_text or "").strip())
        return hashlib.sha256(f"eng_an||{norm_p}".encode("utf-8")).hexdigest()[:24]

    def _load_english_analysis_pool_unlocked(self) -> dict[str, dict[str, Any]]:
        if not self.english_analysis_pool_path.exists():
            return {}
        sig = _file_stat_sig(self.english_analysis_pool_path)
        if sig is None or sig[1] == 0:
            return {}
        cache_key = f"eng_an_pool::{self.english_analysis_pool_path.resolve()}"
        with _GLOBAL_CACHE_LOCK:
            cached = _GLOBAL_JSON_CACHE.get(cache_key)
            if cached is not None and cached[0] == sig:
                return {k: dict(v) for k, v in cached[1].items()}
        try:
            data = json.loads(self.english_analysis_pool_path.read_text(encoding="utf-8"))
            clean = {
                str(k): dict(v)
                for k, v in (data if isinstance(data, dict) else {}).items()
                if isinstance(v, dict)
            }
            with _GLOBAL_CACHE_LOCK:
                _GLOBAL_JSON_CACHE[cache_key] = (sig, clean)
            return {k: dict(v) for k, v in clean.items()}
        except Exception:
            return {}

    def _save_english_analysis_pool_unlocked(self, pool: dict[str, dict[str, Any]]) -> None:
        raw = json.dumps(pool, ensure_ascii=False, indent=2)
        _atomic_write_text(self.english_analysis_pool_path, raw, encoding="utf-8")
        new_sig = _file_stat_sig(self.english_analysis_pool_path)
        if new_sig is not None:
            with _GLOBAL_CACHE_LOCK:
                _GLOBAL_JSON_CACHE[f"eng_an_pool::{self.english_analysis_pool_path.resolve()}"] = (
                    new_sig,
                    {str(k): dict(v) for k, v in pool.items() if isinstance(v, dict)},
                )

    def load_english_analysis_pool(self) -> dict[str, dict[str, Any]]:
        """저장된 영어 지문 분석지 풀(Pool) 전체를 안전하게 조회한다."""
        with self._lock:
            pool = self._load_english_analysis_pool_unlocked()
            return {k: dict(v) for k, v in pool.items() if isinstance(v, dict)}

    def get_cached_english_analysis(self, cache_key: str) -> EnglishPassageAnalysisOutput | None:
        """영어 분석지 풀에서 캐시 히트 시 EnglishPassageAnalysisOutput을 즉시 반환한다."""
        if not cache_key:
            return None
        with self._lock:
            pool = self._load_english_analysis_pool_unlocked()
            entry = pool.get(cache_key)
            if not isinstance(entry, dict):
                return None
            raw_analysis = entry.get("analysis")
            if not isinstance(raw_analysis, dict):
                return None
            from src.models.english import EnglishPassageAnalysisOutput

            try:
                return EnglishPassageAnalysisOutput.model_validate(raw_analysis)
            except Exception:
                return None

    def save_cached_english_analysis(
        self,
        cache_key: str,
        analysis: EnglishPassageAnalysisOutput,
        *,
        exam_name: str = "",
        q_label: str = "",
        question_type: str = "",
    ) -> None:
        """AI로 신규 생성한 영어 지문 분석지를 서버 풀에 영구 저장한다."""
        if not cache_key or analysis is None:
            return
        with self._lock:
            pool = self._load_english_analysis_pool_unlocked()
            pool[cache_key] = {
                "exam_name": (exam_name or "").strip(),
                "q_label": (q_label or "").strip(),
                "question_type": (question_type or "").strip(),
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "analysis": analysis.model_dump(),
            }
            self._save_english_analysis_pool_unlocked(pool)

    def get_english_analysis_pdf_path(self, cache_key: str) -> Path:
        """지문별 영어 분석지 PDF 파일 경로를 반환한다 (v6: PDF 문장 잘림 zero 검증·유형별 배지 격리·하이픈/괄호 완전 보정)."""
        return self.english_analysis_pdfs_root / f"{self._safe_kor_cache_key(cache_key)}_eng_analysis_v6.pdf"

    def _unlink_eng_pdfs_for_key(self, cache_key: str) -> bool:
        safe_key = self._safe_kor_cache_key(cache_key)
        removed_any = False
        for pdf_p in (
            self.get_english_analysis_pdf_path(cache_key),
            self.english_analysis_pdfs_root / f"{safe_key}_eng_analysis_v5.pdf",
            self.english_analysis_pdfs_root / f"{safe_key}_eng_analysis_v4.pdf",
            self.english_analysis_pdfs_root / f"{safe_key}_eng_analysis_v3.pdf",
            self.english_analysis_pdfs_root / f"{safe_key}_eng_analysis_v2.pdf",
            self.english_analysis_pdfs_root / f"{safe_key}_eng_analysis_v1.pdf",
        ):
            if pdf_p.exists():
                pdf_p.unlink(missing_ok=True)
                removed_any = True
        return removed_any

    def has_english_analysis_pdf(self, cache_key: str) -> bool:
        """지문별 영어 분석지 PDF가 서버 DB에 저장되어 있는지 반환한다."""
        return bool(cache_key) and self._has_nonempty_file(self.get_english_analysis_pdf_path(cache_key))

    def get_english_analysis_pdf_bytes(self, cache_key: str) -> bytes | None:
        """서버 DB에 저장된 지문별 영어 분석지 PDF 바이트를 반환한다."""
        if not cache_key:
            return None
        return self._read_nonempty_bytes(self.get_english_analysis_pdf_path(cache_key))

    def save_english_analysis_pdf_bytes(self, cache_key: str, pdf_bytes: bytes) -> None:
        """지문별 영어 분석지 PDF 바이트를 서버 DB에 영구 저장한다."""
        if not cache_key or not pdf_bytes:
            return
        with self._lock:
            self.english_analysis_pdfs_root.mkdir(parents=True, exist_ok=True)
            _atomic_write_bytes(self.get_english_analysis_pdf_path(cache_key), pdf_bytes)
            safe_key = self._safe_kor_cache_key(cache_key)
            for legacy_ver in ("v1", "v2", "v3", "v4", "v5"):
                legacy_p = self.english_analysis_pdfs_root / f"{safe_key}_eng_analysis_{legacy_ver}.pdf"
                if legacy_p.exists():
                    legacy_p.unlink(missing_ok=True)
            with _GLOBAL_CACHE_LOCK:
                _GLOBAL_STORAGE_STATS_CACHE.clear()

    def delete_english_analysis_entry(self, cache_key: str) -> int:
        """영어 지문 분석지 캐시 및 저장된 지문 PDF 1건을 삭제한다."""
        if not cache_key:
            return 0
        with self._lock:
            pool = self._load_english_analysis_pool_unlocked()
            removed = 0
            if cache_key in pool:
                pool.pop(cache_key, None)
                self._save_english_analysis_pool_unlocked(pool)
                removed = 1
            if self._unlink_eng_pdfs_for_key(cache_key):
                removed = 1
            with _GLOBAL_CACHE_LOCK:
                _GLOBAL_STORAGE_STATS_CACHE.clear()
            return removed

    def clear_english_analysis_pool(self) -> int:
        """영어 지문 분석지 캐시 및 저장된 지문별 PDF 전체를 비운다."""
        with self._lock:
            pool = self._load_english_analysis_pool_unlocked()
            removed = len(pool)
            self._save_english_analysis_pool_unlocked({})
            if self.english_analysis_pdfs_root.exists():
                for child in self.english_analysis_pdfs_root.iterdir():
                    if child.is_file():
                        child.unlink(missing_ok=True)
            with _GLOBAL_CACHE_LOCK:
                _GLOBAL_STORAGE_STATS_CACHE.clear()
            return removed

    def delete_english_analyses_by_exam(self, exam_name: str) -> int:
        """특정 회차가 삭제될 때 해당 회차의 영어 분석지 캐시 및 지문별 PDF도 함께 삭제한다."""
        target_exam = (exam_name or "").strip()
        if not target_exam:
            return 0
        with self._lock:
            pool = self._load_english_analysis_pool_unlocked()
            to_del = [
                k
                for k, v in pool.items()
                if isinstance(v, dict) and str(v.get("exam_name") or "").strip() == target_exam
            ]
            if not to_del:
                return 0
            for k in to_del:
                pool.pop(k, None)
                self._unlink_eng_pdfs_for_key(k)
            self._save_english_analysis_pool_unlocked(pool)
            return len(to_del)

    def rename_english_analyses_exam(self, old_exam_name: str, new_exam_name: str) -> int:
        """회차명 변경 시 영어 분석지 캐시의 exam_name 메타데이터를 동기화한다."""
        old_n = (old_exam_name or "").strip()
        new_n = re.sub(r"\s+", " ", (new_exam_name or "").strip())
        if not old_n or not new_n or old_n == new_n:
            return 0
        with self._lock:
            pool = self._load_english_analysis_pool_unlocked()
            changed = 0
            for k, v in pool.items():
                if isinstance(v, dict) and str(v.get("exam_name") or "").strip() == old_n:
                    v["exam_name"] = new_n
                    changed += 1
            if changed > 0:
                self._save_english_analysis_pool_unlocked(pool)
            return changed
