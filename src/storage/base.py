from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

_GLOBAL_CACHE_LOCK = threading.RLock()
_GLOBAL_JSON_CACHE: dict[str, tuple[Any, Any]] = {}
_GLOBAL_STORAGE_STATS_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
STORAGE_STATS_TTL_SECONDS = 60.0


def _file_stat_sig(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
        return (int(st.st_mtime_ns), int(st.st_size))
    except OSError:
        return None


def _dir_json_stat_sig(dir_path: Path, main_file: Path) -> tuple[Any, ...]:
    parts: list[Any] = [str(main_file.resolve()) if main_file.exists() else str(main_file), _file_stat_sig(main_file)]
    if dir_path.exists():
        try:
            for child in sorted(dir_path.glob("*.json")):
                if child.name == main_file.name:
                    continue
                parts.append((child.name, _file_stat_sig(child)))
        except OSError:
            pass
    return tuple(parts)


def _safe_segment(name: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", (name or "").strip()).strip(". ")
    return cleaned or "기타"


def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """임시 파일(.tmp)에 먼저 기록 후 os.replace()로 원자적(Atomic) 교체하여 파일 손상 및 동시성 충돌을 방지한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp_path.write_text(text, encoding=encoding)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """임시 파일(.tmp)에 먼저 기록 후 os.replace()로 원자적(Atomic) 교체하여 파일 손상 및 동시성 충돌을 방지한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp_path.write_bytes(data)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass



class StorageBase:
    """오라클 서버 로컬 영구 디스크(./exam_db, ./concept_db) 전용 저장소 관리자."""

    def __init__(
        self,
        local_root: str | Path = "./exam_db",
        concept_root: str | Path = "./concept_db",
    ) -> None:
        self._lock = threading.RLock()

        self.local_root = Path(local_root)
        self.pdfs_dir = self.local_root / "pdfs"
        self.crops_dir = self.local_root / "crops"
        self.organized_dir = self.local_root / "과목별_기출DB"
        self.db_path = self.local_root / "db.json"
        self.users_path = self.local_root / "users.json"
        self.credit_ledger_path = self.local_root / "credit_ledger.jsonl"
        self.payment_settings_path = self.local_root / "payment_settings.json"
        self.payment_orders_path = self.local_root / "payment_orders.json"
        self.english_pool_path = self.local_root / "english_question_pool.json"
        self.english_analysis_pool_path = self.local_root / "english_analysis_pool.json"
        self.english_analysis_pdfs_root = self.local_root / "english_analysis_pdfs"
        self.korean_analysis_pool_path = self.local_root / "korean_analysis_pool.json"
        self.korean_analysis_pdfs_root = self.local_root / "korean_analysis_pdfs"
        self.korean_pool_path = self.local_root / "korean_question_pool.json"
        self.user_korean_history_path = self.local_root / "user_korean_history.json"
        self.user_english_history_path = self.local_root / "user_english_history.json"
        self.saved_exams_root = self.local_root / "saved_exams"
        self.saved_exams_pdfs_root = self.saved_exams_root / "pdfs"
        self.english_dict_cache_path = self.local_root / "english_dict_cache.json"
        self.english_dict_seed_path = Path(__file__).resolve().parents[2] / "data" / "csat_vocab_seed.json"
        self.user_vocab_root = self.local_root / "user_vocab"

        self.concept_root = Path(concept_root)
        self.concept_books_path = self.concept_root / "concept_books.json"

        self.local_root.mkdir(parents=True, exist_ok=True)
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.organized_dir.mkdir(parents=True, exist_ok=True)
        self.english_analysis_pdfs_root.mkdir(parents=True, exist_ok=True)
        self.korean_analysis_pdfs_root.mkdir(parents=True, exist_ok=True)
        self.saved_exams_root.mkdir(parents=True, exist_ok=True)
        self.saved_exams_pdfs_root.mkdir(parents=True, exist_ok=True)
        self.user_vocab_root.mkdir(parents=True, exist_ok=True)
        self.concept_root.mkdir(parents=True, exist_ok=True)

    def _local_exam_dir(self, subject_category: str, exam_name: str) -> Path:
        with self._lock:
            target = self.organized_dir / _safe_segment(subject_category) / _safe_segment(exam_name)
            target.mkdir(parents=True, exist_ok=True)
            return target
    def _has_nonempty_file(self, path: Path) -> bool:
        try:
            return path.exists() and path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    def _read_nonempty_bytes(self, path: Path) -> bytes | None:
        with self._lock:
            if not path.exists():
                return None
            try:
                data = path.read_bytes()
                return data if data else None
            except OSError:
                return None

    def append_credit_ledger(self, entry: dict[str, Any], max_lines: int = 10_000, keep_lines: int = 5_000) -> None:
        """크레딧 변동 내역을 credit_ledger.jsonl에 스레드 안전하게 추가 기록한다."""
        with self._lock:
            self.credit_ledger_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
            with self.credit_ledger_path.open("a", encoding="utf-8") as fp:
                fp.write(line)
            try:
                if self.credit_ledger_path.stat().st_size > 2 * 1024 * 1024:
                    lines = self.credit_ledger_path.read_text(encoding="utf-8").splitlines()
                    if len(lines) > max_lines:
                        trimmed = "\n".join(lines[-keep_lines:]) + "\n"
                        _atomic_write_text(self.credit_ledger_path, trimmed, encoding="utf-8")
            except OSError:
                pass

    def read_credit_ledger(self, username: str = "", limit: int = 100) -> list[dict[str, Any]]:
        """credit_ledger.jsonl에서 최신순 크레딧 변동 내역을 조회한다."""
        norm_u = (username or "").strip().lower()
        max_n = max(1, int(limit or 100))
        with self._lock:
            if not self.credit_ledger_path.exists():
                return []
            try:
                raw_lines = self.credit_ledger_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                return []

        results: list[dict[str, Any]] = []
        for ln in reversed(raw_lines):
            s = ln.strip()
            if not s:
                continue
            try:
                item = json.loads(s)
            except Exception:
                continue
            if not isinstance(item, dict):
                continue
            if norm_u and str(item.get("username") or "").strip().lower() != norm_u:
                continue
            results.append(item)
            if len(results) >= max_n:
                break
        return results
