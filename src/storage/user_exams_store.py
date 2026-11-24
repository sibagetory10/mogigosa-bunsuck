from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from src.storage.base import (
    _GLOBAL_CACHE_LOCK,
    _GLOBAL_JSON_CACHE,
    _atomic_write_bytes,
    _atomic_write_text,
    _file_stat_sig,
)


class UserExamsStoreMixin:
    def _user_saved_exams_path(self, username: str) -> Path:
        safe_user = re.sub(r"[^a-zA-Z0-9._-]+", "_", (username or "").strip()) or "anonymous"
        return self.saved_exams_root / f"{safe_user}.json"

    def _load_user_exams_unlocked(self, username: str) -> list[dict[str, Any]]:
        p = self._user_saved_exams_path(username)
        if not p.exists():
            return []
        sig = _file_stat_sig(p)
        if sig is None or sig[1] == 0:
            return []
        cache_key = f"user_exams::{p.resolve()}"
        with _GLOBAL_CACHE_LOCK:
            cached = _GLOBAL_JSON_CACHE.get(cache_key)
            if cached is not None and cached[0] == sig:
                return [dict(it) for it in cached[1]]
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            clean = [dict(it) for it in data if isinstance(it, dict)] if isinstance(data, list) else []
            with _GLOBAL_CACHE_LOCK:
                _GLOBAL_JSON_CACHE[cache_key] = (sig, clean)
            return [dict(it) for it in clean]
        except Exception:
            return []

    def _save_user_exams_unlocked(self, username: str, exams: list[dict[str, Any]]) -> None:
        self.saved_exams_root.mkdir(parents=True, exist_ok=True)
        p = self._user_saved_exams_path(username)
        _atomic_write_text(p, json.dumps(exams, ensure_ascii=False, indent=2), encoding="utf-8")
        new_sig = _file_stat_sig(p)
        if new_sig is not None:
            with _GLOBAL_CACHE_LOCK:
                _GLOBAL_JSON_CACHE[f"user_exams::{p.resolve()}"] = (
                    new_sig,
                    [dict(it) for it in exams if isinstance(it, dict)],
                )

    def save_user_exam(
        self,
        username: str,
        exam_record: dict[str, Any],
        pdf_bytes: bytes | None = None,
        max_exams_per_user: int = 100,
    ) -> dict[str, Any]:
        """계정별 '나의 시험지'에 시험지 데이터와 PDF(선택)를 과목별로 자동 저장한다."""
        uname = (username or "").strip()
        if not uname:
            return exam_record
        with self._lock:
            exams = self._load_user_exams_unlocked(uname)
            exam_id = str(exam_record.get("exam_id") or "").strip()
            if not exam_id:
                exam_id = hashlib.sha256(f"{uname}||{time.time()}".encode("utf-8")).hexdigest()[:16]
            now_str = time.strftime("%Y-%m-%d %H:%M:%S")

            existing_idx = next((i for i, e in enumerate(exams) if e.get("exam_id") == exam_id), None)
            merged = dict(exams[existing_idx]) if existing_idx is not None else {}
            merged.update(exam_record)
            merged["exam_id"] = exam_id
            merged.setdefault("created_at", now_str)
            merged["updated_at"] = now_str

            if pdf_bytes:
                self.saved_exams_pdfs_root.mkdir(parents=True, exist_ok=True)
                safe_user = re.sub(r"[^a-zA-Z0-9._-]+", "_", uname) or "anonymous"
                pdf_name = f"{safe_user}__{exam_id}.pdf"
                _atomic_write_bytes(self.saved_exams_pdfs_root / pdf_name, pdf_bytes)
                merged["pdf_filename"] = pdf_name
                merged["has_pdf"] = True

            if existing_idx is not None:
                exams.pop(existing_idx)
            exams.insert(0, merged)

            if len(exams) > max_exams_per_user:
                dropped = exams[max_exams_per_user:]
                exams = exams[:max_exams_per_user]
                for d_item in dropped:
                    p_name = str(d_item.get("pdf_filename") or "")
                    if p_name:
                        try:
                            (self.saved_exams_pdfs_root / p_name).unlink(missing_ok=True)
                        except Exception:
                            pass

            self._save_user_exams_unlocked(uname, exams)
            return merged

    def list_user_exams(self, username: str, subject: str = "전체") -> list[dict[str, Any]]:
        """계정별로 저장된 시험지 목록을 최신순(선택 시 과목 필터)으로 반환한다."""
        uname = (username or "").strip()
        if not uname:
            return []
        with self._lock:
            exams = self._load_user_exams_unlocked(uname)
            if subject and subject != "전체":
                return [e for e in exams if str(e.get("subject") or "기타") == subject]
            return exams

    def get_user_exam_pdf_bytes(self, username: str, exam_id: str) -> bytes | None:
        """저장된 시험지의 PDF 바이트를 반환한다."""
        uname = (username or "").strip()
        if not uname or not exam_id:
            return None
        with self._lock:
            exams = self._load_user_exams_unlocked(uname)
            target = next((e for e in exams if e.get("exam_id") == exam_id), None)
            if not target:
                return None
            p_name = str(target.get("pdf_filename") or "")
            if not p_name:
                return None
            pdf_path = self.saved_exams_pdfs_root / p_name
            if not pdf_path.exists():
                return None
            try:
                return pdf_path.read_bytes()
            except Exception:
                return None

    def delete_user_exam(self, username: str, exam_id: str) -> bool:
        """계정의 저장된 시험지 및 연결된 PDF 파일을 삭제한다."""
        uname = (username or "").strip()
        if not uname or not exam_id:
            return False
        with self._lock:
            exams = self._load_user_exams_unlocked(uname)
            target_idx = next((i for i, e in enumerate(exams) if e.get("exam_id") == exam_id), None)
            if target_idx is None:
                return False
            removed = exams.pop(target_idx)
            p_name = str(removed.get("pdf_filename") or "")
            if p_name:
                try:
                    (self.saved_exams_pdfs_root / p_name).unlink(missing_ok=True)
                except Exception:
                    pass
            self._save_user_exams_unlocked(uname, exams)
            return True

    def rename_user_data(self, old_username: str, new_username: str) -> None:
        """구글 전용 임시 ID에서 정식 아이디로 전환 시 시험지·영어 이력·개념서 작성자를 이관한다."""
        old_u = (old_username or "").strip()
        new_u = (new_username or "").strip()
        if not old_u or not new_u or old_u == new_u:
            return

        with self._lock:
            # 1) 영어 변형 풀이 이력 이관
            try:
                history = self._load_user_english_history_unlocked()
                if old_u in history:
                    old_list = list(history.pop(old_u) or [])
                    new_list = list(history.get(new_u) or [])
                    seen = set(new_list)
                    for vid in old_list:
                        if vid not in seen:
                            seen.add(vid)
                            new_list.append(vid)
                    history[new_u] = new_list
                    self._save_user_english_history_unlocked(history)
            except Exception:
                pass

            # 2) 계정별 '나의 시험지' 이관 (exam_id 기준 중복 없이 병합)
            try:
                old_exams_path = self._user_saved_exams_path(old_u)
                if old_exams_path.exists():
                    old_exams = self._load_user_exams_unlocked(old_u)
                    new_exams = self._load_user_exams_unlocked(new_u)
                    merged_map: dict[str, dict[str, Any]] = {}
                    for ex in old_exams + new_exams:
                        ex_id = str(ex.get("exam_id") or "")
                        if ex_id:
                            merged_map[ex_id] = ex
                    merged_list = sorted(
                        merged_map.values(),
                        key=lambda e: str(e.get("updated_at") or e.get("created_at") or ""),
                        reverse=True,
                    )
                    self._save_user_exams_unlocked(new_u, merged_list)
                    if old_exams_path != self._user_saved_exams_path(new_u):
                        old_exams_path.unlink(missing_ok=True)
            except Exception:
                pass

            # 3) 공용 개념서 DB 작성자 표기 이관
            try:
                for book in self.load_concept_books():
                    if (book.created_by or "").strip() == old_u:
                        book.created_by = new_u
                        self.save_concept_book(book)
            except Exception:
                pass

            # 4) 계정별 개인 영단어장 이관 (normalized_key 기준 중복 없이 병합)
            try:
                old_vocab_path = self._user_vocab_path(old_u)
                if old_vocab_path.exists():
                    old_v = self._load_user_vocab_unlocked(old_u)
                    new_v = self._load_user_vocab_unlocked(new_u)
                    merged_v_map: dict[str, dict[str, Any]] = {}
                    for row in old_v + new_v:
                        nk = str(row.get("normalized_key") or "").strip()
                        if nk and nk not in merged_v_map:
                            merged_v_map[nk] = row
                    self._save_user_vocab_unlocked(new_u, list(merged_v_map.values()))
                    if old_vocab_path != self._user_vocab_path(new_u):
                        old_vocab_path.unlink(missing_ok=True)
            except Exception:
                pass

            # 5) 크레딧 원장(credit_ledger.jsonl) 사용자명 이관
            try:
                if self.credit_ledger_path.exists():
                    raw_lines = self.credit_ledger_path.read_text(encoding="utf-8").splitlines()
                    updated_lines: list[str] = []
                    ledger_changed = False
                    for ln in raw_lines:
                        s = ln.strip()
                        if not s:
                            continue
                        try:
                            row = json.loads(s)
                            if isinstance(row, dict) and str(row.get("username") or "").strip().lower() == old_u.lower():
                                row["username"] = new_u
                                ledger_changed = True
                                updated_lines.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                            else:
                                updated_lines.append(s)
                        except Exception:
                            updated_lines.append(s)
                    if ledger_changed:
                        _atomic_write_text(
                            self.credit_ledger_path,
                            ("\n".join(updated_lines) + "\n") if updated_lines else "",
                            encoding="utf-8",
                        )
            except Exception:
                pass
