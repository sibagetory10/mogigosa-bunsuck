from __future__ import annotations

import hashlib
import json
import random
import re
import time
from typing import TYPE_CHECKING, Any, Sequence

from src.storage.base import (
    _GLOBAL_CACHE_LOCK,
    _GLOBAL_JSON_CACHE,
    _GLOBAL_STORAGE_STATS_CACHE,
    _atomic_write_text,
    _file_stat_sig,
)

if TYPE_CHECKING:
    from src.models import RewrittenQuestion

CURRENT_KOREAN_POOL_VERSION = "v2"


def _is_korean_dialogue_or_poetry_meta(korean_section: str = "", korean_subgenre: str = "") -> bool:
    sec = (korean_section or "").strip()
    sub = (korean_subgenre or "").strip()
    if sec == "화법과 작문":
        return True
    if any(k in sub for k in ("화법", "대화", "토론", "협상", "극", "시나리오", "희곡")):
        return True
    if any(k in sub for k in ("현대시", "고전시가", "시가", "시조", "가사", "향가", "한시")) or sub == "시":
        return True
    return False


def _is_valid_cached_korean_set(
    raw_questions: list[Any],
    expected_count: int,
) -> list[RewrittenQuestion] | None:
    """캐시 풀에서 로드한 국어 변형 문항 세트 전체가 유효한지 무결성을 검증한다."""
    if not isinstance(raw_questions, list) or not raw_questions:
        return None
    if expected_count > 0 and len(raw_questions) != expected_count:
        return None

    from src.models.concept import RewrittenQuestion

    validated: list[RewrittenQuestion] = []
    for item in raw_questions:
        if not isinstance(item, dict):
            return None
        try:
            rw = RewrittenQuestion.model_validate(item)
        except Exception:
            return None
        stem = (rw.rewritten_stem or "").strip()
        shared_p = (rw.shared_passage or "").strip()
        choices = list(rw.rewritten_choices or [])
        ans = str(rw.preserved_answer or "").strip()
        if not stem or not shared_p or len(choices) != 5 or ans not in ("1", "2", "3", "4", "5"):
            return None
        validated.append(rw)
    return validated


class KoreanPoolStoreMixin:
    @staticmethod
    def build_korean_pool_key(
        passage_text: str,
        group_id: str = "",
        korean_section: str = "",
        question_count: int = 0,
        korean_subgenre: str = "",
    ) -> str:
        """정제된 국어 지문·세트 번호·영역·문항 수 조합으로 국어 변형 풀 캐시 키를 생성한다."""
        from src.models.question import normalize_korean_group_id
        from src.text_formatter import clean_korean_exam_passage

        is_keep_lines = _is_korean_dialogue_or_poetry_meta(korean_section, korean_subgenre)
        cleaned = clean_korean_exam_passage(passage_text or "", is_poetry_or_dialogue=is_keep_lines)
        norm_p = re.sub(r"\s+", " ", cleaned.strip())
        gid_norm = normalize_korean_group_id(group_id)
        sec_norm = (korean_section or "").strip()
        q_cnt = max(0, int(question_count or 0))
        raw_key = f"{CURRENT_KOREAN_POOL_VERSION}::{norm_p}||{gid_norm}||{sec_norm}||{q_cnt}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def build_korean_set_variant_id(cache_key: str, stems: Sequence[str]) -> str:
        """캐시 키와 세트 내 문항 발문 목록으로 변형 세트 고유 ID를 생성한다 (선지 재배치 시에도 불변)."""
        norm_stems = "||".join(re.sub(r"\s+", " ", (s or "").strip()) for s in stems)
        return hashlib.sha256(f"{cache_key}||{norm_stems}".encode("utf-8")).hexdigest()[:24]

    def _load_korean_pool_unlocked(self) -> dict[str, list[dict[str, Any]]]:
        if not self.korean_pool_path.exists():
            return {}
        sig = _file_stat_sig(self.korean_pool_path)
        if sig is None or sig[1] == 0:
            return {}
        cache_key = f"kor_q_pool::{self.korean_pool_path.resolve()}"
        with _GLOBAL_CACHE_LOCK:
            cached = _GLOBAL_JSON_CACHE.get(cache_key)
            if cached is not None and cached[0] == sig:
                return {k: [dict(it) for it in v] for k, v in cached[1].items()}
        try:
            data = json.loads(self.korean_pool_path.read_text(encoding="utf-8"))
            clean = {
                str(k): [dict(it) for it in v if isinstance(it, dict)]
                for k, v in (data if isinstance(data, dict) else {}).items()
                if isinstance(v, list)
            }
            with _GLOBAL_CACHE_LOCK:
                _GLOBAL_JSON_CACHE[cache_key] = (sig, clean)
            return {k: [dict(it) for it in v] for k, v in clean.items()}
        except Exception:
            return {}

    def _save_korean_pool_unlocked(self, pool: dict[str, list[dict[str, Any]]]) -> None:
        raw = json.dumps(pool, ensure_ascii=False, indent=2)
        _atomic_write_text(self.korean_pool_path, raw, encoding="utf-8")
        new_sig = _file_stat_sig(self.korean_pool_path)
        if new_sig is not None:
            with _GLOBAL_CACHE_LOCK:
                _GLOBAL_JSON_CACHE[f"kor_q_pool::{self.korean_pool_path.resolve()}"] = (
                    new_sig,
                    {
                        str(k): [dict(it) for it in v if isinstance(it, dict)]
                        for k, v in pool.items()
                        if isinstance(v, list)
                    },
                )
                _GLOBAL_STORAGE_STATS_CACHE.clear()

    def load_korean_pool(self) -> dict[str, list[dict[str, Any]]]:
        """저장된 국어 변형 문제 세트 풀(Pool) 전체를 안전하게 조회한다."""
        with self._lock:
            pool = self._load_korean_pool_unlocked()
            return {
                k: [dict(it) for it in v if isinstance(it, dict)]
                for k, v in pool.items()
                if isinstance(v, list) and v
            }

    def get_korean_pool_set_count(self, cache_key: str) -> int:
        """특정 지문 키에 저장된 국어 변형 세트 개수를 반환한다."""
        if not cache_key:
            return 0
        with self._lock:
            pool = self._load_korean_pool_unlocked()
            entries = pool.get(cache_key)
            return len(entries) if isinstance(entries, list) else 0

    def _load_user_korean_history_unlocked(self) -> dict[str, list[str]]:
        if not self.user_korean_history_path.exists():
            return {}
        sig = _file_stat_sig(self.user_korean_history_path)
        if sig is None or sig[1] == 0:
            return {}
        cache_key = f"kor_hist::{self.user_korean_history_path.resolve()}"
        with _GLOBAL_CACHE_LOCK:
            cached = _GLOBAL_JSON_CACHE.get(cache_key)
            if cached is not None and cached[0] == sig:
                return {k: list(v) for k, v in cached[1].items()}
        try:
            data = json.loads(self.user_korean_history_path.read_text(encoding="utf-8"))
            clean = {
                str(k): [str(x) for x in v]
                for k, v in (data if isinstance(data, dict) else {}).items()
                if isinstance(v, list)
            }
            with _GLOBAL_CACHE_LOCK:
                _GLOBAL_JSON_CACHE[cache_key] = (sig, clean)
            return {k: list(v) for k, v in clean.items()}
        except Exception:
            return {}

    def _save_user_korean_history_unlocked(self, history: dict[str, list[str]]) -> None:
        raw = json.dumps(history, ensure_ascii=False, indent=2)
        _atomic_write_text(self.user_korean_history_path, raw, encoding="utf-8")
        new_sig = _file_stat_sig(self.user_korean_history_path)
        if new_sig is not None:
            with _GLOBAL_CACHE_LOCK:
                _GLOBAL_JSON_CACHE[f"kor_hist::{self.user_korean_history_path.resolve()}"] = (
                    new_sig,
                    {str(k): [str(x) for x in v] for k, v in history.items() if isinstance(v, list)},
                )

    def get_user_seen_korean_sets(self, username: str) -> set[str]:
        """특정 계정이 이미 생성·풀이·PDF로 받은 국어 변형 세트 variant_id 집합을 반환한다."""
        uname = (username or "").strip()
        if not uname:
            return set()
        with self._lock:
            history = self._load_user_korean_history_unlocked()
            items = history.get(uname)
            return set(str(x) for x in items) if isinstance(items, list) else set()

    def mark_user_seen_korean_sets(
        self,
        username: str,
        variant_ids: Sequence[str],
        max_history: int = 3000,
    ) -> int:
        """계정별로 받은 국어 변형 세트의 variant_id를 기록해 동일 세트 중복 출제를 방지한다."""
        uname = (username or "").strip()
        if not uname or not variant_ids:
            return 0
        with self._lock:
            history = self._load_user_korean_history_unlocked()
            existing = list(history.get(uname) or [])
            seen_set = set(existing)
            added = 0
            for vid in variant_ids:
                clean_vid = str(vid or "").strip()
                if not clean_vid or clean_vid in seen_set:
                    continue
                seen_set.add(clean_vid)
                existing.append(clean_vid)
                added += 1
            if added == 0:
                return 0
            if len(existing) > max_history:
                existing = existing[-max_history:]
            history[uname] = existing
            self._save_user_korean_history_unlocked(history)
            return added

    def get_cached_korean_set(
        self,
        cache_key: str,
        expected_count: int = 0,
        username: str = "",
    ) -> tuple[list[RewrittenQuestion], str] | None:
        """국어 변형 풀에서 계정이 아직 보지 않은 유효 변형 세트 1건을 무작위 반환한다."""
        if not cache_key:
            return None
        with self._lock:
            pool = self._load_korean_pool_unlocked()
            entries = pool.get(cache_key)
            if not isinstance(entries, list) or not entries:
                return None

            uname = (username or "").strip()
            seen_ids: set[str] = set()
            if uname:
                history = self._load_user_korean_history_unlocked()
                seen_ids = set(str(x) for x in (history.get(uname) or []))

            valid_candidates: list[tuple[list[RewrittenQuestion], str]] = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                raw_qs = entry.get("questions")
                validated = _is_valid_cached_korean_set(raw_qs, expected_count)
                if not validated:
                    continue
                vid = str(entry.get("variant_id") or "").strip() or self.build_korean_set_variant_id(
                    cache_key, [rw.rewritten_stem for rw in validated]
                )
                if seen_ids and vid in seen_ids:
                    continue
                valid_candidates.append((validated, vid))

            if not valid_candidates:
                return None
            return random.choice(valid_candidates)

    def save_cached_korean_set(
        self,
        cache_key: str,
        rewritten_list: Sequence[RewrittenQuestion],
        *,
        exam_name: str = "",
        group_id: str = "",
        korean_section: str = "",
        korean_subgenre: str = "",
        passage_topic: str = "",
        max_per_key: int = 5,
    ) -> str:
        """2-Gate 검수를 통과한 국어 변형 세트를 키당 최대 max_per_key(기본 5)세트까지 풀에 영구 저장한다."""
        if not cache_key or not rewritten_list:
            return ""
        rw_items = list(rewritten_list)
        vid = self.build_korean_set_variant_id(cache_key, [rw.rewritten_stem for rw in rw_items])
        with self._lock:
            pool = self._load_korean_pool_unlocked()
            entries = list(pool.get(cache_key) or [])
            for existing in entries:
                if not isinstance(existing, dict):
                    continue
                ex_vid = str(existing.get("variant_id") or "").strip()
                if not ex_vid and isinstance(existing.get("questions"), list):
                    ex_vid = self.build_korean_set_variant_id(
                        cache_key,
                        [str(q.get("rewritten_stem") or "") for q in existing["questions"] if isinstance(q, dict)],
                    )
                if ex_vid == vid:
                    return vid

            new_entry = {
                "variant_id": vid,
                "exam_name": (exam_name or "").strip(),
                "group_id": (group_id or "").strip(),
                "section": (korean_section or "").strip(),
                "subgenre": (korean_subgenre or "").strip(),
                "passage_topic": (passage_topic or "").strip(),
                "question_count": len(rw_items),
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "questions": [rw.model_dump() for rw in rw_items],
            }
            entries.append(new_entry)
            if len(entries) > max_per_key:
                entries = entries[-max_per_key:]
            pool[cache_key] = entries
            self._save_korean_pool_unlocked(pool)
            return vid

    def delete_korean_pool_entry(
        self,
        cache_key: str,
        item_idx: int | None = None,
    ) -> int:
        """특정 캐시 키 전체 또는 해당 키 내 특정 인덱스 변형 세트 1건을 삭제한다."""
        if not cache_key:
            return 0
        with self._lock:
            pool = self._load_korean_pool_unlocked()
            entries = pool.get(cache_key)
            if not isinstance(entries, list) or not entries:
                return 0
            if item_idx is None:
                removed = len(entries)
                pool.pop(cache_key, None)
            elif 0 <= item_idx < len(entries):
                entries = list(entries)
                entries.pop(item_idx)
                removed = 1
                if entries:
                    pool[cache_key] = entries
                else:
                    pool.pop(cache_key, None)
            else:
                return 0
            self._save_korean_pool_unlocked(pool)
            return removed

    def clear_korean_pool(self) -> int:
        """국어 변형 문제 세트 풀(Pool) 캐시만 전체 비운다 (기출 DB·회원 DB·분석지 DB는 유지)."""
        with self._lock:
            pool = self._load_korean_pool_unlocked()
            removed = sum(len(v) for v in pool.values() if isinstance(v, list))
            self._save_korean_pool_unlocked({})
            return removed

    def delete_korean_pool_by_exam(self, exam_name: str) -> int:
        """특정 회차가 삭제될 때 해당 회차의 국어 변형 세트 캐시도 함께 삭제한다."""
        target_exam = (exam_name or "").strip()
        if not target_exam:
            return 0
        with self._lock:
            pool = self._load_korean_pool_unlocked()
            removed = 0
            new_pool: dict[str, list[dict[str, Any]]] = {}
            for ck, entries in pool.items():
                if not isinstance(entries, list):
                    continue
                kept = [
                    ent
                    for ent in entries
                    if not (isinstance(ent, dict) and str(ent.get("exam_name") or "").strip() == target_exam)
                ]
                removed += len(entries) - len(kept)
                if kept:
                    new_pool[ck] = kept
            if removed > 0:
                self._save_korean_pool_unlocked(new_pool)
            return removed

    def rename_korean_pool_exam(self, old_exam_name: str, new_exam_name: str) -> int:
        """회차명 변경 시 국어 변형 세트 캐시의 exam_name 메타데이터를 동기화한다."""
        old_n = (old_exam_name or "").strip()
        new_n = re.sub(r"\s+", " ", (new_exam_name or "").strip())
        if not old_n or not new_n or old_n == new_n:
            return 0
        with self._lock:
            pool = self._load_korean_pool_unlocked()
            changed = 0
            for entries in pool.values():
                if not isinstance(entries, list):
                    continue
                for ent in entries:
                    if isinstance(ent, dict) and str(ent.get("exam_name") or "").strip() == old_n:
                        ent["exam_name"] = new_n
                        changed += 1
            if changed > 0:
                self._save_korean_pool_unlocked(pool)
            return changed
