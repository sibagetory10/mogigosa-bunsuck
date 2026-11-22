from __future__ import annotations

import hashlib
import json
import random
import re
from typing import TYPE_CHECKING, Any

from src.storage.base import (
    _GLOBAL_CACHE_LOCK,
    _GLOBAL_JSON_CACHE,
    _atomic_write_text,
    _file_stat_sig,
)

if TYPE_CHECKING:
    from src.models import RewrittenQuestion

CURRENT_ENGLISH_POOL_VERSION = "v2"
_DSL_REMNANT_RE = re.compile(
    r"(?:^\s*(?:SPLIT|INSERT_AT|SENTENCE)\s*:|^\s*[1-5①②③④⑤]\s*[:.)]\s*[A-Za-z\-']+(?:\s*(?:->|→)\s*[A-Za-z\-']+)?\s*$)",
    re.IGNORECASE | re.MULTILINE,
)


def _is_valid_cached_english_question(rw: Any) -> bool:
    """캐시 풀에서 로드한 영어 변형 문항이 정상 조립된 문항인지 빠른 무결성 검사를 수행한다."""
    if rw is None:
        return False
    stem = (getattr(rw, "rewritten_stem", "") or "").strip()
    choices = list(getattr(rw, "rewritten_choices", []) or [])
    ans = str(getattr(rw, "preserved_answer", "") or "").strip()
    t = (getattr(rw, "target_question_type", "") or "").strip()
    if len(stem) < 70 or len(choices) != 5 or ans not in ("1", "2", "3", "4", "5"):
        return False
    if _DSL_REMNANT_RE.search(stem):
        return False
    if "빈칸" in t and "____" not in stem:
        return False
    if "함축" in t and "<u" not in stem.lower():
        return False
    if ("어법" in t or ("어휘" in t and "선택형" not in t and "(A)" not in t)) and len(
        re.findall(r"<u\b[^>]*>.*?</u>", stem, flags=re.I | re.S)
    ) != 5:
        return False
    if "선택형" in t and not all(mk in stem for mk in ("(A)", "(B)", "(C)")):
        return False
    if "순서" in t and not all(mk in stem for mk in ("(A)", "(B)", "(C)")):
        return False
    if ("삽입" in t or "무관" in t) and not all(mk in stem for mk in ("①", "②", "③", "④", "⑤")):
        return False
    if "요약" in t and ("(A)" not in stem or "(B)" not in stem):
        return False
    return True


class EnglishPoolStoreMixin:
    @staticmethod
    def build_english_pool_key(
        source_passage: str,
        target_type: str,
        difficulty: str = "",
        blank_unit: str = "",
        vocab_style: str = "",
    ) -> str:
        """지문 원문과 변형 옵션 조합으로 영어 변형 풀(Pool) v2 캐시 키를 생성한다."""
        norm_p = re.sub(r"\s+", " ", (source_passage or "").strip())
        t = (target_type or "").strip()
        opt = blank_unit if "빈칸" in t else (vocab_style if "어휘" in t else "")
        raw_key = f"{CURRENT_ENGLISH_POOL_VERSION}::{norm_p}||{t}||{(difficulty or '').strip()}||{(opt or '').strip()}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:24]

    def _load_english_pool_unlocked(self) -> dict[str, list[dict[str, Any]]]:
        if not self.english_pool_path.exists():
            return {}
        sig = _file_stat_sig(self.english_pool_path)
        if sig is None or sig[1] == 0:
            return {}
        cache_key = f"eng_pool::{self.english_pool_path.resolve()}"
        with _GLOBAL_CACHE_LOCK:
            cached = _GLOBAL_JSON_CACHE.get(cache_key)
            if cached is not None and cached[0] == sig:
                return {k: [dict(it) for it in v] for k, v in cached[1].items()}
        try:
            data = json.loads(self.english_pool_path.read_text(encoding="utf-8"))
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

    def _save_english_pool_unlocked(self, pool: dict[str, list[dict[str, Any]]]) -> None:
        raw = json.dumps(pool, ensure_ascii=False, indent=2)
        _atomic_write_text(self.english_pool_path, raw, encoding="utf-8")
        new_sig = _file_stat_sig(self.english_pool_path)
        if new_sig is not None:
            with _GLOBAL_CACHE_LOCK:
                _GLOBAL_JSON_CACHE[f"eng_pool::{self.english_pool_path.resolve()}"] = (
                    new_sig,
                    {str(k): [dict(it) for it in v if isinstance(it, dict)] for k, v in pool.items() if isinstance(v, list)},
                )

    def load_english_pool(self) -> dict[str, list[dict[str, Any]]]:
        """저장된 영어 변형 풀(Pool) 전체를 안전하게 조회한다."""
        with self._lock:
            pool = self._load_english_pool_unlocked()
            return {
                k: [dict(it) for it in v if isinstance(it, dict)]
                for k, v in pool.items()
                if isinstance(v, list) and v
            }

    def delete_english_pool_entry(
        self,
        cache_key: str,
        item_idx: int | None = None,
    ) -> int:
        """특정 캐시 키 전체 또는 해당 키 내 특정 인덱스 변형 문항 1개를 삭제한다."""
        if not cache_key:
            return 0
        with self._lock:
            pool = self._load_english_pool_unlocked()
            items = pool.get(cache_key)
            if not isinstance(items, list) or not items:
                return 0
            if item_idx is None:
                removed = len(items)
                pool.pop(cache_key, None)
            elif 0 <= item_idx < len(items):
                items = list(items)
                items.pop(item_idx)
                removed = 1
                if items:
                    pool[cache_key] = items
                else:
                    pool.pop(cache_key, None)
            else:
                return 0
            self._save_english_pool_unlocked(pool)
            return removed

    def clear_english_pool(self) -> int:
        """영어 변형 풀(Pool) 캐시만 전체 비운다 (기출 DB 및 회원 DB는 유지)."""
        with self._lock:
            pool = self._load_english_pool_unlocked()
            removed = sum(len(v) for v in pool.values() if isinstance(v, list))
            self._save_english_pool_unlocked({})
            return removed

    @staticmethod
    def build_english_variant_id(cache_key: str, rewritten_stem: str) -> str:
        """캐시 키와 변형 지문 본문 조합으로 개별 변형 문항 고유 ID를 생성한다."""
        norm_stem = re.sub(r"\s+", " ", (rewritten_stem or "").strip())
        return hashlib.sha256(f"{cache_key}||{norm_stem}".encode("utf-8")).hexdigest()[:24]

    def _load_user_english_history_unlocked(self) -> dict[str, list[str]]:
        if not self.user_english_history_path.exists():
            return {}
        sig = _file_stat_sig(self.user_english_history_path)
        if sig is None or sig[1] == 0:
            return {}
        cache_key = f"eng_hist::{self.user_english_history_path.resolve()}"
        with _GLOBAL_CACHE_LOCK:
            cached = _GLOBAL_JSON_CACHE.get(cache_key)
            if cached is not None and cached[0] == sig:
                return {k: list(v) for k, v in cached[1].items()}
        try:
            data = json.loads(self.user_english_history_path.read_text(encoding="utf-8"))
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

    def _save_user_english_history_unlocked(self, history: dict[str, list[str]]) -> None:
        raw = json.dumps(history, ensure_ascii=False, indent=2)
        _atomic_write_text(self.user_english_history_path, raw, encoding="utf-8")
        new_sig = _file_stat_sig(self.user_english_history_path)
        if new_sig is not None:
            with _GLOBAL_CACHE_LOCK:
                _GLOBAL_JSON_CACHE[f"eng_hist::{self.user_english_history_path.resolve()}"] = (
                    new_sig,
                    {str(k): [str(x) for x in v] for k, v in history.items() if isinstance(v, list)},
                )

    def get_user_seen_english_variants(self, username: str) -> set[str]:
        """특정 계정이 이미 풀었거나 PDF로 생성한 영어 변형 문항 ID 집합을 반환한다."""
        uname = (username or "").strip()
        if not uname:
            return set()
        with self._lock:
            history = self._load_user_english_history_unlocked()
            items = history.get(uname)
            return set(str(x) for x in items) if isinstance(items, list) else set()

    def mark_user_seen_english_variants(
        self,
        username: str,
        pairs: list[tuple[str, str]],
        max_history: int = 5000,
    ) -> int:
        """계정별로 PDF 생성 또는 풀이 완료한 영어 변형 문항을 기록해 중복 출제를 방지한다."""
        uname = (username or "").strip()
        if not uname or not pairs:
            return 0
        with self._lock:
            history = self._load_user_english_history_unlocked()
            existing = list(history.get(uname) or [])
            seen_set = set(existing)
            added = 0
            for cache_key, stem in pairs:
                if not cache_key or not (stem or "").strip():
                    continue
                vid = self.build_english_variant_id(cache_key, stem)
                if vid not in seen_set:
                    seen_set.add(vid)
                    existing.append(vid)
                    added += 1
            if added == 0:
                return 0
            if len(existing) > max_history:
                existing = existing[-max_history:]
            history[uname] = existing
            self._save_user_english_history_unlocked(history)
            return added

    def get_cached_english_question(
        self,
        cache_key: str,
        exclude_stem: str = "",
        username: str = "",
    ) -> RewrittenQuestion | None:
        """검증 완료된 영어 변형 풀에서 계정이 아직 풀거나 PDF로 받지 않은 문항만 즉시 반환한다."""
        if not cache_key:
            return None
        with self._lock:
            pool = self._load_english_pool_unlocked()
            items = pool.get(cache_key)
            if not isinstance(items, list) or not items:
                return None
            candidates = list(items)
            uname = (username or "").strip()
            if uname:
                history = self._load_user_english_history_unlocked()
                seen_ids = set(str(x) for x in (history.get(uname) or []))
                if seen_ids:
                    candidates = [
                        it
                        for it in candidates
                        if self.build_english_variant_id(cache_key, str(it.get("rewritten_stem") or "")) not in seen_ids
                    ]
                    if not candidates:
                        return None
            if exclude_stem and len(candidates) > 1:
                filtered = [it for it in candidates if (it.get("rewritten_stem") or "").strip() != exclude_stem.strip()]
                if filtered:
                    candidates = filtered
            chosen = random.choice(candidates)
            from src.models.concept import RewrittenQuestion

            try:
                rw_obj = RewrittenQuestion.model_validate(chosen)
                return rw_obj if _is_valid_cached_english_question(rw_obj) else None
            except Exception:
                return None

    def save_cached_english_question(
        self,
        cache_key: str,
        rw: RewrittenQuestion,
        max_per_key: int = 5,
    ) -> None:
        """파이썬 검증을 통과한 영어 변형 문항을 키당 최대 max_per_key(기본 5)개까지 풀에 저장한다."""
        if not cache_key or rw is None:
            return
        with self._lock:
            pool = self._load_english_pool_unlocked()
            items = list(pool.get(cache_key) or [])
            stem_key = (rw.rewritten_stem or "").strip()
            if any((it.get("rewritten_stem") or "").strip() == stem_key for it in items):
                return
            items.append(rw.model_dump())
            if len(items) > max_per_key:
                items = items[-max_per_key:]
            pool[cache_key] = items
            self._save_english_pool_unlocked(pool)

