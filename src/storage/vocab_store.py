from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.storage.base import (
    _GLOBAL_CACHE_LOCK,
    _GLOBAL_JSON_CACHE,
    _atomic_write_text,
    _file_stat_sig,
    _safe_segment,
)

if TYPE_CHECKING:
    from src.models import EnglishDictEntry, UserVocabItem


class VocabStoreMixin:
    @staticmethod
    def _read_english_dict_file_unlocked(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"entries": {}, "aliases": {}}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return {"entries": {}, "aliases": {}}
            entries = {
                str(k): dict(v)
                for k, v in (data.get("entries") or {}).items()
                if isinstance(v, dict)
            }
            aliases = {
                str(k): str(v)
                for k, v in (data.get("aliases") or {}).items()
                if isinstance(v, str)
            }
            return {"entries": entries, "aliases": aliases}
        except Exception:
            return {"entries": {}, "aliases": {}}

    def _load_english_dict_ref_unlocked(self) -> dict[str, Any]:
        """영단어 사전 캐시를 mtime/size 캐시에서 읽어 읽기 전용 참조로 반환한다 (O(1))."""
        seed_sig = _file_stat_sig(self.english_dict_seed_path)
        cache_sig = _file_stat_sig(self.english_dict_cache_path)
        has_seed = seed_sig is not None and seed_sig[1] > 0
        has_cache = cache_sig is not None and cache_sig[1] > 0
        if not has_seed and not has_cache:
            return {"entries": {}, "aliases": {}}
        sig = (seed_sig, cache_sig)
        cache_key = f"eng_dict::{self.english_dict_cache_path.resolve()}"
        with _GLOBAL_CACHE_LOCK:
            cached = _GLOBAL_JSON_CACHE.get(cache_key)
            if cached is not None and cached[0] == sig:
                return cached[1]

        entries: dict[str, dict[str, Any]] = {}
        aliases: dict[str, str] = {}
        if has_seed:
            seed_data = self._read_english_dict_file_unlocked(self.english_dict_seed_path)
            entries.update(seed_data["entries"])
            aliases.update(seed_data["aliases"])
        if has_cache:
            cache_data = self._read_english_dict_file_unlocked(self.english_dict_cache_path)
            entries.update(cache_data["entries"])
            aliases.update(cache_data["aliases"])

        clean = {"entries": entries, "aliases": aliases}
        with _GLOBAL_CACHE_LOCK:
            _GLOBAL_JSON_CACHE[cache_key] = (sig, clean)
        return clean

    def _load_english_dict_cache_unlocked(self) -> dict[str, Any]:
        ref = self._load_english_dict_ref_unlocked()
        return {
            "entries": {k: dict(v) for k, v in ref["entries"].items()},
            "aliases": dict(ref["aliases"]),
        }

    def _save_english_dict_cache_unlocked(self, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False, indent=2)
        _atomic_write_text(self.english_dict_cache_path, raw, encoding="utf-8")
        with _GLOBAL_CACHE_LOCK:
            _GLOBAL_JSON_CACHE.pop(f"eng_dict::{self.english_dict_cache_path.resolve()}", None)

    def get_cached_english_dict_entry(self, query: str) -> EnglishDictEntry | None:
        """단어·숙어·한글 검색어에 해당하는 캐시된 영단어 사전 엔트리를 반환한다."""
        from src.models.vocab import EnglishDictEntry, normalize_english_dict_key

        q_norm = normalize_english_dict_key(query)
        if not q_norm:
            return None
        with self._lock:
            payload = self._load_english_dict_ref_unlocked()
            entries = payload["entries"]
            aliases = payload["aliases"]
            raw = entries.get(q_norm)
            if raw is None and q_norm in aliases:
                raw = entries.get(aliases[q_norm])
            if not isinstance(raw, dict):
                return None
            try:
                return EnglishDictEntry.model_validate(raw)
            except Exception:
                return None

    def save_cached_english_dict_entry(
        self,
        entry: EnglishDictEntry,
        aliases: list[str] | None = None,
    ) -> None:
        """영단어 사전 조회 결과를 공용 캐시에 저장하고 검색 별칭을 함께 등록한다."""
        from src.models.vocab import normalize_english_dict_key

        canonical_key = normalize_english_dict_key(entry.headword or entry.query_normalized)
        if not canonical_key:
            return
        with self._lock:
            payload = self._read_english_dict_file_unlocked(self.english_dict_cache_path)
            entry_copy = entry.model_copy(
                update={
                    "query_normalized": canonical_key,
                    "updated_at": entry.updated_at or time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            payload["entries"][canonical_key] = entry_copy.model_dump()
            for al in aliases or []:
                al_norm = normalize_english_dict_key(al)
                if al_norm and al_norm != canonical_key:
                    payload["aliases"][al_norm] = canonical_key
            self._save_english_dict_cache_unlocked(payload)

    def list_cached_english_dict_entries(
        self,
        search: str = "",
        limit: int = 100,
    ) -> list[EnglishDictEntry]:
        """공용 영단어 사전 캐시에 저장된 엔트리를 최신순·검색어 필터로 반환한다."""
        from src.models.vocab import EnglishDictEntry

        with self._lock:
            payload = self._load_english_dict_ref_unlocked()
            raw_vals = list(payload["entries"].values())
        q_low = (search or "").strip().lower()
        filtered_raw: list[dict[str, Any]] = []
        for raw in raw_vals:
            if not isinstance(raw, dict):
                continue
            if q_low:
                senses_raw = raw.get("senses") or []
                sense_meanings = [
                    str(s.get("meaning_ko") or "")
                    for s in senses_raw
                    if isinstance(s, dict)
                ]
                hay = " ".join(
                    [
                        str(raw.get("headword") or ""),
                        str(raw.get("summary_ko") or ""),
                        *sense_meanings,
                    ]
                ).lower()
                if q_low not in hay:
                    continue
            filtered_raw.append(raw)
        filtered_raw.sort(key=lambda r: str(r.get("updated_at") or ""), reverse=True)
        max_items = max(1, int(limit))
        out: list[EnglishDictEntry] = []
        for raw in filtered_raw:
            try:
                out.append(EnglishDictEntry.model_validate(raw))
                if len(out) >= max_items:
                    break
            except Exception:
                continue
        return out

    def count_cached_english_dict_entries(self) -> int:
        with self._lock:
            payload = self._load_english_dict_ref_unlocked()
            return len(payload["entries"])

    def delete_cached_english_dict_entry(self, key: str) -> bool:
        from src.models.vocab import normalize_english_dict_key

        k_norm = normalize_english_dict_key(key)
        if not k_norm:
            return False
        with self._lock:
            payload = self._read_english_dict_file_unlocked(self.english_dict_cache_path)
            target_key = k_norm if k_norm in payload["entries"] else payload["aliases"].get(k_norm, "")
            if not target_key or target_key not in payload["entries"]:
                return False
            payload["entries"].pop(target_key, None)
            payload["aliases"] = {
                ak: av for ak, av in payload["aliases"].items() if av != target_key and ak != target_key
            }
            self._save_english_dict_cache_unlocked(payload)
            return True

    # -----------------------------------------------------------------------
    # 계정별 나만의 영단어장 (user_vocab/{username}.json)
    # -----------------------------------------------------------------------
    def _user_vocab_path(self, username: str) -> Path:
        safe_user = _safe_segment((username or "").strip() or "anonymous")
        return self.user_vocab_root / f"{safe_user}.json"

    def _load_user_vocab_unlocked(self, username: str) -> list[dict[str, Any]]:
        p = self._user_vocab_path(username)
        if not p.exists():
            return []
        sig = _file_stat_sig(p)
        if sig is None or sig[1] == 0:
            return []
        cache_key = f"user_vocab::{p.resolve()}"
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

    def _save_user_vocab_unlocked(self, username: str, rows: list[dict[str, Any]]) -> None:
        self.user_vocab_root.mkdir(parents=True, exist_ok=True)
        p = self._user_vocab_path(username)
        _atomic_write_text(p, json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        new_sig = _file_stat_sig(p)
        if new_sig is not None:
            with _GLOBAL_CACHE_LOCK:
                _GLOBAL_JSON_CACHE[f"user_vocab::{p.resolve()}"] = (
                    new_sig,
                    [dict(it) for it in rows if isinstance(it, dict)],
                )

    def load_user_vocab(self, username: str) -> list[UserVocabItem]:
        """특정 회원의 나만의 단어장 목록을 로드한다."""
        uname = (username or "").strip()
        if not uname:
            return []
        with self._lock:
            rows = self._load_user_vocab_unlocked(uname)
        from src.models.vocab import UserVocabItem

        items: list[UserVocabItem] = []
        for row in rows:
            try:
                items.append(UserVocabItem.model_validate(row))
            except Exception:
                continue
        return items

    def save_user_vocab_item(self, username: str, item: UserVocabItem) -> list[UserVocabItem]:
        """단일 영단어를 개인 단어장에 추가 또는 갱신(기존 암기/오답/메모 보존)하고 최신 목록을 반환한다."""
        from src.models.vocab import normalize_english_dict_key

        uname = (username or "").strip()
        if not uname:
            return []
        nk = normalize_english_dict_key(item.normalized_key or item.headword)
        if not nk:
            return self.load_user_vocab(uname)
        with self._lock:
            rows = self._load_user_vocab_unlocked(uname)
            existing_idx = next(
                (i for i, r in enumerate(rows) if str(r.get("normalized_key") or "") == nk),
                None,
            )
            now_str = item.added_at or time.strftime("%Y-%m-%d %H:%M:%S")
            new_dict = item.model_copy(update={"normalized_key": nk, "added_at": now_str}).model_dump()
            if existing_idx is not None:
                prev = rows.pop(existing_idx)
                new_dict["mastered"] = bool(prev.get("mastered", False))
                new_dict["wrong_count"] = int(prev.get("wrong_count", 0) or 0)
                if not (new_dict.get("memo") or "").strip():
                    new_dict["memo"] = str(prev.get("memo") or "")
                if not (new_dict.get("source_exam") or "").strip():
                    new_dict["source_exam"] = str(prev.get("source_exam") or "")
            rows.insert(0, new_dict)
            self._save_user_vocab_unlocked(uname, rows)
        return self.load_user_vocab(uname)

    def save_user_vocab_batch(self, username: str, items: list[UserVocabItem]) -> tuple[int, int]:
        """여러 영단어를 기존 항목을 덮어쓰지 않고 일괄 추가한다. 반환: (신규 추가 수, 기존 건너뜀 수)."""
        from src.models.vocab import normalize_english_dict_key

        uname = (username or "").strip()
        if not uname or not items:
            return 0, 0
        with self._lock:
            rows = self._load_user_vocab_unlocked(uname)
            existing_keys = {str(r.get("normalized_key") or "") for r in rows}
            added = 0
            skipped = 0
            to_prepend: list[dict[str, Any]] = []
            now_str = time.strftime("%Y-%m-%d %H:%M:%S")
            for item in items:
                nk = normalize_english_dict_key(item.normalized_key or item.headword)
                if not nk:
                    continue
                if nk in existing_keys:
                    skipped += 1
                    continue
                existing_keys.add(nk)
                to_prepend.append(
                    item.model_copy(
                        update={"normalized_key": nk, "added_at": item.added_at or now_str}
                    ).model_dump()
                )
                added += 1
            if added > 0:
                rows = to_prepend + rows
                self._save_user_vocab_unlocked(uname, rows)
            return added, skipped

    def update_user_vocab_item(
        self,
        username: str,
        normalized_key: str,
        *,
        mastered: bool | None = None,
        memo: str | None = None,
        increment_wrong: bool = False,
        reset_wrong: bool = False,
    ) -> bool:
        """개인 단어장 단어의 암기 완료 여부·메모·오답 횟수를 부분 업데이트한다."""
        from src.models.vocab import normalize_english_dict_key

        uname = (username or "").strip()
        nk = normalize_english_dict_key(normalized_key)
        if not uname or not nk:
            return False
        with self._lock:
            rows = self._load_user_vocab_unlocked(uname)
            for row in rows:
                if str(row.get("normalized_key") or "") == nk:
                    if mastered is not None:
                        row["mastered"] = bool(mastered)
                    if memo is not None:
                        row["memo"] = str(memo).strip()
                    if reset_wrong:
                        row["wrong_count"] = 0
                    elif increment_wrong:
                        row["wrong_count"] = int(row.get("wrong_count", 0) or 0) + 1
                    self._save_user_vocab_unlocked(uname, rows)
                    return True
            return False

    def delete_user_vocab_item(self, username: str, normalized_key: str) -> bool:
        """개인 단어장에서 단어 1건을 삭제한다."""
        from src.models.vocab import normalize_english_dict_key

        uname = (username or "").strip()
        nk = normalize_english_dict_key(normalized_key)
        if not uname or not nk:
            return False
        with self._lock:
            rows = self._load_user_vocab_unlocked(uname)
            new_rows = [r for r in rows if str(r.get("normalized_key") or "") != nk]
            if len(new_rows) == len(rows):
                return False
            self._save_user_vocab_unlocked(uname, new_rows)
            return True

    def clear_user_vocab(self, username: str, only_mastered: bool = False) -> int:
        """개인 단어장 전체 또는 암기 완료 항목만 비운다."""
        uname = (username or "").strip()
        if not uname:
            return 0
        with self._lock:
            rows = self._load_user_vocab_unlocked(uname)
            if not rows:
                return 0
            if only_mastered:
                rem = [r for r in rows if not bool(r.get("mastered", False))]
                removed = len(rows) - len(rem)
                if removed > 0:
                    self._save_user_vocab_unlocked(uname, rem)
                return removed
            self._save_user_vocab_unlocked(uname, [])
            return len(rows)

