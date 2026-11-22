from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from src.exporters import build_exam_ai_jsonl, build_exam_html, build_exam_markdown
from src.models import (
    ConceptBookRecord,
    EnglishDictEntry,
    KoreanPassageAnalysisOutput,
    LEGACY_TIER_MAP,
    QuestionRecord,
    RewrittenQuestion,
    UserRecord,
    UserVocabItem,
    normalize_english_dict_key,
)

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


class StorageManager:
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
        self.english_pool_path = self.local_root / "english_question_pool.json"
        self.korean_analysis_pool_path = self.local_root / "korean_analysis_pool.json"
        self.korean_analysis_pdfs_root = self.local_root / "korean_analysis_pdfs"
        self.user_english_history_path = self.local_root / "user_english_history.json"
        self.saved_exams_root = self.local_root / "saved_exams"
        self.saved_exams_pdfs_root = self.saved_exams_root / "pdfs"
        self.english_dict_cache_path = self.local_root / "english_dict_cache.json"
        self.english_dict_seed_path = Path(__file__).resolve().parent.parent / "data" / "csat_vocab_seed.json"
        self.user_vocab_root = self.local_root / "user_vocab"

        self.concept_root = Path(concept_root)
        self.concept_books_path = self.concept_root / "concept_books.json"

        self.local_root.mkdir(parents=True, exist_ok=True)
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.organized_dir.mkdir(parents=True, exist_ok=True)
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

    def load_db(self) -> list[QuestionRecord]:
        with self._lock:
            if not self.db_path.exists():
                return []
            sig = _file_stat_sig(self.db_path)
            if sig is None or sig[1] == 0:
                return []
            cache_key = f"db::{self.db_path.resolve()}"
            with _GLOBAL_CACHE_LOCK:
                cached = _GLOBAL_JSON_CACHE.get(cache_key)
                if cached is not None and cached[0] == sig:
                    return [rec.model_copy(deep=True) for rec in cached[1]]

            last_err: Exception | None = None
            for attempt in range(3):
                try:
                    text = self.db_path.read_text(encoding="utf-8")
                    if not text.strip():
                        time.sleep(0.05)
                        continue
                    raw = json.loads(text)
                    if not isinstance(raw, list):
                        raise ValueError(f"db.json 루트가 리스트가 아닙니다 (type: {type(raw)})")
                    records = [QuestionRecord.model_validate(item) for item in raw]
                    new_sig = _file_stat_sig(self.db_path) or sig
                    with _GLOBAL_CACHE_LOCK:
                        _GLOBAL_JSON_CACHE[cache_key] = (new_sig, records)
                    return [rec.model_copy(deep=True) for rec in records]
                except Exception as exc:
                    last_err = exc
                    if attempt < 2:
                        time.sleep(0.05)

            # 끝까지 실패 시 기존 파일을 빈 리스트로 덮어쓰지 않도록 방어 로직
            raise RuntimeError(
                f"{self.db_path} 파일이 존재하지만 파싱에 실패했습니다 ({last_err}). "
                "기존 데이터를 빈 리스트([])로 덮어쓰는 데이터 증발을 방지하기 위해 중단합니다."
            )

    def export_organized_exam_files(
        self,
        records: list[QuestionRecord],
        target_exams: set[str] | None = None,
    ) -> list[Path]:
        """과목별 · 모의고사 회차별 폴더에 JSON / Markdown / HTML 정리 파일을 생성한다."""
        with self._lock:
            grouped: dict[tuple[str, str], list[QuestionRecord]] = {}
            for rec in records:
                if target_exams is not None and rec.exam_name not in target_exams:
                    continue
                grouped.setdefault((rec.subject_category, rec.exam_name), []).append(rec)

            written_dirs: list[Path] = []
            for (subj, exam), group_recs in grouped.items():
                ordered = sorted(group_recs, key=lambda r: r.q_num)
                exam_dir = self._local_exam_dir(subj, exam)

                json_bytes = json.dumps(
                    [r.model_dump() for r in ordered],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                ai_jsonl_bytes = build_exam_ai_jsonl(ordered).encode("utf-8")
                md_bytes = build_exam_markdown(exam, subj, ordered).encode("utf-8")
                html_bytes = build_exam_html(
                    exam,
                    subj,
                    ordered,
                    crop_loader=lambda fn, s=subj, e=exam: self.get_crop_bytes(
                        fn, subject_category=s, exam_name=e
                    ),
                ).encode("utf-8")

                (exam_dir / "문제목록.json").write_bytes(json_bytes)
                (exam_dir / "ai_index.jsonl").write_bytes(ai_jsonl_bytes)
                (exam_dir / "기출정리.md").write_bytes(md_bytes)
                (exam_dir / "기출정리.html").write_bytes(html_bytes)
                written_dirs.append(exam_dir)

            return written_dirs

    def save_db(
        self,
        records: list[QuestionRecord],
        target_exams: set[str] | None = None,
        sync_local_organized: bool = True,
    ) -> str | None:
        with self._lock:
            # 방어 로직: 기존 파일이 존재하고 유효한 데이터가 있는데 빈 레코드로 덮어쓰려는 경우 방어
            if not records and self.db_path.exists() and self.db_path.stat().st_size > 2:
                raise ValueError("기존 db.json 데이터 증발 방지: 빈 레코드 목록으로 저장을 시도할 수 없습니다.")

            payload = [rec.model_dump() for rec in records]
            raw_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            _atomic_write_text(self.db_path, raw_text, encoding="utf-8")
            new_sig = _file_stat_sig(self.db_path)
            if new_sig is not None:
                with _GLOBAL_CACHE_LOCK:
                    _GLOBAL_JSON_CACHE[f"db::{self.db_path.resolve()}"] = (
                        new_sig,
                        [rec.model_copy(deep=True) for rec in records],
                    )
                    _GLOBAL_STORAGE_STATS_CACHE.clear()

            errors: list[str] = []
            if sync_local_organized:
                try:
                    self.export_organized_exam_files(
                        records,
                        target_exams=target_exams,
                    )
                except Exception as exc:
                    errors.append(f"과목·회차별 정리 파일 생성 오류: {exc}")

            return " / ".join(errors) if errors else None

    def load_users(self) -> list[UserRecord]:
        with self._lock:
            if not self.users_path.exists():
                return []
            sig = _file_stat_sig(self.users_path)
            if sig is None or sig[1] == 0:
                return []
            cache_key = f"users::{self.users_path.resolve()}"
            with _GLOBAL_CACHE_LOCK:
                cached = _GLOBAL_JSON_CACHE.get(cache_key)
                if cached is not None and cached[0] == sig:
                    return [u.model_copy(deep=True) for u in cached[1]]

            last_err: Exception | None = None
            for attempt in range(3):
                try:
                    text = self.users_path.read_text(encoding="utf-8")
                    if not text.strip():
                        time.sleep(0.05)
                        continue
                    raw = json.loads(text)
                    if not isinstance(raw, list):
                        raise ValueError(f"users.json 루트가 리스트가 아닙니다 (type: {type(raw)})")
                    needs_migration = any(
                        isinstance(item, dict)
                        and (
                            str(item.get("tier", "")).strip().upper() in LEGACY_TIER_MAP
                            or "custom_api_key" not in item
                        )
                        for item in raw
                    )
                    users = [UserRecord.model_validate(item) for item in raw]
                    if needs_migration:
                        self.save_users(users)
                    new_sig = _file_stat_sig(self.users_path) or sig
                    with _GLOBAL_CACHE_LOCK:
                        _GLOBAL_JSON_CACHE[cache_key] = (new_sig, users)
                    return [u.model_copy(deep=True) for u in users]
                except Exception as exc:
                    last_err = exc
                    if attempt < 2:
                        time.sleep(0.05)

            # 끝까지 실패 시 기존 파일을 빈 리스트로 덮어쓰지 않도록 방어 로직
            raise RuntimeError(
                f"{self.users_path} 파일이 존재하지만 파싱에 실패했습니다 ({last_err}). "
                "기존 사용자 목록을 빈 리스트([])로 덮어쓰는 데이터 증발을 방지하기 위해 중단합니다."
            )

    def save_users(self, users: list[UserRecord]) -> str | None:
        with self._lock:
            if not users and self.users_path.exists() and self.users_path.stat().st_size > 2:
                raise ValueError("기존 users.json 데이터 증발 방지: 빈 사용자 목록으로 저장을 시도할 수 없습니다.")

            payload = [u.model_dump() for u in users]
            raw_text = json.dumps(payload, ensure_ascii=False, indent=2)
            _atomic_write_text(self.users_path, raw_text, encoding="utf-8")
            new_sig = _file_stat_sig(self.users_path)
            if new_sig is not None:
                with _GLOBAL_CACHE_LOCK:
                    _GLOBAL_JSON_CACHE[f"users::{self.users_path.resolve()}"] = (
                        new_sig,
                        [u.model_copy(deep=True) for u in users],
                    )
            return None

    def load_concept_books(self) -> list[ConceptBookRecord]:
        """로컬 ./concept_db 에 저장된 공용 개념서·교과서 DB 목록을 불러온다."""
        with self._lock:
            dir_sig = _dir_json_stat_sig(self.concept_root, self.concept_books_path)
            cache_key = f"cbooks::{self.concept_root.resolve()}"
            with _GLOBAL_CACHE_LOCK:
                cached = _GLOBAL_JSON_CACHE.get(cache_key)
                if cached is not None and cached[0] == dir_sig:
                    return [b.model_copy(deep=True) for b in cached[1]]

            books_map: dict[str, ConceptBookRecord] = {}
            if self.concept_books_path.exists():
                try:
                    raw = json.loads(self.concept_books_path.read_text(encoding="utf-8"))
                    if isinstance(raw, list):
                        for item in raw:
                            b = ConceptBookRecord.model_validate(item)
                            books_map[b.id] = b
                except Exception:
                    pass

            # 개별 JSON 파일(./concept_db/*.json)도 스캔해 병합
            if self.concept_root.exists():
                for child in sorted(self.concept_root.glob("*.json")):
                    if child.name == "concept_books.json":
                        continue
                    try:
                        item = json.loads(child.read_text(encoding="utf-8"))
                        if isinstance(item, dict) and "book_title" in item and "units" in item:
                            if "id" not in item:
                                item["id"] = child.stem
                            b = ConceptBookRecord.model_validate(item)
                            books_map.setdefault(b.id, b)
                    except Exception:
                        continue

            ordered_books = sorted(books_map.values(), key=lambda x: (x.subject, x.book_title))
            new_sig = _dir_json_stat_sig(self.concept_root, self.concept_books_path)
            with _GLOBAL_CACHE_LOCK:
                _GLOBAL_JSON_CACHE[cache_key] = (new_sig, ordered_books)
            return [b.model_copy(deep=True) for b in ordered_books]

    def save_concept_book(self, book: ConceptBookRecord) -> str | None:
        """단원별 구조화된 개념서 레코드를 로컬(./concept_db)에 영구 저장한다."""
        with self._lock:
            self.concept_root.mkdir(parents=True, exist_ok=True)
            existing = {b.id: b for b in self.load_concept_books()}
            existing[book.id] = book
            all_books = sorted(existing.values(), key=lambda x: (x.subject, x.book_title))

            # 1) 통합 인덱스(./concept_db/concept_books.json) 저장 (원자적 교체)
            all_payload = [b.model_dump() for b in all_books]
            all_raw_text = json.dumps(all_payload, ensure_ascii=False, indent=2)
            _atomic_write_text(self.concept_books_path, all_raw_text, encoding="utf-8")

            # 2) 개별 개념서 JSON 파일(./concept_db/<safe_id>.json) 저장 (원자적 교체)
            book_filename = f"{_safe_segment(book.id)}.json"
            single_payload = book.model_dump()
            single_raw_text = json.dumps(single_payload, ensure_ascii=False, indent=2)
            _atomic_write_text(self.concept_root / book_filename, single_raw_text, encoding="utf-8")
            return None

    def delete_concept_book(self, book_id: str) -> str | None:
        """공용 개념서 DB에서 특정 개념서를 삭제한다."""
        with self._lock:
            remaining = [b for b in self.load_concept_books() if b.id != book_id]
            all_raw_text = json.dumps([b.model_dump() for b in remaining], ensure_ascii=False, indent=2)
            _atomic_write_text(self.concept_books_path, all_raw_text, encoding="utf-8")

            book_filename = f"{_safe_segment(book_id)}.json"
            (self.concept_root / book_filename).unlink(missing_ok=True)
            return None

    def save_crop_png(
        self,
        filename: str,
        png_bytes: bytes,
        subject_category: str = "",
        exam_name: str = "",
    ) -> str | None:
        with self._lock:
            (self.crops_dir / filename).write_bytes(png_bytes)
            if subject_category and exam_name:
                (self._local_exam_dir(subject_category, exam_name) / filename).write_bytes(png_bytes)
            return None

    def save_pdf_file(
        self,
        filename: str,
        pdf_bytes: bytes,
        subject_category: str = "",
        exam_name: str = "",
    ) -> str | None:
        with self._lock:
            (self.pdfs_dir / filename).write_bytes(pdf_bytes)
            if subject_category and exam_name:
                (self._local_exam_dir(subject_category, exam_name) / filename).write_bytes(pdf_bytes)
            return None

    def _resolve_exam_meta_for_file(
        self,
        filename: str,
        subject_category: str = "",
        exam_name: str = "",
    ) -> tuple[str, str]:
        with self._lock:
            if subject_category and exam_name:
                return subject_category, exam_name
            for rec in self.load_db():
                if filename in (rec.image_filename, rec.source_pdf, rec.answer_pdf):
                    return rec.subject_category, rec.exam_name
            return subject_category, exam_name

    def get_crop_bytes(
        self,
        filename: str,
        subject_category: str = "",
        exam_name: str = "",
    ) -> bytes | None:
        with self._lock:
            if not filename:
                return None
            local_path = self.crops_dir / filename
            if local_path.exists():
                return local_path.read_bytes()

            subj, exam = self._resolve_exam_meta_for_file(filename, subject_category, exam_name)
            if subj and exam:
                org_path = self.organized_dir / _safe_segment(subj) / _safe_segment(exam) / filename
                if org_path.exists():
                    data = org_path.read_bytes()
                    local_path.write_bytes(data)
                    return data
            return None

    def get_pdf_bytes(
        self,
        filename: str,
        subject_category: str = "",
        exam_name: str = "",
    ) -> bytes | None:
        with self._lock:
            if not filename:
                return None
            local_path = self.pdfs_dir / filename
            if local_path.exists():
                return local_path.read_bytes()

            subj, exam = self._resolve_exam_meta_for_file(filename, subject_category, exam_name)
            if subj and exam:
                org_path = self.organized_dir / _safe_segment(subj) / _safe_segment(exam) / filename
                if org_path.exists():
                    data = org_path.read_bytes()
                    local_path.write_bytes(data)
                    return data
            return None

    def delete_question(self, question_id: str) -> None:
        with self._lock:
            records = self.load_db()
            target_rec = next((r for r in records if r.id == question_id), None)
            remaining = [r for r in records if r.id != question_id]

            if target_rec and target_rec.image_filename:
                (self.crops_dir / target_rec.image_filename).unlink(missing_ok=True)
                org_img = (
                    self.organized_dir
                    / _safe_segment(target_rec.subject_category)
                    / _safe_segment(target_rec.exam_name)
                    / target_rec.image_filename
                )
                org_img.unlink(missing_ok=True)

            target_exams = {target_rec.exam_name} if target_rec else None
            self.save_db(remaining, target_exams=target_exams)

    def delete_exam(self, exam_name: str) -> int:
        with self._lock:
            records = self.load_db()
            to_delete = [r for r in records if r.exam_name == exam_name]
            remaining = [r for r in records if r.exam_name != exam_name]
            subj_cats = {r.subject_category for r in to_delete}
            pdf_names = {n for r in to_delete for n in (r.source_pdf, r.answer_pdf) if n}

            for rec in to_delete:
                if rec.image_filename:
                    (self.crops_dir / rec.image_filename).unlink(missing_ok=True)

            for pdf_fn in pdf_names:
                (self.pdfs_dir / pdf_fn).unlink(missing_ok=True)

            for subj in subj_cats:
                org_exam_dir = self.organized_dir / _safe_segment(subj) / _safe_segment(exam_name)
                if org_exam_dir.exists():
                    shutil.rmtree(org_exam_dir, ignore_errors=True)

            self.save_db(remaining)
            self.delete_korean_analyses_by_exam(exam_name)
            return len(to_delete)

    def rename_exam(self, old_exam_name: str, new_exam_name: str) -> int:
        """기존 DB에 저장된 회차명을 일괄 변경하고 로컬 정리 폴더도 이동한다."""
        old_n = (old_exam_name or "").strip()
        new_n = re.sub(r"\s+", " ", (new_exam_name or "").strip())
        if not old_n or not new_n or old_n == new_n:
            return 0
        with self._lock:
            records = self.load_db()
            changed = 0
            subj_cats: set[str] = set()
            for idx, rec in enumerate(records):
                if rec.exam_name == old_n:
                    subj_cats.add(rec.subject_category)
                    records[idx] = rec.model_copy(update={"exam_name": new_n})
                    changed += 1
            if changed == 0:
                return 0
            for subj in subj_cats:
                old_dir = self.organized_dir / _safe_segment(subj) / _safe_segment(old_n)
                new_dir = self.organized_dir / _safe_segment(subj) / _safe_segment(new_n)
                if old_dir.exists() and not new_dir.exists():
                    try:
                        new_dir.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(old_dir), str(new_dir))
                    except Exception:
                        pass
            self.save_db(records)
            self.rename_korean_analyses_exam(old_n, new_n)
            return changed

    def reset_exam_database(self, delete_users: bool = False) -> dict[str, Any]:
        """로컬 영구 디스크에 누적된 기출 DB·이미지·PDF 파일을 모두 삭제(초기화)한다."""
        with self._lock:
            removed_local = 0
            for folder in (self.crops_dir, self.pdfs_dir, self.korean_analysis_pdfs_root):
                if folder.exists():
                    for child in folder.iterdir():
                        if child.is_file():
                            child.unlink(missing_ok=True)
                            removed_local += 1
                        elif child.is_dir():
                            shutil.rmtree(child, ignore_errors=True)
                            removed_local += 1

            if self.organized_dir.exists():
                for child in list(self.organized_dir.iterdir()):
                    shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink(missing_ok=True)
                    removed_local += 1

            _atomic_write_text(self.db_path, "[]", encoding="utf-8")
            if self.english_pool_path.exists():
                _atomic_write_text(self.english_pool_path, "{}", encoding="utf-8")
            if self.korean_analysis_pool_path.exists():
                _atomic_write_text(self.korean_analysis_pool_path, "{}", encoding="utf-8")
            if delete_users and self.users_path.exists():
                self.users_path.unlink(missing_ok=True)

            return {
                "removed_local": removed_local,
            }

    @staticmethod
    def build_english_pool_key(
        source_passage: str,
        target_type: str,
        difficulty: str = "",
        blank_unit: str = "",
        vocab_style: str = "",
    ) -> str:
        """지문 원문과 변형 옵션 조합으로 영어 변형 풀(Pool) 캐시 키를 생성한다."""
        norm_p = re.sub(r"\s+", " ", (source_passage or "").strip())
        t = (target_type or "").strip()
        opt = blank_unit if "빈칸" in t else (vocab_style if "어휘" in t else "")
        raw_key = f"{norm_p}||{t}||{(difficulty or '').strip()}||{(opt or '').strip()}"
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
        try:
            data = json.loads(self.user_english_history_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_user_english_history_unlocked(self, history: dict[str, list[str]]) -> None:
        raw = json.dumps(history, ensure_ascii=False, indent=2)
        _atomic_write_text(self.user_english_history_path, raw, encoding="utf-8")

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
            try:
                return RewrittenQuestion.model_validate(chosen)
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

    # -----------------------------------------------------------------------
    # 계정별 나의 시험지 자동 저장 / 조회 / PDF / 삭제
    # -----------------------------------------------------------------------
    def _user_saved_exams_path(self, username: str) -> Path:
        safe_user = re.sub(r"[^a-zA-Z0-9._-]+", "_", (username or "").strip()) or "anonymous"
        return self.saved_exams_root / f"{safe_user}.json"

    def _load_user_exams_unlocked(self, username: str) -> list[dict[str, Any]]:
        p = self._user_saved_exams_path(username)
        if not p.exists():
            return []
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save_user_exams_unlocked(self, username: str, exams: list[dict[str, Any]]) -> None:
        self.saved_exams_root.mkdir(parents=True, exist_ok=True)
        p = self._user_saved_exams_path(username)
        _atomic_write_text(p, json.dumps(exams, ensure_ascii=False, indent=2), encoding="utf-8")

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

    # -----------------------------------------------------------------------
    # 서버 디스크 및 앱 저장공간 통계 (관리자 모니터링용)
    # -----------------------------------------------------------------------
    def get_server_storage_stats(self, force_refresh: bool = False) -> dict[str, Any]:
        """서버 디스크 전체/사용/여유 용량과 앱 DB 폴더별 파일 수 및 용량을 반환한다 (60초 TTL 캐시 적용)."""
        mount_path = self.local_root.resolve()
        cache_key = str(mount_path)
        now_ts = time.monotonic()
        if not force_refresh:
            with _GLOBAL_CACHE_LOCK:
                cached_stats = _GLOBAL_STORAGE_STATS_CACHE.get(cache_key)
                if cached_stats is not None and (now_ts - cached_stats[0]) < STORAGE_STATS_TTL_SECONDS:
                    return dict(cached_stats[1])
        try:
            usage = shutil.disk_usage(mount_path)
            disk_total = int(usage.total)
            disk_used = int(usage.used)
            disk_free = int(usage.free)
        except OSError:
            disk_total = 0
            disk_used = 0
            disk_free = 0
        disk_used_pct = round((disk_used / disk_total) * 100.0, 1) if disk_total > 0 else 0.0

        def _dir_stats(dir_path: Path) -> tuple[int, int]:
            if not dir_path.exists():
                return 0, 0
            total_b = 0
            file_cnt = 0
            try:
                for p in dir_path.rglob("*"):
                    try:
                        if p.is_file() and not p.is_symlink():
                            total_b += int(p.stat().st_size)
                            file_cnt += 1
                    except OSError:
                        continue
            except OSError:
                pass
            return total_b, file_cnt

        def _root_files_stats(dir_path: Path) -> tuple[int, int]:
            if not dir_path.exists():
                return 0, 0
            total_b = 0
            file_cnt = 0
            try:
                for p in dir_path.iterdir():
                    try:
                        if p.is_file() and not p.is_symlink():
                            total_b += int(p.stat().st_size)
                            file_cnt += 1
                    except OSError:
                        continue
            except OSError:
                pass
            return total_b, file_cnt

        specs: list[tuple[str, str, tuple[int, int]]] = [
            ("기출 원본 PDF", str(self.pdfs_dir), _dir_stats(self.pdfs_dir)),
            ("크롭 문항 이미지", str(self.crops_dir), _dir_stats(self.crops_dir)),
            ("과목별 정리 파일", str(self.organized_dir), _dir_stats(self.organized_dir)),
            ("국어 지문 분석지 PDF", str(self.korean_analysis_pdfs_root), _dir_stats(self.korean_analysis_pdfs_root)),
            ("나의 시험지 보관함", str(self.saved_exams_root), _dir_stats(self.saved_exams_root)),
            ("개인 영단어장", str(self.user_vocab_root), _dir_stats(self.user_vocab_root)),
            ("DB JSON · 캐시 파일", str(self.local_root), _root_files_stats(self.local_root)),
            ("공용 개념서 DB", str(self.concept_root), _dir_stats(self.concept_root)),
        ]

        breakdown = [
            {
                "label": label,
                "path": path_str,
                "bytes": int(b_cnt),
                "files": int(f_cnt),
            }
            for label, path_str, (b_cnt, f_cnt) in specs
        ]
        app_total_bytes = sum(item["bytes"] for item in breakdown)
        app_total_files = sum(item["files"] for item in breakdown)

        stats_out = {
            "mount_path": str(mount_path),
            "disk_total_bytes": disk_total,
            "disk_used_bytes": disk_used,
            "disk_free_bytes": disk_free,
            "disk_used_percent": disk_used_pct,
            "app_total_bytes": app_total_bytes,
            "app_total_files": app_total_files,
            "breakdown": breakdown,
        }
        with _GLOBAL_CACHE_LOCK:
            _GLOBAL_STORAGE_STATS_CACHE[cache_key] = (now_ts, stats_out)
        return dict(stats_out)

    # -----------------------------------------------------------------------
    # 국어 정밀 지문 분석지 캐시 풀(Pool) 영구 저장 / 조회 / 관리
    # -----------------------------------------------------------------------
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
    # 공용 영단어 사전 캐시 DB (data/csat_vocab_seed.json + english_dict_cache.json)
    # -----------------------------------------------------------------------
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

    def _load_english_dict_cache_unlocked(self) -> dict[str, Any]:
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
                return {
                    "entries": {k: dict(v) for k, v in cached[1]["entries"].items()},
                    "aliases": dict(cached[1]["aliases"]),
                }

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
        return {
            "entries": {k: dict(v) for k, v in entries.items()},
            "aliases": dict(aliases),
        }

    def _save_english_dict_cache_unlocked(self, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False, indent=2)
        _atomic_write_text(self.english_dict_cache_path, raw, encoding="utf-8")
        with _GLOBAL_CACHE_LOCK:
            _GLOBAL_JSON_CACHE.pop(f"eng_dict::{self.english_dict_cache_path.resolve()}", None)

    def get_cached_english_dict_entry(self, query: str) -> EnglishDictEntry | None:
        """단어·숙어·한글 검색어에 해당하는 캐시된 영단어 사전 엔트리를 반환한다."""
        q_norm = normalize_english_dict_key(query)
        if not q_norm:
            return None
        with self._lock:
            payload = self._load_english_dict_cache_unlocked()
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
        with self._lock:
            payload = self._load_english_dict_cache_unlocked()
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
            payload = self._load_english_dict_cache_unlocked()
            return len(payload["entries"])

    def delete_cached_english_dict_entry(self, key: str) -> bool:
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
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return [dict(it) for it in data if isinstance(it, dict)] if isinstance(data, list) else []
        except Exception:
            return []

    def _save_user_vocab_unlocked(self, username: str, rows: list[dict[str, Any]]) -> None:
        self.user_vocab_root.mkdir(parents=True, exist_ok=True)
        p = self._user_vocab_path(username)
        _atomic_write_text(p, json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_user_vocab(self, username: str) -> list[UserVocabItem]:
        """특정 회원의 나만의 단어장 목록을 로드한다."""
        uname = (username or "").strip()
        if not uname:
            return []
        with self._lock:
            rows = self._load_user_vocab_unlocked(uname)
        items: list[UserVocabItem] = []
        for row in rows:
            try:
                items.append(UserVocabItem.model_validate(row))
            except Exception:
                continue
        return items

    def save_user_vocab_item(self, username: str, item: UserVocabItem) -> list[UserVocabItem]:
        """단일 영단어를 개인 단어장에 추가 또는 갱신(기존 암기/오답/메모 보존)하고 최신 목록을 반환한다."""
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

