from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.storage.base import (
    _GLOBAL_CACHE_LOCK,
    _GLOBAL_JSON_CACHE,
    _GLOBAL_STORAGE_STATS_CACHE,
    STORAGE_STATS_TTL_SECONDS,
    _atomic_write_bytes,
    _atomic_write_text,
    _dir_json_stat_sig,
    _file_stat_sig,
    _safe_segment,
)

if TYPE_CHECKING:
    from src.models import ConceptBookRecord, QuestionRecord, UserRecord


class ExamStoreMixin:
    def _load_db_ref_unlocked(self) -> tuple[QuestionRecord, ...]:
        """db.json을 mtime/size 캐시에서 읽어 불변 튜플 참조로 반환한다 (deep-copy 없음)."""
        from src.models.question import QuestionRecord

        if not self.db_path.exists():
            return ()
        sig = _file_stat_sig(self.db_path)
        if sig is None or sig[1] == 0:
            return ()
        cache_key = f"db::{self.db_path.resolve()}"
        with _GLOBAL_CACHE_LOCK:
            cached = _GLOBAL_JSON_CACHE.get(cache_key)
            if cached is not None and cached[0] == sig:
                return cached[1]

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
                records = tuple(QuestionRecord.model_validate(item) for item in raw)
                new_sig = _file_stat_sig(self.db_path) or sig
                with _GLOBAL_CACHE_LOCK:
                    _GLOBAL_JSON_CACHE[cache_key] = (new_sig, records)
                return records
            except Exception as exc:
                last_err = exc
                if attempt < 2:
                    time.sleep(0.05)

        raise RuntimeError(
            f"{self.db_path} 파일이 존재하지만 파싱에 실패했습니다 ({last_err}). "
            "기존 데이터를 빈 리스트([])로 덮어쓰는 데이터 증발을 방지하기 위해 중단합니다."
        )

    def load_db_readonly(self) -> tuple[QuestionRecord, ...]:
        """읽기 전용 화면(목록/필터/대시보드)을 위해 deep-copy 없이 캐시된 기출 문항 튜플을 반환한다."""
        with self._lock:
            return self._load_db_ref_unlocked()

    def load_db(self) -> list[QuestionRecord]:
        with self._lock:
            return [rec.model_copy(deep=True) for rec in self._load_db_ref_unlocked()]

    def export_organized_exam_files(
        self,
        records: list[QuestionRecord],
        target_exams: set[str] | None = None,
    ) -> list[Path]:
        """과목별 · 모의고사 회차별 폴더에 JSON / Markdown / HTML 정리 파일을 생성한다."""
        from src.exporters import build_exam_ai_jsonl, build_exam_html, build_exam_markdown

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
                        tuple(rec.model_copy(deep=True) for rec in records),
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
        from src.models.user import LEGACY_TIER_MAP, UserRecord

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

    def _load_concept_books_ref_unlocked(self) -> tuple[ConceptBookRecord, ...]:
        """./concept_db 공용 개념서 목록을 mtime/size 캐시에서 불변 튜플 참조로 반환한다."""
        from src.models.concept import ConceptBookRecord

        dir_sig = _dir_json_stat_sig(self.concept_root, self.concept_books_path)
        cache_key = f"cbooks::{self.concept_root.resolve()}"
        with _GLOBAL_CACHE_LOCK:
            cached = _GLOBAL_JSON_CACHE.get(cache_key)
            if cached is not None and cached[0] == dir_sig:
                return cached[1]

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

        ordered_books = tuple(sorted(books_map.values(), key=lambda x: (x.subject, x.book_title)))
        new_sig = _dir_json_stat_sig(self.concept_root, self.concept_books_path)
        with _GLOBAL_CACHE_LOCK:
            _GLOBAL_JSON_CACHE[cache_key] = (new_sig, ordered_books)
        return ordered_books

    def load_concept_books(self) -> list[ConceptBookRecord]:
        """로컬 ./concept_db 에 저장된 공용 개념서·교과서 DB 목록을 불러온다."""
        with self._lock:
            return [b.model_copy(deep=True) for b in self._load_concept_books_ref_unlocked()]

    def get_db_summary_counts(self) -> tuple[int, int, int]:
        """사이드바 요약용 (기출 문항 수, 등록 회차 수, 공용 개념서 수)를 deep-copy 없이 O(1) 캐시로 반환한다."""
        with self._lock:
            records = self._load_db_ref_unlocked()
            books = self._load_concept_books_ref_unlocked()
            unique_exams = {r.exam_name for r in records if r.exam_name}
            return len(records), len(unique_exams), len(books)

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
            self.delete_english_analyses_by_exam(exam_name)
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
            self.rename_english_analyses_exam(old_n, new_n)
            return changed

    def reset_exam_database(self, delete_users: bool = False) -> dict[str, Any]:
        """로컬 영구 디스크에 누적된 기출 DB·이미지·PDF 파일을 모두 삭제(초기화)한다."""
        with self._lock:
            removed_local = 0
            for folder in (
                self.crops_dir,
                self.pdfs_dir,
                self.korean_analysis_pdfs_root,
                self.english_analysis_pdfs_root,
            ):
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
            if self.english_analysis_pool_path.exists():
                _atomic_write_text(self.english_analysis_pool_path, "{}", encoding="utf-8")
            if self.korean_analysis_pool_path.exists():
                _atomic_write_text(self.korean_analysis_pool_path, "{}", encoding="utf-8")
            if delete_users and self.users_path.exists():
                self.users_path.unlink(missing_ok=True)

            return {
                "removed_local": removed_local,
            }
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
            ("영어 지문 분석지 PDF", str(self.english_analysis_pdfs_root), _dir_stats(self.english_analysis_pdfs_root)),
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
