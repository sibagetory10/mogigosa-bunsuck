from __future__ import annotations

import hashlib
import io
import json
import os
import random
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials as OAuthCredentials
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

from src.exporters import build_exam_ai_jsonl, build_exam_html, build_exam_markdown
from src.models import ConceptBookRecord, LEGACY_TIER_MAP, QuestionRecord, RewrittenQuestion, UserRecord

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]
FOLDER_MIME = "application/vnd.google-apps.folder"
KNOWN_SUBJECT_FOLDERS = ("수학", "사회", "과학", "영어")


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
    """Google Drive(OAuth 2.0 개인 드라이브 / 서비스 계정 공유 드라이브) 및 로컬 ./exam_db 폴백 영구 저장소 관리자."""

    def __init__(
        self,
        local_root: str | Path = "./exam_db",
        service_account_info: dict[str, Any] | None = None,
        drive_folder_id: str = "",
        oauth_info: dict[str, Any] | None = None,
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
        self.user_english_history_path = self.local_root / "user_english_history.json"
        self.saved_exams_root = self.local_root / "saved_exams"
        self.saved_exams_pdfs_root = self.saved_exams_root / "pdfs"

        self.concept_root = Path(concept_root)
        self.concept_books_path = self.concept_root / "concept_books.json"

        self.local_root.mkdir(parents=True, exist_ok=True)
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.organized_dir.mkdir(parents=True, exist_ok=True)
        self.saved_exams_root.mkdir(parents=True, exist_ok=True)
        self.saved_exams_pdfs_root.mkdir(parents=True, exist_ok=True)
        self.concept_root.mkdir(parents=True, exist_ok=True)

        self.drive_folder_id = drive_folder_id.strip()
        self.service_account_info = service_account_info
        self.oauth_info = oauth_info
        self.drive_auth_mode: str = ""
        self.drive_service: Any | None = None
        self.drive_connected: bool = False
        self.drive_folder_name: str = ""
        self.drive_error: str = ""

        self._subfolder_ids: dict[tuple[str, str], str] = {}

        if (self.oauth_info or self.service_account_info) and self.drive_folder_id:
            self._init_drive_client()

    def _init_drive_client(self) -> None:
        with self._lock:
            try:
                if self.oauth_info:
                    creds = OAuthCredentials(
                        token=None,
                        refresh_token=self.oauth_info["refresh_token"],
                        token_uri=self.oauth_info.get("token_uri", "https://oauth2.googleapis.com/token"),
                        client_id=self.oauth_info["client_id"],
                        client_secret=self.oauth_info["client_secret"],
                        scopes=DRIVE_SCOPES,
                    )
                    if not creds.valid and creds.refresh_token:
                        creds.refresh(GoogleAuthRequest())
                    self.drive_auth_mode = "oauth"
                else:
                    creds = ServiceAccountCredentials.from_service_account_info(
                        self.service_account_info,
                        scopes=DRIVE_SCOPES,
                    )
                    self.drive_auth_mode = "service_account"
                service = build("drive", "v3", credentials=creds, cache_discovery=False)
                meta = (
                    service.files()
                    .get(
                        fileId=self.drive_folder_id,
                        fields="id, name, mimeType, driveId",
                        supportsAllDrives=True,
                    )
                    .execute()
                )
                if meta.get("mimeType") != FOLDER_MIME:
                    self.drive_error = f"지정한 ID({self.drive_folder_id})가 폴더가 아닙니다."
                    return

                self.drive_service = service
                self.drive_connected = True
                self.drive_folder_name = meta.get("name", self.drive_folder_id)
                self.drive_error = ""
            except Exception as exc:
                self.drive_service = None
                self.drive_connected = False
                self.drive_error = str(exc)

    def _list_children(self, parent_id: str, extra_query: str = "") -> list[dict[str, Any]]:
        """지정 폴더 하위 항목을 페이지네이션까지 모두 조회한다."""
        with self._lock:
            if not self.drive_service:
                return []

            q = f"'{parent_id}' in parents and trashed = false"
            if extra_query:
                q = f"{q} and {extra_query}"

            items: list[dict[str, Any]] = []
            page_token: str | None = None
            while True:
                resp = (
                    self.drive_service.files()
                    .list(
                        q=q,
                        fields="nextPageToken, files(id, name, mimeType, size)",
                        pageSize=200,
                        pageToken=page_token,
                        supportsAllDrives=True,
                        includeItemsFromAllDrives=True,
                    )
                    .execute()
                )
                items.extend(resp.get("files", []))
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break
            return items

    def _find_drive_subfolder(self, folder_name: str, parent_id: str | None = None) -> str | None:
        with self._lock:
            if not self.drive_connected or not self.drive_service:
                return None
            pid = parent_id or self.drive_folder_id
            cache_key = (pid, folder_name)
            if cache_key in self._subfolder_ids:
                return self._subfolder_ids[cache_key]

            safe_name = folder_name.replace("'", "\\'")
            found = self._list_children(
                pid,
                extra_query=f"mimeType = '{FOLDER_MIME}' and name = '{safe_name}'",
            )
            if found:
                fid = found[0]["id"]
                self._subfolder_ids[cache_key] = fid
                return fid
            return None

    def _ensure_drive_subfolder(self, folder_name: str, parent_id: str | None = None) -> str | None:
        with self._lock:
            if not self.drive_connected or not self.drive_service:
                return None
            pid = parent_id or self.drive_folder_id
            existing_id = self._find_drive_subfolder(folder_name, parent_id=pid)
            if existing_id:
                return existing_id

            created = (
                self.drive_service.files()
                .create(
                    body={
                        "name": folder_name,
                        "mimeType": FOLDER_MIME,
                        "parents": [pid],
                    },
                    fields="id",
                    supportsAllDrives=True,
                )
                .execute()
            )
            fid = created["id"]
            self._subfolder_ids[(pid, folder_name)] = fid
            return fid

    def _ensure_drive_exam_folder(self, subject_category: str, exam_name: str) -> str | None:
        with self._lock:
            subj_id = self._ensure_drive_subfolder(_safe_segment(subject_category))
            if not subj_id:
                return None
            return self._ensure_drive_subfolder(_safe_segment(exam_name), parent_id=subj_id)

    def _find_drive_exam_folder(self, subject_category: str, exam_name: str) -> str | None:
        with self._lock:
            subj_id = self._find_drive_subfolder(_safe_segment(subject_category))
            if not subj_id:
                return None
            return self._find_drive_subfolder(_safe_segment(exam_name), parent_id=subj_id)

    def _local_exam_dir(self, subject_category: str, exam_name: str) -> Path:
        with self._lock:
            target = self.organized_dir / _safe_segment(subject_category) / _safe_segment(exam_name)
            target.mkdir(parents=True, exist_ok=True)
            return target

    def _trash_drive_item(self, file_id: str) -> None:
        with self._lock:
            if not self.drive_connected or not self.drive_service:
                return
            self.drive_service.files().update(
                fileId=file_id,
                body={"trashed": True},
                supportsAllDrives=True,
            ).execute()

    def _download_file_bytes(self, file_id: str) -> bytes:
        with self._lock:
            assert self.drive_service is not None
            request = self.drive_service.files().get_media(
                fileId=file_id,
                supportsAllDrives=True,
            )
            buf = io.BytesIO()
            downloader = MediaIoBaseDownload(buf, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            return buf.getvalue()

    def _upsert_drive_file(
        self,
        parent_id: str,
        filename: str,
        data: bytes,
        mime_type: str,
        known_existing_id: str | None = None,
        skip_lookup: bool = False,
    ) -> str | None:
        with self._lock:
            if not self.drive_connected or not self.drive_service:
                return None

            file_id = known_existing_id
            if not file_id and not skip_lookup:
                safe_name = filename.replace("'", "\\'")
                existing = self._list_children(
                    parent_id,
                    extra_query=f"mimeType != '{FOLDER_MIME}' and name = '{safe_name}'",
                )
                if existing:
                    file_id = existing[0]["id"]

            media = MediaIoBaseUpload(
                io.BytesIO(data),
                mimetype=mime_type,
                resumable=len(data) > 512 * 1024,
            )
            if file_id:
                self.drive_service.files().update(
                    fileId=file_id,
                    media_body=media,
                    supportsAllDrives=True,
                ).execute()
                return file_id

            created = (
                self.drive_service.files()
                .create(
                    body={"name": filename, "parents": [parent_id]},
                    media_body=media,
                    fields="id",
                    supportsAllDrives=True,
                )
                .execute()
            )
            return created.get("id")

    def sync_on_startup(self, sync_media: bool = False) -> dict[str, Any]:
        """앱 시작 시 Google Drive의 db.json, users.json, 개념서 DB를 로컬과 동기화한다.

        기본값(sync_media=False)에서는 메타데이터 JSON만 단일 루트 조회로 빠르게 동기화하며,
        개별 PNG/PDF는 get_crop_bytes/get_pdf_bytes 호출 시 온디맨드로 내려받는다.
        """
        with self._lock:
            result: dict[str, Any] = {
                "connected": self.drive_connected,
                "folder_name": self.drive_folder_name,
                "db_synced": False,
                "users_synced": False,
                "crops_downloaded": 0,
                "pdfs_downloaded": 0,
                "error": self.drive_error,
            }
            if not self.drive_connected or not self.drive_service:
                return result

            try:
                # 루트 폴더를 1회만 조회해 불필요한 반복 API 왕복을 제거
                root_children = self._list_children(self.drive_folder_id)
                root_files_by_name: dict[str, dict[str, Any]] = {}
                root_folders: list[dict[str, Any]] = []
                root_folders_by_name: dict[str, dict[str, Any]] = {}
                for item in root_children:
                    if item.get("mimeType") == FOLDER_MIME:
                        root_folders.append(item)
                        root_folders_by_name[item["name"]] = item
                    else:
                        root_files_by_name[item["name"]] = item

                # 1) db.json 동기화
                if "db.json" in root_files_by_name:
                    db_bytes = self._download_file_bytes(root_files_by_name["db.json"]["id"])
                    _atomic_write_bytes(self.db_path, db_bytes)
                    result["db_synced"] = True
                elif self.db_path.exists():
                    self._upsert_drive_file(
                        self.drive_folder_id,
                        "db.json",
                        self.db_path.read_bytes(),
                        "application/json",
                    )
                    result["db_synced"] = True

                # 1-b) users.json 동기화
                if "users.json" in root_files_by_name:
                    users_bytes = self._download_file_bytes(root_files_by_name["users.json"]["id"])
                    _atomic_write_bytes(self.users_path, users_bytes)
                    result["users_synced"] = True
                elif self.users_path.exists():
                    self._upsert_drive_file(
                        self.drive_folder_id,
                        "users.json",
                        self.users_path.read_bytes(),
                        "application/json",
                    )
                    result["users_synced"] = True

                # 1-c) 공용 개념서 DB (concept_books.json 및 concept_db 하위 JSON) 동기화
                if "concept_books.json" in root_files_by_name:
                    cb_bytes = self._download_file_bytes(root_files_by_name["concept_books.json"]["id"])
                    _atomic_write_bytes(self.concept_books_path, cb_bytes)
                    result["concept_books_synced"] = True
                elif self.concept_books_path.exists():
                    self._upsert_drive_file(
                        self.drive_folder_id,
                        "concept_books.json",
                        self.concept_books_path.read_bytes(),
                        "application/json",
                    )
                    result["concept_books_synced"] = True

                concept_folder = root_folders_by_name.get("concept_db")
                if concept_folder and not result.get("concept_books_synced"):
                    for item in self._list_children(concept_folder["id"], extra_query=f"mimeType != '{FOLDER_MIME}'"):
                        if not item["name"].lower().endswith(".json"):
                            continue
                        target = self.concept_root / item["name"]
                        remote_size = int(item.get("size", 0) or 0)
                        if not target.exists() or (remote_size > 0 and target.stat().st_size != remote_size):
                            target.write_bytes(self._download_file_bytes(item["id"]))

                if not sync_media:
                    return result

                # 2) 과목별/회차별 폴더 동기화 (수학/사회/과학/<회차명>/*)
                for root_item in root_folders:
                    fname = root_item["name"]
                    if fname in ("crops", "pdfs", "concept_db"):
                        continue
                    subj_dir_name = _safe_segment(fname)
                    for exam_item in self._list_children(
                        root_item["id"],
                        extra_query=f"mimeType = '{FOLDER_MIME}'",
                    ):
                        local_exam_dir = self.organized_dir / subj_dir_name / _safe_segment(exam_item["name"])
                        local_exam_dir.mkdir(parents=True, exist_ok=True)
                        for leaf in self._list_children(
                            exam_item["id"],
                            extra_query=f"mimeType != '{FOLDER_MIME}'",
                        ):
                            leaf_name = leaf["name"]
                            remote_size = int(leaf.get("size", 0) or 0)
                            org_target = local_exam_dir / leaf_name
                            need_dl = not org_target.exists() or (
                                remote_size > 0 and org_target.stat().st_size != remote_size
                            )
                            if need_dl:
                                data = self._download_file_bytes(leaf["id"])
                                org_target.write_bytes(data)
                                lower = leaf_name.lower()
                                if lower.endswith(".png"):
                                    (self.crops_dir / leaf_name).write_bytes(data)
                                    result["crops_downloaded"] += 1
                                elif lower.endswith(".pdf"):
                                    (self.pdfs_dir / leaf_name).write_bytes(data)
                                    result["pdfs_downloaded"] += 1

                # 3) 레거시 crops / pdfs 폴더 하위 호환 동기화 (폴더가 존재하는 경우만)
                legacy_crops = root_folders_by_name.get("crops")
                if legacy_crops:
                    for item in self._list_children(legacy_crops["id"], extra_query=f"mimeType != '{FOLDER_MIME}'"):
                        target = self.crops_dir / item["name"]
                        remote_size = int(item.get("size", 0) or 0)
                        if not target.exists() or (remote_size > 0 and target.stat().st_size != remote_size):
                            target.write_bytes(self._download_file_bytes(item["id"]))
                            result["crops_downloaded"] += 1

                legacy_pdfs = root_folders_by_name.get("pdfs")
                if legacy_pdfs:
                    for item in self._list_children(legacy_pdfs["id"], extra_query=f"mimeType != '{FOLDER_MIME}'"):
                        target = self.pdfs_dir / item["name"]
                        remote_size = int(item.get("size", 0) or 0)
                        if not target.exists() or (remote_size > 0 and target.stat().st_size != remote_size):
                            target.write_bytes(self._download_file_bytes(item["id"]))
                            result["pdfs_downloaded"] += 1

            except Exception as exc:
                result["error"] = str(exc)

            return result

    def load_db(self) -> list[QuestionRecord]:
        with self._lock:
            if not self.db_path.exists():
                return []
            if self.db_path.stat().st_size == 0:
                return []

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
                    return [QuestionRecord.model_validate(item) for item in raw]
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
        sync_drive: bool = True,
        target_exams: set[str] | None = None,
    ) -> list[Path]:
        """과목별 · 모의고사 회차별 폴더에 JSON / Markdown / HTML 정리 파일을 생성하고 Drive에 업로드한다."""
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

                if sync_drive and self.drive_connected:
                    drive_exam_id = self._ensure_drive_exam_folder(subj, exam)
                    if drive_exam_id:
                        self._upsert_drive_file(drive_exam_id, "문제목록.json", json_bytes, "application/json")
                        self._upsert_drive_file(drive_exam_id, "ai_index.jsonl", ai_jsonl_bytes, "application/x-ndjson")
                        self._upsert_drive_file(drive_exam_id, "기출정리.md", md_bytes, "text/markdown")
                        self._upsert_drive_file(drive_exam_id, "기출정리.html", html_bytes, "text/html")

            return written_dirs

    def save_db(
        self,
        records: list[QuestionRecord],
        sync_organized_drive: bool = True,
        target_exams: set[str] | None = None,
        sync_db_drive: bool = True,
        sync_local_organized: bool = True,
    ) -> str | None:
        with self._lock:
            # 방어 로직: 기존 파일이 존재하고 유효한 데이터가 있는데 빈 레코드로 덮어쓰려는 경우 방어
            if not records and self.db_path.exists() and self.db_path.stat().st_size > 2:
                raise ValueError("기존 db.json 데이터 증발 방지: 빈 레코드 목록으로 저장을 시도할 수 없습니다.")

            payload = [rec.model_dump() for rec in records]
            raw_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            _atomic_write_text(self.db_path, raw_text, encoding="utf-8")

            errors: list[str] = []
            if sync_local_organized or sync_organized_drive:
                try:
                    self.export_organized_exam_files(
                        records,
                        sync_drive=sync_organized_drive,
                        target_exams=target_exams,
                    )
                except Exception as exc:
                    errors.append(f"과목·회차별 정리 파일 생성/동기화 오류: {exc}")

            if sync_db_drive and self.drive_connected:
                try:
                    self._upsert_drive_file(
                        self.drive_folder_id,
                        "db.json",
                        raw_text.encode("utf-8"),
                        "application/json",
                    )
                except Exception as exc:
                    errors.append(f"Drive db.json 동기화 오류: {exc}")

            return " / ".join(errors) if errors else None

    def load_users(self) -> list[UserRecord]:
        with self._lock:
            if not self.users_path.exists():
                return []
            if self.users_path.stat().st_size == 0:
                return []

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
                    return users
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

            if self.drive_connected:
                try:
                    self._upsert_drive_file(
                        self.drive_folder_id,
                        "users.json",
                        raw_text.encode("utf-8"),
                        "application/json",
                    )
                except Exception as exc:
                    return f"로컬 users.json 저장 완료 / Drive 동기화 오류: {exc}"
            return None

    def load_concept_books(self) -> list[ConceptBookRecord]:
        """로컬 ./concept_db 에 저장된 공용 개념서·교과서 DB 목록을 불러온다."""
        with self._lock:
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

            return sorted(books_map.values(), key=lambda x: (x.subject, x.book_title))

    def save_concept_book(self, book: ConceptBookRecord) -> str | None:
        """
        단원별 구조화된 개념서 레코드를 로컬(./concept_db) 및 Google Drive 공용 폴더에 자동 동기화 저장한다.
        """
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

            if self.drive_connected:
                errors: list[str] = []
                try:
                    self._upsert_drive_file(
                        self.drive_folder_id,
                        "concept_books.json",
                        all_raw_text.encode("utf-8"),
                        "application/json",
                    )
                except Exception as exc:
                    errors.append(f"concept_books.json 업로드 오류: {exc}")

                try:
                    concept_folder_id = self._ensure_drive_subfolder("concept_db")
                    if concept_folder_id:
                        self._upsert_drive_file(
                            concept_folder_id,
                            book_filename,
                            single_raw_text.encode("utf-8"),
                            "application/json",
                        )
                except Exception as exc:
                    errors.append(f"concept_db/{book_filename} 업로드 오류: {exc}")

                if errors:
                    return f"로컬 ./concept_db 저장 완료 / Drive 동기화 경고: {' / '.join(errors)}"
            return None

    def delete_concept_book(self, book_id: str) -> str | None:
        """공용 개념서 DB에서 특정 개념서를 삭제하고 Google Drive에도 반영한다."""
        with self._lock:
            remaining = [b for b in self.load_concept_books() if b.id != book_id]
            all_raw_text = json.dumps([b.model_dump() for b in remaining], ensure_ascii=False, indent=2)
            _atomic_write_text(self.concept_books_path, all_raw_text, encoding="utf-8")

            book_filename = f"{_safe_segment(book_id)}.json"
            (self.concept_root / book_filename).unlink(missing_ok=True)

            if self.drive_connected and self.drive_service:
                try:
                    self._upsert_drive_file(
                        self.drive_folder_id,
                        "concept_books.json",
                        all_raw_text.encode("utf-8"),
                        "application/json",
                    )
                    concept_folder_id = self._find_drive_subfolder("concept_db")
                    if concept_folder_id:
                        safe_name = book_filename.replace("'", "\\'")
                        for item in self._list_children(concept_folder_id, extra_query=f"name = '{safe_name}'"):
                            self._trash_drive_item(item["id"])
                except Exception as exc:
                    return f"로컬 삭제 완료 / Drive 반영 경고: {exc}"
            return None

    def save_crop_png(
        self,
        filename: str,
        png_bytes: bytes,
        subject_category: str = "",
        exam_name: str = "",
        sync_drive: bool = True,
    ) -> str | None:
        with self._lock:
            (self.crops_dir / filename).write_bytes(png_bytes)
            if subject_category and exam_name:
                (self._local_exam_dir(subject_category, exam_name) / filename).write_bytes(png_bytes)

            if sync_drive and self.drive_connected:
                try:
                    if subject_category and exam_name:
                        folder_id = self._ensure_drive_exam_folder(subject_category, exam_name)
                    else:
                        folder_id = self._ensure_drive_subfolder("crops")
                    if folder_id:
                        self._upsert_drive_file(folder_id, filename, png_bytes, "image/png")
                except Exception as exc:
                    return f"로컬 크롭 저장 완료 / Drive 업로드 오류: {exc}"
            return None

    def save_pdf_file(
        self,
        filename: str,
        pdf_bytes: bytes,
        subject_category: str = "",
        exam_name: str = "",
        sync_drive: bool = True,
    ) -> str | None:
        with self._lock:
            (self.pdfs_dir / filename).write_bytes(pdf_bytes)
            if subject_category and exam_name:
                (self._local_exam_dir(subject_category, exam_name) / filename).write_bytes(pdf_bytes)

            if sync_drive and self.drive_connected:
                try:
                    if subject_category and exam_name:
                        folder_id = self._ensure_drive_exam_folder(subject_category, exam_name)
                    else:
                        folder_id = self._ensure_drive_subfolder("pdfs")
                    if folder_id:
                        self._upsert_drive_file(folder_id, filename, pdf_bytes, "application/pdf")
                except Exception as exc:
                    return f"로컬 PDF 저장 완료 / Drive 업로드 오류: {exc}"
            return None

    def sync_exam_batch(
        self,
        subject_category: str,
        exam_name: str,
        records: list[QuestionRecord],
        crop_filenames: list[str] | None = None,
        pdf_filenames: list[str] | None = None,
        include_organized_exports: bool = True,
    ) -> str | None:
        """
        로컬에 축적된 회차별 크롭·PDF·정리 파일(HTML/MD/JSON/JSONL)과 db.json을
        Google Drive에 한 번에 일괄 동기화한다(중간 백업 및 최종 일괄 업로드용).
        """
        with self._lock:
            payload = [rec.model_dump() for rec in records]
            raw_text = json.dumps(payload, ensure_ascii=False, indent=2)
            _atomic_write_text(self.db_path, raw_text, encoding="utf-8")

            errors: list[str] = []
            if include_organized_exports:
                try:
                    self.export_organized_exam_files(
                        records,
                        sync_drive=False,
                        target_exams={exam_name},
                    )
                except Exception as exc:
                    errors.append(f"로컬 정리 파일 생성 오류: {exc}")

            if not self.drive_connected:
                return " / ".join(errors) if errors else None

            try:
                exam_folder_id = self._ensure_drive_exam_folder(subject_category, exam_name)
                if exam_folder_id:
                    existing_map = {
                        item["name"]: item["id"]
                        for item in self._list_children(
                            exam_folder_id,
                            extra_query=f"mimeType != '{FOLDER_MIME}'",
                        )
                    }
                    local_exam_dir = self._local_exam_dir(subject_category, exam_name)

                    for pdf_fn in pdf_filenames or []:
                        src_pdf = self.pdfs_dir / pdf_fn
                        if not src_pdf.exists():
                            src_pdf = local_exam_dir / pdf_fn
                        if src_pdf.exists():
                            fid = self._upsert_drive_file(
                                exam_folder_id,
                                pdf_fn,
                                src_pdf.read_bytes(),
                                "application/pdf",
                                known_existing_id=existing_map.get(pdf_fn),
                                skip_lookup=True,
                            )
                            if fid:
                                existing_map[pdf_fn] = fid

                    for crop_fn in crop_filenames or []:
                        src_crop = self.crops_dir / crop_fn
                        if not src_crop.exists():
                            src_crop = local_exam_dir / crop_fn
                        if src_crop.exists():
                            fid = self._upsert_drive_file(
                                exam_folder_id,
                                crop_fn,
                                src_crop.read_bytes(),
                                "image/png",
                                known_existing_id=existing_map.get(crop_fn),
                                skip_lookup=True,
                            )
                            if fid:
                                existing_map[crop_fn] = fid

                    if include_organized_exports:
                        for export_fn, mime in (
                            ("문제목록.json", "application/json"),
                            ("ai_index.jsonl", "application/x-ndjson"),
                            ("기출정리.md", "text/markdown"),
                            ("기출정리.html", "text/html"),
                        ):
                            export_path = local_exam_dir / export_fn
                            if export_path.exists():
                                fid = self._upsert_drive_file(
                                    exam_folder_id,
                                    export_fn,
                                    export_path.read_bytes(),
                                    mime,
                                    known_existing_id=existing_map.get(export_fn),
                                    skip_lookup=True,
                                )
                                if fid:
                                    existing_map[export_fn] = fid

                self._upsert_drive_file(
                    self.drive_folder_id,
                    "db.json",
                    raw_text.encode("utf-8"),
                    "application/json",
                )
            except Exception as exc:
                errors.append(f"Google Drive 일괄 동기화 오류: {exc}")

            return " / ".join(errors) if errors else None

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

            if self.drive_connected:
                try:
                    safe_name = filename.replace("'", "\\'")
                    search_folders: list[str] = []
                    if subj and exam:
                        exam_folder = self._find_drive_exam_folder(subj, exam)
                        if exam_folder:
                            search_folders.append(exam_folder)
                    legacy_crops = self._find_drive_subfolder("crops")
                    if legacy_crops:
                        search_folders.append(legacy_crops)

                    for fid in search_folders:
                        found = self._list_children(fid, extra_query=f"name = '{safe_name}'")
                        if found:
                            data = self._download_file_bytes(found[0]["id"])
                            local_path.write_bytes(data)
                            if subj and exam:
                                (self._local_exam_dir(subj, exam) / filename).write_bytes(data)
                            return data
                except Exception:
                    return None
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

            if self.drive_connected:
                try:
                    safe_name = filename.replace("'", "\\'")
                    search_folders: list[str] = []
                    if subj and exam:
                        exam_folder = self._find_drive_exam_folder(subj, exam)
                        if exam_folder:
                            search_folders.append(exam_folder)
                    legacy_pdfs = self._find_drive_subfolder("pdfs")
                    if legacy_pdfs:
                        search_folders.append(legacy_pdfs)

                    for fid in search_folders:
                        found = self._list_children(fid, extra_query=f"name = '{safe_name}'")
                        if found:
                            data = self._download_file_bytes(found[0]["id"])
                            local_path.write_bytes(data)
                            if subj and exam:
                                (self._local_exam_dir(subj, exam) / filename).write_bytes(data)
                            return data
                except Exception:
                    return None
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

                if self.drive_connected and self.drive_service:
                    try:
                        safe_name = target_rec.image_filename.replace("'", "\\'")
                        folders = []
                        exam_fid = self._find_drive_exam_folder(target_rec.subject_category, target_rec.exam_name)
                        if exam_fid:
                            folders.append(exam_fid)
                        legacy_fid = self._find_drive_subfolder("crops")
                        if legacy_fid:
                            folders.append(legacy_fid)
                        for fid in folders:
                            for item in self._list_children(fid, extra_query=f"name = '{safe_name}'"):
                                self._trash_drive_item(item["id"])
                    except Exception:
                        pass

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
                    if self.drive_connected and self.drive_service:
                        try:
                            legacy_fid = self._find_drive_subfolder("crops")
                            if legacy_fid:
                                safe_name = rec.image_filename.replace("'", "\\'")
                                for item in self._list_children(legacy_fid, extra_query=f"name = '{safe_name}'"):
                                    self._trash_drive_item(item["id"])
                        except Exception:
                            pass

            for pdf_fn in pdf_names:
                (self.pdfs_dir / pdf_fn).unlink(missing_ok=True)
                if self.drive_connected and self.drive_service:
                    try:
                        legacy_pdfs = self._find_drive_subfolder("pdfs")
                        if legacy_pdfs:
                            safe_name = pdf_fn.replace("'", "\\'")
                            for item in self._list_children(legacy_pdfs, extra_query=f"name = '{safe_name}'"):
                                self._trash_drive_item(item["id"])
                    except Exception:
                        pass

            for subj in subj_cats:
                org_exam_dir = self.organized_dir / _safe_segment(subj) / _safe_segment(exam_name)
                if org_exam_dir.exists():
                    shutil.rmtree(org_exam_dir, ignore_errors=True)
                if self.drive_connected and self.drive_service:
                    try:
                        exam_fid = self._find_drive_exam_folder(subj, exam_name)
                        if exam_fid:
                            self._trash_drive_item(exam_fid)
                            self._subfolder_ids.pop(
                                (self._find_drive_subfolder(_safe_segment(subj)) or "", _safe_segment(exam_name)),
                                None,
                            )
                    except Exception:
                        pass

            self.save_db(remaining, sync_organized_drive=False)
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
            self.save_db(records, sync_organized_drive=False)
            return changed

    def reset_exam_database(self, delete_users: bool = False) -> dict[str, Any]:
        """로컬 및 Google Drive에 누적된 기출 DB·이미지·PDF 파일을 모두 삭제(초기화)한다."""
        with self._lock:
            removed_local = 0
            for folder in (self.crops_dir, self.pdfs_dir):
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
            if delete_users and self.users_path.exists():
                self.users_path.unlink(missing_ok=True)

            trashed_drive = 0
            drive_err = ""
            if self.drive_connected and self.drive_service:
                try:
                    for item in self._list_children(self.drive_folder_id):
                        item_name = item.get("name", "")
                        if not delete_users and item_name == "users.json":
                            continue
                        if item_name in ("concept_books.json", "concept_db"):
                            continue
                        self._trash_drive_item(item["id"])
                        trashed_drive += 1
                    self._subfolder_ids.clear()
                    self._upsert_drive_file(
                        self.drive_folder_id,
                        "db.json",
                        b"[]",
                        "application/json",
                    )
                except Exception as exc:
                    drive_err = str(exc)

            return {
                "removed_local": removed_local,
                "trashed_drive": trashed_drive,
                "drive_error": drive_err,
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
        try:
            data = json.loads(self.english_pool_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_english_pool_unlocked(self, pool: dict[str, list[dict[str, Any]]]) -> None:
        raw = json.dumps(pool, ensure_ascii=False, indent=2)
        _atomic_write_text(self.english_pool_path, raw, encoding="utf-8")

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
