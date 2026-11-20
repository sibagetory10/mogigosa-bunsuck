from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import streamlit as st

HYDROGEN_MODEL: str = "gemini-4-hydrogen"
HYDROGEN_LABEL: str = "Gemini 4 Hydrogen (낮음)"

AVAILABLE_MODELS: list[str] = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    HYDROGEN_MODEL,
    "gemini-3.8-flash",
]
MODEL_LABELS: dict[str, str] = {
    "gemini-3.5-flash": "Gemini 3.5 Flash",
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
    HYDROGEN_MODEL: HYDROGEN_LABEL,
    "gemini-3.8-flash": "Gemini 3.8 Flash",
}
DEFAULT_MODEL: str = "gemini-3.5-flash"
EXAM_TAGGING_MODEL: str = "gemini-3.5-flash-lite"
EXAM_NO_ANSWER_SOLVER_MODEL: str = "gemini-3.8-flash"
CONCEPT_DB_BUILD_MODEL: str = "gemini-3.8-flash"
CONCEPT_SCOPE_MODEL: str = "gemini-3.5-flash-lite"
REWRITE_MODEL: str = HYDROGEN_MODEL


def resolve_api_model_name(model_choice: str | None) -> str:
    """UI 표시명 또는 모델 ID를 실제 Gemini API 모델명으로 정규화한다."""
    raw = (model_choice or "").strip()
    if not raw:
        return DEFAULT_MODEL
    for model_id, label in MODEL_LABELS.items():
        if raw == label:
            return model_id
    return raw


@dataclass
class ResolvedCredentials:
    gemini_api_key: str
    gemini_source: str
    drive_folder_id: str
    drive_folder_source: str
    service_account_info: dict[str, Any] | None
    service_account_source: str
    oauth_credentials_info: dict[str, Any] | None = None
    oauth_credentials_source: str = "미설정"


@dataclass
class AdminConfig:
    admin_usernames: set[str]
    admin_password: str
    source: str


def _safe_get_secret(candidate_keys: list[str]) -> Any:
    """st.secrets가 없거나 키가 없는 환경에서도 예외 없이 값을 조회한다."""
    try:
        secrets_obj = st.secrets
        for key in candidate_keys:
            if key in secrets_obj:
                return secrets_obj[key]
    except Exception:
        return None
    return None


def _parse_json_mapping(raw_val: Any) -> dict[str, Any] | None:
    """Mapping, JSON 문자열, 또는 JSON 파일 경로를 dict로 정규화한다."""
    if not raw_val:
        return None
    if isinstance(raw_val, Mapping):
        return {k: v for k, v in raw_val.items()}
    if isinstance(raw_val, str):
        text = raw_val.strip()
        if not text:
            return None
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            if os.path.isfile(text):
                try:
                    with open(text, "r", encoding="utf-8") as f:
                        obj = json.load(f)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    return None
    return None


def _normalize_service_account(raw_val: Any) -> dict[str, Any] | None:
    """문자열 JSON 또는 TOML Mapping을 google-auth용 dict로 정규화한다."""
    parsed = _parse_json_mapping(raw_val)
    if not parsed:
        return None

    # private_key 내부 이스케이프 개행 복구
    if isinstance(parsed.get("private_key"), str):
        parsed["private_key"] = parsed["private_key"].replace("\\n", "\n")

    required_keys = {"client_email", "private_key", "token_uri"}
    if not required_keys.issubset(parsed.keys()):
        return None
    return parsed


def _normalize_oauth_credentials(raw_val: Any) -> dict[str, Any] | None:
    """OAuth 2.0 사용자 인증(client_id, client_secret, refresh_token)을 dict로 정규화한다."""
    parsed = _parse_json_mapping(raw_val)
    if not parsed:
        return None

    if not parsed.get("client_id") or not parsed.get("client_secret") or not parsed.get("refresh_token"):
        return None

    return {
        "client_id": str(parsed["client_id"]).strip(),
        "client_secret": str(parsed["client_secret"]).strip(),
        "refresh_token": str(parsed["refresh_token"]).strip(),
        "token_uri": str(parsed.get("token_uri") or "https://oauth2.googleapis.com/token").strip(),
    }


def resolve_credentials(
    manual_gemini_key: str = "",
    manual_folder_id: str = "",
    manual_sa_json: str = "",
) -> ResolvedCredentials:
    """st.secrets -> 환경변수 -> 사이드바 수동 입력 우선순위로 인증 정보를 로드한다."""

    # 1. GEMINI_API_KEY
    gemini_key = ""
    gemini_source = "미설정"
    sec_gemini = _safe_get_secret(["GEMINI_API_KEY", "gemini_api_key"])
    if isinstance(sec_gemini, str) and sec_gemini.strip():
        gemini_key = sec_gemini.strip()
        gemini_source = "st.secrets"
    elif os.environ.get("GEMINI_API_KEY", "").strip():
        gemini_key = os.environ["GEMINI_API_KEY"].strip()
        gemini_source = "환경변수"
    elif manual_gemini_key.strip():
        gemini_key = manual_gemini_key.strip()
        gemini_source = "사이드바 수동 입력"

    # 사이드바 수동 오버라이드가 명시적으로 입력되었을 때(secrets/env가 있어도 덮어쓸 수 있도록 별도 선택 가능하지만,
    # 요구사항 우선순위는 st.secrets -> 환경변수 -> 사이드바 수동 입력 순서임)

    # 2. GOOGLE_DRIVE_FOLDER_ID
    folder_id = ""
    folder_source = "미설정"
    sec_folder = _safe_get_secret(["GOOGLE_DRIVE_FOLDER_ID", "google_drive_folder_id"])
    if isinstance(sec_folder, str) and sec_folder.strip():
        folder_id = sec_folder.strip()
        folder_source = "st.secrets"
    elif os.environ.get("GOOGLE_DRIVE_FOLDER_ID", "").strip():
        folder_id = os.environ["GOOGLE_DRIVE_FOLDER_ID"].strip()
        folder_source = "환경변수"
    elif manual_folder_id.strip():
        folder_id = manual_folder_id.strip()
        folder_source = "사이드바 수동 입력"

    # 3. GCP_SERVICE_ACCOUNT
    sa_info: dict[str, Any] | None = None
    sa_source = "미설정"
    sec_sa = _safe_get_secret(["GCP_SERVICE_ACCOUNT", "gcp_service_account"])
    if sec_sa is not None:
        sa_info = _normalize_service_account(sec_sa)
        if sa_info:
            sa_source = "st.secrets"

    if sa_info is None:
        env_sa = os.environ.get("GCP_SERVICE_ACCOUNT", "").strip()
        if env_sa:
            sa_info = _normalize_service_account(env_sa)
            if sa_info:
                sa_source = "환경변수"

    if sa_info is None and manual_sa_json.strip():
        sa_info = _normalize_service_account(manual_sa_json)
        if sa_info:
            sa_source = "사이드바 수동 입력"

    # 4. GOOGLE_OAUTH (개인 Google 계정 OAuth 2.0 Refresh Token)
    oauth_info: dict[str, Any] | None = None
    oauth_source = "미설정"
    sec_oauth = _safe_get_secret(["GOOGLE_OAUTH", "google_oauth", "GOOGLE_OAUTH_JSON"])
    if sec_oauth is not None:
        oauth_info = _normalize_oauth_credentials(sec_oauth)
        if oauth_info:
            oauth_source = "st.secrets"

    if oauth_info is None:
        env_oauth = os.environ.get("GOOGLE_OAUTH", "").strip()
        if env_oauth:
            oauth_info = _normalize_oauth_credentials(env_oauth)
            if oauth_info:
                oauth_source = "환경변수"

    return ResolvedCredentials(
        gemini_api_key=gemini_key,
        gemini_source=gemini_source,
        drive_folder_id=folder_id,
        drive_folder_source=folder_source,
        service_account_info=sa_info,
        service_account_source=sa_source,
        oauth_credentials_info=oauth_info,
        oauth_credentials_source=oauth_source,
    )


DEFAULT_ADMIN_USERNAMES: set[str] = {"sibagetory10"}


def resolve_admin_config() -> AdminConfig:
    """st.secrets -> 환경변수 순서로 관리자 아이디 목록 및 관리자 비밀번호를 로드한다."""
    raw_users = _safe_get_secret(["ADMIN_USERNAMES", "ADMIN_USERNAME", "admin_usernames", "admin_username"])
    source = "기본 설정"
    usernames: set[str] = set(DEFAULT_ADMIN_USERNAMES)

    if raw_users is not None:
        source = "st.secrets"
    else:
        env_users = os.environ.get("ADMIN_USERNAMES") or os.environ.get("ADMIN_USERNAME") or ""
        if env_users.strip():
            raw_users = env_users
            source = "환경변수"

    if isinstance(raw_users, str):
        usernames |= {u.strip().lower() for u in raw_users.split(",") if u.strip()}
    elif isinstance(raw_users, (list, tuple, set)):
        usernames |= {str(u).strip().lower() for u in raw_users if str(u).strip()}

    admin_pw = ""
    sec_pw = _safe_get_secret(["ADMIN_PASSWORD", "admin_password"])
    if isinstance(sec_pw, str) and sec_pw.strip():
        admin_pw = sec_pw.strip()
        source = "st.secrets"
    elif os.environ.get("ADMIN_PASSWORD", "").strip():
        admin_pw = os.environ["ADMIN_PASSWORD"].strip()
        if source == "기본 설정":
            source = "환경변수"

    return AdminConfig(
        admin_usernames=usernames,
        admin_password=admin_pw,
        source=source,
    )
