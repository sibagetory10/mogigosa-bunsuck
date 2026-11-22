from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None  # type: ignore[assignment]

FLASH_LITE_MODEL: str = "gemini-3.5-flash-lite"
FLASH_LITE_LABEL: str = "Gemini 3.5 Flash-Lite"

# 하위 호환용 별칭 (전 기능 gemini-3.5-flash-lite 단일 모델 통일)
HYDROGEN_MODEL: str = FLASH_LITE_MODEL
HYDROGEN_LABEL: str = FLASH_LITE_LABEL

AVAILABLE_MODELS: list[str] = [
    FLASH_LITE_MODEL,
]
MODEL_LABELS: dict[str, str] = {
    FLASH_LITE_MODEL: FLASH_LITE_LABEL,
}
DEFAULT_MODEL: str = FLASH_LITE_MODEL
EXAM_TAGGING_MODEL: str = FLASH_LITE_MODEL
EXAM_NO_ANSWER_SOLVER_MODEL: str = FLASH_LITE_MODEL
CONCEPT_DB_BUILD_MODEL: str = FLASH_LITE_MODEL
CONCEPT_SCOPE_MODEL: str = FLASH_LITE_MODEL
REWRITE_MODEL: str = FLASH_LITE_MODEL

# Gemini 3.5 Flash-Lite 공식 API 단가 (USD / 1M tokens, 출력 단가는 사고 토큰 포함)
FLASH_LITE_INPUT_USD_PER_1M: float = 0.30
FLASH_LITE_CACHED_INPUT_USD_PER_1M: float = 0.03
FLASH_LITE_OUTPUT_USD_PER_1M: float = 2.50
MODEL_TOKEN_PRICING_USD_PER_1M: dict[str, dict[str, float]] = {
    FLASH_LITE_MODEL: {
        "input": FLASH_LITE_INPUT_USD_PER_1M,
        "cached_input": FLASH_LITE_CACHED_INPUT_USD_PER_1M,
        "output": FLASH_LITE_OUTPUT_USD_PER_1M,
    },
}
DEFAULT_USD_TO_KRW: float = 1335.0


def resolve_api_model_name(model_choice: str | None) -> str:
    """모든 모델 요청을 단일 모델(gemini-3.5-flash-lite)로 정규화한다."""
    _ = model_choice
    return FLASH_LITE_MODEL


def resolve_usd_to_krw_rate() -> float:
    """st.secrets('USD_TO_KRW') -> 환경변수 -> 기본값(1335.0) 순으로 USD->KRW 환율을 로드한다."""
    raw = _safe_get_secret(["USD_TO_KRW", "usd_to_krw"])
    if raw is None:
        raw = os.environ.get("USD_TO_KRW", "").strip() or None
    if raw is not None:
        try:
            val = float(str(raw).replace(",", "").strip())
            if 500.0 <= val <= 5000.0:
                return val
        except (ValueError, TypeError):
            pass
    return DEFAULT_USD_TO_KRW


def resolve_billing_api_key() -> str:
    """선택 사항인 Google Cloud Billing Catalog API 키(st.secrets 또는 환경변수)를 로드한다."""
    sec_val = _safe_get_secret(
        ["GOOGLE_BILLING_API_KEY", "CLOUD_BILLING_API_KEY", "google_billing_api_key"]
    )
    if isinstance(sec_val, str) and sec_val.strip():
        return sec_val.strip()
    for env_k in ("GOOGLE_BILLING_API_KEY", "CLOUD_BILLING_API_KEY"):
        v = os.environ.get(env_k, "").strip()
        if v:
            return v
    return ""


@dataclass
class ResolvedCredentials:
    gemini_api_key: str
    gemini_source: str


@dataclass
class AdminConfig:
    admin_usernames: set[str]
    admin_password: str
    source: str


def _safe_get_secret(candidate_keys: list[str]):
    """st.secrets가 없거나 키가 없는 환경에서도 예외 없이 값을 조회한다."""
    try:
        secrets_obj = st.secrets
        for key in candidate_keys:
            if key in secrets_obj:
                return secrets_obj[key]
    except Exception:
        return None
    return None


def resolve_credentials(
    manual_gemini_key: str = "",
) -> ResolvedCredentials:
    """st.secrets -> 환경변수 -> 사이드바 수동 입력 우선순위로 Gemini API 인증 정보를 로드한다."""
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

    return ResolvedCredentials(
        gemini_api_key=gemini_key,
        gemini_source=gemini_source,
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


@dataclass
class GoogleLoginConfig:
    """사용자 웹 로그인용 Google OAuth 2.0 클라이언트 설정."""

    client_id: str
    client_secret: str
    redirect_uri: str
    source: str

    @property
    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret and self.redirect_uri)


def resolve_google_login_config() -> GoogleLoginConfig:
    """st.secrets -> 환경변수 순서로 웹 로그인용 Google OAuth 2.0 설정을 로드한다.

    Drive용 GOOGLE_OAUTH(Desktop 앱 토큰)와는 별도로 관리한다.
    """
    sec_section = _safe_get_secret(["google_login", "GOOGLE_LOGIN"])
    section_map = sec_section if isinstance(sec_section, Mapping) else {}

    def _pick(sec_keys: list[str], env_key: str, section_key: str) -> tuple[str, str]:
        raw_sec = _safe_get_secret(sec_keys)
        if isinstance(raw_sec, str) and raw_sec.strip():
            return raw_sec.strip(), "st.secrets"
        sec_inner = section_map.get(section_key) or section_map.get(section_key.upper())
        if isinstance(sec_inner, str) and sec_inner.strip():
            return sec_inner.strip(), "st.secrets"
        env_val = os.environ.get(env_key, "").strip()
        if env_val:
            return env_val, "환경변수"
        return "", ""

    client_id, src_id = _pick(
        ["GOOGLE_LOGIN_CLIENT_ID", "google_login_client_id"],
        "GOOGLE_LOGIN_CLIENT_ID",
        "client_id",
    )
    client_secret, src_sec = _pick(
        ["GOOGLE_LOGIN_CLIENT_SECRET", "google_login_client_secret"],
        "GOOGLE_LOGIN_CLIENT_SECRET",
        "client_secret",
    )
    redirect_uri, src_uri = _pick(
        ["GOOGLE_LOGIN_REDIRECT_URI", "google_login_redirect_uri"],
        "GOOGLE_LOGIN_REDIRECT_URI",
        "redirect_uri",
    )

    source = next((s for s in (src_id, src_sec, src_uri) if s), "미설정")
    return GoogleLoginConfig(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        source=source,
    )
