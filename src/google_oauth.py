from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
import urllib.parse
from typing import Any

import requests

from src.auth import (
    _b64url_decode,
    _b64url_encode,
    _get_session_signing_key,
    normalize_username,
)
from src.config import GoogleLoginConfig
from src.storage import StorageManager

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
OAUTH_STATE_TTL_SECONDS = 600


def create_google_oauth_state(
    action: str,
    username: str = "",
    storage: StorageManager | None = None,
    ttl_seconds: int = OAUTH_STATE_TTL_SECONDS,
) -> str:
    """CSRF 방지 및 로그인/계정연동 분기를 위한 HMAC 서명 OAuth state 토큰을 생성한다."""
    act = "link" if action == "link" else "login"
    payload = {
        "act": act,
        "u": normalize_username(username),
        "exp": int(time.time()) + int(ttl_seconds),
        "n": secrets.token_hex(8),
    }
    payload_b64 = _b64url_encode(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    )
    sig = hmac.new(
        _get_session_signing_key(storage),
        payload_b64.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return f"{payload_b64}.{_b64url_encode(sig)}"


def verify_google_oauth_state(
    state_token: str,
    storage: StorageManager | None = None,
) -> dict[str, Any] | None:
    """OAuth 콜백으로 돌아온 state 토큰의 서명과 만료 시각을 검증한다."""
    raw = (state_token or "").strip()
    if "." not in raw:
        return None
    payload_b64, sig_b64 = raw.split(".", 1)
    if not payload_b64 or not sig_b64:
        return None

    expected_sig = hmac.new(
        _get_session_signing_key(storage),
        payload_b64.encode("ascii"),
        hashlib.sha256,
    ).digest()
    try:
        actual_sig = _b64url_decode(sig_b64)
        if not hmac.compare_digest(actual_sig, expected_sig):
            return None
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    act = str(payload.get("act") or "")
    if act not in ("login", "link"):
        return None
    exp = payload.get("exp")
    if not isinstance(exp, (int, float)) or time.time() > float(exp):
        return None
    username = normalize_username(str(payload.get("u") or ""))
    if act == "link" and not username:
        return None
    return {"act": act, "u": username, "exp": float(exp)}


def build_google_authorize_url(config: GoogleLoginConfig, state: str) -> str:
    """Google OAuth 2.0 동의 화면 이동 URL을 생성한다."""
    params = {
        "client_id": config.client_id,
        "redirect_uri": config.redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "online",
        "prompt": "select_account",
    }
    return f"{GOOGLE_AUTH_URL}?{urllib.parse.urlencode(params)}"


def exchange_google_code_for_userinfo(
    config: GoogleLoginConfig,
    code: str,
    timeout: int = 15,
) -> dict[str, Any]:
    """Authorization Code를 액세스 토큰으로 교환한 뒤 Google OIDC 사용자 정보를 조회한다."""
    clean_code = (code or "").strip()
    if not clean_code:
        raise ValueError("Google 인증 코드(code)가 비어 있습니다.")

    token_resp = requests.post(
        GOOGLE_TOKEN_URL,
        data={
            "code": clean_code,
            "client_id": config.client_id,
            "client_secret": config.client_secret,
            "redirect_uri": config.redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=timeout,
    )
    if token_resp.status_code != 200:
        err_detail = ""
        try:
            err_json = token_resp.json()
            err_detail = str(err_json.get("error_description") or err_json.get("error") or "")
        except Exception:
            err_detail = token_resp.text[:200]
        raise ValueError(f"Google 토큰 교환 실패 (HTTP {token_resp.status_code}): {err_detail}")

    token_data = token_resp.json()
    access_token = str(token_data.get("access_token") or "").strip()
    if not access_token:
        raise ValueError("Google 응답에 access_token이 없습니다.")

    info_resp = requests.get(
        GOOGLE_USERINFO_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=timeout,
    )
    if info_resp.status_code != 200:
        raise ValueError(f"Google 사용자 정보 조회 실패 (HTTP {info_resp.status_code})")

    info = info_resp.json()
    sub = str(info.get("sub") or "").strip()
    if not sub:
        raise ValueError("Google 사용자 고유 ID(sub)를 확인할 수 없습니다.")

    return {
        "sub": sub,
        "email": str(info.get("email") or "").strip(),
        "email_verified": bool(info.get("email_verified", False)),
        "name": str(info.get("name") or "").strip(),
        "picture": str(info.get("picture") or "").strip(),
    }
