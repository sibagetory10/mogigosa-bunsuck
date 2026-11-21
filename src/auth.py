from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone

from src.config import (
    DEFAULT_USD_TO_KRW,
    FLASH_LITE_MODEL,
    _safe_get_secret,
    resolve_admin_config,
)
from src.models import TierType, UserRecord
from src.storage import StorageManager


def _storage_lock(storage: StorageManager | None):
    return getattr(storage, "_lock", None) or nullcontext()

KST = timezone(timedelta(hours=9))
FREE_MAX_BATCH_QUESTIONS = 5

# 월간 토큰 한도 (Gemini 3.5 Flash-Lite 단일 모델 · PRO 대비 ULTRA 정확히 4배)
FREE_MONTHLY_TOKEN_LIMIT = 500_000
PRO_MONTHLY_TOKEN_LIMIT = 3_000_000
ULTRA_MONTHLY_TOKEN_LIMIT = 12_000_000
MONTHLY_TOKEN_LIMITS: dict[str, int] = {
    "FREE": FREE_MONTHLY_TOKEN_LIMIT,
    "PRO": PRO_MONTHLY_TOKEN_LIMIT,
    "ULTRA": ULTRA_MONTHLY_TOKEN_LIMIT,
}

# 월 구독료 (관리자 대시보드 예상 매출 대비 API 원가율 계산용)
PRO_MONTHLY_PRICE_KRW = 9_900
ULTRA_MONTHLY_PRICE_KRW = 19_900

# 하위 호환용 상수 (고급 모델 폐지 완료)
PRO_WEEKLY_PREMIUM_LIMIT = 50
ULTRA_WEEKLY_PREMIUM_LIMIT = 200
WEEKLY_PREMIUM_LIMITS: dict[str, int] = {}
FREE_EXAMS_PER_UNLOCK = 3
PREMIUM_MODEL_IDS: tuple[str, ...] = ()

VALID_TIERS: tuple[TierType, ...] = ("FREE", "CUSTOM", "PRO", "ULTRA", "ADMIN")
TIER_LABELS: dict[TierType, str] = {
    "FREE": "무료 (0원)",
    "CUSTOM": "개인 키 (0원)",
    "PRO": "PRO (9,900원/월)",
    "ULTRA": "ULTRA (19,900원/월)",
    "ADMIN": "최고 관리자",
}
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{3,30}$")


def get_kst_today_str() -> str:
    """한국 표준시(KST) 기준 오늘 날짜 YYYY-MM-DD를 반환한다."""
    return datetime.now(KST).strftime("%Y-%m-%d")


def get_kst_week_str() -> str:
    """한국 표준시(KST) 기준 ISO 주차 YYYY-Www를 반환한다 (매주 월요일 초기화)."""
    return datetime.now(KST).strftime("%G-W%V")


def get_kst_month_str() -> str:
    """한국 표준시(KST) 기준 현재 월 YYYY-MM을 반환한다 (매월 1일 초기화)."""
    return datetime.now(KST).strftime("%Y-%m")


def format_token_count(tokens: int | float) -> str:
    """토큰 수를 읽기 쉬운 한국어 축약 표기로 변환한다 (예: 3,000,000 -> 300만, 12,500 -> 12,500)."""
    n = max(0, int(round(tokens or 0)))
    if n >= 100_000 and n % 10_000 == 0:
        return f"{n // 10_000:,}만"
    return f"{n:,}"


def format_krw(cost_usd: float, usd_to_krw: float = DEFAULT_USD_TO_KRW) -> str:
    """USD 비용을 원화(₩) 문자열로 변환한다."""
    krw = max(0.0, float(cost_usd or 0.0)) * float(usd_to_krw or DEFAULT_USD_TO_KRW)
    if 0 < krw < 1.0:
        return f"₩{krw:.2f}"
    if krw < 100.0 and krw != int(krw):
        return f"₩{krw:,.1f}"
    return f"₩{round(krw):,}"


def is_premium_model(model_name: str | None) -> bool:
    """전 기능 Gemini 3.5 Flash-Lite 단일 모델 통일로 인해 고급 모델 구분을 사용하지 않는다."""
    _ = model_name
    return False


PBKDF2_ITERATIONS = 240_000
_PBKDF2_PREFIX = "pbkdf2_sha256"


def hash_password(password: str, salt: str | None = None) -> str:
    """PBKDF2-HMAC-SHA256(24만 회) + random salt로 비밀번호를 해싱한다. 형식: pbkdf2_sha256$iters$salt$hex"""
    actual_salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), actual_salt.encode("utf-8"), PBKDF2_ITERATIONS
    ).hex()
    return f"{_PBKDF2_PREFIX}${PBKDF2_ITERATIONS}${actual_salt}${digest}"


def is_legacy_password_hash(stored_hash: str) -> bool:
    return bool(stored_hash) and not stored_hash.startswith(f"{_PBKDF2_PREFIX}$")


def verify_password(password: str, stored_hash: str) -> bool:
    """PBKDF2 해시(및 기존 salt$sha256 레거시 해시)를 상수 시간 비교로 검증한다."""
    if not stored_hash or "$" not in stored_hash:
        return False
    if stored_hash.startswith(f"{_PBKDF2_PREFIX}$"):
        try:
            _, iters_s, salt, expected = stored_hash.split("$", 3)
            candidate = hashlib.pbkdf2_hmac(
                "sha256", password.encode("utf-8"), salt.encode("utf-8"), int(iters_s)
            ).hex()
        except (ValueError, TypeError):
            return False
        return hmac.compare_digest(candidate, expected)
    salt, expected_digest = stored_hash.split("$", 1)
    candidate_digest = hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()
    return hmac.compare_digest(candidate_digest, expected_digest)


def normalize_username(username: str) -> str:
    return (username or "").strip().lower()


def mask_api_key(api_key: str) -> str:
    """API 키 전체를 노출하지 않고 앞 6자리 + 마스킹 기호로 반환한다."""
    key = (api_key or "").strip()
    if not key:
        return ""
    if len(key) <= 6:
        return key[:2] + "••••••••"
    return f"{key[:6]}••••••••"


SESSION_TOKEN_TTL_SECONDS: int = 7 * 24 * 3600
_PROCESS_FALLBACK_SIGNING_KEY: bytes | None = None


def _get_session_signing_key(storage: StorageManager | None = None) -> bytes:
    """새로고침 이후에도 유지되는 서명용 비밀키를 결정론적으로 로드한다."""
    parts: list[str] = []
    for key_name in ("SESSION_SECRET", "ADMIN_PASSWORD", "GEMINI_API_KEY"):
        sec_val = _safe_get_secret([key_name, key_name.lower()])
        if isinstance(sec_val, str) and sec_val.strip():
            parts.append(f"{key_name}:{sec_val.strip()}")
        elif os.environ.get(key_name, "").strip():
            parts.append(f"{key_name}:{os.environ[key_name].strip()}")

    if parts:
        seed = "|".join(parts)
        return hashlib.sha256(f"mogigosa-session-v1:{seed}".encode("utf-8")).digest()

    if storage is not None:
        secret_file = storage.local_root / ".session_secret"
        try:
            if secret_file.exists():
                existing = secret_file.read_text(encoding="utf-8").strip()
                if existing:
                    return hashlib.sha256(existing.encode("utf-8")).digest()
            generated = secrets.token_hex(32)
            secret_file.write_text(generated, encoding="utf-8")
            return hashlib.sha256(generated.encode("utf-8")).digest()
        except Exception:
            pass

    # 비밀값이 전혀 설정되지 않은 환경: 코드에 박힌 고정 키 대신 프로세스별 무작위 키 사용 (재시작 시 세션만 만료)
    global _PROCESS_FALLBACK_SIGNING_KEY
    if _PROCESS_FALLBACK_SIGNING_KEY is None:
        _PROCESS_FALLBACK_SIGNING_KEY = secrets.token_bytes(32)
    return _PROCESS_FALLBACK_SIGNING_KEY


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def create_session_token(
    username: str,
    storage: StorageManager | None = None,
    ttl_seconds: int = SESSION_TOKEN_TTL_SECONDS,
) -> str:
    """브라우저 새로고침 시 로그인 상태를 안전하게 복원하기 위한 HMAC 서명 세션 토큰을 생성한다."""
    norm = normalize_username(username)
    payload = {"u": norm, "exp": int(time.time()) + max(60, int(ttl_seconds))}
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
    sig = hmac.new(
        _get_session_signing_key(storage),
        payload_b64.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return f"{payload_b64}.{_b64url_encode(sig)}"


def verify_session_token(token: str, storage: StorageManager | None = None) -> str | None:
    """HMAC 서명 세션 토큰을 검증하고 유효한 경우 정규화된 username을 반환한다."""
    raw = (token or "").strip()
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
    username = normalize_username(str(payload.get("u", "")))
    exp = payload.get("exp")
    if not username or not isinstance(exp, (int, float)):
        return None
    if time.time() > float(exp):
        return None
    return username


def _sync_user_state(user: UserRecord, today: str, admin_usernames: set[str]) -> bool:
    """자정(KST) · 주간(KST) · 월간(KST) 초기화 및 관리자 아이디 자동 ADMIN 승급을 동기화한다."""
    changed = False
    current_week = get_kst_week_str()
    current_month = get_kst_month_str()
    if getattr(user, "last_used_date", "") != today:
        user.daily_count = 0
        user.last_used_date = today
        changed = True
    if hasattr(user, "last_used_week") and getattr(user, "last_used_week", "") != current_week:
        if hasattr(user, "weekly_premium_count"):
            user.weekly_premium_count = 0
        user.last_used_week = current_week
        changed = True
    if hasattr(user, "last_used_month") and getattr(user, "last_used_month", "") != current_month:
        user.monthly_prompt_tokens = 0
        user.monthly_cached_tokens = 0
        user.monthly_output_tokens = 0
        user.monthly_total_tokens = 0
        user.monthly_api_calls = 0
        user.monthly_cost_usd = 0.0
        user.monthly_server_cost_usd = 0.0
        user.last_used_month = current_month
        changed = True
    if user.username in admin_usernames and user.tier != "ADMIN":
        user.tier = "ADMIN"
        changed = True
    return changed


def get_user_by_username(storage: StorageManager, username: str) -> UserRecord | None:
    norm = normalize_username(username)
    with _storage_lock(storage):
        users = storage.load_users()
        today = get_kst_today_str()
        admin_names = resolve_admin_config().admin_usernames
        changed = False
        found: UserRecord | None = None

        for u in users:
            if u.username == norm:
                if _sync_user_state(u, today, admin_names):
                    changed = True
                found = u
                break

        if changed:
            storage.save_users(users)
        return found


def register_user(
    storage: StorageManager,
    username: str,
    nickname: str,
    password: str,
) -> tuple[bool, str, UserRecord | None]:
    raw_id = (username or "").strip()
    if not USERNAME_RE.match(raw_id):
        return (
            False,
            "아이디는 3~30자의 영문, 숫자, 밑줄(_), 마침표(.), 하이픈(-)만 사용할 수 있습니다.",
            None,
        )

    clean_nick = (nickname or "").strip()
    if not (1 <= len(clean_nick) <= 20):
        return False, "이름은 1~20자 이내로 입력해 주세요.", None

    if len(password or "") < 4:
        return False, "비밀번호는 최소 4자 이상이어야 합니다.", None

    norm_id = normalize_username(raw_id)
    with _storage_lock(storage):
        users = storage.load_users()
        if any(u.username == norm_id for u in users):
            return False, "이미 사용 중인 아이디입니다. 다른 아이디를 입력해 주세요.", None

        today = get_kst_today_str()
        current_month = get_kst_month_str()
        admin_cfg = resolve_admin_config()
        initial_tier: TierType = "ADMIN" if norm_id in admin_cfg.admin_usernames else "FREE"
        new_user = UserRecord(
            username=norm_id,
            nickname=clean_nick,
            password_hash=hash_password(password),
            tier=initial_tier,
            custom_api_key="",
            daily_count=0,
            last_used_date=today,
            last_used_month=current_month,
            created_at=datetime.now(KST).isoformat(timespec="seconds"),
        )
        users.append(new_user)
        err = storage.save_users(users)
    tier_desc = TIER_LABELS[initial_tier]
    if err:
        return True, f"회원가입 완료! ({err})", new_user
    return True, f"환영합니다, {clean_nick}님! (등급: {tier_desc} · {initial_tier})", new_user


def authenticate_user(
    storage: StorageManager,
    username: str,
    password: str,
) -> tuple[bool, str, UserRecord | None]:
    norm_id = normalize_username(username)
    if not norm_id or not password:
        return False, "아이디와 비밀번호를 모두 입력해 주세요.", None

    with _storage_lock(storage):
        users = storage.load_users()
        today = get_kst_today_str()
        admin_names = resolve_admin_config().admin_usernames
        for u in users:
            if u.username == norm_id:
                if not u.password_hash or not getattr(u, "has_local_credentials", True):
                    return (
                        False,
                        "이 계정은 Google 로그인 전용입니다. 'Google 계정으로 로그인' 후 아이디·비밀번호를 등록해 주세요.",
                        None,
                    )
                if not verify_password(password, u.password_hash):
                    return False, "비밀번호가 일치하지 않습니다.", None
                rehashed = False
                if is_legacy_password_hash(u.password_hash):
                    u.password_hash = hash_password(password)
                    rehashed = True
                if _sync_user_state(u, today, admin_names) or rehashed:
                    storage.save_users(users)
                return True, f"{u.nickname}님, 로그인되었습니다.", u

    return False, "등록되지 않은 아이디입니다. 회원가입 탭에서 가입해 주세요.", None


def get_user_by_google_sub(storage: StorageManager, google_sub: str) -> UserRecord | None:
    """Google OIDC sub 식별자로 연동된 회원을 조회한다."""
    clean_sub = (google_sub or "").strip()
    if not clean_sub:
        return None
    with _storage_lock(storage):
        users = storage.load_users()
        today = get_kst_today_str()
        admin_names = resolve_admin_config().admin_usernames
        changed = False
        found: UserRecord | None = None

        for u in users:
            if (getattr(u, "google_sub", "") or "").strip() == clean_sub:
                if _sync_user_state(u, today, admin_names):
                    changed = True
                found = u
                break

        if changed:
            storage.save_users(users)
        return found


def authenticate_google_user(
    storage: StorageManager,
    google_sub: str,
    google_email: str = "",
) -> tuple[bool, str, UserRecord | None]:
    """연동된 Google sub로 회원을 인증한다. 미연동 신규 구글 계정이면 (False, '', None)을 반환한다."""
    clean_sub = (google_sub or "").strip()
    clean_email = (google_email or "").strip()
    if not clean_sub:
        return False, "Google 계정 식별자를 확인할 수 없습니다.", None

    with _storage_lock(storage):
        users = storage.load_users()
        today = get_kst_today_str()
        admin_names = resolve_admin_config().admin_usernames
        for u in users:
            if (getattr(u, "google_sub", "") or "").strip() == clean_sub:
                changed = _sync_user_state(u, today, admin_names)
                if clean_email and getattr(u, "google_email", "") != clean_email:
                    u.google_email = clean_email
                    changed = True
                if changed:
                    storage.save_users(users)
                return True, f"{u.nickname}님, Google 계정으로 로그인되었습니다.", u

    return False, "", None


def register_google_user(
    storage: StorageManager,
    google_sub: str,
    google_email: str,
    nickname: str,
    desired_username: str = "",
    desired_password: str = "",
) -> tuple[bool, str, UserRecord | None]:
    """Google 계정으로 신규 회원가입한다. 아이디·비밀번호는 즉시 입력하거나 나중에 등록할 수 있다."""
    clean_sub = (google_sub or "").strip()
    clean_email = (google_email or "").strip()
    if not clean_sub:
        return False, "Google 인증 정보가 유효하지 않습니다.", None

    clean_nick = (nickname or "").strip()
    if not (1 <= len(clean_nick) <= 20):
        return False, "이름은 1~20자 이내로 입력해 주세요.", None

    raw_id = (desired_username or "").strip()
    raw_pw = desired_password or ""
    has_local = bool(raw_id or raw_pw)
    if has_local:
        if not USERNAME_RE.match(raw_id):
            return (
                False,
                "아이디는 3~30자의 영문, 숫자, 밑줄(_), 마침표(.), 하이픈(-)만 사용할 수 있습니다.",
                None,
            )
        if len(raw_pw) < 4:
            return False, "비밀번호는 최소 4자 이상이어야 합니다.", None

    with _storage_lock(storage):
        users = storage.load_users()
        if any((getattr(u, "google_sub", "") or "").strip() == clean_sub for u in users):
            return False, "이미 가입된 Google 계정입니다. 로그인해 주세요.", None

        existing_ids = {u.username for u in users}
        if has_local:
            norm_id = normalize_username(raw_id)
            if norm_id in existing_ids:
                return False, "이미 사용 중인 아이디입니다. 다른 아이디를 입력해 주세요.", None
            pw_hash = hash_password(raw_pw)
        else:
            norm_id = f"g_{secrets.token_hex(5)}"
            while norm_id in existing_ids:
                norm_id = f"g_{secrets.token_hex(5)}"
            pw_hash = ""

        today = get_kst_today_str()
        current_month = get_kst_month_str()
        admin_cfg = resolve_admin_config()
        initial_tier: TierType = "ADMIN" if norm_id in admin_cfg.admin_usernames else "FREE"
        new_user = UserRecord(
            username=norm_id,
            nickname=clean_nick,
            password_hash=pw_hash,
            google_sub=clean_sub,
            google_email=clean_email,
            has_local_credentials=has_local,
            tier=initial_tier,
            custom_api_key="",
            daily_count=0,
            last_used_date=today,
            last_used_month=current_month,
            created_at=datetime.now(KST).isoformat(timespec="seconds"),
        )
        users.append(new_user)
        err = storage.save_users(users)

    tier_desc = TIER_LABELS[initial_tier]
    if err:
        return True, f"Google 회원가입 완료! ({err})", new_user
    return True, f"환영합니다, {clean_nick}님! (Google 계정 가입 · {tier_desc})", new_user


def link_google_account(
    storage: StorageManager,
    username: str,
    google_sub: str,
    google_email: str,
) -> tuple[bool, str, UserRecord | None]:
    """기존 계정에 Google 계정을 연동한다 (1개의 Google 계정은 1개 회원 계정에만 연동 가능)."""
    norm_id = normalize_username(username)
    clean_sub = (google_sub or "").strip()
    clean_email = (google_email or "").strip()
    if not norm_id or not clean_sub:
        return False, "계정 또는 Google 인증 정보가 올바르지 않습니다.", None

    with _storage_lock(storage):
        users = storage.load_users()
        for other in users:
            if (getattr(other, "google_sub", "") or "").strip() == clean_sub and other.username != norm_id:
                return False, "이 Google 계정은 이미 다른 회원 계정에 연동되어 있습니다.", None

        target: UserRecord | None = None
        for u in users:
            if u.username == norm_id:
                u.google_sub = clean_sub
                u.google_email = clean_email
                target = u
                break

        if target is None:
            return False, "연동할 회원 계정을 찾을 수 없습니다.", None

        err = storage.save_users(users)

    if err:
        return True, f"Google 계정({clean_email}) 연동 완료 ({err})", target
    return True, f"Google 계정({clean_email})이 연동되었습니다!", target


def unlink_google_account(
    storage: StorageManager,
    username: str,
) -> tuple[bool, str, UserRecord | None]:
    """기존 계정의 Google 연동을 해제한다 (아이디·비밀번호가 등록된 계정만 가능)."""
    norm_id = normalize_username(username)
    with _storage_lock(storage):
        users = storage.load_users()
        target: UserRecord | None = None
        for u in users:
            if u.username == norm_id:
                if not (getattr(u, "has_local_credentials", True) and bool(u.password_hash)):
                    return (
                        False,
                        "아이디·비밀번호를 먼저 등록해야 Google 연동을 해제할 수 있습니다.",
                        None,
                    )
                u.google_sub = ""
                u.google_email = ""
                target = u
                break

        if target is None:
            return False, "회원 계정을 찾을 수 없습니다.", None

        err = storage.save_users(users)

    if err:
        return True, f"Google 연동 해제 완료 ({err})", target
    return True, "Google 계정 연동이 해제되었습니다.", target


def register_local_credentials_for_user(
    storage: StorageManager,
    current_username: str,
    new_username: str,
    password: str,
) -> tuple[bool, str, UserRecord | None]:
    """Google 전용 가입자가 나중에 일반 로그인용 아이디·비밀번호를 등록한다."""
    old_norm = normalize_username(current_username)
    raw_new = (new_username or "").strip()
    if not USERNAME_RE.match(raw_new):
        return (
            False,
            "아이디는 3~30자의 영문, 숫자, 밑줄(_), 마침표(.), 하이픈(-)만 사용할 수 있습니다.",
            None,
        )
    if len(password or "") < 4:
        return False, "비밀번호는 최소 4자 이상이어야 합니다.", None

    norm_new = normalize_username(raw_new)
    with _storage_lock(storage):
        users = storage.load_users()
        if any(u.username == norm_new and u.username != old_norm for u in users):
            return False, "이미 사용 중인 아이디입니다. 다른 아이디를 입력해 주세요.", None

        admin_names = resolve_admin_config().admin_usernames
        target: UserRecord | None = None
        for u in users:
            if u.username == old_norm:
                u.username = norm_new
                u.password_hash = hash_password(password)
                u.has_local_credentials = True
                if norm_new in admin_names and u.tier != "ADMIN":
                    u.tier = "ADMIN"
                target = u
                break

        if target is None:
            return False, "회원 계정을 찾을 수 없습니다.", None

        err = storage.save_users(users)

    if old_norm != norm_new:
        storage.rename_user_data(old_norm, norm_new)

    if err:
        return True, f"아이디(@{norm_new})·비밀번호 등록 완료 ({err})", target
    return True, f"아이디(@{norm_new})와 비밀번호가 등록되었습니다! 이제 일반 로그인도 가능합니다.", target


def save_user_custom_api_key(
    storage: StorageManager,
    username: str,
    api_key: str,
) -> tuple[bool, str, UserRecord | None]:
    """개인 Gemini API 키를 저장하고 즉시 CUSTOM 등급으로 전환한다."""
    cleaned_key = (api_key or "").strip()
    if len(cleaned_key) < 12 or " " in cleaned_key:
        return False, "올바른 형식의 Gemini API 키(`AIza...`)를 입력해 주세요.", None

    norm_id = normalize_username(username)
    with _storage_lock(storage):
        users = storage.load_users()
        target: UserRecord | None = None

        for u in users:
            if u.username == norm_id:
                u.custom_api_key = cleaned_key
                if u.tier != "ADMIN":
                    u.tier = "CUSTOM"
                target = u
                break

        if target is None:
            return False, "사용자 계정을 찾을 수 없습니다.", None

        err = storage.save_users(users)
    if err:
        return True, f"개인 API 키 저장 완료 ({err})", target
    return True, "개인 API 키가 계정에 영구 저장되어 '개인 키 모드(CUSTOM)'로 전환되었습니다!", target


def clear_user_custom_api_key(
    storage: StorageManager,
    username: str,
) -> tuple[bool, str, UserRecord | None]:
    """저장된 개인 API 키를 삭제하고 FREE(무료 회원) 등급으로 복귀시킨다."""
    norm_id = normalize_username(username)
    with _storage_lock(storage):
        users = storage.load_users()
        target: UserRecord | None = None

        for u in users:
            if u.username == norm_id:
                u.custom_api_key = ""
                if u.tier != "ADMIN":
                    u.tier = "FREE"
                target = u
                break

        if target is None:
            return False, "사용자 계정을 찾을 수 없습니다.", None

        err = storage.save_users(users)
    if err:
        return True, f"개인 키 삭제 완료 ({err})", target
    return True, "개인 API 키를 삭제하고 무료 회원(FREE)으로 복귀했습니다.", target


def is_server_api_key_user(user: UserRecord | None) -> bool:
    """해당 사용자가 서버 Gemini API 키를 사용하는지(CUSTOM 개인 키가 아닌지) 반환한다."""
    if user is None:
        return True
    return user.tier != "CUSTOM"


def resolve_active_gemini_api_key(
    user: UserRecord,
    server_api_key: str,
    model_name: str | None = None,
    force_server_key: bool = False,
) -> tuple[str, str]:
    """
    등급에 따라 실제 Gemini 호출용 API 키를 라우팅한다 (전 기능 Gemini 3.5 Flash-Lite 단일 모델).
    - CUSTOM 등급: 본인이 등록한 개인 API 키를 모든 작업에 사용 (서버 원가 0원)
    - FREE / PRO / ULTRA / ADMIN: 서버 공용 Gemini API 키 사용 (월간 토큰 한도 적용)
    """
    _ = model_name
    server_key = (server_api_key or "").strip()
    if user.tier == "CUSTOM":
        custom_key = (user.custom_api_key or "").strip()
        if custom_key:
            return custom_key, "개인 키 모드 (CUSTOM · 본인 API 키)"
        if force_server_key:
            return server_key, "서버 키 (업로드·태깅 공용)"
        return "", "개인 키 미등록 (사이드바에서 Gemini API 키를 등록해 주세요)"
    if force_server_key:
        return server_key, "서버 키 (업로드·태깅 공용)"
    return server_key, f"서버 키 ({TIER_LABELS.get(user.tier, user.tier)})"


def get_remaining_free_quota(user: UserRecord) -> int | None:
    """하위 호환용; 월간 토큰 한도는 get_remaining_monthly_tokens()를 사용한다."""
    return get_remaining_monthly_tokens(user)


def get_max_batch_questions(user: UserRecord | None) -> int | None:
    """등급별 1회 최대 생성·변형 문항 수 제한을 반환한다 (FREE=5문항, 유료·개인키·관리자=제한 없음 None)."""
    if user is None or user.tier == "FREE":
        return FREE_MAX_BATCH_QUESTIONS
    return None


def get_monthly_token_limit(user: UserRecord | None) -> int | None:
    """등급별 월간 Gemini 3.5 Flash-Lite 토큰 한도를 반환한다 (FREE=50만, PRO=300만, ULTRA=1,200만, CUSTOM/ADMIN=None)."""
    if user is None:
        return FREE_MONTHLY_TOKEN_LIMIT
    return MONTHLY_TOKEN_LIMITS.get(user.tier)


def get_used_monthly_tokens(user: UserRecord | None) -> int:
    """이번 달(KST YYYY-MM) 누적 사용 토큰 수를 반환한다."""
    if user is None:
        return 0
    current_month = get_kst_month_str()
    if getattr(user, "last_used_month", "") != current_month:
        return 0
    return max(0, int(getattr(user, "monthly_total_tokens", 0) or 0))


def get_remaining_monthly_tokens(user: UserRecord | None) -> int | None:
    """이번 달(KST) 잔여 토큰 수를 반환한다. 무제한(CUSTOM/ADMIN)은 None."""
    limit = get_monthly_token_limit(user)
    if limit is None:
        return None
    used = get_used_monthly_tokens(user)
    return max(0, limit - used)


def check_token_quota(
    storage: StorageManager,
    username: str,
) -> tuple[bool, int | None, int | None, str]:
    """
    API 호출 직전 사용자의 이번 달 토큰 한도 초과 여부를 확인한다.
    반환: (허용 여부, 잔여 토큰 또는 None, 월간 한도 또는 None, 차단/안내 메시지)
    """
    user = get_user_by_username(storage, username)
    if user is None:
        return False, 0, FREE_MONTHLY_TOKEN_LIMIT, "로그인 계정 정보를 확인할 수 없습니다. 다시 로그인해 주세요."

    limit = get_monthly_token_limit(user)
    if limit is None:
        return True, None, None, ""

    used = get_used_monthly_tokens(user)
    rem = max(0, limit - used)
    if rem <= 0:
        tier_lbl = TIER_LABELS.get(user.tier, user.tier)
        if user.tier == "FREE":
            upgrade_hint = "사이드바에서 개인 API 키(CUSTOM)를 등록하거나 PRO(월 300만)/ULTRA(월 1,200만) 등급으로 업그레이드해 주세요."
        elif user.tier == "PRO":
            upgrade_hint = "매월 1일(KST) 초기화되며, 더 넉넉한 한도가 필요하시면 ULTRA(월 1,200만 토큰 · 4배)로 업그레이드해 주세요."
        else:
            upgrade_hint = "매월 1일(KST) 자동 초기화됩니다. 급한 경우 관리자에게 월간 한도 초기화를 요청해 주세요."
        return (
            False,
            0,
            limit,
            f"🚫 이번 달 토큰 한도를 모두 사용했습니다 ({tier_lbl} · {used:,} / {limit:,} 토큰). {upgrade_hint}",
        )
    return True, rem, limit, ""


def record_token_usage(
    storage: StorageManager,
    username: str,
    prompt_tokens: int,
    cached_tokens: int,
    output_tokens: int,
    total_tokens: int,
    cost_usd: float,
    api_calls: int = 1,
    used_server_key: bool = True,
    sync_drive: bool = True,
) -> UserRecord | None:
    """실제 Gemini API 응답에서 수집된 토큰 수와 USD 예상 비용을 사용자 레코드(월간+누적)에 기록한다."""
    p_tok = max(0, int(prompt_tokens or 0))
    c_tok = max(0, int(cached_tokens or 0))
    o_tok = max(0, int(output_tokens or 0))
    t_tok = max(0, int(total_tokens or 0))
    if t_tok <= 0:
        t_tok = p_tok + c_tok + o_tok
    calls = max(0, int(api_calls or 0))
    usd = max(0.0, float(cost_usd or 0.0))
    if t_tok == 0 and calls == 0 and usd == 0.0:
        return None

    norm_id = normalize_username(username)
    if not norm_id:
        return None

    with _storage_lock(storage):
        users = storage.load_users()
        today = get_kst_today_str()
        admin_names = resolve_admin_config().admin_usernames
        target: UserRecord | None = None

        for u in users:
            if u.username == norm_id:
                _sync_user_state(u, today, admin_names)
                u.monthly_prompt_tokens = int(getattr(u, "monthly_prompt_tokens", 0) or 0) + p_tok
                u.monthly_cached_tokens = int(getattr(u, "monthly_cached_tokens", 0) or 0) + c_tok
                u.monthly_output_tokens = int(getattr(u, "monthly_output_tokens", 0) or 0) + o_tok
                u.monthly_total_tokens = int(getattr(u, "monthly_total_tokens", 0) or 0) + t_tok
                u.monthly_api_calls = int(getattr(u, "monthly_api_calls", 0) or 0) + calls
                u.monthly_cost_usd = round(float(getattr(u, "monthly_cost_usd", 0.0) or 0.0) + usd, 8)
                if used_server_key:
                    u.monthly_server_cost_usd = round(
                        float(getattr(u, "monthly_server_cost_usd", 0.0) or 0.0) + usd, 8
                    )

                u.total_prompt_tokens = int(getattr(u, "total_prompt_tokens", 0) or 0) + p_tok
                u.total_cached_tokens = int(getattr(u, "total_cached_tokens", 0) or 0) + c_tok
                u.total_output_tokens = int(getattr(u, "total_output_tokens", 0) or 0) + o_tok
                u.total_total_tokens = int(getattr(u, "total_total_tokens", 0) or 0) + t_tok
                u.total_api_calls = int(getattr(u, "total_api_calls", 0) or 0) + calls
                u.total_cost_usd = round(float(getattr(u, "total_cost_usd", 0.0) or 0.0) + usd, 8)
                if used_server_key:
                    u.total_server_cost_usd = round(
                        float(getattr(u, "total_server_cost_usd", 0.0) or 0.0) + usd, 8
                    )
                target = u
                break

        if target is not None:
            _ = sync_drive
            storage.save_users(users)
        return target


def reset_user_monthly_tokens(
    storage: StorageManager,
    target_usernames: list[str],
) -> tuple[int, str | None]:
    """관리자가 지정한 회원의 이번 달 토큰 사용량을 0으로 초기화한다 (누적 통계는 보존)."""
    targets = {normalize_username(u) for u in target_usernames if normalize_username(u)}
    if not targets:
        return 0, None

    current_month = get_kst_month_str()
    with _storage_lock(storage):
        users = storage.load_users()
        reset_cnt = 0
        for u in users:
            if u.username in targets:
                u.monthly_prompt_tokens = 0
                u.monthly_cached_tokens = 0
                u.monthly_output_tokens = 0
                u.monthly_total_tokens = 0
                u.monthly_api_calls = 0
                u.monthly_cost_usd = 0.0
                u.monthly_server_cost_usd = 0.0
                u.last_used_month = current_month
                reset_cnt += 1
        if reset_cnt > 0:
            err = storage.save_users(users)
            return reset_cnt, err
        return 0, None


def get_weekly_premium_limit(user: UserRecord | None) -> int | None:
    """하위 호환용 (고급 모델 폐지 완료)."""
    _ = user
    return None


def get_remaining_pro_weekly_quota(user: UserRecord) -> int | None:
    """하위 호환용 (고급 모델 폐지 완료)."""
    _ = user
    return None


def can_use_premium_features(user: UserRecord) -> bool:
    """문제 자동 개조 및 맞춤형 시험지 PDF 병합 허용 여부 (전 등급 허용, 월간 토큰 한도 적용)."""
    return user.tier in ("FREE", "CUSTOM", "PRO", "ULTRA", "ADMIN")


def can_access_shared_concept_db(user: UserRecord | None) -> bool:
    """공용 개념서·교과서 DB 구축(업로드) 및 단원별 불러오기 권한 여부 (PRO / ULTRA / ADMIN 전용)."""
    if user is None:
        return False
    return user.tier in ("PRO", "ULTRA", "ADMIN")


def resolve_model_for_user(user: UserRecord, requested_model: str) -> tuple[str, str]:
    """전 등급·전 기능을 Gemini 3.5 Flash-Lite 단일 모델로 라우팅한다."""
    _ = (user, requested_model)
    return FLASH_LITE_MODEL, ""


def record_exam_upload_for_user(
    storage: StorageManager,
    username: str,
) -> tuple[UserRecord | None, bool]:
    """기출 PDF 업로드·태깅 완료 시 회원의 누적 업로드 횟수를 +1 한다."""
    norm_id = normalize_username(username)
    with _storage_lock(storage):
        users = storage.load_users()
        today = get_kst_today_str()
        admin_names = resolve_admin_config().admin_usernames
        target: UserRecord | None = None

        for u in users:
            if u.username == norm_id:
                _sync_user_state(u, today, admin_names)
                cur_uploads = int(getattr(u, "uploaded_exam_count", 0) or 0) + 1
                if hasattr(u, "uploaded_exam_count"):
                    u.uploaded_exam_count = cur_uploads
                target = u
                break

        if target is not None:
            storage.save_users(users)
        return target, False


def consume_premium_model_usage(
    storage: StorageManager,
    username: str,
    used_model: str,
    count: int = 1,
) -> UserRecord | None:
    """하위 호환용 no-op (실제 사용량은 record_token_usage에서 토큰 단위로 자동 기록됨)."""
    _ = (storage, username, used_model, count)
    return None


def consume_extraction_quota(
    storage: StorageManager,
    username: str,
    extracted_count: int,
) -> UserRecord | None:
    """
    추출한 문항 수만큼 서버 사용량(daily_count)을 기록한다 (하위 호환용; 개편 후 FREE 한도 제한 없음).
    """
    norm_id = normalize_username(username)
    with _storage_lock(storage):
        users = storage.load_users()
        today = get_kst_today_str()
        admin_names = resolve_admin_config().admin_usernames
        target: UserRecord | None = None

        for u in users:
            if u.username == norm_id:
                _sync_user_state(u, today, admin_names)
                if u.tier != "CUSTOM":
                    u.daily_count += max(0, extracted_count)
                    u.last_used_date = today
                target = u
                break

        if target is not None:
            storage.save_users(users)
        return target


def batch_update_user_tiers(
    storage: StorageManager,
    tier_updates: dict[str, TierType],
) -> tuple[int, str | None]:
    """관리자가 여러 회원의 등급을 한 번에 업데이트해 users.json에 저장한다."""
    with _storage_lock(storage):
        users = storage.load_users()
        changed_count = 0
        for u in users:
            new_tier = tier_updates.get(u.username)
            if new_tier in VALID_TIERS and u.tier != new_tier:
                u.tier = new_tier
                changed_count += 1

        if changed_count > 0:
            err = storage.save_users(users)
            return changed_count, err
        return 0, None


def delete_users_by_usernames(
    storage: StorageManager,
    target_usernames: list[str],
    actor_username: str = "",
) -> tuple[int, str | None, str]:
    """관리자가 지정한 회원 계정을 users.json 및 Google Drive에서 영구 삭제한다.

    본인 계정 및 마지막 남은 ADMIN 계정은 삭제할 수 없다.
    반환: (삭제된 계정 수, 저장 오류 메시지 또는 None, 안내/경고 메시지)
    """
    actor_norm = normalize_username(actor_username)
    targets = {normalize_username(u) for u in target_usernames if normalize_username(u)}
    if not targets:
        return 0, None, "삭제할 계정을 선택해 주세요."

    with _storage_lock(storage):
        users = storage.load_users()
        admin_count = sum(1 for u in users if u.tier == "ADMIN")
        remaining: list[UserRecord] = []
        deleted_count = 0
        skipped_reasons: list[str] = []

        for u in users:
            if u.username in targets:
                if actor_norm and u.username == actor_norm:
                    skipped_reasons.append(f"@{u.username}(본인 계정)")
                    remaining.append(u)
                    continue
                if u.tier == "ADMIN" and admin_count <= 1:
                    skipped_reasons.append(f"@{u.username}(마지막 관리자)")
                    remaining.append(u)
                    continue
                if u.tier == "ADMIN":
                    admin_count -= 1
                deleted_count += 1
            else:
                remaining.append(u)

        err: str | None = None
        if deleted_count > 0:
            err = storage.save_users(remaining)

    note = ""
    if skipped_reasons:
        note = f"보호됨(삭제 제외): {', '.join(skipped_reasons)}"
    return deleted_count, err, note
