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

from src.config import _safe_get_secret, resolve_admin_config
from src.models import TierType, UserRecord
from src.storage import StorageManager


def _storage_lock(storage: StorageManager | None):
    return getattr(storage, "_lock", None) or nullcontext()

KST = timezone(timedelta(hours=9))
FREE_MAX_BATCH_QUESTIONS = 5
PAID_MAX_BATCH_QUESTIONS = 20
PRO_WEEKLY_PREMIUM_LIMIT = 50
ULTRA_WEEKLY_PREMIUM_LIMIT = 200
WEEKLY_PREMIUM_LIMITS: dict[str, int] = {
    "PRO": PRO_WEEKLY_PREMIUM_LIMIT,
    "ULTRA": ULTRA_WEEKLY_PREMIUM_LIMIT,
}
FREE_EXAMS_PER_UNLOCK = 3
PREMIUM_MODEL_IDS: tuple[str, ...] = ("gemini-3.8-flash", "gemini-4-hydrogen")
VALID_TIERS: tuple[TierType, ...] = ("FREE", "CUSTOM", "PRO", "ULTRA", "ADMIN")
TIER_LABELS: dict[TierType, str] = {
    "FREE": "무료 (0원)",
    "CUSTOM": "개인 키 (0원)",
    "PRO": "PRO (1,990원/월)",
    "ULTRA": "ULTRA (4,900원/월)",
    "ADMIN": "최고 관리자",
}
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{3,30}$")


def get_kst_today_str() -> str:
    """한국 표준시(KST) 기준 오늘 날짜 YYYY-MM-DD를 반환한다."""
    return datetime.now(KST).strftime("%Y-%m-%d")


def get_kst_week_str() -> str:
    """한국 표준시(KST) 기준 ISO 주차 YYYY-Www를 반환한다 (매주 월요일 초기화)."""
    return datetime.now(KST).strftime("%G-W%V")


def is_premium_model(model_name: str | None) -> bool:
    """3.8 이상 또는 Hydrogen 고급 추론 모델인지 여부를 반환한다."""
    raw = (model_name or "").strip().lower()
    return raw in PREMIUM_MODEL_IDS or "3.8" in raw or "hydrogen" in raw


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
    for key_name in ("SESSION_SECRET", "ADMIN_PASSWORD", "GEMINI_API_KEY", "GOOGLE_DRIVE_FOLDER_ID"):
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
    """자정(KST) · 주간(KST) 초기화 및 관리자 아이디 자동 ADMIN 승급을 동기화한다."""
    changed = False
    current_week = get_kst_week_str()
    if getattr(user, "last_used_date", "") != today:
        user.daily_count = 0
        user.last_used_date = today
        changed = True
    if hasattr(user, "last_used_week") and getattr(user, "last_used_week", "") != current_week:
        if hasattr(user, "weekly_premium_count"):
            user.weekly_premium_count = 0
        user.last_used_week = current_week
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


def resolve_active_gemini_api_key(
    user: UserRecord,
    server_api_key: str,
    model_name: str | None = None,
    force_server_key: bool = False,
) -> tuple[str, str]:
    """
    등급 및 작업 유형(업로드·태깅 vs 고급 추론 모델)에 따라 실제 Gemini 호출용 API 키를 라우팅한다.
    - force_server_key=True (탭 1 업로드/표지인식/크롭/태깅 전체): CUSTOM 포함 모든 등급 서버 키 사용
    - CUSTOM 등급: 고급 추론 모델(3.8 / Hydrogen, is_premium_model=True)만 개인 API 키 사용,
      기본 경량 모델(3.5 Flash-Lite 등)과 업로드·태깅 작업은 서버 키 사용
    - FREE / PRO / ULTRA / ADMIN: 서버 키 사용
    """
    server_key = (server_api_key or "").strip()
    if force_server_key:
        return server_key, "서버 키 (업로드·태깅 공용)"
    if user.tier == "CUSTOM":
        if model_name is None or is_premium_model(model_name):
            return user.custom_api_key.strip(), "개인 키 모드 (CUSTOM · 고급 추론)"
        return server_key, "서버 키 (기본·업로드 공용)"
    return server_key, f"서버 키 ({TIER_LABELS.get(user.tier, user.tier)})"


def get_remaining_free_quota(user: UserRecord) -> int | None:
    """개편 후 FREE 등급은 기본 추출·업로드가 무제한(3.5 Flash-Lite 고정)이므로 None을 반환한다."""
    return None


def get_max_batch_questions(user: UserRecord | None) -> int:
    """등급별 1회 최대 생성·변형 문항 수 제한을 반환한다 (FREE=5문항, PRO/ULTRA/CUSTOM/ADMIN=20문항)."""
    if user is None or user.tier == "FREE":
        return FREE_MAX_BATCH_QUESTIONS
    return PAID_MAX_BATCH_QUESTIONS


def get_weekly_premium_limit(user: UserRecord | None) -> int | None:
    """등급별 주간 고급 모델(3.8 / Hydrogen) 생성 문항 한도를 반환한다 (PRO=50, ULTRA=200, 그 외 None)."""
    if user is None:
        return None
    return WEEKLY_PREMIUM_LIMITS.get(user.tier)


def get_remaining_pro_weekly_quota(user: UserRecord) -> int | None:
    """PRO/ULTRA 등급의 이번 주(KST) 남은 고급 모델 생성 문항 수를 반환한다. 타 등급은 None."""
    limit = get_weekly_premium_limit(user)
    if limit is None:
        return None
    current_week = get_kst_week_str()
    used_week = (
        getattr(user, "weekly_premium_count", 0)
        if getattr(user, "last_used_week", "") == current_week
        else 0
    )
    return max(0, limit - int(used_week or 0))


def can_use_premium_features(user: UserRecord) -> bool:
    """문제 자동 개조 및 맞춤형 시험지 PDF 병합 허용 여부 (개편 후 전 등급 허용, FREE는 3.5 Flash-Lite 기본)."""
    return user.tier in ("FREE", "CUSTOM", "PRO", "ULTRA", "ADMIN")


def can_access_shared_concept_db(user: UserRecord | None) -> bool:
    """공용 개념서·교과서 DB 구축(업로드) 및 단원별 불러오기 권한 여부 (PRO / ULTRA / ADMIN 전용)."""
    if user is None:
        return False
    return user.tier in ("PRO", "ULTRA", "ADMIN")


def resolve_model_for_user(user: UserRecord, requested_model: str) -> tuple[str, str]:
    """
    회원 등급과 잔여 해금/주간 한도에 따라 실제 호출할 Gemini 모델 ID와 안내 메시지를 결정한다.
    - FREE: 기본 추론은 항상 gemini-3.5-flash-lite. 단, 기출 3회 업로드당 1회 해금된 크레딧이 있으면 3.8/Hydrogen 1회 허용.
    - CUSTOM / ADMIN: 선택한 모든 모델 무제한 허용 (CUSTOM은 개인 키 사용).
    - PRO(주간 50문항) / ULTRA(주간 200문항): 3.5 계열 무제한, 3.8/Hydrogen은 주간 문항 한도 내 허용(소진 시 gemini-3.5-flash-lite로 우회).
    """
    req = (requested_model or "").strip() or "gemini-3.5-flash-lite"
    if user.tier in ("CUSTOM", "ADMIN"):
        return req, ""

    if user.tier in ("PRO", "ULTRA"):
        if is_premium_model(req):
            rem = get_remaining_pro_weekly_quota(user) or 0
            if rem <= 0:
                return (
                    "gemini-3.5-flash-lite",
                    f"{user.tier} 주간 고급 모델 한도({get_weekly_premium_limit(user)}문항)를 모두 사용해 Gemini 3.5 Flash-Lite로 실행됩니다.",
                )
        return req, ""

    # FREE 등급
    credits = int(getattr(user, "premium_unlock_credits", 0) or 0)
    uploaded_cnt = int(getattr(user, "uploaded_exam_count", 0) or 0)
    if is_premium_model(req):
        if credits > 0:
            return req, ""
        progress = uploaded_cnt % FREE_EXAMS_PER_UNLOCK
        return (
            "gemini-3.5-flash-lite",
            f"FREE 등급은 기출 PDF {FREE_EXAMS_PER_UNLOCK}회 업로드마다 고급 모델(3.8/Hydrogen) 1회 해금됩니다 "
            f"(현재 {progress}/{FREE_EXAMS_PER_UNLOCK}회). Gemini 3.5 Flash-Lite로 실행됩니다.",
        )

    if req != "gemini-3.5-flash-lite":
        return "gemini-3.5-flash-lite", ""
    return req, ""


def record_exam_upload_for_user(
    storage: StorageManager,
    username: str,
) -> tuple[UserRecord | None, bool]:
    """
    기출 PDF 업로드·태깅 완료 시 회원의 누적 업로드 횟수를 +1 하고,
    FREE 등급인 경우 3회마다 고급 모델(3.8/Hydrogen) 해금 크레딧 1회를 지급한다.
    """
    norm_id = normalize_username(username)
    with _storage_lock(storage):
        users = storage.load_users()
        today = get_kst_today_str()
        admin_names = resolve_admin_config().admin_usernames
        target: UserRecord | None = None
        unlocked = False

        for u in users:
            if u.username == norm_id:
                _sync_user_state(u, today, admin_names)
                cur_uploads = int(getattr(u, "uploaded_exam_count", 0) or 0) + 1
                if hasattr(u, "uploaded_exam_count"):
                    u.uploaded_exam_count = cur_uploads
                if u.tier == "FREE" and cur_uploads % FREE_EXAMS_PER_UNLOCK == 0:
                    if hasattr(u, "premium_unlock_credits"):
                        u.premium_unlock_credits = int(getattr(u, "premium_unlock_credits", 0) or 0) + 1
                    unlocked = True
                target = u
                break

        if target is not None:
            storage.save_users(users)
        return target, unlocked


def consume_premium_model_usage(
    storage: StorageManager,
    username: str,
    used_model: str,
    count: int = 1,
) -> UserRecord | None:
    """
    고급 모델(3.8 / Hydrogen) 작업 실행 시 등급별 크레딧 또는 주간 문항 사용량을 차감/누적한다.
    - FREE: 실행당 해금 크레딧 1회 차감 (1회 최대 5문항)
    - PRO / ULTRA: 실제 생성된 문항 수(count)만큼 주간 한도(PRO 50 / ULTRA 200문항)에서 차감
    - CUSTOM / ADMIN 또는 일반(3.5) 모델 실행 시에는 차감하지 않는다.
    """
    if not is_premium_model(used_model):
        return None
    step_count = max(1, int(count or 1))
    norm_id = normalize_username(username)
    with _storage_lock(storage):
        users = storage.load_users()
        today = get_kst_today_str()
        admin_names = resolve_admin_config().admin_usernames
        target: UserRecord | None = None
        changed = False

        for u in users:
            if u.username == norm_id:
                if _sync_user_state(u, today, admin_names):
                    changed = True
                cur_credits = int(getattr(u, "premium_unlock_credits", 0) or 0)
                if u.tier == "FREE" and cur_credits > 0:
                    if hasattr(u, "premium_unlock_credits"):
                        u.premium_unlock_credits = cur_credits - 1
                    changed = True
                elif u.tier in ("PRO", "ULTRA"):
                    if hasattr(u, "weekly_premium_count"):
                        u.weekly_premium_count = int(getattr(u, "weekly_premium_count", 0) or 0) + step_count
                    if hasattr(u, "last_used_week"):
                        u.last_used_week = get_kst_week_str()
                    changed = True
                target = u
                break

        if changed and target is not None:
            storage.save_users(users)
        return target


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
