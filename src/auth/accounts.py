from __future__ import annotations

import secrets
import time
from datetime import datetime
from typing import TYPE_CHECKING

from src.auth.tokens import (
    KST,
    TIER_LABELS,
    USERNAME_RE,
    VALID_TIERS,
    _storage_lock,
    get_kst_month_str,
    get_kst_today_str,
    get_kst_week_str,
    hash_password,
    is_legacy_password_hash,
    normalize_username,
    verify_password,
)
from src.config import resolve_admin_config

if TYPE_CHECKING:
    from src.models import TierType, UserRecord
    from src.storage import StorageManager

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
        from src.models.user import UserRecord

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
        from src.models.user import UserRecord

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
