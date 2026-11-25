from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from src.auth.accounts import (
    _flush_ledger_events,
    _sync_user_state,
    get_effective_tier,
    get_user_by_username,
)
from src.auth.tokens import (
    DAILY_FREE_AI_LIMITS,
    DAILY_SHARED_UPLOAD_LIMITS,
    FREE_MAX_BATCH_QUESTIONS,
    FREE_MONTHLY_CREDIT_LIMIT,
    FREE_MONTHLY_TOKEN_LIMIT,
    KST,
    MONTHLY_CREDIT_LIMITS,
    MONTHLY_TOKEN_LIMITS,
    TIER_LABELS,
    _storage_lock,
    get_kst_month_str,
    get_kst_today_str,
    normalize_username,
)
from src.config import FLASH_LITE_MODEL, resolve_admin_config

if TYPE_CHECKING:
    from src.models import UserRecord
    from src.storage import StorageManager

_ACTIVE_JOB_LOCK = threading.Lock()
_ACTIVE_CREDIT_USERS: set[str] = set()


@dataclass
class CreditReservation:
    """동시 실행 방지 및 선예치·자동 환불 정산을 위한 크레딧 예약 핸들."""

    username: str
    is_admin: bool
    reserved_total: int
    reserved_from_base: int
    reserved_from_bonus: int
    reason: str
    settled: bool = False


def is_server_api_key_user(user: UserRecord | None) -> bool:
    """전 등급이 서버 공용 Gemini API 키를 단일 사용한다."""
    _ = user
    return True


def resolve_active_gemini_api_key(
    user: UserRecord,
    server_api_key: str,
    model_name: str | None = None,
    force_server_key: bool = False,
) -> tuple[str, str]:
    """전 등급(FREE / PRO / ULTRA / ADMIN)에 대해 서버 공용 Gemini API 키를 라우팅한다."""
    _ = (model_name, force_server_key)
    server_key = (server_api_key or "").strip()
    eff_tier = get_effective_tier(user)
    return server_key, f"서버 키 ({TIER_LABELS.get(eff_tier, eff_tier)})"


def get_remaining_free_quota(user: UserRecord) -> int | None:
    """하위 호환용; 가용 크레딧은 get_total_available_credits()를 사용한다."""
    return get_total_available_credits(user)


def get_max_batch_questions(user: UserRecord | None) -> int | None:
    """등급별 1회 최대 생성·변형 문항 수 제한을 반환한다 (FREE=5문항, PRO/ULTRA/ADMIN=제한 없음 None)."""
    if user is None or get_effective_tier(user) == "FREE":
        return FREE_MAX_BATCH_QUESTIONS
    return None


def get_monthly_credit_limit(user: UserRecord | None) -> int | None:
    """등급별 월간 기본 크레딧 한도를 반환한다 (FREE=30, PRO=600, ULTRA=2,400, ADMIN=None)."""
    if user is None:
        return FREE_MONTHLY_CREDIT_LIMIT
    eff_tier = get_effective_tier(user)
    if eff_tier == "ADMIN":
        return None
    return MONTHLY_CREDIT_LIMITS.get(eff_tier, FREE_MONTHLY_CREDIT_LIMIT)


def get_used_monthly_credits(user: UserRecord | None) -> int:
    """이번 달(KST YYYY-MM) 사용한 기본 크레딧 수를 반환한다."""
    if user is None:
        return 0
    current_month = get_kst_month_str()
    if getattr(user, "last_used_month", "") != current_month:
        return 0
    return max(0, int(getattr(user, "monthly_used_credits", 0) or 0))


def get_remaining_monthly_base_credits(user: UserRecord | None) -> int | None:
    """이번 달(KST) 기본 크레딧 잔여량을 반환한다. 무제한(ADMIN)은 None."""
    limit = get_monthly_credit_limit(user)
    if limit is None:
        return None
    used = get_used_monthly_credits(user)
    return max(0, limit - used)


def get_bonus_credits(user: UserRecord | None) -> int:
    """월초에 소멸되지 않는 보너스 크레딧 잔액을 반환한다."""
    if user is None:
        return 0
    return max(0, int(getattr(user, "bonus_credits", 0) or 0))


def get_total_available_credits(user: UserRecord | None) -> int | None:
    """이번 달 기본 잔여 크레딧 + 보너스 크레딧 합계를 반환한다. ADMIN은 None."""
    base_rem = get_remaining_monthly_base_credits(user)
    if base_rem is None:
        return None
    return base_rem + get_bonus_credits(user)


def get_daily_free_ai_used(user: UserRecord | None) -> int:
    """금일(KST) 사용한 0C 무료 AI 보조 기능(해설+채팅+신규영단어) 횟수를 반환한다."""
    if user is None:
        return 0
    if getattr(user, "last_used_date", "") != get_kst_today_str():
        return 0
    return max(0, int(getattr(user, "daily_free_ai_count", 0) or 0))


def get_daily_free_ai_limit(user: UserRecord | None) -> int | None:
    """등급별 0C 무료 AI 보조 기능 일일 한도를 반환한다 (ADMIN=None)."""
    if user is None:
        return DAILY_FREE_AI_LIMITS["FREE"]
    eff_tier = get_effective_tier(user)
    if eff_tier == "ADMIN":
        return None
    return DAILY_FREE_AI_LIMITS.get(eff_tier, DAILY_FREE_AI_LIMITS["FREE"])


def get_daily_shared_upload_used(user: UserRecord | None) -> int:
    """금일(KST) 수행한 공용 DB 신규 업로드 횟수를 반환한다."""
    if user is None:
        return 0
    if getattr(user, "last_used_date", "") != get_kst_today_str():
        return 0
    return max(0, int(getattr(user, "daily_shared_upload_count", 0) or 0))


def get_daily_shared_upload_limit(user: UserRecord | None) -> int | None:
    """등급별 공용 DB 신규 업로드 일일 한도를 반환한다 (FREE=0, PRO=3, ULTRA=8, ADMIN=None)."""
    if user is None:
        return 0
    eff_tier = get_effective_tier(user)
    if eff_tier == "ADMIN":
        return None
    return DAILY_SHARED_UPLOAD_LIMITS.get(eff_tier, 0)


def _append_ledger_safe(storage: StorageManager, entry: dict[str, object]) -> None:
    if hasattr(storage, "append_credit_ledger"):
        try:
            storage.append_credit_ledger(entry)
        except Exception:
            pass


def check_credit_quota(
    storage: StorageManager,
    username: str,
    required_credits: int = 1,
    action_label: str = "",
) -> tuple[bool, int | None, int | None, str]:
    """
    작업 실행 전 가용 크레딧(월간 기본 잔여 + 보너스) 충족 여부를 조회한다.
    반환: (허용 여부, 가용 크레딧 또는 None, 월간 기본 한도 또는 None, 안내 메시지)
    """
    user = get_user_by_username(storage, username)
    if user is None:
        return False, 0, FREE_MONTHLY_CREDIT_LIMIT, "로그인 계정 정보를 확인할 수 없습니다. 다시 로그인해 주세요."

    limit = get_monthly_credit_limit(user)
    if limit is None:
        return True, None, None, ""

    need = max(0, int(required_credits or 0))
    avail = get_total_available_credits(user) or 0
    if need <= 0:
        return True, avail, limit, ""

    if avail < need:
        eff_tier = get_effective_tier(user)
        tier_lbl = TIER_LABELS.get(eff_tier, eff_tier)
        act_part = f"{action_label} · " if action_label else ""
        if eff_tier == "FREE":
            upgrade_hint = "PRO(월 600C · 9,900원) 또는 ULTRA(월 2,400C · 19,900원) 등급으로 업그레이드해 주세요."
        elif eff_tier == "PRO":
            upgrade_hint = "매월 1일(KST) 초기화되며, 더 넉넉한 한도가 필요하시면 ULTRA(월 2,400C · 4배)로 업그레이드해 주세요."
        else:
            upgrade_hint = "매월 1일(KST) 자동 초기화됩니다. 급한 경우 관리자에게 보너스 크레딧 지급을 요청해 주세요."
        return (
            False,
            avail,
            limit,
            f"🚫 크레딧이 부족합니다 ({act_part}필요 {need:,}C / 현재 가용 {avail:,}C · {tier_lbl}). {upgrade_hint}",
        )
    return True, avail, limit, ""


def acquire_credit_reservation(
    storage: StorageManager,
    username: str,
    reserve_credits: int,
    reason: str = "",
) -> tuple[bool, CreditReservation | None, str]:
    """
    동일 계정의 동시 유료 실행을 차단하고, 가용 크레딧을 원자적으로 선예치(선차감)한다.
    반드시 settle_credit_reservation()으로 짝을 맞춰 종료/환불해야 한다.
    """
    norm_id = normalize_username(username)
    if not norm_id:
        return False, None, "로그인 계정 정보를 확인할 수 없습니다. 다시 로그인해 주세요."

    need = max(0, int(reserve_credits or 0))
    clean_reason = (reason or "").strip() or "크레딧 사용"

    with _ACTIVE_JOB_LOCK:
        if norm_id in _ACTIVE_CREDIT_USERS:
            return (
                False,
                None,
                "⏳ 현재 계정의 이전 출제·분석 작업이 진행 중입니다. 완료될 때까지 잠시만 기다려 주세요.",
            )
        _ACTIVE_CREDIT_USERS.add(norm_id)

    try:
        with _storage_lock(storage):
            users = storage.load_users()
            today = get_kst_today_str()
            now_iso = datetime.now(KST).isoformat(timespec="seconds")
            admin_names = resolve_admin_config().admin_usernames

            target: UserRecord | None = None
            changed = False
            expired_events: list[dict[str, object]] = []
            for u in users:
                if u.username == norm_id:
                    if _sync_user_state(u, today, admin_names, expired_events_out=expired_events):
                        changed = True
                    target = u
                    break

            if target is None:
                if changed:
                    storage.save_users(users)
                    _flush_ledger_events(storage, expired_events)
                _ACTIVE_CREDIT_USERS.discard(norm_id)
                return False, None, "로그인 계정 정보를 확인할 수 없습니다. 다시 로그인해 주세요."

            if target.tier == "ADMIN" or need == 0:
                if changed:
                    storage.save_users(users)
                    _flush_ledger_events(storage, expired_events)
                return (
                    True,
                    CreditReservation(
                        username=norm_id,
                        is_admin=(target.tier == "ADMIN"),
                        reserved_total=need,
                        reserved_from_base=0,
                        reserved_from_bonus=0,
                        reason=clean_reason,
                    ),
                    "",
                )

            limit = MONTHLY_CREDIT_LIMITS.get(target.tier, FREE_MONTHLY_CREDIT_LIMIT)
            base_used = max(0, int(getattr(target, "monthly_used_credits", 0) or 0))
            base_rem = max(0, limit - base_used)
            bonus_rem = max(0, int(getattr(target, "bonus_credits", 0) or 0))
            avail = base_rem + bonus_rem

            if avail < need:
                if changed:
                    storage.save_users(users)
                    _flush_ledger_events(storage, expired_events)
                tier_lbl = TIER_LABELS.get(target.tier, target.tier)
                if target.tier == "FREE":
                    hint = "PRO(월 600C) 또는 ULTRA(월 2,400C) 등급으로 업그레이드해 주세요."
                elif target.tier == "PRO":
                    hint = "매월 1일(KST) 초기화되며, ULTRA(월 2,400C) 업그레이드 또는 보너스 크레딧을 이용해 주세요."
                else:
                    hint = "매월 1일(KST) 자동 초기화됩니다. 관리자에게 보너스 크레딧 지급을 요청해 주세요."
                with _ACTIVE_JOB_LOCK:
                    _ACTIVE_CREDIT_USERS.discard(norm_id)
                return (
                    False,
                    None,
                    f"🚫 크레딧이 부족합니다 ({clean_reason} · 필요 {need:,}C / 현재 가용 {avail:,}C · {tier_lbl}). {hint}",
                )

            from_base = min(need, base_rem)
            from_bonus = need - from_base
            target.monthly_used_credits = base_used + from_base
            target.bonus_credits = bonus_rem - from_bonus
            target.total_used_credits = max(0, int(getattr(target, "total_used_credits", 0) or 0)) + need
            target.total_base_credits_used = (
                max(0, int(getattr(target, "total_base_credits_used", 0) or 0)) + from_base
            )
            storage.save_users(users)
            _flush_ledger_events(storage, expired_events)

            avail_after = max(0, limit - target.monthly_used_credits) + target.bonus_credits
            _append_ledger_safe(
                storage,
                {
                    "ts": now_iso,
                    "username": norm_id,
                    "event": "USE",
                    "delta": -need,
                    "reason": clean_reason,
                    "base_used_after": target.monthly_used_credits,
                    "bonus_after": target.bonus_credits,
                    "avail_after": avail_after,
                    "actor": norm_id,
                },
            )
            return (
                True,
                CreditReservation(
                    username=norm_id,
                    is_admin=False,
                    reserved_total=need,
                    reserved_from_base=from_base,
                    reserved_from_bonus=from_bonus,
                    reason=clean_reason,
                ),
                "",
            )
    except Exception:
        with _ACTIVE_JOB_LOCK:
            _ACTIVE_CREDIT_USERS.discard(norm_id)
        raise


def settle_credit_reservation(
    storage: StorageManager,
    reservation: CreditReservation | None,
    actual_credits: int,
    detail_note: str = "",
) -> None:
    """
    선예치된 크레딧을 최종 성공 건수 기준으로 정산하고, 미사용분은 즉시 자동 환불하며 실행 락을 해제한다.
    """
    if reservation is None or reservation.settled:
        return
    reservation.settled = True

    try:
        actual = max(0, min(reservation.reserved_total, int(actual_credits or 0)))
        if reservation.is_admin:
            if actual > 0:
                with _storage_lock(storage):
                    users = storage.load_users()
                    today = get_kst_today_str()
                    now_iso = datetime.now(KST).isoformat(timespec="seconds")
                    admin_names = resolve_admin_config().admin_usernames
                    for u in users:
                        if u.username == reservation.username:
                            _sync_user_state(u, today, admin_names)
                            u.monthly_used_credits = max(0, int(getattr(u, "monthly_used_credits", 0) or 0)) + actual
                            u.total_used_credits = max(0, int(getattr(u, "total_used_credits", 0) or 0)) + actual
                            storage.save_users(users)
                            _append_ledger_safe(
                                storage,
                                {
                                    "ts": now_iso,
                                    "username": reservation.username,
                                    "event": "USE",
                                    "delta": -actual,
                                    "reason": detail_note or reservation.reason,
                                    "base_used_after": u.monthly_used_credits,
                                    "bonus_after": int(getattr(u, "bonus_credits", 0) or 0),
                                    "avail_after": None,
                                    "actor": reservation.username,
                                },
                            )
                            break
            return

        refund = reservation.reserved_total - actual
        if refund <= 0:
            return

        with _storage_lock(storage):
            users = storage.load_users()
            today = get_kst_today_str()
            now_iso = datetime.now(KST).isoformat(timespec="seconds")
            admin_names = resolve_admin_config().admin_usernames
            for u in users:
                if u.username == reservation.username:
                    expired_events: list[dict[str, object]] = []
                    _sync_user_state(u, today, admin_names, expired_events_out=expired_events)
                    refund_to_bonus = min(refund, reservation.reserved_from_bonus)
                    refund_to_base = refund - refund_to_bonus
                    u.bonus_credits = max(0, int(getattr(u, "bonus_credits", 0) or 0)) + refund_to_bonus
                    u.monthly_used_credits = max(
                        0, int(getattr(u, "monthly_used_credits", 0) or 0) - refund_to_base
                    )
                    u.total_used_credits = max(
                        0, int(getattr(u, "total_used_credits", 0) or 0) - refund
                    )
                    u.total_base_credits_used = max(
                        0, int(getattr(u, "total_base_credits_used", 0) or 0) - refund_to_base
                    )
                    storage.save_users(users)
                    _flush_ledger_events(storage, expired_events)
                    limit = MONTHLY_CREDIT_LIMITS.get(u.tier, FREE_MONTHLY_CREDIT_LIMIT)
                    avail_after = max(0, limit - u.monthly_used_credits) + u.bonus_credits
                    _append_ledger_safe(
                        storage,
                        {
                            "ts": now_iso,
                            "username": reservation.username,
                            "event": "REFUND",
                            "delta": refund,
                            "reason": detail_note or f"{reservation.reason} 미성공분 자동 환불 ({actual:,}C 확정)",
                            "base_used_after": u.monthly_used_credits,
                            "bonus_after": u.bonus_credits,
                            "avail_after": avail_after,
                            "actor": "system",
                        },
                    )
                    break
    finally:
        with _ACTIVE_JOB_LOCK:
            _ACTIVE_CREDIT_USERS.discard(reservation.username)


def consume_daily_free_ai_quota(
    storage: StorageManager,
    username: str,
    action_label: str = "무료 AI 보조 기능",
) -> tuple[bool, str]:
    """0C 무료 AI 기능(해설·1:1채팅·미등록 영단어 AI조회)의 등급별 일일 호출 한도를 검사하고 +1 기록한다."""
    norm_id = normalize_username(username)
    if not norm_id:
        return False, "로그인 계정 정보를 확인할 수 없습니다. 다시 로그인해 주세요."

    with _storage_lock(storage):
        users = storage.load_users()
        today = get_kst_today_str()
        admin_names = resolve_admin_config().admin_usernames
        for u in users:
            if u.username == norm_id:
                expired_events: list[dict[str, object]] = []
                changed = _sync_user_state(
                    u, today, admin_names, expired_events_out=expired_events
                )
                if u.tier == "ADMIN":
                    if changed:
                        storage.save_users(users)
                        _flush_ledger_events(storage, expired_events)
                    return True, ""
                limit = DAILY_FREE_AI_LIMITS.get(u.tier, DAILY_FREE_AI_LIMITS["FREE"])
                used = max(0, int(getattr(u, "daily_free_ai_count", 0) or 0))
                if used >= limit:
                    if changed:
                        storage.save_users(users)
                        _flush_ledger_events(storage, expired_events)
                    tier_lbl = TIER_LABELS.get(u.tier, u.tier)
                    return (
                        False,
                        f"🚫 오늘의 무료 AI 보조 기능 한도를 모두 사용했습니다 ({used}/{limit}회 · {action_label} · {tier_lbl}). 매일 자정(KST) 자동 초기화됩니다.",
                    )
                u.daily_free_ai_count = used + 1
                storage.save_users(users)
                _flush_ledger_events(storage, expired_events)
                return True, ""
    return False, "로그인 계정 정보를 확인할 수 없습니다. 다시 로그인해 주세요."


def refund_daily_free_ai_quota(storage: StorageManager, username: str) -> None:
    """0C 무료 AI 호출이 예외로 실패했을 때 일일 사용 횟수를 1회 복원한다."""
    norm_id = normalize_username(username)
    if not norm_id:
        return
    with _storage_lock(storage):
        users = storage.load_users()
        today = get_kst_today_str()
        admin_names = resolve_admin_config().admin_usernames
        for u in users:
            if u.username == norm_id:
                expired_events: list[dict[str, object]] = []
                changed = _sync_user_state(
                    u, today, admin_names, expired_events_out=expired_events
                )
                cur = int(getattr(u, "daily_free_ai_count", 0) or 0)
                if cur > 0:
                    u.daily_free_ai_count = cur - 1
                    changed = True
                if changed:
                    storage.save_users(users)
                    _flush_ledger_events(storage, expired_events)
                break


def check_daily_shared_upload_quota(
    storage: StorageManager,
    username: str,
    count_needed: int = 1,
) -> tuple[bool, int | None, str]:
    """PRO/ULTRA 등급의 공용 DB(기출·개념서) 일일 신규 업로드 한도 초과 여부를 확인한다."""
    user = get_user_by_username(storage, username)
    if user is None:
        return False, 0, "로그인 계정 정보를 확인할 수 없습니다. 다시 로그인해 주세요."
    eff_tier = get_effective_tier(user)
    if eff_tier == "ADMIN":
        return True, None, ""
    if eff_tier not in ("PRO", "ULTRA"):
        return False, 0, "🔒 공용 DB 신규 업로드는 PRO · ULTRA · ADMIN 전용 기능입니다."

    limit = DAILY_SHARED_UPLOAD_LIMITS.get(eff_tier, 0)
    used = get_daily_shared_upload_used(user)
    rem = max(0, limit - used)
    need = max(1, int(count_needed or 1))
    if used + need > limit:
        tier_lbl = TIER_LABELS.get(eff_tier, eff_tier)
        return (
            False,
            rem,
            f"🚫 오늘의 공용 DB 신규 업로드 한도를 초과했습니다 (오늘 {used}/{limit}회 사용 · 요청 {need}건 · {tier_lbl}). 매일 자정(KST) 초기화됩니다.",
        )
    return True, rem, ""


def consume_daily_shared_upload_quota(
    storage: StorageManager,
    username: str,
    count: int = 1,
) -> None:
    """공용 DB 신규 업로드(기출/개념서) 성공 시 금일 업로드 카운터를 증가시킨다."""
    norm_id = normalize_username(username)
    add_n = max(0, int(count or 0))
    if not norm_id or add_n <= 0:
        return
    with _storage_lock(storage):
        users = storage.load_users()
        today = get_kst_today_str()
        admin_names = resolve_admin_config().admin_usernames
        for u in users:
            if u.username == norm_id:
                expired_events: list[dict[str, object]] = []
                changed = _sync_user_state(
                    u, today, admin_names, expired_events_out=expired_events
                )
                if u.tier != "ADMIN":
                    u.daily_shared_upload_count = max(
                        0, int(getattr(u, "daily_shared_upload_count", 0) or 0)
                    ) + add_n
                    changed = True
                if changed:
                    storage.save_users(users)
                    _flush_ledger_events(storage, expired_events)
                break


def reset_user_monthly_credits(
    storage: StorageManager,
    target_usernames: list[str],
    actor_username: str = "admin",
) -> tuple[int, str | None]:
    """관리자가 지정한 회원의 이번 달 사용 크레딧만 0으로 초기화한다 (보너스 크레딧·누적 원가 통계는 보존)."""
    targets = {normalize_username(u) for u in target_usernames if normalize_username(u)}
    if not targets:
        return 0, None

    today = get_kst_today_str()
    current_month = get_kst_month_str()
    now_iso = datetime.now(KST).isoformat(timespec="seconds")
    admin_names = resolve_admin_config().admin_usernames
    actor_norm = normalize_username(actor_username) or "admin"

    with _storage_lock(storage):
        users = storage.load_users()
        reset_cnt = 0
        ledger_entries: list[dict[str, object]] = []
        expired_events: list[dict[str, object]] = []
        for u in users:
            if u.username in targets:
                _sync_user_state(u, today, admin_names, expired_events_out=expired_events)
                prev_used = max(0, int(getattr(u, "monthly_used_credits", 0) or 0))
                u.monthly_used_credits = 0
                u.last_used_month = current_month
                reset_cnt += 1
                bonus_now = max(0, int(getattr(u, "bonus_credits", 0) or 0))
                avail_after: int | None = (
                    None
                    if u.tier == "ADMIN"
                    else MONTHLY_CREDIT_LIMITS.get(u.tier, FREE_MONTHLY_CREDIT_LIMIT) + bonus_now
                )
                ledger_entries.append(
                    {
                        "ts": now_iso,
                        "username": u.username,
                        "event": "MONTHLY_RESET",
                        "delta": prev_used,
                        "reason": f"관리자 월간 사용 크레딧 초기화 (기존 {prev_used:,}C → 0C)",
                        "base_used_after": 0,
                        "bonus_after": bonus_now,
                        "avail_after": avail_after,
                        "actor": actor_norm,
                    }
                )
        if reset_cnt > 0 or expired_events:
            err = storage.save_users(users)
            _flush_ledger_events(storage, expired_events)
            for entry in ledger_entries:
                _append_ledger_safe(storage, entry)
            return reset_cnt, err
        return 0, None


# ---------------------------------------------------------------------------
# 권한 판별 헬퍼 (공용 DB 기여 vs 관리자 단독 수정·삭제)
# ---------------------------------------------------------------------------

def can_contribute_shared_db(user: UserRecord | None) -> bool:
    """공용 기출·개념서 DB 신규 업로드 및 공용 개념서 불러오기 권한 여부 (PRO / ULTRA / ADMIN)."""
    if user is None:
        return False
    return get_effective_tier(user) in ("PRO", "ULTRA", "ADMIN")


def can_access_shared_concept_db(user: UserRecord | None) -> bool:
    """공용 개념서·교과서 DB 구축(신규 업로드) 및 단원별 불러오기 권한 여부 (PRO / ULTRA / ADMIN)."""
    return can_contribute_shared_db(user)


def can_admin_shared_db(user: UserRecord | None) -> bool:
    """공용 DB 수정·삭제·회차명 변경·다시 태깅·NCIC 기준표 관리 권한 여부 (ADMIN 단독 전용)."""
    if user is None:
        return False
    return get_effective_tier(user) == "ADMIN"


def can_use_premium_features(user: UserRecord) -> bool:
    """문제 자동 개조 및 맞춤형 시험지 PDF 병합 허용 여부 (전 등급 허용, 크레딧 차감 적용)."""
    return get_effective_tier(user) in ("FREE", "PRO", "ULTRA", "ADMIN")


def resolve_model_for_user(user: UserRecord, requested_model: str) -> tuple[str, str]:
    """전 등급·전 기능을 Gemini 3.5 Flash-Lite 단일 모델로 라우팅한다."""
    _ = (user, requested_model)
    return FLASH_LITE_MODEL, ""


# ---------------------------------------------------------------------------
# 백엔드 실제 API 토큰 · 원화 원가 추적 (관리자 모니터링용)
# ---------------------------------------------------------------------------

def get_monthly_token_limit(user: UserRecord | None) -> int | None:
    """하위 호환용 월간 토큰 참고 한도 반환."""
    if user is None:
        return FREE_MONTHLY_TOKEN_LIMIT
    return MONTHLY_TOKEN_LIMITS.get(get_effective_tier(user))


def get_used_monthly_tokens(user: UserRecord | None) -> int:
    """이번 달(KST YYYY-MM) 누적 사용 토큰 수를 반환한다."""
    if user is None:
        return 0
    current_month = get_kst_month_str()
    if getattr(user, "last_used_month", "") != current_month:
        return 0
    return max(0, int(getattr(user, "monthly_total_tokens", 0) or 0))


def get_remaining_monthly_tokens(user: UserRecord | None) -> int | None:
    """하위 호환용; 실제 잔액은 get_total_available_credits()를 사용한다."""
    limit = get_monthly_token_limit(user)
    if limit is None:
        return None
    used = get_used_monthly_tokens(user)
    return max(0, limit - used)


def check_token_quota(
    storage: StorageManager,
    username: str,
) -> tuple[bool, int | None, int | None, str]:
    """하위 호환용 래퍼: 0C 이상 계정 유효성 및 최소 크레딧 확인으로 위임한다."""
    return check_credit_quota(storage, username, required_credits=0)


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
    """하위 호환용: 월간 사용 크레딧 초기화(reset_user_monthly_credits)를 함께 수행한다."""
    return reset_user_monthly_credits(storage, target_usernames)


def get_weekly_premium_limit(user: UserRecord | None) -> int | None:
    _ = user
    return None


def get_remaining_pro_weekly_quota(user: UserRecord) -> int | None:
    _ = user
    return None


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
    _ = (storage, username, used_model, count)
    return None


def consume_extraction_quota(
    storage: StorageManager,
    username: str,
    extracted_count: int,
) -> UserRecord | None:
    """하위 호환용: 추출 문항 통계(daily_count)만 기록한다."""
    norm_id = normalize_username(username)
    with _storage_lock(storage):
        users = storage.load_users()
        today = get_kst_today_str()
        admin_names = resolve_admin_config().admin_usernames
        target: UserRecord | None = None

        for u in users:
            if u.username == norm_id:
                _sync_user_state(u, today, admin_names)
                u.daily_count += max(0, extracted_count)
                u.last_used_date = today
                target = u
                break

        if target is not None:
            storage.save_users(users)
        return target
