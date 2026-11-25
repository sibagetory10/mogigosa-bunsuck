from __future__ import annotations

import secrets
import threading
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from src.auth.accounts import (
    _flush_ledger_events,
    _sync_user_state,
    get_effective_tier,
    get_user_by_username,
    parse_kst_iso,
)
from src.auth.tokens import (
    FREE_MONTHLY_CREDIT_LIMIT,
    KST,
    MONTHLY_CREDIT_LIMITS,
    PRO_MONTHLY_CREDIT_LIMIT,
    PRO_MONTHLY_PRICE_KRW,
    TIER_LABELS,
    ULTRA_MONTHLY_CREDIT_LIMIT,
    ULTRA_MONTHLY_PRICE_KRW,
    _storage_lock,
    normalize_username,
)
from src.config import resolve_admin_config

if TYPE_CHECKING:
    from src.models import PaymentOrder, PaymentPlanType, UserRecord
    from src.storage import StorageManager

PAYMENT_PLAN_SPECS: dict[PaymentPlanType, dict[str, Any]] = {
    "PRO": {
        "tier": "PRO",
        "label": "PRO 30일권",
        "price_krw": PRO_MONTHLY_PRICE_KRW,
        "monthly_credits": PRO_MONTHLY_CREDIT_LIMIT,
        "duration_days": 30,
    },
    "ULTRA": {
        "tier": "ULTRA",
        "label": "ULTRA 30일권",
        "price_krw": ULTRA_MONTHLY_PRICE_KRW,
        "monthly_credits": ULTRA_MONTHLY_CREDIT_LIMIT,
        "duration_days": 30,
    },
}


def _resolve_cur_kst(now_dt: datetime | None = None) -> datetime:
    if now_dt is None:
        return datetime.now(KST)
    if now_dt.tzinfo is None:
        return now_dt.replace(tzinfo=KST)
    return now_dt.astimezone(KST)


def format_tier_expiry_badge(user: UserRecord | None, now_dt: datetime | None = None) -> str:
    """유료 회원의 만료일/잔여일수 배지 문자열을 반환한다 (무기한 수동 승급 포함)."""
    if user is None:
        return ""
    eff_tier = get_effective_tier(user, now_dt=now_dt)
    if eff_tier not in ("PRO", "ULTRA"):
        return ""
    raw_exp = str(getattr(user, "tier_expires_at", "") or "").strip()
    if not raw_exp:
        return "무기한 (관리자 지정)"
    exp_dt = parse_kst_iso(raw_exp)
    if exp_dt is None:
        return ""
    cur_dt = _resolve_cur_kst(now_dt)
    rem_sec = (exp_dt - cur_dt).total_seconds()
    if rem_sec <= 0:
        return ""
    days_left = int(rem_sec // 86400)
    if days_left >= 1:
        return f"{exp_dt.strftime('%Y-%m-%d')} 만료 (D-{days_left})"
    hours_left = max(1, int((rem_sec + 3599) // 3600))
    return f"{exp_dt.strftime('%Y-%m-%d %H:%M')} 만료 (D-Day · {hours_left}시간 남음)"


def get_order_consumed_base_credits(
    storage: StorageManager,
    order: PaymentOrder,
    user: UserRecord | None = None,
) -> int:
    """승인 완료된 주문에 대해 승인 시점 이후 회원이 실제 소비한 기본 크레딧을 계산한다."""
    if order.status != "APPROVED":
        return 0
    target_u = user if user is not None else get_user_by_username(storage, order.username)
    if target_u is None:
        return 0
    cur_total = max(0, int(getattr(target_u, "total_base_credits_used", 0) or 0))
    snap_total = max(0, int(getattr(order, "base_used_at_approval", 0) or 0))
    return max(0, cur_total - snap_total)


def send_discord_payment_notification(
    webhook_url: str,
    order: PaymentOrder | None = None,
    is_test: bool = False,
    timeout: int = 4,
) -> tuple[bool, str]:
    """관리자 Discord 웹훅 URL로 입금 확인 신청 또는 테스트 알림 Embed를 전송한다."""
    url = (webhook_url or "").strip()
    if not url:
        return False, "Discord 웹훅 URL이 설정되지 않았습니다."
    if not (
        url.startswith("https://discord.com/api/webhooks/")
        or url.startswith("https://discordapp.com/api/webhooks/")
    ):
        return False, "유효한 Discord 웹훅 URL 형식이 아닙니다."

    import requests

    if is_test or order is None:
        payload: dict[str, Any] = {
            "username": "MogiMogi 결제 알림",
            "embeds": [
                {
                    "title": "🔔 MogiMogi Discord 결제 알림 연동 테스트",
                    "description": "웹훅 연결이 정상 동작합니다! 회원이 입금 확인을 요청하면 이 채널로 즉시 알림이 도착합니다.",
                    "color": 0x2563EB,
                    "timestamp": datetime.now(KST).isoformat(timespec="seconds"),
                }
            ],
        }
    else:
        plan_spec = PAYMENT_PLAN_SPECS.get(order.plan_tier, {})
        plan_label = str(plan_spec.get("label") or f"{order.plan_tier} 30일권")
        fields: list[dict[str, Any]] = [
            {"name": "신청 회원", "value": f"@{order.username} ({order.user_nickname or '-'})", "inline": True},
            {"name": "요금제 · 금액", "value": f"{plan_label} · ₩{order.amount_krw:,}", "inline": True},
            {"name": "입금자명", "value": order.depositor_name, "inline": True},
            {"name": "주문 번호", "value": f"`{order.order_id}`", "inline": True},
        ]
        if (order.user_note or "").strip():
            fields.append({"name": "회원 메모", "value": order.user_note[:120], "inline": False})
        payload = {
            "username": "MogiMogi 결제 알림",
            "embeds": [
                {
                    "title": "💳 새 멤버십 입금 확인 요청 도착!",
                    "description": "관리자 탭의 **[입금 확인 승인 대기열]**에서 입금 내역 확인 후 원클릭 승인해 주세요.",
                    "color": 0x16A34A if order.plan_tier == "PRO" else 0xEA580C,
                    "fields": fields,
                    "timestamp": order.requested_at or datetime.now(KST).isoformat(timespec="seconds"),
                }
            ],
        }

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        if 200 <= resp.status_code < 300:
            return True, "Discord 알림을 성공적으로 전송했습니다."
        return False, f"Discord 응답 오류 (HTTP {resp.status_code}): {resp.text[:150]}"
    except Exception as exc:
        return False, f"Discord 알림 전송 실패: {exc}"


def test_discord_webhook(webhook_url: str) -> tuple[bool, str]:
    """관리자 탭 테스트 버튼용 Discord 웹훅 확인 함수."""
    return send_discord_payment_notification(webhook_url, order=None, is_test=True, timeout=5)


def submit_payment_order(
    storage: StorageManager,
    username: str,
    plan_tier: PaymentPlanType,
    depositor_name: str,
    user_note: str = "",
    now_dt: datetime | None = None,
) -> tuple[bool, str, PaymentOrder | None]:
    """회원의 PRO/ULTRA 30일권 입금 확인 요청을 원자적으로 생성하고 Discord 실시간 알림을 보낸다."""
    from src.models.payment import PaymentOrder

    norm_id = normalize_username(username)
    if not norm_id:
        return False, "로그인 계정 정보를 확인할 수 없습니다. 다시 로그인해 주세요.", None
    if plan_tier not in PAYMENT_PLAN_SPECS:
        return False, "지원하지 않는 요금제입니다. (PRO 또는 ULTRA만 선택 가능)", None

    clean_depositor = (depositor_name or "").strip()
    if not (1 <= len(clean_depositor) <= 30):
        return False, "입금자명은 1~30자 이내로 정확히 입력해 주세요.", None
    clean_note = (user_note or "").strip()[:120]

    cur_dt = _resolve_cur_kst(now_dt)
    now_iso = cur_dt.isoformat(timespec="seconds")
    plan_spec = PAYMENT_PLAN_SPECS[plan_tier]
    webhook_url = ""

    with _storage_lock(storage):
        user = get_user_by_username(storage, norm_id)
        if user is None:
            return False, "회원 계정 정보를 확인할 수 없습니다.", None
        if get_effective_tier(user, now_dt=cur_dt) == "ADMIN":
            return False, "최고 관리자(ADMIN) 계정은 별도 결제 신청이 필요하지 않습니다.", None

        settings = storage.load_payment_settings()
        if not settings.enabled or not settings.has_transfer_method:
            return False, "현재 관리자가 입금 계좌 정보를 준비 중입니다. 잠시 후 다시 시도해 주세요.", None

        orders = storage.load_payment_orders()
        if any(o.username == norm_id and o.status == "PENDING" for o in orders):
            return (
                False,
                "이미 승인 대기 중인 입금 확인 요청이 있습니다. 기존 요청이 처리되거나 취소한 뒤 다시 신청해 주세요.",
                None,
            )

        existing_ids = {o.order_id for o in orders}
        date_part = cur_dt.strftime("%Y%m%d")
        order_id = f"PAY-{date_part}-{secrets.token_hex(3).upper()}"
        while order_id in existing_ids:
            order_id = f"PAY-{date_part}-{secrets.token_hex(3).upper()}"

        new_order = PaymentOrder(
            order_id=order_id,
            username=norm_id,
            user_nickname=user.nickname,
            plan_tier=plan_tier,
            duration_days=int(plan_spec["duration_days"]),
            amount_krw=int(plan_spec["price_krw"]),
            depositor_name=clean_depositor,
            user_note=clean_note,
            status="PENDING",
            requested_at=now_iso,
        )
        orders.insert(0, new_order)
        storage.save_payment_orders(orders)
        webhook_url = (settings.discord_webhook_url or "").strip()

    if webhook_url:
        t = threading.Thread(
            target=send_discord_payment_notification,
            args=(webhook_url, new_order),
            kwargs={"is_test": False, "timeout": 4},
            name="discord-pay-notify",
            daemon=True,
        )
        t.start()

    return (
        True,
        f"입금 확인 요청({order_id})이 접수되었습니다! 관리자 확인 후 즉시 멤버십이 활성화됩니다.",
        new_order,
    )


def cancel_my_pending_order(
    storage: StorageManager,
    username: str,
    order_id: str,
    now_dt: datetime | None = None,
) -> tuple[bool, str]:
    """회원이 본인의 승인 대기(PENDING) 중인 입금 요청을 직접 취소한다."""
    norm_id = normalize_username(username)
    clean_oid = (order_id or "").strip()
    if not norm_id or not clean_oid:
        return False, "취소할 신청 정보를 확인할 수 없습니다."

    cur_dt = _resolve_cur_kst(now_dt)
    now_iso = cur_dt.isoformat(timespec="seconds")

    with _storage_lock(storage):
        orders = storage.load_payment_orders()
        target_ord = next(
            (o for o in orders if o.order_id == clean_oid and o.username == norm_id),
            None,
        )
        if target_ord is None:
            return False, "해당 입금 신청 내역을 찾을 수 없습니다."
        if target_ord.status != "PENDING":
            return False, f"승인 대기 상태인 신청만 취소할 수 있습니다. (현재 상태: {target_ord.status})"

        target_ord.status = "CANCELLED"
        target_ord.processed_at = now_iso
        target_ord.processed_by = norm_id
        target_ord.admin_memo = "회원 본인 신청 취소"
        storage.save_payment_orders(orders)

    return True, f"입금 신청({clean_oid})을 취소했습니다."


def approve_payment_order(
    storage: StorageManager,
    order_id: str,
    admin_username: str,
    admin_memo: str = "",
    now_dt: datetime | None = None,
) -> tuple[bool, str]:
    """대기 중인 입금 요청을 원클릭 승인하여 등급 전환·만료일(+30일/연장합산)·크레딧 리셋(부채 상계)을 원자적으로 처리한다."""
    clean_oid = (order_id or "").strip()
    admin_norm = normalize_username(admin_username) or "admin"
    clean_memo = (admin_memo or "").strip()[:120]
    cur_dt = _resolve_cur_kst(now_dt)
    today = cur_dt.strftime("%Y-%m-%d")
    current_month = cur_dt.strftime("%Y-%m")
    now_iso = cur_dt.isoformat(timespec="seconds")

    with _storage_lock(storage):
        orders = storage.load_payment_orders()
        target_ord = next((o for o in orders if o.order_id == clean_oid), None)
        if target_ord is None:
            return False, "해당 결제 주문을 찾을 수 없습니다."
        if target_ord.status != "PENDING":
            return False, f"이미 처리된 주문입니다. (현재 상태: {target_ord.status})"

        users = storage.load_users()
        admin_names = resolve_admin_config().admin_usernames
        target_user = next((u for u in users if u.username == target_ord.username), None)
        if target_user is None:
            return False, f"신청 회원(@{target_ord.username}) 계정이 존재하지 않습니다."

        expired_events: list[dict[str, object]] = []
        _sync_user_state(
            target_user,
            today,
            admin_names,
            now_dt=cur_dt,
            expired_events_out=expired_events,
        )
        if target_user.tier == "ADMIN":
            return False, "최고 관리자(ADMIN) 계정은 요금제 승인 대상이 아닙니다."

        # 환불 역산 및 오입금 방어용 스냅샷 저장
        target_ord.prev_tier = str(target_user.tier)
        target_ord.prev_tier_expires_at = str(getattr(target_user, "tier_expires_at", "") or "")
        target_ord.prev_monthly_used_credits = max(
            0, int(getattr(target_user, "monthly_used_credits", 0) or 0)
        )
        target_ord.prev_last_used_month = str(getattr(target_user, "last_used_month", "") or current_month)
        target_ord.base_used_at_approval = max(
            0, int(getattr(target_user, "total_base_credits_used", 0) or 0)
        )

        # 동일 등급 연장은 기존 유효 만료일에 +30일 합산, 신규/타 등급 변경은 승인 시각 + 30일
        if target_user.tier == target_ord.plan_tier and target_ord.prev_tier_expires_at.strip():
            cur_exp = parse_kst_iso(target_ord.prev_tier_expires_at)
            base_dt = cur_exp if (cur_exp is not None and cur_exp > cur_dt) else cur_dt
            new_exp_dt = base_dt + timedelta(days=int(target_ord.duration_days or 30))
        else:
            new_exp_dt = cur_dt + timedelta(days=int(target_ord.duration_days or 30))

        plan_limit = MONTHLY_CREDIT_LIMITS.get(target_ord.plan_tier, PRO_MONTHLY_CREDIT_LIMIT)
        debt = max(0, int(getattr(target_user, "refund_credit_debt", 0) or 0))
        applied_debt = min(plan_limit, debt)

        target_user.tier = target_ord.plan_tier
        target_user.tier_expires_at = new_exp_dt.isoformat(timespec="seconds")
        target_user.monthly_used_credits = applied_debt
        target_user.refund_credit_debt = debt - applied_debt
        target_user.last_used_month = current_month

        target_ord.applied_refund_debt = applied_debt
        target_ord.status = "APPROVED"
        target_ord.processed_at = now_iso
        target_ord.processed_by = admin_norm
        target_ord.granted_expires_at = target_user.tier_expires_at
        target_ord.admin_memo = clean_memo or "입금 확인 완료"

        storage.save_users(users)
        storage.save_payment_orders(orders)
        _flush_ledger_events(storage, expired_events)

        bonus_now = max(0, int(getattr(target_user, "bonus_credits", 0) or 0))
        avail_after = max(0, plan_limit - applied_debt) + bonus_now
        debt_note = f" · 기존 환불부채 {applied_debt:,}C 상계" if applied_debt > 0 else ""
        exp_disp = new_exp_dt.strftime("%Y-%m-%d")
        _flush_ledger_events(
            storage,
            [
                {
                    "ts": now_iso,
                    "username": target_user.username,
                    "event": "PAYMENT_APPROVE",
                    "delta": plan_limit - applied_debt,
                    "reason": (
                        f"{target_ord.plan_tier} {target_ord.duration_days}일권 승인 "
                        f"({target_ord.order_id} · ₩{target_ord.amount_krw:,} · 만료 {exp_disp}{debt_note})"
                    ),
                    "base_used_after": applied_debt,
                    "bonus_after": bonus_now,
                    "avail_after": avail_after,
                    "actor": admin_norm,
                }
            ],
        )

    msg_extra = f" (기존 환불 초과사용분 {applied_debt:,}C 자동 상계)" if applied_debt > 0 else ""
    return (
        True,
        f"@{target_ord.username} 회원의 {target_ord.plan_tier} 30일권(₩{target_ord.amount_krw:,}) 승인을 완료했습니다! "
        f"(만료일: {new_exp_dt.strftime('%Y-%m-%d')} · 가용 {avail_after:,}C{msg_extra})",
    )


def reject_payment_order(
    storage: StorageManager,
    order_id: str,
    admin_username: str,
    reason: str = "",
    now_dt: datetime | None = None,
) -> tuple[bool, str]:
    """대기 중인 입금 요청을 반려 처리한다."""
    clean_oid = (order_id or "").strip()
    admin_norm = normalize_username(admin_username) or "admin"
    clean_reason = (reason or "").strip()[:120] or "입금 내역 미확인"
    cur_dt = _resolve_cur_kst(now_dt)
    now_iso = cur_dt.isoformat(timespec="seconds")

    with _storage_lock(storage):
        orders = storage.load_payment_orders()
        target_ord = next((o for o in orders if o.order_id == clean_oid), None)
        if target_ord is None:
            return False, "해당 결제 주문을 찾을 수 없습니다."
        if target_ord.status != "PENDING":
            return False, f"대기 상태인 주문만 반려할 수 있습니다. (현재 상태: {target_ord.status})"

        target_ord.status = "REJECTED"
        target_ord.processed_at = now_iso
        target_ord.processed_by = admin_norm
        target_ord.admin_memo = clean_reason
        storage.save_payment_orders(orders)

    return True, f"입금 신청({clean_oid})을 반려 처리했습니다. (사유: {clean_reason})"


def refund_payment_order(
    storage: StorageManager,
    order_id: str,
    admin_username: str,
    reason: str = "",
    now_dt: datetime | None = None,
) -> tuple[bool, str]:
    """승인된 주문을 환불(승인취소)하고 직전 구독 상태·당월 사용량 복원 및 초과 소비 부채를 기록한다."""
    clean_oid = (order_id or "").strip()
    admin_norm = normalize_username(admin_username) or "admin"
    clean_reason = (reason or "").strip()[:120] or "관리자 환불 · 승인취소"
    cur_dt = _resolve_cur_kst(now_dt)
    today = cur_dt.strftime("%Y-%m-%d")
    current_month = cur_dt.strftime("%Y-%m")
    now_iso = cur_dt.isoformat(timespec="seconds")

    with _storage_lock(storage):
        orders = storage.load_payment_orders()
        target_ord = next((o for o in orders if o.order_id == clean_oid), None)
        if target_ord is None:
            return False, "해당 결제 주문을 찾을 수 없습니다."
        if target_ord.status != "APPROVED":
            return False, f"승인 완료된 주문만 환불·승인취소할 수 있습니다. (현재 상태: {target_ord.status})"

        approved_for_user = [
            o for o in orders if o.username == target_ord.username and o.status == "APPROVED"
        ]
        approved_for_user.sort(
            key=lambda o: (o.processed_at or "", o.requested_at or "", o.order_id),
            reverse=True,
        )
        if approved_for_user and approved_for_user[0].order_id != target_ord.order_id:
            return (
                False,
                f"가장 최근에 승인된 결제 건({approved_for_user[0].order_id})부터 순서대로만 환불·승인취소할 수 있습니다.",
            )

        users = storage.load_users()
        admin_names = resolve_admin_config().admin_usernames
        target_user = next((u for u in users if u.username == target_ord.username), None)
        expired_events: list[dict[str, object]] = []
        used_during_order = 0
        restored_tier = "FREE"
        new_debt = 0
        avail_after: int | None = 0

        if target_user is not None:
            _sync_user_state(
                target_user,
                today,
                admin_names,
                now_dt=cur_dt,
                expired_events_out=expired_events,
            )
            if target_user.tier != "ADMIN":
                used_during_order = max(
                    0,
                    int(getattr(target_user, "total_base_credits_used", 0) or 0)
                    - int(getattr(target_ord, "base_used_at_approval", 0) or 0),
                )

                # 직전 등급·만료일이 현재 시점에도 유효하면 원상 복구, 아니면 FREE 복귀
                prev_tier = str(getattr(target_ord, "prev_tier", "") or "FREE")
                prev_exp_str = str(getattr(target_ord, "prev_tier_expires_at", "") or "").strip()
                prev_exp_dt = parse_kst_iso(prev_exp_str)
                if prev_tier in ("PRO", "ULTRA") and (
                    not prev_exp_str or (prev_exp_dt is not None and prev_exp_dt > cur_dt)
                ):
                    target_user.tier = prev_tier  # type: ignore[assignment]
                    target_user.tier_expires_at = prev_exp_str
                else:
                    target_user.tier = "FREE"
                    target_user.tier_expires_at = ""

                restored_tier = target_user.tier
                restored_limit = MONTHLY_CREDIT_LIMITS.get(restored_tier, FREE_MONTHLY_CREDIT_LIMIT)
                debt_pool = max(0, int(getattr(target_user, "refund_credit_debt", 0) or 0)) + max(
                    0, int(getattr(target_ord, "applied_refund_debt", 0) or 0)
                )

                if current_month == str(getattr(target_ord, "prev_last_used_month", "") or ""):
                    combined = max(0, int(getattr(target_ord, "prev_monthly_used_credits", 0) or 0)) + used_during_order
                    target_user.monthly_used_credits = min(combined, restored_limit)
                    overflow = max(0, combined - restored_limit)
                    target_user.refund_credit_debt = debt_pool + overflow
                else:
                    prev_tier_limit = MONTHLY_CREDIT_LIMITS.get(prev_tier, FREE_MONTHLY_CREDIT_LIMIT)
                    unused_prev_month = max(
                        0,
                        prev_tier_limit - max(0, int(getattr(target_ord, "prev_monthly_used_credits", 0) or 0)),
                    )
                    unpaid_used = max(0, used_during_order - unused_prev_month)
                    cur_combined = max(0, int(getattr(target_user, "monthly_used_credits", 0) or 0)) + unpaid_used
                    target_user.monthly_used_credits = min(cur_combined, restored_limit)
                    overflow = max(0, cur_combined - restored_limit)
                    target_user.refund_credit_debt = debt_pool + overflow

                new_debt = target_user.refund_credit_debt
                bonus_now = max(0, int(getattr(target_user, "bonus_credits", 0) or 0))
                avail_after = max(0, restored_limit - target_user.monthly_used_credits) + bonus_now
                storage.save_users(users)
                _flush_ledger_events(storage, expired_events)

                _flush_ledger_events(
                    storage,
                    [
                        {
                            "ts": now_iso,
                            "username": target_user.username,
                            "event": "PAYMENT_REFUND",
                            "delta": -used_during_order if used_during_order > 0 else 0,
                            "reason": (
                                f"{target_ord.plan_tier} 환불·승인취소 ({target_ord.order_id} · "
                                f"승인후 소비 {used_during_order:,}C · 복귀 {restored_tier}"
                                + (f" · 미정산부채 {new_debt:,}C" if new_debt > 0 else "")
                                + ")"
                            ),
                            "base_used_after": target_user.monthly_used_credits,
                            "bonus_after": bonus_now,
                            "avail_after": avail_after,
                            "actor": admin_norm,
                        }
                    ],
                )
            elif expired_events:
                storage.save_users(users)
                _flush_ledger_events(storage, expired_events)

        target_ord.status = "REFUNDED"
        target_ord.refunded_at = now_iso
        target_ord.refunded_by = admin_norm
        target_ord.admin_memo = clean_reason
        storage.save_payment_orders(orders)

    debt_txt = f" · 초과사용 미정산 부채 {new_debt:,}C 기록(재결제·익월 시 상계)" if new_debt > 0 else ""
    return (
        True,
        f"주문({clean_oid})을 환불·승인취소했습니다. (@{target_ord.username} → {restored_tier} 복귀 · "
        f"승인후 소비 {used_during_order:,}C 반영{debt_txt})",
    )


def summarize_payment_revenue(
    orders: list[PaymentOrder],
    current_month_kst: str,
) -> dict[str, int]:
    """승인(APPROVED) 건만을 기준으로 이번 달·누적 실매출, 환불 합계, 대기 건수를 집계한다."""
    pending_count = 0
    month_approved_krw = 0
    month_approved_count = 0
    total_approved_krw = 0
    total_approved_count = 0
    total_refunded_krw = 0

    for ord_item in orders:
        if ord_item.status == "PENDING":
            pending_count += 1
        elif ord_item.status == "APPROVED":
            amt = max(0, int(ord_item.amount_krw or 0))
            total_approved_krw += amt
            total_approved_count += 1
            proc_month = str(ord_item.processed_at or "")[:7]
            if proc_month == current_month_kst:
                month_approved_krw += amt
                month_approved_count += 1
        elif ord_item.status == "REFUNDED":
            total_refunded_krw += max(0, int(ord_item.amount_krw or 0))

    return {
        "pending_count": pending_count,
        "month_approved_krw": month_approved_krw,
        "month_approved_count": month_approved_count,
        "total_approved_krw": total_approved_krw,
        "total_approved_count": total_approved_count,
        "total_refunded_krw": total_refunded_krw,
    }
