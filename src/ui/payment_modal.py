from __future__ import annotations

import html as _html
from typing import TYPE_CHECKING

import streamlit as st

from src.auth import (
    FREE_MAX_BATCH_QUESTIONS,
    FREE_MONTHLY_CREDIT_LIMIT,
    PAYMENT_PLAN_SPECS,
    PRO_MONTHLY_CREDIT_LIMIT,
    PRO_MONTHLY_PRICE_KRW,
    TIER_LABELS,
    ULTRA_MONTHLY_CREDIT_LIMIT,
    ULTRA_MONTHLY_PRICE_KRW,
    cancel_my_pending_order,
    format_tier_expiry_badge,
    get_effective_tier,
    get_total_available_credits,
    get_user_by_username,
    submit_payment_order,
)

if TYPE_CHECKING:
    from src.models import PaymentPlanType, UserRecord
    from src.storage import StorageManager

_STATUS_BADGES: dict[str, tuple[str, str]] = {
    "PENDING": ("⏳ 승인 대기", "#B45309"),
    "APPROVED": ("✅ 승인 완료", "#15803D"),
    "REJECTED": ("❌ 반려됨", "#B91C1C"),
    "REFUNDED": ("↩️ 환불 처리됨", "#6D28D9"),
    "CANCELLED": ("🚫 본인 취소", "#64748B"),
}


@st.dialog("💎 멤버십 요금제 결제 · 연장", width="large")
def open_payment_dialog(storage: StorageManager, current_user: UserRecord) -> None:
    """계좌이체·간편송금 안내 및 입금 확인 신청·내역 조회 모달."""
    user = get_user_by_username(storage, current_user.username) or current_user
    settings = storage.load_payment_settings()
    user_orders = [o for o in storage.load_payment_orders() if o.username == user.username]

    eff_tier = get_effective_tier(user)
    tier_kor = TIER_LABELS.get(eff_tier, eff_tier)
    total_avail = get_total_available_credits(user)
    exp_badge = format_tier_expiry_badge(user)
    debt = max(0, int(getattr(user, "refund_credit_debt", 0) or 0))

    avail_text = "무제한" if total_avail is None else f"{total_avail:,}C"
    exp_part = f" · 📅 {exp_badge}" if exp_badge else ""
    st.markdown(
        f"<div style='background:#F8FAFC; border:1px solid #CBD5E1; border-radius:10px; "
        f"padding:0.7rem 0.95rem; margin-bottom:0.8rem;'>"
        f"<div style='display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:8px;'>"
        f"<div><b>현재 내 멤버십</b>: <span class='tier-badge tier-{eff_tier}'>{_html.escape(tier_kor)}</span>"
        f"{_html.escape(exp_part)}</div>"
        f"<div>가용 크레딧: <b>{avail_text}</b></div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )
    if debt > 0:
        st.warning(
            f"⚠️ 이전 환불 건에서 초과 소비한 미정산 크레딧 **{debt:,}C**가 있습니다. "
            "다음 유료 멤버십 승인 또는 월초 초기화 시 해당 분량만큼 자동 상계(선차감)됩니다."
        )

    c_pro, c_ultra = st.columns(2)
    with c_pro:
        with st.container(border=True):
            st.markdown("<span class='tier-badge tier-PRO'>PRO · 30일권</span>", unsafe_allow_html=True)
            st.markdown(f"#### ₩{PRO_MONTHLY_PRICE_KRW:,} <span style='font-size:0.82rem; color:#64748B;'>/ 30일</span>", unsafe_allow_html=True)
            st.write(f"- 월간 기본 크레딧 **{PRO_MONTHLY_CREDIT_LIMIT:,}C** *(FREE {FREE_MONTHLY_CREDIT_LIMIT}C 대비 20배)*")
            st.write(f"- 1회 생성 **제한 없음** *(FREE {FREE_MAX_BATCH_QUESTIONS}문항 제한 해제)*")
            st.write("- 무료 AI 보조 일 **40회** · 공용 DB 업로드 일 **3회**")
    with c_ultra:
        with st.container(border=True):
            st.markdown("<span class='tier-badge tier-ULTRA'>ULTRA · 30일권 (4배)</span>", unsafe_allow_html=True)
            st.markdown(f"#### ₩{ULTRA_MONTHLY_PRICE_KRW:,} <span style='font-size:0.82rem; color:#64748B;'>/ 30일</span>", unsafe_allow_html=True)
            st.write(f"- 월간 기본 크레딧 **{ULTRA_MONTHLY_CREDIT_LIMIT:,}C** *(PRO 대비 4배)*")
            st.write("- PRO의 모든 기능 및 배치 제한 해제 포함")
            st.write("- 무료 AI 보조 일 **100회** · 공용 DB 업로드 일 **8회**")

    st.caption(
        "ℹ️ **이용 기간·크레딧 적용 안내**: 승인 즉시 당월 사용 크레딧이 초기화되어 요금제 크레딧 전액을 바로 쓰실 수 있습니다. "
        "이용 중인 동일 등급을 연장하시면 **남은 기간에 +30일이 합산**되며, 다른 등급으로 변경하시면 **승인일 기준 +30일**이 새로 적용됩니다."
    )

    st.markdown("---")
    st.markdown("#### 🏦 1단계: 아래 계좌 또는 간편송금으로 입금해 주세요")

    if not settings.enabled or not settings.has_transfer_method:
        st.warning("현재 관리자가 입금 계좌 정보를 준비 중입니다. 설정이 완료되면 바로 신청하실 수 있습니다.")
    else:
        if (settings.guide_notice or "").strip():
            st.info(settings.guide_notice.strip())

        if settings.bank_name.strip() and settings.account_number.strip():
            holder_txt = settings.account_holder.strip() or "-"
            st.caption("우측 복사 아이콘(📋)을 누르면 은행·계좌번호가 복사됩니다.")
            st.code(
                f"{settings.bank_name.strip()} {settings.account_number.strip()} (예금주: {holder_txt})",
                language=None,
            )

        link_btns: list[str] = []
        if settings.kakaopay_link.strip():
            safe_kp = _html.escape(settings.kakaopay_link.strip(), quote=True)
            link_btns.append(
                f"<a href='{safe_kp}' target='_blank' rel='noopener noreferrer' "
                "style='display:inline-flex; align-items:center; justify-content:center; gap:6px; "
                "padding:0.52rem 1rem; border-radius:9px; background:#FEE500; color:#191919; "
                "font-weight:800; font-size:0.88rem; text-decoration:none; border:1px solid #EAB308;'>"
                "💛 카카오페이로 송금하기</a>"
            )
        if settings.toss_link.strip():
            safe_ts = _html.escape(settings.toss_link.strip(), quote=True)
            link_btns.append(
                f"<a href='{safe_ts}' target='_blank' rel='noopener noreferrer' "
                "style='display:inline-flex; align-items:center; justify-content:center; gap:6px; "
                "padding:0.52rem 1rem; border-radius:9px; background:#0064FF; color:#FFFFFF; "
                "font-weight:800; font-size:0.88rem; text-decoration:none;'>"
                "💙 토스로 송금하기</a>"
            )
        if link_btns:
            st.markdown(
                f"<div style='display:flex; flex-wrap:wrap; gap:8px; margin:0.4rem 0 0.8rem;'>{''.join(link_btns)}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("#### 📩 2단계: 입금 확인 요청 보내기")
    pending_ord = next((o for o in user_orders if o.status == "PENDING"), None)

    if pending_ord is not None:
        req_disp = str(pending_ord.requested_at or "")[:16].replace("T", " ")
        st.info(
            f"⏳ **승인 대기 중인 입금 확인 요청이 있습니다.**\n\n"
            f"- **주문 번호**: `{pending_ord.order_id}`\n"
            f"- **신청 요금제**: {pending_ord.plan_tier} 30일권 (`₩{pending_ord.amount_krw:,}`)\n"
            f"- **입금자명**: {pending_ord.depositor_name} · **신청 시각**: {req_disp}\n\n"
            "관리자가 입금 내역을 확인하는 즉시 멤버십이 자동으로 활성화됩니다."
        )
        if st.button(
            "🗑️ 이 입금 신청 취소하기",
            key=f"modal_cancel_pending_{pending_ord.order_id}",
            width="stretch",
        ):
            ok_c, msg_c = cancel_my_pending_order(storage, user.username, pending_ord.order_id)
            if ok_c:
                st.session_state["auth_flash_msg"] = ("info", msg_c)
                st.rerun()
            else:
                st.error(msg_c)
    elif settings.enabled and settings.has_transfer_method:
        default_plan: PaymentPlanType = "PRO" if eff_tier != "ULTRA" else "ULTRA"
        plan_options: list[PaymentPlanType] = ["PRO", "ULTRA"]
        with st.form("user_submit_payment_order_form"):
            selected_plan = st.radio(
                "신청할 요금제 선택",
                options=plan_options,
                index=plan_options.index(default_plan),
                format_func=lambda t: (
                    f"{PAYMENT_PLAN_SPECS[t]['label']} — ₩{PAYMENT_PLAN_SPECS[t]['price_krw']:,} "
                    f"(월 {PAYMENT_PLAN_SPECS[t]['monthly_credits']:,}C)"
                ),
                horizontal=True,
            )
            depositor_name = st.text_input(
                "입금자명 (실제 계좌이체·송금에 표시된 성함)",
                value=user.nickname,
                max_chars=30,
                placeholder="학부모님 등 다른 분 성함으로 송금하신 경우 실제 입금자명으로 수정해 주세요.",
            )
            user_note = st.text_input(
                "연락처 또는 메모 (선택)",
                max_chars=120,
                placeholder="예: 010-1234-5678 / 타인 명의 입금 확인 부탁드립니다.",
            )
            submitted = st.form_submit_button(
                "✅ 입금 완료 · 승인 요청 보내기",
                type="primary",
                width="stretch",
            )

        if submitted:
            ok_s, msg_s, _ = submit_payment_order(
                storage,
                username=user.username,
                plan_tier=selected_plan,
                depositor_name=depositor_name,
                user_note=user_note,
            )
            if ok_s:
                st.session_state["auth_flash_msg"] = ("success", msg_s)
                st.rerun()
            else:
                st.error(msg_s)

    if user_orders:
        st.markdown("---")
        with st.expander(f"📋 나의 결제 · 입금 신청 내역 ({len(user_orders)}건)", expanded=False):
            for ord_item in user_orders[:10]:
                badge_lbl, badge_color = _STATUS_BADGES.get(
                    ord_item.status, (ord_item.status, "#334155")
                )
                req_t = str(ord_item.requested_at or "")[:16].replace("T", " ")
                extra_line = ""
                if ord_item.status == "APPROVED" and ord_item.granted_expires_at:
                    extra_line = f" · 만료일 `{ord_item.granted_expires_at[:10]}`"
                elif ord_item.status in ("REJECTED", "REFUNDED") and ord_item.admin_memo:
                    extra_line = f" · 사유: {_html.escape(ord_item.admin_memo)}"
                st.markdown(
                    f"<div style='font-size:0.82rem; padding:0.45rem 0.65rem; margin-bottom:0.4rem; "
                    f"background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px;'>"
                    f"<b style='color:{badge_color};'>[{badge_lbl}]</b> "
                    f"<b>{ord_item.plan_tier} 30일권</b> (₩{ord_item.amount_krw:,}) · "
                    f"입금자: {_html.escape(ord_item.depositor_name)} · "
                    f"<span style='color:#64748B;'>{req_t}</span>{extra_line}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
