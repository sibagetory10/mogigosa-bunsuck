from __future__ import annotations

import html as _html
import json
from datetime import datetime
from typing import Any

import streamlit as st

from src.auth import (
    PRO_MONTHLY_PRICE_KRW,
    TIER_LABELS,
    ULTRA_MONTHLY_PRICE_KRW,
    VALID_TIERS,
    adjust_user_bonus_credits,
    batch_update_user_tiers,
    delete_users_by_usernames,
    format_krw,
    get_bonus_credits,
    get_daily_free_ai_limit,
    get_daily_free_ai_used,
    get_kst_month_str,
    get_monthly_credit_limit,
    get_total_available_credits,
    get_used_monthly_credits,
    reset_user_monthly_credits,
    sync_all_users_state,
)
from src.config import (
    FLASH_LITE_CACHED_INPUT_USD_PER_1M,
    FLASH_LITE_INPUT_USD_PER_1M,
    FLASH_LITE_OUTPUT_USD_PER_1M,
    resolve_billing_api_key,
)
from src.fx_rate import get_daily_gemini_fx_rate
from src.ui.common import create_lazy_expander, is_container_open
from src.ui.context import AppContext


def _render_admin_panel(ctx: AppContext) -> None:
    st.subheader("🛡️ 관리자 회원 관리 (등급 설정 · 실제 API 토큰 & 원화 비용 모니터링)")
    st.caption(
        "가입한 회원 목록 및 실제 API 토큰·비용 내역은 `./exam_db/users.json`에 영구 저장되며 Google Drive 연동 시 자동 동기화됩니다."
    )

    col_top1, col_top2 = st.columns([2, 1])
    with col_top1:
        search_q = st.text_input(
            "회원 검색 (아이디 또는 이름)",
            placeholder="검색어를 입력하세요...",
            key="admin_user_search",
        ).strip().lower()
    with col_top2:
        st.markdown("<div style='height: 1.65rem;'></div>", unsafe_allow_html=True)
        if st.button("🔄 회원 목록 새로고침", width="stretch", key="admin_refresh_btn"):
            st.rerun()

    users_list = sync_all_users_state(ctx.storage)
    if not users_list:
        st.info("아직 가입한 회원이 없습니다.")
        return

    tier_counts = {t: sum(1 for u in users_list if u.tier == t) for t in VALID_TIERS}
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("전체 회원", f"{len(users_list)}명")
    m2.metric("FREE (0원 · 30C)", f"{tier_counts['FREE']}명")
    m3.metric("PRO (9,900원 · 600C)", f"{tier_counts['PRO']}명")
    m4.metric("ULTRA (19,900원 · 2,400C)", f"{tier_counts['ULTRA']}명")
    m5.metric("ADMIN", f"{tier_counts['ADMIN']}명")

    # ------------------------------------------------------------------
    # 💰 실제 API 토큰 사용량 & 원화(₩) 환산 대시보드 (매일 00:00 KST 자동 환율 적용)
    # ------------------------------------------------------------------
    current_month = get_kst_month_str()
    fx_api_keys = [k for k in (resolve_billing_api_key(), ctx.creds.gemini_api_key) if k]
    force_fx_now = bool(st.session_state.pop("admin_force_fx_refresh", False))
    fx_info = get_daily_gemini_fx_rate(
        cache_dir=ctx.storage.local_root,
        api_keys=fx_api_keys,
        force_refresh=force_fx_now,
    )

    with st.container(border=True):
        rate_c1, rate_c2 = st.columns([2.2, 1.2])
        with rate_c1:
            st.markdown(f"#### 💰 API 토큰 사용량 & 원화(₩) 환산 ({current_month} KST 기준)")
            st.caption(
                f"적용 단가 (Gemini 3.5 Flash-Lite): "
                f"입력 `${FLASH_LITE_INPUT_USD_PER_1M:.2f}` · "
                f"캐시 `${FLASH_LITE_CACHED_INPUT_USD_PER_1M:.2f}` · "
                f"출력(+사고) `${FLASH_LITE_OUTPUT_USD_PER_1M:.2f}` / 1M 토큰"
            )
            badge_icon = "✅" if fx_info.is_google_official else "💱"
            ref_part = f" · 기준일 `{fx_info.reference_date}`" if fx_info.reference_date else ""
            st.caption(
                f"{badge_icon} **자동 적용 환율: 1 USD = ₩{fx_info.rate:,.2f}** "
                f"({fx_info.source}{ref_part}) | "
                f"최근 확인: `{fx_info.checked_at}` · **자동 갱신: 매일 00:00 KST**"
            )
        with rate_c2:
            include_vat = st.checkbox(
                "🇰🇷 한국 부가가치세(VAT 10%) 포함",
                value=True,
                key="admin_include_kr_vat",
                help="대한민국 Google Cloud / Gemini API 결제 계정(Google Cloud Korea LLC)은 환산 원화 금액에 VAT 10%가 별도 청구됩니다.",
            )
            if st.button("🔄 환율 지금 다시 확인", key="admin_refresh_fx_btn", width="stretch"):
                st.session_state["admin_force_fx_refresh"] = True
                st.rerun()

        vat_mult = 1.10 if include_vat else 1.0
        usd_to_krw = float(fx_info.rate) * vat_mult
        vat_label = "VAT 10% 포함" if include_vat else "VAT 별도"

        def _u_month_attr(u_obj: Any, attr: str, default: Any = 0) -> Any:
            if getattr(u_obj, "last_used_month", "") != current_month:
                return default
            return getattr(u_obj, attr, default) or default

        month_prompt_sum = sum(int(_u_month_attr(u, "monthly_prompt_tokens", 0)) for u in users_list)
        month_cached_sum = sum(int(_u_month_attr(u, "monthly_cached_tokens", 0)) for u in users_list)
        month_output_sum = sum(int(_u_month_attr(u, "monthly_output_tokens", 0)) for u in users_list)
        month_total_tok_sum = sum(int(_u_month_attr(u, "monthly_total_tokens", 0)) for u in users_list)
        month_calls_sum = sum(int(_u_month_attr(u, "monthly_api_calls", 0)) for u in users_list)
        month_server_usd = sum(float(_u_month_attr(u, "monthly_server_cost_usd", 0.0)) for u in users_list)
        month_all_usd = sum(float(_u_month_attr(u, "monthly_cost_usd", 0.0)) for u in users_list)

        total_tok_sum = sum(int(getattr(u, "total_total_tokens", 0) or 0) for u in users_list)
        total_server_usd = sum(float(getattr(u, "total_server_cost_usd", 0.0) or 0.0) for u in users_list)
        total_all_usd = sum(float(getattr(u, "total_cost_usd", 0.0) or 0.0) for u in users_list)

        est_monthly_rev_krw = (
            tier_counts["PRO"] * PRO_MONTHLY_PRICE_KRW
            + tier_counts["ULTRA"] * ULTRA_MONTHLY_PRICE_KRW
        )
        month_server_krw = month_server_usd * usd_to_krw
        cost_ratio_str = (
            f"{(month_server_krw / est_monthly_rev_krw) * 100:.1f}%"
            if est_monthly_rev_krw > 0
            else "-"
        )

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "이번 달 서버 API 비용",
            format_krw(month_server_usd, usd_to_krw),
            help=f"서버 키 부담분 기준 (${month_server_usd:.4f} · {vat_label} / 개인키 포함 전체 ${month_all_usd:.4f})",
        )
        k1.caption(f"USD `${month_server_usd:.4f}` ({vat_label} · 개인키 제외)")
        k2.metric(
            "이번 달 전체 사용 토큰",
            f"{month_total_tok_sum:,}",
        )
        k2.caption(
            f"입력 {month_prompt_sum + month_cached_sum:,} · 출력 {month_output_sum:,} · {month_calls_sum:,}회"
        )
        k3.metric(
            "누적 서버 API 비용",
            format_krw(total_server_usd, usd_to_krw),
            help=f"전체 누적 서버 키 부담분 (${total_server_usd:.4f} · {vat_label} / 개인키 포함 전체 ${total_all_usd:.4f})",
        )
        k3.caption(f"USD `${total_server_usd:.4f}` · 누적 {total_tok_sum:,} 토큰")
        k4.metric(
            "예상 월 매출 대비 API 원가율",
            cost_ratio_str,
        )
        k4.caption(f"예상 월 매출 ₩{est_monthly_rev_krw:,}")

    filtered_users = [
        u
        for u in users_list
        if not search_q
        or search_q in u.username.lower()
        or search_q in u.nickname.lower()
        or search_q in (getattr(u, "google_email", "") or "").lower()
    ]
    filtered_users.sort(key=lambda u: u.created_at or u.username, reverse=True)

    table_rows = []
    for u in filtered_users:
        u_m_tok = int(_u_month_attr(u, "monthly_total_tokens", 0))
        u_m_usd = float(_u_month_attr(u, "monthly_cost_usd", 0.0))
        u_t_usd = float(getattr(u, "total_cost_usd", 0.0) or 0.0)
        u_m_cred = get_used_monthly_credits(u)
        u_cred_lim = get_monthly_credit_limit(u)
        u_bonus_c = get_bonus_credits(u)
        u_avail_c = get_total_available_credits(u)
        u_tot_c = int(getattr(u, "total_used_credits", 0) or 0)
        u_free_ai_used = get_daily_free_ai_used(u)
        u_free_ai_lim = get_daily_free_ai_limit(u)
        avail_disp = "무제한" if u_avail_c is None else f"{u_avail_c:,}C (월 {u_cred_lim:,}C)"
        free_ai_disp = "무제한" if u_free_ai_lim is None else f"{u_free_ai_used}/{u_free_ai_lim}"
        table_rows.append(
            {
                "아이디": u.username,
                "이름": u.nickname,
                "Google 연동": (getattr(u, "google_email", "") or "").strip() or "-",
                "현재 등급": u.tier,
                "가용 크레딧": avail_disp,
                "이번 달 사용(C)": u_m_cred,
                "보너스(C)": u_bonus_c,
                "누적 사용(C)": u_tot_c,
                "오늘 무료AI": free_ai_disp,
                "이번 달 토큰": u_m_tok,
                "이번 달 비용(₩)": format_krw(u_m_usd, usd_to_krw),
                "누적 비용(₩)": format_krw(u_t_usd, usd_to_krw),
                "기출 업로드": int(getattr(u, "uploaded_exam_count", 0) or 0),
            }
        )

    edited_rows = st.data_editor(
        table_rows,
        column_config={
            "아이디": st.column_config.TextColumn("아이디", disabled=True),
            "이름": st.column_config.TextColumn("이름", disabled=True),
            "Google 연동": st.column_config.TextColumn("Google 연동", disabled=True),
            "현재 등급": st.column_config.SelectboxColumn(
                "현재 등급 (변경 가능)",
                options=list(VALID_TIERS),
                required=True,
            ),
            "가용 크레딧": st.column_config.TextColumn("가용 크레딧", disabled=True),
            "이번 달 사용(C)": st.column_config.NumberColumn("이번 달 사용(C)", format="%d", disabled=True),
            "보너스(C)": st.column_config.NumberColumn("보너스(C)", format="%d", disabled=True),
            "누적 사용(C)": st.column_config.NumberColumn("누적 사용(C)", format="%d", disabled=True),
            "오늘 무료AI": st.column_config.TextColumn("오늘 무료AI", disabled=True),
            "이번 달 토큰": st.column_config.NumberColumn("이번 달 토큰", format="%d", disabled=True),
            "이번 달 비용(₩)": st.column_config.TextColumn("이번 달 비용(₩)", disabled=True),
            "누적 비용(₩)": st.column_config.TextColumn("누적 비용(₩)", disabled=True),
            "기출 업로드": st.column_config.NumberColumn("기출 업로드", disabled=True),
        },
        hide_index=True,
        width="stretch",
        key="admin_users_editor",
    )

    if st.button("💾 회원 등급 변경사항 저장", type="primary", key="admin_save_tiers_btn"):
        updates = {
            row["아이디"]: row["현재 등급"]
            for row in edited_rows
            if row.get("아이디") and row.get("현재 등급") in VALID_TIERS
        }
        changed_cnt, err = batch_update_user_tiers(ctx.storage, updates)
        if err:
            st.warning(f"{changed_cnt}명 변경됨 / 저장 경고: {err}")
        elif changed_cnt > 0:
            st.success(f"✅ {changed_cnt}명의 회원 등급을 즉시 변경하여 `users.json`에 저장했습니다!")
            st.rerun()
        else:
            st.info("변경된 등급이 없습니다.")

    actor_id_admin = ctx.current_user.username if ctx.current_user is not None else "admin"

    with st.expander("🎁 관리자: 회원 보너스 크레딧 지급 · 차감", expanded=False):
        bonus_options = [u.username for u in filtered_users]
        bonus_label_map = {
            u.username: f"@{u.username} ({u.nickname} · {u.tier} · 보너스 {get_bonus_credits(u):,}C)"
            for u in filtered_users
        }
        bonus_targets = st.multiselect(
            "보너스 크레딧을 지급하거나 차감할 회원 선택 (월초 초기화 시 소멸되지 않음)",
            options=bonus_options,
            format_func=lambda un: bonus_label_map.get(un, un),
            key="admin_bonus_credit_select",
        )
        bc1, bc2 = st.columns([1.0, 2.0])
        with bc1:
            bonus_delta = int(
                st.number_input(
                    "증감 크레딧 (+지급 / -차감)",
                    min_value=-100_000,
                    max_value=100_000,
                    value=50,
                    step=10,
                    key="admin_bonus_delta_input",
                )
            )
        with bc2:
            bonus_reason = st.text_input(
                "지급 · 차감 사유 (원장 기록용)",
                value="관리자 이벤트 지급",
                key="admin_bonus_reason_input",
            )
        if st.button(
            "🎁 보너스 크레딧 반영하기",
            disabled=not (bonus_targets and bonus_delta != 0),
            key="admin_exec_bonus_credits",
        ):
            b_cnt, b_err = adjust_user_bonus_credits(
                ctx.storage,
                bonus_targets,
                delta_credits=bonus_delta,
                reason=bonus_reason,
                actor_username=actor_id_admin,
            )
            if b_err:
                st.warning(f"{b_cnt}명 반영됨 / 저장 경고: {b_err}")
            elif b_cnt > 0:
                st.success(f"✅ {b_cnt}명에게 보너스 크레딧({bonus_delta:+,}C)을 반영하고 원장에 기록했습니다.")
                st.rerun()
            else:
                st.info("변경된 보너스 크레딧이 없습니다.")

    with st.expander("🔄 관리자: 회원 이번 달 사용 크레딧 초기화", expanded=False):
        reset_options = [u.username for u in filtered_users]
        reset_label_map = {
            u.username: f"@{u.username} ({u.nickname} · {u.tier} · 이번 달 사용 {get_used_monthly_credits(u):,}C)"
            for u in filtered_users
        }
        reset_targets = st.multiselect(
            "이번 달 사용 크레딧을 0C로 초기화할 회원 선택 (보너스 크레딧·누적 API 원가 기록은 유지됨)",
            options=reset_options,
            format_func=lambda un: reset_label_map.get(un, un),
            key="admin_reset_credit_select",
        )
        if st.button(
            "🔄 선택한 회원 월간 크레딧 초기화",
            disabled=not reset_targets,
            key="admin_exec_reset_credits",
        ):
            r_cnt, r_err = reset_user_monthly_credits(
                ctx.storage,
                reset_targets,
                actor_username=actor_id_admin,
            )
            if r_err:
                st.warning(f"{r_cnt}명 초기화됨 / 저장 경고: {r_err}")
            elif r_cnt > 0:
                st.success(f"✅ {r_cnt}명의 이번 달 사용 크레딧을 0C로 초기화했습니다.")
                st.rerun()

    with st.expander("📜 크레딧 원장 감사 로그 (credit_ledger.jsonl · 최근 200건)", expanded=False):
        ledger_filter_u = st.text_input(
            "아이디 필터 (비워두면 전체 회원 조회)",
            placeholder="예: student01",
            key="admin_ledger_user_filter",
        ).strip()
        ledger_rows = ctx.storage.read_credit_ledger(username=ledger_filter_u, limit=200)
        if not ledger_rows:
            st.info("기록된 크레딧 원장 내역이 없습니다.")
        else:
            disp_rows = [
                {
                    "시각": str(r.get("ts") or "").replace("T", " ")[:19],
                    "아이디": str(r.get("username") or ""),
                    "유형": str(r.get("event") or ""),
                    "증감(C)": int(r.get("delta") or 0),
                    "사유": str(r.get("reason") or ""),
                    "가용 잔액(C)": "무제한" if r.get("avail_after") is None else f"{int(r.get('avail_after') or 0):,}",
                    "보너스(C)": int(r.get("bonus_after") or 0),
                    "실행자": str(r.get("actor") or ""),
                }
                for r in ledger_rows
            ]
            st.dataframe(disp_rows, hide_index=True, width="stretch")

    with st.expander("🗑️ 관리자: 회원 계정 삭제", expanded=False):
        actor_id = ctx.current_user.username if ctx.current_user is not None else ""
        deletable_users = [u for u in filtered_users if u.username != actor_id]
        if not deletable_users:
            st.info("삭제 가능한 다른 회원 계정이 없습니다.")
        else:
            del_options = [u.username for u in deletable_users]
            user_label_map = {
                u.username: f"@{u.username} ({u.nickname} · {u.tier})"
                for u in deletable_users
            }
            del_targets = st.multiselect(
                "삭제할 회원 계정 선택 (본인 계정 및 마지막 관리자는 보호됨)",
                options=del_options,
                format_func=lambda un: user_label_map.get(un, un),
                key="admin_delete_user_select",
            )
            confirm_del_users = st.checkbox(
                "선택한 회원 계정을 영구 삭제합니다. (복구 불가)",
                key="admin_confirm_delete_users",
            )
            if st.button(
                "🗑️ 선택한 계정 영구 삭제",
                disabled=not (del_targets and confirm_del_users),
                key="admin_exec_delete_users",
            ):
                del_cnt, del_err, del_note = delete_users_by_usernames(
                    ctx.storage,
                    del_targets,
                    actor_username=actor_id,
                )
                if del_err:
                    st.warning(f"{del_cnt}개 계정 삭제됨 / Drive 저장 경고: {del_err}")
                elif del_cnt > 0:
                    st.success(f"🗑️ {del_cnt}개 회원 계정을 영구 삭제했습니다.")
                    if del_note:
                        st.info(del_note)
                    st.rerun()
                elif del_note:
                    st.warning(del_note)

    storage_exp = create_lazy_expander(
        "💾 서버 디스크 저장공간 · 앱 DB 용량 모니터링",
        expanded=False,
        key="admin_storage_stats_exp",
    )
    with storage_exp:
        if is_container_open(storage_exp):
            from src.ui.admin_pools import _render_server_storage_admin_section

            _render_server_storage_admin_section(ctx)

    git_exp = create_lazy_expander(
        "🔄 서버 Git 동기화 · 자동 업데이트 제어",
        expanded=False,
        key="admin_git_update_exp",
    )
    with git_exp:
        if is_container_open(git_exp):
            from src.ui.admin_pools import _render_server_git_update_admin_section

            _render_server_git_update_admin_section()

    eng_pool_exp = create_lazy_expander(
        "🔤 영어 변형 문제 풀(Pool) 캐시 관리",
        expanded=False,
        key="admin_eng_pool_exp",
    )
    with eng_pool_exp:
        if is_container_open(eng_pool_exp):
            from src.ui.admin_pools import _render_english_pool_admin_section

            _render_english_pool_admin_section(ctx)

    kor_pool_exp = create_lazy_expander(
        "📖 국어 지문 분석지 캐시 · PDF DB 관리",
        expanded=False,
        key="admin_kor_analysis_pool_exp",
    )
    with kor_pool_exp:
        if is_container_open(kor_pool_exp):
            from src.ui.admin_pools import _render_korean_analysis_pool_admin_section

            _render_korean_analysis_pool_admin_section(ctx)

    eng_an_pool_exp = create_lazy_expander(
        "🔤 영어 지문 분석지 캐시 · PDF DB 관리",
        expanded=False,
        key="admin_eng_analysis_pool_exp",
    )
    with eng_an_pool_exp:
        if is_container_open(eng_an_pool_exp):
            from src.ui.admin_pools import _render_english_analysis_pool_admin_section

            _render_english_analysis_pool_admin_section(ctx)


render_admin_panel = _render_admin_panel
