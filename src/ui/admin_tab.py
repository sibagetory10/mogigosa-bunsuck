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
    batch_update_user_tiers,
    delete_users_by_usernames,
    format_krw,
    format_token_count,
    get_kst_month_str,
    get_monthly_token_limit,
    mask_api_key,
    reset_user_monthly_tokens,
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

    users_list = ctx.storage.load_users()
    if not users_list:
        st.info("아직 가입한 회원이 없습니다.")
        return

    tier_counts = {t: sum(1 for u in users_list if u.tier == t) for t in VALID_TIERS}
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("전체 회원", f"{len(users_list)}명")
    m2.metric("FREE (0원)", f"{tier_counts['FREE']}명")
    m3.metric("CUSTOM (0원)", f"{tier_counts['CUSTOM']}명")
    m4.metric("PRO (9,900원/월)", f"{tier_counts['PRO']}명")
    m5.metric("ULTRA (19,900원/월)", f"{tier_counts['ULTRA']}명")
    m6.metric("ADMIN", f"{tier_counts['ADMIN']}명")

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
        u_m_calls = int(_u_month_attr(u, "monthly_api_calls", 0))
        u_m_usd = float(_u_month_attr(u, "monthly_cost_usd", 0.0))
        u_t_tok = int(getattr(u, "total_total_tokens", 0) or 0)
        u_t_usd = float(getattr(u, "total_cost_usd", 0.0) or 0.0)
        u_lim = get_monthly_token_limit(u)
        if u_lim is not None and u_lim > 0:
            usage_pct = f"{min(999.9, (u_m_tok / u_lim) * 100):.1f}% ({format_token_count(u_lim)})"
        else:
            usage_pct = "무제한"
        table_rows.append(
            {
                "아이디": u.username,
                "이름": u.nickname,
                "Google 연동": (getattr(u, "google_email", "") or "").strip() or "-",
                "현재 등급": u.tier,
                "이번 달 토큰": u_m_tok,
                "한도 사용률": usage_pct,
                "이번 달 비용(₩)": format_krw(u_m_usd, usd_to_krw),
                "누적 토큰": u_t_tok,
                "누적 비용(₩)": format_krw(u_t_usd, usd_to_krw),
                "월 호출수": u_m_calls,
                "기출 업로드": int(getattr(u, "uploaded_exam_count", 0) or 0),
                "개인 키": mask_api_key(u.custom_api_key) or "-",
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
            "이번 달 토큰": st.column_config.NumberColumn("이번 달 토큰", format="%d", disabled=True),
            "한도 사용률": st.column_config.TextColumn("한도 사용률", disabled=True),
            "이번 달 비용(₩)": st.column_config.TextColumn("이번 달 비용(₩)", disabled=True),
            "누적 토큰": st.column_config.NumberColumn("누적 토큰", format="%d", disabled=True),
            "누적 비용(₩)": st.column_config.TextColumn("누적 비용(₩)", disabled=True),
            "월 호출수": st.column_config.NumberColumn("월 호출수", format="%d", disabled=True),
            "기출 업로드": st.column_config.NumberColumn("기출 업로드", disabled=True),
            "개인 키": st.column_config.TextColumn("개인 키", disabled=True),
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

    with st.expander("🔄 관리자: 회원 이번 달 토큰 사용량 초기화", expanded=False):
        reset_options = [u.username for u in filtered_users]
        reset_label_map = {
            u.username: f"@{u.username} ({u.nickname} · {u.tier} · 이번 달 {int(_u_month_attr(u, 'monthly_total_tokens', 0)):,} 토큰)"
            for u in filtered_users
        }
        reset_targets = st.multiselect(
            "이번 달 토큰 사용량을 0으로 초기화할 회원 선택 (누적 토큰·비용 기록은 유지됨)",
            options=reset_options,
            format_func=lambda un: reset_label_map.get(un, un),
            key="admin_reset_token_select",
        )
        if st.button(
            "🔄 선택한 회원 월간 토큰 초기화",
            disabled=not reset_targets,
            key="admin_exec_reset_tokens",
        ):
            r_cnt, r_err = reset_user_monthly_tokens(ctx.storage, reset_targets)
            if r_err:
                st.warning(f"{r_cnt}명 초기화됨 / Drive 저장 경고: {r_err}")
            elif r_cnt > 0:
                st.success(f"✅ {r_cnt}명의 이번 달 토큰 사용량을 0으로 초기화했습니다.")
                st.rerun()

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
