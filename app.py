from __future__ import annotations

import streamlit as st

from src.config import (
    resolve_billing_api_key,
    resolve_credentials,
    resolve_google_login_config,
)
from src.fx_rate import start_daily_midnight_fx_scheduler
from src.ui.auth_sidebar import render_auth_and_sidebar, render_guest_landing
from src.ui.common import (
    create_lazy_tabs,
    get_cached_storage,
    init_session_defaults,
    is_container_open,
)
from src.ui_styles import inject_responsive_css

st.set_page_config(
    page_title="2022 개정 교육과정 맞춤형 기출 플랫폼",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="auto",
)

inject_responsive_css()
init_session_defaults()

creds = resolve_credentials()
storage = get_cached_storage(local_root="./exam_db")
_fx_api_keys = [k for k in (resolve_billing_api_key(), creds.gemini_api_key) if k]
start_daily_midnight_fx_scheduler(
    cache_dir=storage.local_root,
    api_keys=_fx_api_keys,
    storage=storage,
)

google_login_cfg = resolve_google_login_config()
ctx = render_auth_and_sidebar(storage, creds, google_login_cfg)

st.markdown(
    """
    <div class="app-hero">
      <h1>📘 2022 개정 교육과정 맞춤형 기출 추출 & 상위 문제 개조 플랫폼</h1>
      <p>개념서·교과서 PDF에 실제로 나온 원리와 용어만 남기고, 원리는 같지만 용어가 어려운 상위 기출은 내 개념서 언어로 자동 개조합니다.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if ctx.current_user is None:
    render_guest_landing()
    st.stop()

if ctx.is_admin:
    _pending_pay_cnt = storage.count_pending_payment_orders()
    _admin_tab_label = (
        f"🛡️ 관리자: 회원·결제 관리 (🔔 {_pending_pay_cnt})"
        if _pending_pay_cnt > 0
        else "🛡️ 관리자: 회원·결제 관리"
    )
    tab1, tab2, tab3, tab4, my_exams_tab, admin_tab = create_lazy_tabs(
        [
            "📥 탭 1: 모의고사 수집 & 2022 원리·용어 분해",
            "🎯 탭 2: 개념서 범위 추출 & 상위 문제 개조",
            "📚 탭 3: 공용 개념서·교과서 DB 구축",
            "📖 탭 4: 영단어 사전",
            "📁 나의 시험지",
            _admin_tab_label,
        ],
        key="main_app_tabs",
    )
else:
    tab1, tab2, tab3, tab4, my_exams_tab = create_lazy_tabs(
        [
            "📥 탭 1: 모의고사 수집 & 2022 원리·용어 분해",
            "🎯 탭 2: 개념서 범위 추출 & 상위 문제 개조",
            "📚 탭 3: 공용 개념서·교과서 DB 구축",
            "📖 탭 4: 영단어 사전",
            "📁 나의 시험지",
        ],
        key="main_app_tabs",
    )
    admin_tab = None

with tab1:
    if is_container_open(tab1):
        from src.ui.tab1_upload import render_tab1

        render_tab1(ctx)

with tab2:
    if is_container_open(tab2):
        from src.ui.tab2_transform import render_tab2

        render_tab2(ctx)

with tab3:
    if is_container_open(tab3):
        from src.ui.tab3_concept import render_tab3

        render_tab3(ctx)

with tab4:
    if is_container_open(tab4):
        from src.ui.tab4_dictionary import render_tab4

        render_tab4(ctx)

with my_exams_tab:
    if is_container_open(my_exams_tab):
        from src.ui.my_exams_tab import render_my_exams_tab

        render_my_exams_tab(ctx)

if admin_tab is not None:
    with admin_tab:
        if is_container_open(admin_tab):
            from src.ui.admin_tab import render_admin_panel

            render_admin_panel(ctx)
