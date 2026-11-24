from __future__ import annotations

import html as _html
from typing import Any

import streamlit as st

from src.auth import (
    CREDIT_COST_DIRECT_CONCEPT_PDF,
    CREDIT_COST_ENGLISH_ANALYSIS,
    CREDIT_COST_ENGLISH_VARIANT,
    CREDIT_COST_KOREAN_ANALYSIS,
    CREDIT_COST_KOREAN_SET_TRANSFORM,
    FREE_MAX_BATCH_QUESTIONS,
    FREE_MONTHLY_CREDIT_LIMIT,
    PRO_MONTHLY_CREDIT_LIMIT,
    TIER_LABELS,
    ULTRA_MONTHLY_CREDIT_LIMIT,
    authenticate_google_user,
    authenticate_user,
    create_session_token,
    format_credit_count,
    get_bonus_credits,
    get_daily_free_ai_limit,
    get_daily_free_ai_used,
    get_monthly_credit_limit,
    get_remaining_monthly_base_credits,
    get_total_available_credits,
    get_used_monthly_credits,
    get_user_by_username,
    link_google_account,
    register_google_user,
    register_local_credentials_for_user,
    register_user,
    resolve_active_gemini_api_key,
    unlink_google_account,
    verify_session_token,
)
from src.config import DEFAULT_MODEL, FLASH_LITE_MODEL
from src.auth.oauth import (
    build_google_authorize_url,
    create_google_oauth_state,
    exchange_google_code_for_userinfo,
    verify_google_oauth_state,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.storage import StorageManager
from src.ui.context import AppContext


def _render_google_oauth_link_button(auth_url: str, label: str) -> None:
    """현재 탭(target=_self)에서 Google OAuth 동의 화면으로 이동하는 공식 스타일 버튼을 렌더링한다."""
    safe_url = _html.escape(auth_url, quote=True)
    safe_label = _html.escape(label)
    google_g_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 48 48" '
        'style="flex-shrink:0;">'
        '<path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 '
        '14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>'
        '<path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 '
        '5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>'
        '<path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19'
        'C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>'
        '<path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 '
        '2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>'
        '</svg>'
    )
    st.markdown(
        f"""
        <a href="{safe_url}" target="_self" style="
            display:flex; align-items:center; justify-content:center; gap:9px;
            width:100%; padding:0.52rem 0.75rem; margin-top:0.25rem; margin-bottom:0.25rem;
            background:#FFFFFF; color:#1F2937; border:1px solid #CBD5E1; border-radius:0.5rem;
            font-size:0.88rem; font-weight:600; text-decoration:none;
            box-shadow:0 1px 2px rgba(15,23,42,0.06); box-sizing:border-box;">
          {google_g_svg}
          <span>{safe_label}</span>
        </a>
        """,
        unsafe_allow_html=True,
    )


def _qp_scalar(key: str) -> str:
    val = st.query_params.get(key)
    if isinstance(val, list):
        return str(val[-1]) if val else ""
    return str(val) if val is not None else ""




def render_auth_and_sidebar(
    storage: StorageManager,
    creds: Any,
    google_login_cfg: Any,
) -> AppContext:
    """Google OAuth 콜백 소비, 세션 복원 및 좌측 계정·멤버십 사이드바를 렌더링하고 AppContext를 반환한다."""
    # Google OAuth 2.0 콜백 처리 (code/state 또는 error 파라미터가 있으면 즉시 소비 후 재실행)
    _oauth_code = _qp_scalar("code")
    _oauth_state = _qp_scalar("state")
    _oauth_error = _qp_scalar("error")
    if _oauth_code or _oauth_state or _oauth_error:
        for _qk in ("code", "state", "error", "scope", "authuser", "prompt", "hd"):
            st.query_params.pop(_qk, None)

        if _oauth_error:
            st.session_state["auth_flash_msg"] = (
                "error",
                f"Google 인증이 취소되었거나 실패했습니다: {_oauth_error}",
            )
            st.rerun()

        if not google_login_cfg.is_configured:
            st.session_state["auth_flash_msg"] = (
                "error",
                "서버에 Google OAuth 설정이 아직 등록되지 않았습니다.",
            )
            st.rerun()

        _state_payload = verify_google_oauth_state(_oauth_state, storage=storage)
        if _state_payload is None:
            st.session_state["auth_flash_msg"] = (
                "error",
                "Google 인증 상태 토큰이 만료되었거나 유효하지 않습니다. 다시 시도해 주세요.",
            )
            st.rerun()

        try:
            _g_info = exchange_google_code_for_userinfo(google_login_cfg, _oauth_code)
        except Exception as exc:
            if _state_payload.get("act") == "link" and _state_payload.get("u"):
                _tu = str(_state_payload["u"])
                st.session_state["logged_in_username"] = _tu
                st.query_params["session"] = create_session_token(_tu, storage=storage)
            st.session_state["auth_flash_msg"] = ("error", f"Google 로그인 처리 실패: {exc}")
            st.rerun()

        if _state_payload["act"] == "link":
            _target_u = str(_state_payload["u"])
            _ok_link, _msg_link, _ = link_google_account(
                storage,
                _target_u,
                _g_info["sub"],
                _g_info["email"],
            )
            st.session_state["logged_in_username"] = _target_u
            st.query_params["session"] = create_session_token(_target_u, storage=storage)
            st.session_state["auth_flash_msg"] = ("success" if _ok_link else "error", _msg_link)
            st.rerun()
        else:
            _ok_g, _msg_g, _g_user = authenticate_google_user(
                storage,
                _g_info["sub"],
                _g_info["email"],
            )
            if _ok_g and _g_user is not None:
                st.session_state["logged_in_username"] = _g_user.username
                st.session_state["pending_google_signup"] = None
                st.query_params["session"] = create_session_token(_g_user.username, storage=storage)
                st.session_state["auth_flash_msg"] = ("success", _msg_g)
                st.rerun()
            else:
                st.session_state["pending_google_signup"] = {
                    "sub": _g_info["sub"],
                    "email": _g_info["email"],
                    "suggested_name": _g_info.get("name") or "",
                }
                st.rerun()

    if not st.session_state["logged_in_username"]:
        raw_session_tok = st.query_params.get("session")
        if isinstance(raw_session_tok, list):
            raw_session_tok = raw_session_tok[-1] if raw_session_tok else ""
        if raw_session_tok:
            restored_username = verify_session_token(str(raw_session_tok), storage=storage)
            if restored_username:
                st.session_state["logged_in_username"] = restored_username
            else:
                st.query_params.pop("session", None)

    current_user = (
        get_user_by_username(storage, st.session_state["logged_in_username"])
        if st.session_state["logged_in_username"]
        else None
    )
    if st.session_state["logged_in_username"] and current_user is None:
        st.session_state["logged_in_username"] = ""
        st.query_params.pop("session", None)
    elif current_user is not None and not st.query_params.get("session"):
        st.query_params["session"] = create_session_token(current_user.username, storage=storage)

    is_admin = bool(current_user is not None and current_user.tier == "ADMIN")
    selected_model = DEFAULT_MODEL
    active_gemini_key = ""
    active_key_label = ""
    upload_gemini_key = ""
    upload_key_label = ""

    # ---------------------------------------------------------------------------
    # 사이드바: 비로그인 시 [로그인 / 회원가입] 탭만 표시, 로그인 시 프로필·개인키·설정 표시
    # ---------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("## 👤 계정 & 멤버십")
        _flash = st.session_state.pop("auth_flash_msg", None)
        if isinstance(_flash, (list, tuple)) and len(_flash) == 2:
            _f_level, _f_text = _flash
            if _f_level == "success":
                st.success(_f_text)
            elif _f_level == "info":
                st.info(_f_text)
            else:
                st.error(_f_text)

        if current_user is None:
            _pending_g = st.session_state.get("pending_google_signup")
            if isinstance(_pending_g, dict) and _pending_g.get("sub"):
                st.markdown("### ✨ Google 회원가입 마무리")
                _g_email_disp = _html.escape(str(_pending_g.get("email") or ""))
                st.info(f"연동 Google 계정: **{_g_email_disp or '(이메일 미제공)'}**")
                _add_local_creds = st.checkbox(
                    "아이디·비밀번호도 지금 함께 등록하기 (나중에 등록 가능)",
                    value=False,
                    key="pending_google_add_local_creds",
                )
                with st.form("sidebar_google_complete_signup_form"):
                    _g_nick = st.text_input(
                        "이름 (필수, 1~20자)",
                        value=str(_pending_g.get("suggested_name") or "")[:20],
                        placeholder="예: 김민준",
                    )
                    _g_id = ""
                    _g_pw = ""
                    _g_pw2 = ""
                    if _add_local_creds:
                        _g_id = st.text_input("아이디 (영문·숫자 3~30자)", placeholder="예: student01")
                        _g_pw = st.text_input("비밀번호 (4자 이상)", type="password")
                        _g_pw2 = st.text_input("비밀번호 확인", type="password")
                    _g_submit = st.form_submit_button("가입 완료하기", type="primary", width="stretch")

                if _g_submit:
                    if _add_local_creds and _g_pw != _g_pw2:
                        st.error("비밀번호 확인이 일치하지 않습니다.")
                    else:
                        _ok_rg, _msg_rg, _new_gu = register_google_user(
                            storage,
                            google_sub=str(_pending_g["sub"]),
                            google_email=str(_pending_g.get("email") or ""),
                            nickname=_g_nick,
                            desired_username=_g_id if _add_local_creds else "",
                            desired_password=_g_pw if _add_local_creds else "",
                        )
                        if _ok_rg and _new_gu is not None:
                            st.session_state["logged_in_username"] = _new_gu.username
                            st.session_state["pending_google_signup"] = None
                            st.query_params["session"] = create_session_token(
                                _new_gu.username, storage=storage
                            )
                            st.session_state["auth_flash_msg"] = ("success", _msg_rg)
                            st.rerun()
                        else:
                            st.error(_msg_rg)

                if st.button("취소하고 로그인 화면으로 돌아가기", key="cancel_google_signup_btn", width="stretch"):
                    st.session_state["pending_google_signup"] = None
                    st.rerun()
            else:
                auth_tab_login, auth_tab_signup = st.tabs(["🔑 로그인", "📝 회원가입"])
                with auth_tab_login:
                    with st.form("sidebar_login_form"):
                        login_id = st.text_input("아이디 (학번 또는 영문 ID)", placeholder="예: 2026101")
                        login_pw = st.text_input("비밀번호", type="password")
                        login_submit = st.form_submit_button("로그인", type="primary", width="stretch")
                    if login_submit:
                        ok, msg, logged_user = authenticate_user(storage, login_id, login_pw)
                        if ok and logged_user is not None:
                            st.session_state["logged_in_username"] = logged_user.username
                            st.query_params["session"] = create_session_token(
                                logged_user.username, storage=storage
                            )
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
                    if google_login_cfg.is_configured:
                        st.markdown(
                            "<div style='text-align:center; font-size:0.76rem; color:#94A3B8; margin:0.4rem 0 0.25rem;'>또는</div>",
                            unsafe_allow_html=True,
                        )
                        _login_state = create_google_oauth_state("login", storage=storage)
                        _render_google_oauth_link_button(
                            build_google_authorize_url(google_login_cfg, _login_state),
                            "Google 계정으로 로그인",
                        )

                with auth_tab_signup:
                    with st.form("sidebar_signup_form"):
                        reg_id = st.text_input("아이디 (영문·숫자 3~30자)", placeholder="예: student01")
                        reg_nick = st.text_input("이름", placeholder="예: 김민준")
                        reg_pw = st.text_input("비밀번호 (4자 이상)", type="password")
                        reg_pw2 = st.text_input("비밀번호 확인", type="password")
                        reg_submit = st.form_submit_button("회원가입", type="primary", width="stretch")
                    if reg_submit:
                        if reg_pw != reg_pw2:
                            st.error("비밀번호 확인이 일치하지 않습니다.")
                        else:
                            ok, msg, new_user = register_user(storage, reg_id, reg_nick, reg_pw)
                            if ok and new_user is not None:
                                st.session_state["logged_in_username"] = new_user.username
                                st.query_params["session"] = create_session_token(
                                    new_user.username, storage=storage
                                )
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                    if google_login_cfg.is_configured:
                        st.markdown(
                            "<div style='text-align:center; font-size:0.76rem; color:#94A3B8; margin:0.4rem 0 0.25rem;'>또는</div>",
                            unsafe_allow_html=True,
                        )
                        _signup_state = create_google_oauth_state("login", storage=storage)
                        _render_google_oauth_link_button(
                            build_google_authorize_url(google_login_cfg, _signup_state),
                            "Google 계정으로 간편 가입",
                        )
        else:
            tier_code = current_user.tier
            tier_kor = TIER_LABELS.get(tier_code, tier_code)
            active_gemini_key, active_key_label = resolve_active_gemini_api_key(
                current_user, creds.gemini_api_key
            )
            upload_gemini_key, upload_key_label = resolve_active_gemini_api_key(
                current_user, creds.gemini_api_key, force_server_key=True
            )
            credit_limit = get_monthly_credit_limit(current_user)
            credit_used = get_used_monthly_credits(current_user)
            base_rem = get_remaining_monthly_base_credits(current_user)
            bonus_rem = get_bonus_credits(current_user)
            total_avail = get_total_available_credits(current_user)
            free_ai_used = get_daily_free_ai_used(current_user)
            free_ai_lim = get_daily_free_ai_limit(current_user)
            bonus_html = f" <span style='color:#059669;'>(+보너스 {bonus_rem:,}C)</span>" if bonus_rem > 0 else ""

            if tier_code == "FREE":
                status_line = (
                    f"<div style='font-size:0.83rem; color:#334155; margin-top:0.45rem;'>"
                    f"가용 크레딧: <b>{total_avail:,}C</b> "
                    f"<span style='color:#64748B;'>(월간 {base_rem:,}/{credit_limit or 0:,}C)</span>{bonus_html}<br/>"
                    f"1회 최대 {FREE_MAX_BATCH_QUESTIONS}문항 · 오늘 무료 AI {free_ai_used}/{free_ai_lim}회"
                    f"</div>"
                )
            elif tier_code == "PRO":
                status_line = (
                    f"<div style='font-size:0.83rem; color:#1D4ED8; margin-top:0.45rem;'>"
                    f"✨ 가용 크레딧: <b>{total_avail:,}C</b> "
                    f"<span style='color:#64748B;'>(월간 {base_rem:,}/{credit_limit or 0:,}C)</span>{bonus_html}<br/>"
                    f"배치 무제한 · 오늘 무료 AI {free_ai_used}/{free_ai_lim}회"
                    f"</div>"
                )
            elif tier_code == "ULTRA":
                status_line = (
                    f"<div style='font-size:0.83rem; color:#C2410C; font-weight:600; margin-top:0.45rem;'>"
                    f"🔥 가용 크레딧: <b>{total_avail:,}C</b> "
                    f"<span style='color:#9A3412; font-weight:400;'>(월간 {base_rem:,}/{credit_limit or 0:,}C)</span>{bonus_html}<br/>"
                    f"PRO 대비 4배 · 오늘 무료 AI {free_ai_used}/{free_ai_lim}회"
                    f"</div>"
                )
            else:
                status_line = (
                    "<div style='font-size:0.83rem; color:#1D4ED8; font-weight:600; margin-top:0.45rem;'>"
                    "🛡️ 최고 관리자 · 크레딧 & 공용 DB 무제한<br/>"
                    f"<span style='color:#64748B; font-weight:400;'>이번 달 누적 사용: {credit_used:,}C</span></div>"
                )

            _has_local = bool(getattr(current_user, "has_local_credentials", True) and current_user.password_hash)
            _linked_email = (getattr(current_user, "google_email", "") or "").strip()
            if _has_local:
                _sub_line_html = f"@{_html.escape(current_user.username)}"
                if _linked_email:
                    _sub_line_html += f"<div style='font-size:0.72rem; color:#2563EB; margin-top:2px;'>🔗 {_html.escape(_linked_email)}</div>"
            else:
                _disp_email = _html.escape(_linked_email) if _linked_email else "미등록"
                _sub_line_html = f"Google 계정 · {_disp_email}"

            st.markdown(
                f"""
                <div class="account-card">
                  <div style="display:flex; justify-content:space-between; align-items:center; gap:6px;">
                    <div>
                      <div style="font-size:1rem; font-weight:800; color:#0F172A;">{_html.escape(current_user.nickname)}</div>
                      <div style="font-size:0.78rem; color:#64748B;">{_sub_line_html}</div>
                    </div>
                    <span class="tier-badge tier-{tier_code}">{tier_kor}</span>
                  </div>
                  {status_line}
                </div>
                """,
                unsafe_allow_html=True,
            )

            if credit_limit is not None and base_rem is not None:
                rem_ratio = max(0.0, min(1.0, base_rem / max(1, credit_limit)))
                st.progress(rem_ratio)

            _show_credit_ledger = st.toggle(
                "📜 내 크레딧 사용 내역 (최근 20건)",
                key="sidebar_toggle_my_credit_ledger",
            )
            if _show_credit_ledger:
                ledger_rows = storage.read_credit_ledger(username=current_user.username, limit=20)
                if not ledger_rows:
                    st.caption("아직 크레딧 사용·지급 내역이 없습니다.")
                else:
                    for lr in ledger_rows:
                        ts_disp = str(lr.get("ts") or "")[5:16].replace("T", " ")
                        delta_v = int(lr.get("delta") or 0)
                        delta_str = f"+{delta_v:,}C" if delta_v > 0 else f"{delta_v:,}C"
                        ev = str(lr.get("event") or "")
                        ev_badge = {
                            "USE": "차감",
                            "REFUND": "환불",
                            "BONUS_GRANT": "보너스",
                            "BONUS_DEDUCT": "회수",
                            "MONTHLY_RESET": "초기화",
                        }.get(ev, ev)
                        avail_v = lr.get("avail_after")
                        bal_txt = "무제한" if avail_v is None else f"{int(avail_v):,}C"
                        rsn = _html.escape(str(lr.get("reason") or ""))
                        st.markdown(
                            f"<div style='font-size:0.75rem; padding:4px 6px; margin-bottom:4px; "
                            f"background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px;'>"
                            f"<b>[{ev_badge}] {delta_str}</b> · 잔여 {bal_txt} "
                            f"<span style='color:#64748B;'>({ts_disp})</span><br/>"
                            f"<span style='color:#334155;'>{rsn}</span></div>",
                            unsafe_allow_html=True,
                        )

            # 계정 · Google 연동 관리 UI
            _show_acc_link = st.toggle(
                "⚙️ 계정 · 로그인 연동 관리",
                key="sidebar_toggle_account_link",
            )
            if _show_acc_link:
                if not _has_local:
                    st.caption(
                        "현재 Google 로그인 전용 계정입니다. 아이디와 비밀번호를 등록하면 일반 로그인도 사용할 수 있습니다."
                    )
                    with st.form("sidebar_register_local_creds_form"):
                        _new_local_id = st.text_input("등록할 아이디 (영문·숫자 3~30자)", placeholder="예: student01")
                        _new_local_pw = st.text_input("비밀번호 (4자 이상)", type="password")
                        _new_local_pw2 = st.text_input("비밀번호 확인", type="password")
                        _reg_local_btn = st.form_submit_button(
                            "💾 아이디·비밀번호 등록",
                            type="primary",
                            width="stretch",
                        )
                    if _reg_local_btn:
                        if _new_local_pw != _new_local_pw2:
                            st.error("비밀번호 확인이 일치하지 않습니다.")
                        else:
                            _ok_lc, _msg_lc, _updated_lc = register_local_credentials_for_user(
                                storage,
                                current_user.username,
                                _new_local_id,
                                _new_local_pw,
                            )
                            if _ok_lc and _updated_lc is not None:
                                st.session_state["logged_in_username"] = _updated_lc.username
                                st.query_params["session"] = create_session_token(
                                    _updated_lc.username, storage=storage
                                )
                                st.session_state["auth_flash_msg"] = ("success", _msg_lc)
                                st.rerun()
                            else:
                                st.error(_msg_lc)

                _has_google = bool((getattr(current_user, "google_sub", "") or "").strip())
                if _has_google:
                    st.caption(f"✅ 연동된 Google 계정: `{_linked_email or '연동됨'}`")
                    if _has_local:
                        if st.button(
                            "🔓 Google 연동 해제",
                            key="unlink_google_btn",
                            width="stretch",
                        ):
                            _ok_ul, _msg_ul, _ = unlink_google_account(storage, current_user.username)
                            if _ok_ul:
                                st.session_state["auth_flash_msg"] = ("info", _msg_ul)
                                st.rerun()
                            else:
                                st.error(_msg_ul)
                    else:
                        st.caption("아이디·비밀번호를 먼저 등록해야 Google 연동을 해제할 수 있습니다.")
                else:
                    if google_login_cfg.is_configured:
                        _link_state = create_google_oauth_state(
                            "link",
                            username=current_user.username,
                            storage=storage,
                        )
                        _render_google_oauth_link_button(
                            build_google_authorize_url(google_login_cfg, _link_state),
                            "Google 계정 연동하기",
                        )
                    else:
                        st.caption("서버에 Google OAuth 설정이 아직 등록되지 않았습니다.")
                        if is_admin:
                            st.caption(
                                "`.streamlit/secrets.toml`에 `GOOGLE_LOGIN_CLIENT_ID`, `GOOGLE_LOGIN_CLIENT_SECRET`, `GOOGLE_LOGIN_REDIRECT_URI`를 추가해 주세요."
                            )

            if st.button("🚪 로그아웃", key="logout_btn", width="stretch"):
                st.session_state["logged_in_username"] = ""
                st.session_state["pending_google_signup"] = None
                st.session_state["matched_items"] = []
                st.session_state["generated_exam_pdf"] = None
                st.query_params.pop("session", None)
                st.rerun()

            st.markdown("---")
            # 전 등급 · 전 기능 Gemini 3.5 Flash-Lite 단일 모델 통일
            selected_model = FLASH_LITE_MODEL

            # 저장소 & DB 요약
            q_count, exam_count, concept_book_count = storage.get_db_summary_counts()
            st.markdown("### 📊 기출 · 개념서 DB & 저장소")
            c1, c2, c3 = st.columns(3)
            c1.metric("기출 문항", f"{q_count}개")
            c2.metric("등록 회차", f"{exam_count}회")
            c3.metric("공용 개념서", f"{concept_book_count}권")
            st.caption("💾 서버 영구 저장소 (`./exam_db`, `./concept_db`)")


    return AppContext(
        storage=storage,
        creds=creds,
        google_login_cfg=google_login_cfg,
        current_user=current_user,
        is_admin=is_admin,
        selected_model=selected_model,
        active_gemini_key=active_gemini_key,
        active_key_label=active_key_label,
        upload_gemini_key=upload_gemini_key,
        upload_key_label=upload_key_label,
    )


def render_guest_landing() -> None:
    """비로그인 상태일 때 멤버십 등급 안내 카드 4종 및 정액 크레딧 단가표를 출력한다."""
    st.info("👈 좌측 사이드바 상단의 **[로그인 / 회원가입]** 탭에서 로그인하시면 기출 탐색·변형문제 출제·지문 분석지 PDF 기능을 이용하실 수 있습니다.")
    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
    with t_col1:
        with st.container(border=True):
            st.markdown("<span class='tier-badge tier-FREE'>FREE · 0원</span>", unsafe_allow_html=True)
            st.markdown("#### 무료 회원")
            st.write(f"- **월 {format_credit_count(FREE_MONTHLY_CREDIT_LIMIT)}** 크레딧 제공")
            st.write(f"- 1회 생성 **최대 {FREE_MAX_BATCH_QUESTIONS}문항** 제한")
            st.write("- 기출 DB 열람 · AI 해설/영단어 무료")
    with t_col2:
        with st.container(border=True):
            st.markdown("<span class='tier-badge tier-PRO'>PRO · 9,900원/월</span>", unsafe_allow_html=True)
            st.markdown("#### 스탠다드 유료")
            st.write(f"- **월 {format_credit_count(PRO_MONTHLY_CREDIT_LIMIT)}** 크레딧 제공")
            st.write("- 1회 생성 문항 수 **제한 없음**")
            st.write("- **공용 기출·개념서 DB** 구축/이용 (0C)")
    with t_col3:
        with st.container(border=True):
            st.markdown("<span class='tier-badge tier-ULTRA'>ULTRA · 19,900원/월</span>", unsafe_allow_html=True)
            st.markdown("#### 대용량 유료 (4배)")
            st.write("- PRO 모든 혜택 포함")
            st.write(f"- **월 {format_credit_count(ULTRA_MONTHLY_CREDIT_LIMIT)}** 크레딧 (PRO 4배)")
            st.write("- 학원·교사용 대용량 출제 최적화")
    with t_col4:
        with st.container(border=True):
            st.markdown("<span class='tier-badge tier-ADMIN'>ADMIN</span>", unsafe_allow_html=True)
            st.markdown("#### 최고 관리자")
            st.write("- 모든 기능·크레딧 **무제한**")
            st.write("- 공용 DB 수정·삭제 · NCIC 기준표 관리")
            st.write("- 회원 등급 · 보너스 · 원화 비용 관리")

    st.caption(
        f"💳 **정액 크레딧 안내 (신규 생성 · 캐시 풀 동일)**: "
        f"영어 변형 **{CREDIT_COST_ENGLISH_VARIANT}C/문항** · 영어 분석지 **{CREDIT_COST_ENGLISH_ANALYSIS}C/지문** · "
        f"국어 변형/분석지 **각 {CREDIT_COST_KOREAN_SET_TRANSFORM}C/세트** · "
        f"수·사·과·한 **개조 {CREDIT_COST_CONCEPT_REWRITE}C / 무개조 {CREDIT_COST_CONCEPT_MATCH_ORIG}C** · "
        f"개인 개념서 PDF **{CREDIT_COST_DIRECT_CONCEPT_PDF}C/회** · "
        f"AI 해설·1:1질문·영단어 사전 **0C 무료**"
    )
    st.stop()
