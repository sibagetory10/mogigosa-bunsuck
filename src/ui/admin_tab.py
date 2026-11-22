from __future__ import annotations

import html as _html
import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
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

    _render_server_storage_admin_section(ctx)
    _render_server_git_update_admin_section()
    _render_english_pool_admin_section(ctx)
    _render_korean_analysis_pool_admin_section(ctx)


def _format_bytes_human(num_bytes: int | float) -> str:
    val = float(max(0, num_bytes or 0))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if val < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(val):,} B"
            if val >= 100:
                return f"{val:.1f} {unit}"
            return f"{val:.2f} {unit}"
        val /= 1024.0
    return "0 B"


def _render_server_storage_admin_section(ctx: AppContext) -> None:
    st.divider()
    hdr_c1, hdr_c2 = st.columns([3.5, 1])
    with hdr_c1:
        st.subheader("💾 서버 저장공간 & 앱 DB 용량 모니터링")
    with hdr_c2:
        st.markdown("<div style='height: 0.45rem;'></div>", unsafe_allow_html=True)
        refresh_storage = st.button("🔄 용량 새로고침", key="admin_refresh_storage_btn", width="stretch")

    stats = ctx.storage.get_server_storage_stats(force_refresh=bool(refresh_storage))
    st.caption(f"기준 디스크 경로: `{stats['mount_path']}`")

    disk_total = int(stats["disk_total_bytes"])
    disk_used = int(stats["disk_used_bytes"])
    disk_free = int(stats["disk_free_bytes"])
    disk_pct = float(stats["disk_used_percent"])
    app_bytes = int(stats["app_total_bytes"])
    app_files = int(stats["app_total_files"])

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("서버 전체 저장공간", _format_bytes_human(disk_total))
    s2.metric(
        "서버 현재 사용 중",
        _format_bytes_human(disk_used),
        delta=f"{disk_pct:.1f}% 사용",
        delta_color="off",
    )
    s3.metric("서버 여유 공간", _format_bytes_human(disk_free))
    s4.metric(
        "앱 DB 사용량",
        _format_bytes_human(app_bytes),
        delta=f"파일 {app_files:,}개",
        delta_color="off",
    )

    st.progress(
        min(1.0, max(0.0, disk_pct / 100.0)),
        text=(
            f"서버 디스크 사용률: {disk_pct:.1f}% "
            f"({_format_bytes_human(disk_used)} / {_format_bytes_human(disk_total)})"
        ),
    )

    if disk_pct >= 85.0:
        st.warning(
            "⚠️ 서버 디스크 사용률이 85%를 초과했습니다. "
            "불필요한 캐시나 기출 PDF 정리를 권장합니다."
        )

    with st.expander("📂 앱 폴더별 저장공간 상세 내역", expanded=False):
        sorted_bd = sorted(stats["breakdown"], key=lambda it: it["bytes"], reverse=True)
        rows = [
            {
                "항목": it["label"],
                "경로": it["path"],
                "파일 수": f"{int(it['files']):,}개",
                "사용 용량": _format_bytes_human(it["bytes"]),
                "앱 내 비중(%)": f"{(it['bytes'] / app_bytes * 100.0):.1f}%" if app_bytes > 0 else "0.0%",
            }
            for it in sorted_bd
        ]
        st.dataframe(rows, hide_index=True, width="stretch")


def _run_git_cmd(repo_dir: Path, args: list[str], timeout: int = 15) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        return proc.returncode == 0, (out or err)
    except Exception as exc:
        return False, str(exc)


@st.cache_data(ttl=30, show_spinner=False)
def _get_cached_git_status_snapshot(repo_dir_str: str) -> dict[str, Any]:
    repo_p = Path(repo_dir_str)
    ok_head, local_head = _run_git_cmd(repo_p, ["rev-parse", "--short", "HEAD"])
    ok_msg, local_msg = _run_git_cmd(repo_p, ["log", "-1", "--format=%cd · %s", "--date=short"])
    ok_st, dirty_raw = _run_git_cmd(repo_p, ["status", "--short"])
    ok_rem, remote_head = _run_git_cmd(repo_p, ["rev-parse", "--short", "origin/main"])
    dirty_lines = [ln for ln in dirty_raw.splitlines() if ln.strip()] if ok_st else []
    return {
        "ok_head": ok_head,
        "local_head": local_head,
        "ok_msg": ok_msg,
        "local_msg": local_msg,
        "dirty_lines": dirty_lines,
        "ok_rem": ok_rem,
        "remote_head": remote_head,
    }


def _render_server_git_update_admin_section() -> None:
    st.divider()
    repo_dir = Path(__file__).resolve().parents[2]
    hdr_c1, hdr_c2 = st.columns([3.5, 1])
    with hdr_c1:
        st.subheader("🔄 서버 Git 상태 확인 · 원클릭 동기화")
    with hdr_c2:
        st.markdown("<div style='height: 0.45rem;'></div>", unsafe_allow_html=True)
        check_remote = st.button("🔍 최신 상태 확인", key="admin_git_check_btn", width="stretch")

    if check_remote:
        _run_git_cmd(repo_dir, ["fetch", "origin", "main", "--quiet"], timeout=20)
        _get_cached_git_status_snapshot.clear()

    git_snap = _get_cached_git_status_snapshot(str(repo_dir))
    ok_head = bool(git_snap["ok_head"])
    local_head = str(git_snap["local_head"])
    ok_msg = bool(git_snap["ok_msg"])
    local_msg = str(git_snap["local_msg"])
    dirty_lines = list(git_snap["dirty_lines"])
    ok_rem = bool(git_snap["ok_rem"])
    remote_head = str(git_snap["remote_head"])

    g1, g2, g3 = st.columns(3)
    g1.metric("현재 서버 커밋", local_head if ok_head else "확인 불가")
    g2.metric(
        "원격 origin/main",
        remote_head if ok_rem else "확인 불가",
        delta=("최신 일치" if ok_head and ok_rem and local_head == remote_head else ("업데이트 있음" if ok_head and ok_rem else None)),
        delta_color=("normal" if ok_head and ok_rem and local_head == remote_head else "inverse"),
    )
    g3.metric(
        "로컬 작업 트리",
        "깨끗함" if not dirty_lines else f"변경 {len(dirty_lines)}건",
    )
    if ok_msg and local_msg:
        st.caption(f"현재 적용된 커밋: `{local_msg}`")
    if dirty_lines:
        with st.expander(f"⚠️ 커밋되지 않은 로컬 변경 파일 ({len(dirty_lines)}건)", expanded=False):
            st.code("\n".join(dirty_lines), language="text")

    uc1, uc2 = st.columns([1.4, 2.2])
    with uc1:
        force_clean_curriculum = st.checkbox(
            "curriculum_2022 충돌 자동 이관·해제",
            value=True,
            key="admin_git_clean_curriculum",
        )
    with uc2:
        if st.button("🚀 최신 코드 동기화 · 서버 재시작", key="admin_git_pull_restart_btn", type="primary", width="stretch"):
            if force_clean_curriculum:
                from src.curriculum_2022 import _DATA_PATH, _load_runtime_standards, _save_runtime_standards
                if _DATA_PATH.exists():
                    try:
                        cur_json = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
                        rt_stds = _load_runtime_standards()
                        for k, v in (cur_json.get("categories") or {}).items():
                            stds = v.get("standards") or []
                            if stds:
                                rt_stds[k] = stds
                        _save_runtime_standards(rt_stds)
                    except Exception:
                        pass
                _run_git_cmd(repo_dir, ["checkout", "--", "data/curriculum_2022.json"])

            is_linux = platform.system().lower() == "linux"
            script_path = repo_dir / "scripts" / "oracle_auto_update.sh"
            if is_linux and script_path.exists():
                try:
                    proc = subprocess.run(
                        ["bash", str(script_path)],
                        cwd=str(repo_dir),
                        capture_output=True,
                        text=True,
                        timeout=60,
                        encoding="utf-8",
                        errors="replace",
                    )
                    log_txt = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
                    if proc.returncode == 0:
                        st.success("✅ 동기화 명령 실행 완료! 서비스가 재시작되면 몇 초 후 새로고침하세요.")
                    else:
                        st.error(f"❌ 동기화 실패 (exit={proc.returncode})")
                    if log_txt:
                        st.code(log_txt, language="text")
                except Exception as exc:
                    st.error(f"❌ 실행 오류: {exc}")
                _get_cached_git_status_snapshot.clear()
            else:
                ok_pull, pull_out = _run_git_cmd(repo_dir, ["pull", "--ff-only", "origin", "main"], timeout=30)
                _get_cached_git_status_snapshot.clear()
                if ok_pull:
                    st.success(f"✅ git pull 완료: {pull_out or 'Already up to date.'}")
                else:
                    st.error(f"❌ git pull 실패: {pull_out}")


def _render_english_pool_admin_section(ctx: AppContext) -> None:
    st.divider()
    st.subheader("🔤 영어 변형 문제 풀(Pool) 캐시 관리")
    st.caption(
        "파이썬 검증을 통과한 영어 변형 문항은 서버의 `./exam_db/english_question_pool.json`에 "
        "지문·유형·옵션 해시별로 최대 3개씩 저장됩니다."
    )

    pool = ctx.storage.load_english_pool()
    total_keys = len(pool)
    flat_entries: list[tuple[str, int, dict]] = []
    type_counts: dict[str, int] = {}
    for c_key, items in pool.items():
        for idx, item in enumerate(items):
            t_name = str(item.get("target_type") or "기타").strip() or "기타"
            type_counts[t_name] = type_counts.get(t_name, 0) + 1
            flat_entries.append((c_key, idx, item))

    total_items = len(flat_entries)
    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("캐시 키(지문×유형) 수", f"{total_keys}개")
    c_m2.metric("저장된 변형 문항 수", f"{total_items}문항")
    c_m3.metric("저장된 변형 유형 수", f"{len(type_counts)}종")

    if type_counts:
        summary_parts = [f"{k}({v})" for k, v in sorted(type_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        st.caption("유형별 분포: " + " · ".join(summary_parts))

    if total_items == 0:
        st.info("현재 저장된 영어 변형 캐시 문항이 없습니다.")
        return

    col_clr1, col_clr2 = st.columns([2, 1])
    with col_clr1:
        confirm_clear = st.checkbox(
            "영어 변형 캐시만 전체 비우기 확인 (기출 DB·회원 DB는 유지됨)",
            key="admin_confirm_clear_eng_pool",
        )
    with col_clr2:
        if st.button(
            "🧹 영어 변형 캐시 전체 비우기",
            disabled=not confirm_clear,
            width="stretch",
            key="admin_clear_eng_pool_btn",
        ):
            removed_cnt = ctx.storage.clear_english_pool()
            st.success(f"🧹 영어 변형 캐시 {removed_cnt}문항을 모두 비웠습니다.")
            st.rerun()

    eng_pool_exp = create_lazy_expander(
        f"📋 저장된 영어 변형 캐시 상세 조회 / 개별 삭제 ({total_items}문항)",
        expanded=False,
        key="admin_eng_pool_detail_exp",
    )
    with eng_pool_exp:
        if not is_container_open(eng_pool_exp):
            return
        f_col1, f_col2 = st.columns([1, 2])
        with f_col1:
            type_options = ["전체"] + sorted(type_counts.keys())
            sel_type = st.selectbox("유형 필터", options=type_options, key="admin_eng_pool_type_filter")
        with f_col2:
            pool_q = st.text_input(
                "지문/해설/키 검색",
                placeholder="키워드 또는 캐시 키 입력...",
                key="admin_eng_pool_search",
            ).strip().lower()

        filtered_entries = [
            (ck, idx, it)
            for ck, idx, it in flat_entries
            if (sel_type == "전체" or (it.get("target_type") or "기타").strip() == sel_type)
            and (
                not pool_q
                or pool_q in ck.lower()
                or pool_q in str(it.get("rewritten_stem") or "").lower()
                or pool_q in str(it.get("pdf_grounded_explanation") or "").lower()
            )
        ]

        st.caption(f"검색 결과: {len(filtered_entries)}문항 (최대 25건 표시)")
        for ck, idx, it in filtered_entries[:25]:
            t_label = str(it.get("target_type") or "기타")
            ans_label = str(it.get("answer") or "-")
            stem_preview = " ".join(str(it.get("rewritten_stem") or "").split())[:130]
            r_col1, r_col2 = st.columns([5, 1])
            with r_col1:
                st.markdown(f"**[{t_label}]** · 키 `{ck}` (#{idx + 1}) · 정답 `{ans_label}`")
                if stem_preview:
                    st.caption(f"{stem_preview}...")
            with r_col2:
                if st.button("🗑️ 삭제", key=f"del_eng_pool_{ck}_{idx}", width="stretch"):
                    removed = ctx.storage.delete_english_pool_entry(ck, idx)
                    if removed > 0:
                        st.success(f"캐시 항목(`{ck}` #{idx + 1})을 삭제했습니다.")
                    st.rerun()


def _render_korean_analysis_pool_admin_section(ctx: AppContext) -> None:
    st.divider()
    st.subheader("📖 국어 지문 분석지 캐시 · PDF DB 관리")
    st.caption(
        "최초 생성된 국어 지문 분석지는 `./exam_db/korean_analysis_pool.json`과 "
        "지문별 PDF(`./exam_db/korean_analysis_pdfs/*.pdf`)로 영구 저장되어 "
        "동일 지문 선택 시 재렌더링 없이 즉시 이어 붙여 제공합니다."
    )

    pool = ctx.storage.load_korean_analysis_pool()
    total_cnt = len(pool)
    pdf_saved_cnt = sum(1 for ck in pool if ctx.storage.has_korean_analysis_pdf(ck))
    exam_names: set[str] = set()
    sec_counts: dict[str, int] = {}
    entries_list: list[tuple[str, dict[str, Any]]] = []
    for ck, entry in pool.items():
        if not isinstance(entry, dict):
            continue
        ex_name = str(entry.get("exam_name") or "미분류 회차").strip() or "미분류 회차"
        sec_name = str(entry.get("section") or "기타").strip() or "기타"
        exam_names.add(ex_name)
        sec_counts[sec_name] = sec_counts.get(sec_name, 0) + 1
        entries_list.append((ck, entry))

    k_m1, k_m2, k_m3, k_m4 = st.columns(4)
    k_m1.metric("저장된 분석지 세트 수", f"{total_cnt}세트")
    k_m2.metric("저장된 지문 PDF", f"{pdf_saved_cnt}개")
    k_m3.metric("포함된 회차 수", f"{len(exam_names)}회차")
    k_m4.metric("영역 수", f"{len(sec_counts)}개 영역")

    if sec_counts:
        summary_parts = [f"{k}({v})" for k, v in sorted(sec_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        st.caption("영역별 분포: " + " · ".join(summary_parts))

    if total_cnt == 0:
        st.info("현재 저장된 국어 지문 분석지 캐시가 없습니다.")
        return

    col_clr1, col_clr2 = st.columns([2, 1])
    with col_clr1:
        confirm_clear = st.checkbox(
            "국어 분석지 캐시 전체 비우기 확인 (기출 DB·회원 DB는 유지됨)",
            key="admin_confirm_clear_kor_an_pool",
        )
    with col_clr2:
        if st.button(
            "🧹 국어 분석지 캐시 전체 비우기",
            disabled=not confirm_clear,
            width="stretch",
            key="admin_clear_kor_an_pool_btn",
        ):
            removed_cnt = ctx.storage.clear_korean_analysis_pool()
            st.success(f"🧹 국어 지문 분석지 캐시 {removed_cnt}세트를 모두 비웠습니다.")
            st.rerun()

    kor_pool_exp = create_lazy_expander(
        f"📋 저장된 국어 지문 분석지 상세 조회 / 개별 삭제 ({total_cnt}세트)",
        expanded=False,
        key="admin_kor_pool_detail_exp",
    )
    with kor_pool_exp:
        if not is_container_open(kor_pool_exp):
            return
        f_col1, f_col2, f_col3 = st.columns([1.2, 1.0, 1.5])
        with f_col1:
            sel_exam = st.selectbox(
                "회차 필터",
                options=["전체"] + sorted(exam_names, reverse=True),
                key="admin_kor_an_pool_exam_filter",
            )
        with f_col2:
            sel_sec = st.selectbox(
                "영역 필터",
                options=["전체"] + sorted(sec_counts.keys()),
                key="admin_kor_an_pool_sec_filter",
            )
        with f_col3:
            kor_q = st.text_input(
                "제목/주제/회차/키 검색",
                placeholder="키워드 또는 캐시 키 입력...",
                key="admin_kor_an_pool_search",
            ).strip().lower()

        sorted_entries = sorted(
            entries_list,
            key=lambda kv: (str(kv[1].get("saved_at") or ""), str(kv[1].get("exam_name") or "")),
            reverse=True,
        )
        filtered_entries = []
        for ck, ent in sorted_entries:
            ex_n = str(ent.get("exam_name") or "미분류 회차").strip() or "미분류 회차"
            sec_n = str(ent.get("section") or "기타").strip() or "기타"
            if sel_exam != "전체" and ex_n != sel_exam:
                continue
            if sel_sec != "전체" and sec_n != sel_sec:
                continue
            an_dict = ent.get("analysis") if isinstance(ent.get("analysis"), dict) else {}
            an_title = str(an_dict.get("title") or "")
            an_theme = str(an_dict.get("core_theme") or "")
            if kor_q and not (
                kor_q in ck.lower()
                or kor_q in ex_n.lower()
                or kor_q in an_title.lower()
                or kor_q in an_theme.lower()
                or kor_q in str(ent.get("group_id") or "").lower()
            ):
                continue
            filtered_entries.append((ck, ent, an_dict))

        st.caption(f"검색 결과: {len(filtered_entries)}세트 (최대 25건 표시)")
        for ck, ent, an_dict in filtered_entries[:25]:
            ex_n = str(ent.get("exam_name") or "미분류 회차")
            gid_n = str(ent.get("group_id") or "-")
            sec_n = str(ent.get("section") or "기타")
            sub_n = str(ent.get("subgenre") or "")
            sec_disp = f"{sec_n}({sub_n})" if sub_n else sec_n
            an_title = str(an_dict.get("title") or "제목 없음")
            theme_prev = " ".join(str(an_dict.get("core_theme") or "").split())[:120]
            saved_at = str(ent.get("saved_at") or "")

            r_col1, r_col2 = st.columns([5, 1])
            with r_col1:
                pdf_badge = "📑 분석PDF" if ctx.storage.has_korean_analysis_pdf(ck) else "📝 JSON만"
                pas_badge = " · 📄 원문PDF" if ctx.storage.has_korean_passage_pdf(ck) else ""
                st.markdown(f"**[{ex_n} · {gid_n}번]** {an_title} (`{sec_disp}` · {pdf_badge}{pas_badge}) · 키 `{ck}`")
                cap_parts = [p for p in (f"저장: {saved_at}" if saved_at else "", theme_prev) if p]
                if cap_parts:
                    st.caption(" · ".join(cap_parts))
            with r_col2:
                if st.button("🗑️ 삭제", key=f"del_kor_an_pool_{ck}", width="stretch"):
                    removed = ctx.storage.delete_korean_analysis_entry(ck)
                    if removed > 0:
                        st.success(f"국어 분석지 캐시(`{ck}`)를 삭제했습니다.")
                    st.rerun()




render_admin_panel = _render_admin_panel
