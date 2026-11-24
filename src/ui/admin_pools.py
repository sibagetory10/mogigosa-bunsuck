from __future__ import annotations

import html as _html
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import streamlit as st

from src.ui.common import create_lazy_expander, is_container_open
from src.ui.context import AppContext

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


def _render_english_batch_builder_expander(ctx: AppContext) -> None:
    """관리자 전용: Gemini Batch API(50% 할인)로 영어 v3 변형 문제 풀을 비동기 대량 구축한다."""
    batch_exp = create_lazy_expander(
        "📦 영어 v3 비동기 Batch API 대량 풀 구축 (50% 할인)",
        expanded=False,
        key="admin_eng_v3_batch_exp",
    )
    with batch_exp:
        if not is_container_open(batch_exp):
            return
        from src.auth import resolve_active_gemini_api_key
        from src.english_v3.batch_runner import (
            load_batch_jobs,
            submit_generation_batch,
            sync_and_advance_batch_job,
        )
        from src.english_v3.routing import ROUTING_PROFILES
        from src.models import ENGLISH_DEFAULT_EXCLUDED_Q_NUMS, ENGLISH_TRANSFORM_TYPES

        api_key, key_lbl = resolve_active_gemini_api_key(ctx.current_user, ctx.creds.gemini_api_key)
        all_db = ctx.storage.load_db_readonly()
        eng_recs = [
            r
            for r in all_db
            if r.subject_category == "영어"
            and r.q_num not in ENGLISH_DEFAULT_EXCLUDED_Q_NUMS
            and bool((getattr(r, "passage_text", "") or "").strip())
        ]
        avail_exams = sorted({r.exam_name for r in eng_recs}, reverse=True)

        if not avail_exams:
            st.info("Batch API에 투입할 영어 독해 기출 지문이 DB에 없습니다.")
        else:
            b_c1, b_c2 = st.columns([1.4, 1.0])
            with b_c1:
                sel_exam = st.selectbox("대상 영어 모의고사 회차", options=avail_exams, key="admin_v3_batch_exam")
            with b_c2:
                sel_profile = st.selectbox(
                    "라우팅 프로필",
                    options=list(ROUTING_PROFILES.keys()),
                    index=0,
                    key="admin_v3_batch_profile",
                )
            sel_types = st.multiselect(
                "사전 적재할 변형 유형",
                options=list(ENGLISH_TRANSFORM_TYPES),
                default=["주제 파악", "빈칸 추론", "어법상 틀린 것 (밑줄 ①~⑤)", "글의 순서 ((A)-(B)-(C))"],
                key="admin_v3_batch_types",
            )
            if st.button("🚀 비동기 Batch 작업 제출 (50% 할인)", key="admin_v3_batch_submit_btn", type="primary"):
                if not api_key:
                    st.error(f"Gemini API 키({key_lbl})가 설정되지 않았습니다.")
                elif not sel_types:
                    st.warning("최소 1개 이상의 변형 유형을 선택하세요.")
                else:
                    target_recs = [r for r in eng_recs if r.exam_name == sel_exam]
                    b_jobs = [
                        {
                            "passage_text": r.passage_text,
                            "target_type": t_type,
                            "difficulty": "고3 평가원·수능 수준",
                            "blank_unit": "구 (3~6단어)",
                            "vocab_style": "밑줄형 (①~⑤)",
                            "source_label": f"출처: {r.exam_name} {r.q_num}번",
                            "key_vocabulary": list(getattr(r, "key_vocabulary", []) or []),
                        }
                        for r in target_recs
                        for t_type in sel_types
                    ]
                    try:
                        rec_b = submit_generation_batch(
                            api_key=api_key,
                            storage=ctx.storage,
                            jobs=b_jobs,
                            display_label=f"{sel_exam} ({len(sel_types)}유형)",
                            routing_profile=sel_profile,
                        )
                        st.success(
                            f"✅ Batch 작업 `{rec_b['job_id']}` 제출 완료! "
                            f"(신규 {rec_b['submitted_count']}건 / 기존 캐시 스킵 {rec_b['skipped_cached']}건)"
                        )
                    except Exception as exc:
                        st.error(f"Batch 제출 실패: {exc}")

        tracked = load_batch_jobs(ctx.storage)
        if tracked:
            st.markdown("**📋 추적 중인 Batch API 작업 목록**")
            for b_item in tracked[:10]:
                jid = str(b_item.get("job_id") or "")
                col_i, col_b = st.columns([4, 1.3])
                with col_i:
                    st.markdown(
                        f"- `{jid}` **{b_item.get('display_label')}** · 단계 `{b_item.get('phase')}` · "
                        f"상태 `{b_item.get('state')}` · 적재 `{b_item.get('committed_count', 0)}건` / "
                        f"탈락 `{b_item.get('rejected_count', 0)}건`"
                    )
                with col_b:
                    if b_item.get("phase") not in ("completed", "failed"):
                        if st.button("🔄 동기화/수확", key=f"sync_v3_batch_{jid}", width="stretch"):
                            if not api_key:
                                st.error("API 키가 필요합니다.")
                            else:
                                try:
                                    updated = sync_and_advance_batch_job(api_key, ctx.storage, jid)
                                    st.success(
                                        f"`{jid}` 동기화 완료 → {updated.get('phase')} "
                                        f"(적재 {updated.get('committed_count', 0)}건)"
                                    )
                                    st.rerun()
                                except Exception as exc:
                                    st.error(f"동기화 오류: {exc}")


def _render_english_pool_admin_section(ctx: AppContext) -> None:
    st.divider()
    st.subheader("🔤 영어 변형 문제 풀(Pool) 캐시 관리 (v3)")
    st.caption(
        "파이썬 하드 규칙 + Blind Solver 검수를 통과한 영어 변형 문항은 `./exam_db/english_question_pool.json`에 "
        "지문·유형·옵션 해시(v3)별로 최대 5개씩 저장됩니다."
    )

    pool = ctx.storage.load_english_pool()
    total_keys = len(pool)
    flat_entries: list[tuple[str, int, dict]] = []
    type_counts: dict[str, int] = {}
    for c_key, items in pool.items():
        for idx, item in enumerate(items):
            t_name = str(item.get("target_question_type") or item.get("target_type") or "기타").strip() or "기타"
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

    _render_english_batch_builder_expander(ctx)

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
            t_label = str(it.get("target_question_type") or it.get("target_type") or "기타")
            ans_label = str(it.get("preserved_answer") or it.get("answer") or "-")
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




def _render_english_analysis_pool_admin_section(ctx: AppContext) -> None:
    st.divider()
    st.subheader("🔤 영어 지문 분석지 캐시 · PDF DB 관리")
    st.caption(
        "최초 생성된 영어 지문 분석지는 `./exam_db/english_analysis_pool.json`과 "
        "지문별 PDF(`./exam_db/english_analysis_pdfs/*.pdf`)로 영구 저장되어 "
        "동일 지문 선택 시 재렌더링 없이 즉시 이어 붙여 제공합니다."
    )

    pool = ctx.storage.load_english_analysis_pool()
    total_cnt = len(pool)
    pdf_saved_cnt = sum(1 for ck in pool if ctx.storage.has_english_analysis_pdf(ck))
    exam_names: set[str] = set()
    type_counts: dict[str, int] = {}
    entries_list: list[tuple[str, dict[str, Any]]] = []
    for ck, entry in pool.items():
        if not isinstance(entry, dict):
            continue
        ex_name = str(entry.get("exam_name") or "직접 입력 / 미분류").strip() or "직접 입력 / 미분류"
        q_type = str(entry.get("question_type") or "기타").strip() or "기타"
        exam_names.add(ex_name)
        type_counts[q_type] = type_counts.get(q_type, 0) + 1
        entries_list.append((ck, entry))

    e_m1, e_m2, e_m3, e_m4 = st.columns(4)
    e_m1.metric("저장된 분석지 수", f"{total_cnt}지문")
    e_m2.metric("저장된 지문 PDF", f"{pdf_saved_cnt}개")
    e_m3.metric("포함된 출처/회차", f"{len(exam_names)}종")
    e_m4.metric("원래 유형 수", f"{len(type_counts)}종")

    if total_cnt == 0:
        st.info("현재 저장된 영어 지문 분석지 캐시가 없습니다.")
        return

    col_clr1, col_clr2 = st.columns([2, 1])
    with col_clr1:
        confirm_clear = st.checkbox(
            "영어 분석지 캐시 전체 비우기 확인 (기출 DB·회원 DB는 유지됨)",
            key="admin_confirm_clear_eng_an_pool",
        )
    with col_clr2:
        if st.button(
            "🧹 영어 분석지 캐시 전체 비우기",
            disabled=not confirm_clear,
            width="stretch",
            key="admin_clear_eng_an_pool_btn",
        ):
            removed_cnt = ctx.storage.clear_english_analysis_pool()
            st.success(f"🧹 영어 지문 분석지 캐시 {removed_cnt}건을 모두 비웠습니다.")
            st.rerun()

    eng_an_exp = create_lazy_expander(
        f"📋 저장된 영어 지문 분석지 상세 조회 / 개별 삭제 ({total_cnt}건)",
        expanded=False,
        key="admin_eng_an_pool_detail_exp",
    )
    with eng_an_exp:
        if not is_container_open(eng_an_exp):
            return
        f_col1, f_col2 = st.columns([1.2, 2.0])
        with f_col1:
            sel_exam = st.selectbox(
                "회차/출처 필터",
                options=["전체"] + sorted(exam_names, reverse=True),
                key="admin_eng_an_pool_exam_filter",
            )
        with f_col2:
            eng_q = st.text_input(
                "제목/주제/출처/키 검색",
                placeholder="키워드 또는 캐시 키 입력...",
                key="admin_eng_an_pool_search",
            ).strip().lower()

        sorted_entries = sorted(
            entries_list,
            key=lambda kv: (str(kv[1].get("saved_at") or ""), str(kv[1].get("exam_name") or "")),
            reverse=True,
        )
        filtered_entries = []
        for ck, ent in sorted_entries:
            ex_n = str(ent.get("exam_name") or "직접 입력 / 미분류").strip() or "직접 입력 / 미분류"
            if sel_exam != "전체" and ex_n != sel_exam:
                continue
            an_dict = ent.get("analysis") if isinstance(ent.get("analysis"), dict) else {}
            an_title = str(an_dict.get("title") or "")
            an_theme = str(an_dict.get("core_theme") or "")
            q_lbl = str(ent.get("q_label") or "")
            if eng_q and not (
                eng_q in ck.lower()
                or eng_q in ex_n.lower()
                or eng_q in q_lbl.lower()
                or eng_q in an_title.lower()
                or eng_q in an_theme.lower()
            ):
                continue
            filtered_entries.append((ck, ent, an_dict))

        st.caption(f"검색 결과: {len(filtered_entries)}건 (최대 25건 표시)")
        for ck, ent, an_dict in filtered_entries[:25]:
            q_lbl = str(ent.get("q_label") or ent.get("exam_name") or "미분류")
            q_type = str(ent.get("question_type") or "독해")
            an_title = str(an_dict.get("title") or "제목 없음")
            theme_prev = " ".join(str(an_dict.get("core_theme") or "").split())[:120]
            saved_at = str(ent.get("saved_at") or "")
            sent_cnt = len(an_dict.get("sentence_analyses") or [])

            r_col1, r_col2 = st.columns([5, 1])
            with r_col1:
                pdf_badge = "📑 분석PDF" if ctx.storage.has_english_analysis_pdf(ck) else "📝 JSON만"
                st.markdown(
                    f"**[{q_lbl}]** {an_title} (`{q_type}` · {sent_cnt}문장 · {pdf_badge}) · 키 `{ck}`"
                )
                cap_parts = [p for p in (f"저장: {saved_at}" if saved_at else "", theme_prev) if p]
                if cap_parts:
                    st.caption(" · ".join(cap_parts))
            with r_col2:
                if st.button("🗑️ 삭제", key=f"del_eng_an_pool_{ck}", width="stretch"):
                    removed = ctx.storage.delete_english_analysis_entry(ck)
                    if removed > 0:
                        st.success(f"영어 분석지 캐시(`{ck}`)를 삭제했습니다.")
                    st.rerun()
