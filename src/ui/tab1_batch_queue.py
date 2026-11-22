from __future__ import annotations

import os
import re
from typing import Any

import streamlit as st

from src.auth import record_exam_upload_for_user, resolve_active_gemini_api_key
from src.models import infer_subject_category, normalize_exam_title_year
from src.ui.common import _ensure_token_quota
from src.ui.context import AppContext
from src.ui.tab1_db_explorer import _normalize_exam_identity
from src.ui.tab1_pairing import (
    _compute_pairing_signature,
    _compute_pairing_widget_hash,
    _detect_single_exam_cover,
    _invalidate_auto_pair_cache,
    _pdf_display_basename,
    _resolve_auto_answer_matches,
)
from src.ui.tab1_tagging import _crop_single_exam_pdf, _tag_and_save_single_exam


def render_tab1_batch_queue(
    ctx: AppContext,
    prob_files: list[Any],
    ans_files: list[Any],
) -> None:
    """다중 문제지 PDF 대기열 매칭 확인 및 순차 분석·저장 루프를 렌더링한다."""
    auto_matches, match_reasons, match_source, match_err = _resolve_auto_answer_matches(
        prob_files,
        ans_files,
        ctx.upload_gemini_key,
    )
    matched_cnt = sum(1 for m in auto_matches if m is not None)
    pair_sig = _compute_pairing_signature(prob_files, ans_files)
    pair_ver = int(st.session_state.get("tab1_ai_pair_ver", 0))
    pair_hash = _compute_pairing_widget_hash(pair_sig, pair_ver)

    if match_source == "ai":
        pair_badge = f"🤖 Gemini 3.5 Flash-Lite 해설지 자동 매칭 **{matched_cnt}/{len(prob_files)}개**"
    elif ans_files:
        pair_badge = f"해설지 자동 매칭 **{matched_cnt}/{len(prob_files)}개**"
    else:
        pair_badge = "해설지 미업로드"

    st.info(
        f"📦 **총 {len(prob_files)}개 문제지 PDF**가 대기열에 등록되었습니다 ({pair_badge}). "
        "시작 버튼을 누르면 **PDF 1개씩 순서대로** 표지 인식 → 문항 크롭 → Gemini 원리·용어 태깅 → DB/Drive 저장을 실행합니다."
    )
    if match_source == "fallback" and match_err and ans_files:
        st.warning(f"⚠️ AI 자동 매칭 일시 오류로 파일명 기준 매칭을 적용했습니다: `{match_err}`")

    ans_option_indices: list[int | None] = [None] + list(range(len(ans_files)))
    selected_ans_indices: list[int | None] = []
    with st.expander(f"📋 대기열 목록 및 해설지 1:1 매칭 확인 ({len(prob_files)}개)", expanded=len(prob_files) <= 12):
        if ans_files:
            hdr_l, hdr_r = st.columns([2.2, 1.0])
            with hdr_l:
                st.caption("🤖 AI가 파일명과 1페이지 내용을 대조해 해설지를 자동 지정했습니다. 필요 시 드롭다운에서 수동 변경할 수 있습니다.")
            with hdr_r:
                if st.button(
                    "🔄 AI(3.5 Flash-Lite) 다시 매칭",
                    key=f"batch_rematch_{pair_hash}",
                    width="stretch",
                ):
                    _invalidate_auto_pair_cache(pair_sig, ctx.upload_gemini_key)
                    st.rerun()

        for q_idx, pf in enumerate(prob_files):
            c_left, c_right = st.columns([1.2, 1.0])
            p_base = _pdf_display_basename(pf.name)
            with c_left:
                st.markdown(f"**{q_idx + 1}. `{p_base}`**")
                if p_base != pf.name:
                    st.caption(f"경로: `{pf.name}`")
            with c_right:
                default_ans = auto_matches[q_idx]
                default_opt_pos = (
                    ans_option_indices.index(default_ans)
                    if default_ans in ans_option_indices
                    else 0
                )
                chosen_ans_idx = st.selectbox(
                    f"해설지 #{q_idx + 1}",
                    options=ans_option_indices,
                    index=default_opt_pos,
                    format_func=lambda idx: (
                        "(해설지 없음)"
                        if idx is None
                        else _pdf_display_basename(ans_files[idx].name)
                    ),
                    key=f"batch_ans_map_{pair_hash}_{q_idx}_{pf.name}",
                    label_visibility="collapsed",
                )
                if match_reasons.get(q_idx):
                    st.caption(f"🤖 {match_reasons[q_idx]}")
                selected_ans_indices.append(chosen_ans_idx)

    b_opt1, b_opt2 = st.columns(2)
    with b_opt1:
        batch_vision_chk = st.checkbox(
            "스캔(이미지) PDF인 경우 Gemini Vision 좌표 탐지 보조 사용",
            value=True,
            key="tab1_batch_vision_chk",
        )
    with b_opt2:
        batch_skip_dup_chk = st.checkbox(
            "이미 DB에 등록된 회차명 자동 건너뛰기 (중복 방지)",
            value=True,
            key="tab1_batch_skip_dup_chk",
        )

    matched_queue_count = sum(1 for x in selected_ans_indices if x is not None)
    unmatched_queue_count = len(prob_files) - matched_queue_count
    if matched_queue_count == 0:
        st.error(
            "🚫 해설지가 매칭된 문제지가 없습니다. 해설지 PDF를 함께 업로드해야 태깅을 시작할 수 있습니다."
        )
    elif unmatched_queue_count > 0:
        st.warning(
            f"⚠️ 해설지가 매칭되지 않은 문제지 **{unmatched_queue_count}개**는 자동 건너뛰기되며, "
            f"해설지가 연결된 **{matched_queue_count}개**만 순차 태깅합니다."
        )

    batch_btn_label = (
        f"🚀 대기열 {len(prob_files)}개 PDF 순차 분석·저장 시작 (하나씩 처리)"
        if unmatched_queue_count == 0
        else f"🚀 해설지 매칭된 {matched_queue_count}/{len(prob_files)}개 PDF 순차 분석·저장 시작 (미매칭 {unmatched_queue_count}개 건너뜀)"
    )
    run_batch_btn = st.button(
        batch_btn_label,
        type="primary",
        width="stretch",
        key="tab1_run_batch_btn",
        disabled=(matched_queue_count == 0),
    )

    if run_batch_btn and matched_queue_count > 0:
        verified_key, verified_key_label = resolve_active_gemini_api_key(
            ctx.current_user, ctx.creds.gemini_api_key, force_server_key=True
        )
        if not verified_key:
            st.error(
                f"서버 업로드·태깅용 API 키가 설정되지 않았습니다 ({verified_key_label}). "
                "`.streamlit/secrets.toml`의 `GEMINI_API_KEY`를 확인해 주세요."
            )
        elif not _ensure_token_quota(ctx):
            pass
        else:
            db_batch_snapshot = ctx.storage.load_db_readonly()
            existing_exam_names = {
                re.sub(r"\s+", " ", r.exam_name.strip())
                for r in db_batch_snapshot
                if r.exam_name
            }
            existing_exam_ids = {
                _normalize_exam_identity(r.exam_name)
                for r in db_batch_snapshot
                if r.exam_name
            }
            total_n = len(prob_files)
            overall_prog = st.progress(0.0)
            overall_status = st.empty()
            item_prog = st.progress(0.0)
            item_status = st.empty()
            item_log = st.empty()
            batch_summary: list[dict[str, Any]] = []
            live_logs: list[str] = []

            def _append_batch_log(msg: str) -> None:
                live_logs.append(msg)
                item_log.code("\n".join(live_logs[-18:]), language="text")

            for idx, prob_f in enumerate(prob_files, start=1):
                if not _ensure_token_quota(ctx):
                    _append_batch_log("🚫 월간 토큰 한도 도달로 남은 대기열을 중단했습니다.")
                    break
                prob_base = _pdf_display_basename(prob_f.name)
                matched_ans_i = selected_ans_indices[idx - 1]
                ans_f = ans_files[matched_ans_i] if matched_ans_i is not None else None
                ans_base = _pdf_display_basename(ans_f.name) if ans_f is not None else ""

                if ans_f is None:
                    _append_batch_log(f"⏭️ [{idx}/{total_n}] {prob_base} — 해설지 미매칭 → 건너뜀")
                    batch_summary.append(
                        {
                            "file": prob_base,
                            "exam_name": os.path.splitext(prob_base)[0].replace("_", " ").strip(),
                            "status": "skipped",
                            "saved_count": 0,
                            "message": "해설지 미첨부(건너뜀 · 해설지 필수)",
                        }
                    )
                    overall_prog.progress(idx / total_n)
                    continue

                overall_status.info(
                    f"📦 **[{idx}/{total_n}]** `{prob_base}` 순차 처리 중... (해설지: `{ans_base}`)"
                )
                item_prog.progress(0.05)
                _append_batch_log(f"--- [{idx}/{total_n}] {prob_base} 시작 ---")

                try:
                    prob_bytes = prob_f.getvalue()
                    ans_bytes = ans_f.getvalue() if ans_f is not None else None

                    item_status.info("🔍 1단계: 표지 메타데이터 판독 중...")
                    parsed_local, _, _ = _detect_single_exam_cover(prob_bytes, prob_base, verified_key)
                    raw_title = (parsed_local.get("exam_title") or "").strip()
                    if not raw_title:
                        raw_title = os.path.splitext(prob_base)[0].replace("_", " ").strip()
                    norm_exam_name = normalize_exam_title_year(raw_title, filename=prob_base)
                    subj_cat = infer_subject_category(parsed_local.get("subject", ""), exam_title=norm_exam_name)
                    _append_batch_log(f"🏷️ 회차명: {norm_exam_name} ({subj_cat})")

                    norm_exam_id = _normalize_exam_identity(norm_exam_name)
                    if norm_exam_name in existing_exam_names or (norm_exam_id and norm_exam_id in existing_exam_ids):
                        _append_batch_log(f"🚫 이미 DB에 등록된 회차 → 업로드 차단: {norm_exam_name}")
                        batch_summary.append(
                            {
                                "file": prob_base,
                                "exam_name": norm_exam_name,
                                "status": "skipped",
                                "saved_count": 0,
                                "message": "이미 DB에 등록된 회차(중복 차단)",
                            }
                        )
                        overall_prog.progress(idx / total_n)
                        continue

                    item_status.info("✂️ 2단계: 2단 편집 분석 및 문항 크롭 중...")
                    item_prog.progress(0.20)
                    crops, _, _ = _crop_single_exam_pdf(
                        prob_bytes,
                        use_vision_fallback=batch_vision_chk,
                        api_key=verified_key,
                        subject_hint=f"{subj_cat} {norm_exam_name}",
                        status_cb=lambda m: item_status.info(m),
                        progress_cb=lambda v: item_prog.progress(0.20 + 0.20 * v),
                    )
                    if not crops:
                        _append_batch_log(f"❌ 문항 번호 탐지 실패 → 건너뜀: {prob_base}")
                        batch_summary.append(
                            {
                                "file": prob_base,
                                "exam_name": norm_exam_name,
                                "status": "failed",
                                "saved_count": 0,
                                "message": "문항 번호를 탐지하지 못했습니다.",
                            }
                        )
                        overall_prog.progress(idx / total_n)
                        continue

                    _append_batch_log(f"✂️ 크롭 완료: {len(crops)}문항")
                    res = _tag_and_save_single_exam(
                        storage=ctx.storage,
                        api_key=verified_key,
                        meta={
                            "exam_name": norm_exam_name,
                            "subject_category": subj_cat,
                            "prob_bytes": prob_bytes,
                            "prob_name": prob_base,
                            "ans_bytes": ans_bytes,
                            "ans_name": ans_base,
                        },
                        parsed_crops=crops,
                        target_q_nums=[c.q_num for c in crops],
                        is_retry=False,
                        push_status=lambda m: item_status.info(m),
                        append_log=_append_batch_log,
                        on_progress=lambda v: item_prog.progress(0.40 + 0.60 * v),
                        on_warning=lambda w: _append_batch_log(f"⚠️ {w}"),
                    )
                    newly_saved_items = res["newly_saved"]
                    if newly_saved_items:
                        existing_exam_names.add(norm_exam_name)
                        if norm_exam_id:
                            existing_exam_ids.add(norm_exam_id)
                        record_exam_upload_for_user(ctx.storage, ctx.current_user.username)
                    failed_qs = res["failed_q_nums"]
                    sync_err = res["sync_err"]
                    msg_parts = [f"{len(newly_saved_items)}문항 저장 ({res['elapsed_s']:.1f}초)"]
                    if failed_qs:
                        msg_parts.append(f"누락 {len(failed_qs)}문항")
                    if sync_err:
                        msg_parts.append(f"동기화 경고: {sync_err}")
                    batch_summary.append(
                        {
                            "file": prob_base,
                            "exam_name": norm_exam_name,
                            "status": "success" if newly_saved_items else "failed",
                            "saved_count": len(newly_saved_items),
                            "message": " · ".join(msg_parts),
                        }
                    )
                except Exception as exc:
                    _append_batch_log(f"❌ 처리 오류 ({prob_base}): {exc}")
                    batch_summary.append(
                        {
                            "file": prob_base,
                            "exam_name": prob_base,
                            "status": "failed",
                            "saved_count": 0,
                            "message": str(exc),
                        }
                    )

                overall_prog.progress(idx / total_n)

            overall_status.success(f"🏁 대기열 {total_n}개 PDF 순차 처리 완료!")
            item_status.empty()
            st.session_state["tab1_batch_summary"] = batch_summary

    saved_batch_summary = st.session_state.get("tab1_batch_summary") or []
    if saved_batch_summary:
        ok_cnt = sum(1 for r in saved_batch_summary if r["status"] == "success")
        skip_cnt = sum(1 for r in saved_batch_summary if r["status"] == "skipped")
        fail_cnt = sum(1 for r in saved_batch_summary if r["status"] == "failed")
        st.markdown(
            f"#### 📊 순차 처리 결과 요약 — ✅ 완료 **{ok_cnt}건** · ⏭️ 중복 건너뜀 **{skip_cnt}건** · ❌ 실패 **{fail_cnt}건**"
        )
        for row in saved_batch_summary:
            icon = "✅" if row["status"] == "success" else ("⏭️" if row["status"] == "skipped" else "❌")
            st.write(f"{icon} **{row['exam_name']}** (`{row['file']}`) — {row['message']}")
