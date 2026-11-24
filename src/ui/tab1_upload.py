from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any

import streamlit as st

from src.auth import (
    MAX_SHARED_UPLOAD_PDF_BYTES,
    can_admin_shared_db,
    can_contribute_shared_db,
    check_daily_shared_upload_quota,
    consume_daily_shared_upload_quota,
    record_exam_upload_for_user,
    resolve_active_gemini_api_key,
)
from src.config import EXAM_TAGGING_MODEL, MODEL_LABELS
from src.models import infer_subject_category, normalize_exam_title_year
from src.ui.common import (
    _ensure_token_quota,
    create_lazy_expander,
    is_container_open,
)
from src.ui.context import AppContext
from src.ui.tab1_db_explorer import (
    _normalize_exam_identity,
    _render_exam_coverage_board,
    _render_question_tag_details,
    render_tab1_db_explorer,
)

if TYPE_CHECKING:
    from src.pdf_parser import CroppedQuestion

def render_tab1(ctx: AppContext) -> None:
    st.subheader("1️⃣ 기출 PDF 수집 및 고해상도 자동 크롭 (250 DPI)")
    db_records = ctx.storage.load_db_readonly()
    _render_exam_coverage_board(ctx, db_records=db_records)

    if not can_contribute_shared_db(ctx.current_user):
        st.info(
            "🔒 **기출 모의고사 PDF 업로드 및 AI 자동 태깅은 PRO · ULTRA · ADMIN 전용 기능입니다.**\n\n"
            "무료(`FREE`) 등급에서는 아래 구축된 **공용 기출 DB 탐색 및 시험지·해설지 내려받기(0C 무료)**를 이용하실 수 있습니다."
        )
        render_tab1_db_explorer(ctx, db_records=db_records)
        return

    collect_mode = st.radio(
        "수집 방식 선택",
        options=["📤 수동 업로드 (기본)", "🔗 EBSi 링크 수집 (보조)"],
        horizontal=True,
    )

    if collect_mode == "🔗 EBSi 링크 수집 (보조)":
        with st.container(border=True):
            ebsi_url = st.text_input(
                "EBSi 기출 문제지 URL 입력",
                placeholder="https://www.ebsi.co.kr/...",
            )
            if st.button("🌐 EBSi 링크에서 공개 PDF 추출 시도", width="content"):
                with st.spinner("EBSi 페이지에서 공개 PDF 링크 확인 중..."):
                    from src.ebsi_fetcher import attempt_ebsi_pdf_fetch

                    ebsi_res = attempt_ebsi_pdf_fetch(ebsi_url)
                if ebsi_res.success and ebsi_res.problem_pdf_bytes:
                    st.session_state["ebsi_problem_bytes"] = ebsi_res.problem_pdf_bytes
                    st.session_state["ebsi_problem_name"] = ebsi_res.problem_filename
                    if ebsi_res.answer_pdf_bytes:
                        st.session_state["ebsi_answer_bytes"] = ebsi_res.answer_pdf_bytes
                        st.session_state["ebsi_answer_name"] = ebsi_res.answer_filename
                    st.success(ebsi_res.message)
                else:
                    st.warning(ebsi_res.message)
                    st.info("👉 아래 [수동 업로드] 영역에 내려받은 문제지·해설지 PDF를 직접 올려주세요.")

    with st.container(border=True):
        upload_mode_label = st.radio(
            "업로드 방식 선택",
            options=["📄 파일 선택 (1개 또는 여러 개)", "📁 폴더 통째로 업로드 (하위 폴더 포함)"],
            horizontal=True,
            key="tab1_upload_mode",
            help="여러 개의 PDF를 드래그/다중 선택하거나, 모의고사 폴더 자체를 선택해 한 번에 대기열에 넣을 수 있습니다. 분석·태깅·저장은 PDF 1개씩 순차 진행됩니다.",
        )
        is_dir_upload = "폴더" in upload_mode_label
        accept_mode: bool | str = "directory" if is_dir_upload else True
        unified_up_label = (
            "📁 모의고사 폴더 통째로 업로드 (문제지·해설지 구분 없이 한 번에 선택 — AI 자동 분류 및 표지 매칭)"
            if is_dir_upload
            else "📄 모의고사 PDF 한 번에 업로드 (문제지 + 해설지를 구분 없이 모두 선택 — AI 자동 분류 및 표지 매칭)"
        )

        raw_all_uploads = st.file_uploader(
            unified_up_label,
            type=["pdf"],
            accept_multiple_files=accept_mode,
            key="tab1_unified_pdf_dir" if is_dir_upload else "tab1_unified_pdf_files",
        )

        raw_all_files = sorted(
            [f for f in (raw_all_uploads or []) if f.name.lower().endswith(".pdf")],
            key=lambda f: f.name,
        )
        if raw_all_files and not can_admin_shared_db(ctx.current_user):
            oversized = [f.name for f in raw_all_files if int(getattr(f, "size", 0) or 0) > MAX_SHARED_UPLOAD_PDF_BYTES]
            if oversized:
                st.error(
                    f"🚫 파일당 최대 크기(50MB)를 초과한 PDF가 제외되었습니다: {', '.join(oversized[:5])}"
                )
                raw_all_files = [
                    f for f in raw_all_files if int(getattr(f, "size", 0) or 0) <= MAX_SHARED_UPLOAD_PDF_BYTES
                ]

        prob_files: list[Any] = []
        ans_files: list[Any] = []

        if len(raw_all_files) == 1:
            prob_files = [raw_all_files[0]]
        elif len(raw_all_files) >= 2:
            from src.ui.tab1_pairing import _resolve_mixed_pdf_split_with_ai

            split_prob_idxs, split_ans_idxs, split_source = _resolve_mixed_pdf_split_with_ai(
                raw_all_files, ctx.upload_gemini_key
            )
            prob_files = [raw_all_files[i] for i in split_prob_idxs]
            ans_files = [raw_all_files[i] for i in split_ans_idxs]

            # 만약 AI/파일명 분류에서 문제지가 0개이거나 해설지가 0개인 예외 상황 안내
            split_badge = "🤖 Gemini AI 표지 판별" if split_source == "ai" else "📁 파일명 판별"
            sc_info, sc_btn = st.columns([3.2, 1.0])
            with sc_info:
                st.caption(
                    f"{split_badge}: 총 {len(raw_all_files)}개 PDF → "
                    f"**문제지 {len(prob_files)}개** · **해설지 {len(ans_files)}개** 자동 분리 완료"
                )
            with sc_btn:
                if st.button("🔄 AI 표지 다시 분류", key="tab1_resplit_btn", width="stretch"):
                    raw_sig = (
                        tuple((f.name, int(getattr(f, "size", 0))) for f in raw_all_files),
                        bool(ctx.upload_gemini_key),
                    )
                    st.session_state.get("tab1_ai_split_cache", {}).pop(raw_sig, None)
                    st.rerun()

            if not prob_files and ans_files:
                st.warning("⚠️ 업로드된 파일이 모두 해설지로 분류되었습니다. 문제지 PDF도 함께 선택해 주세요.")

        is_batch_queue_mode = len(prob_files) >= 2

        if is_batch_queue_mode:
            from src.ui.tab1_batch_queue import render_tab1_batch_queue

            render_tab1_batch_queue(ctx, prob_files, ans_files)
        else:
            from src.ui.tab1_pairing import (
                _compute_pairing_signature,
                _compute_pairing_widget_hash,
                _detect_single_exam_cover,
                _invalidate_auto_pair_cache,
                _pdf_display_basename,
                _resolve_auto_answer_matches,
            )

            prob_file = prob_files[0] if prob_files else None
            ans_file = None
            if ans_files:
                if prob_file is not None and len(ans_files) > 1:
                    s_matches, s_reasons, s_source, s_err = _resolve_auto_answer_matches(
                        [prob_file],
                        ans_files,
                        ctx.upload_gemini_key,
                    )
                    s_pair_sig = _compute_pairing_signature([prob_file], ans_files)
                    s_pair_ver = int(st.session_state.get("tab1_ai_pair_ver", 0))
                    s_pair_hash = _compute_pairing_widget_hash(s_pair_sig, s_pair_ver)
                    s_options: list[int | None] = [None] + list(range(len(ans_files)))
                    s_default = s_matches[0]
                    s_pos = s_options.index(s_default) if s_default in s_options else 0

                    sc_l, sc_r = st.columns([2.2, 1.0])
                    with sc_l:
                        chosen_single_idx = st.selectbox(
                            "🤖 AI 자동 매칭 해설지 (3.5 Flash-Lite · 필요 시 변경 가능)",
                            options=s_options,
                            index=s_pos,
                            format_func=lambda idx: (
                                "(해설지 없음)"
                                if idx is None
                                else _pdf_display_basename(ans_files[idx].name)
                            ),
                            key=f"single_ans_map_{s_pair_hash}_{prob_file.name}",
                        )
                        if s_reasons.get(0):
                            st.caption(f"🤖 {s_reasons[0]}")
                        elif s_source == "fallback" and s_err:
                            st.caption(f"⚠️ AI 자동 매칭 일시 오류로 파일명 기준 선택됨: {s_err}")
                    with sc_r:
                        st.write("")
                        if st.button(
                            "🔄 AI 다시 매칭",
                            key=f"single_rematch_{s_pair_hash}",
                            width="stretch",
                        ):
                            _invalidate_auto_pair_cache(s_pair_sig, ctx.upload_gemini_key)
                            st.rerun()

                    ans_file = ans_files[chosen_single_idx] if chosen_single_idx is not None else None
                else:
                    ans_file = ans_files[0]

            active_prob_bytes: bytes | None = (
                prob_file.getvalue() if prob_file is not None else st.session_state.get("ebsi_problem_bytes")
            )
            active_prob_name: str = (
                _pdf_display_basename(prob_file.name)
                if prob_file is not None
                else st.session_state.get("ebsi_problem_name", "problem.pdf")
            )
            active_ans_bytes: bytes | None = (
                ans_file.getvalue() if ans_file is not None else st.session_state.get("ebsi_answer_bytes")
            )
            active_ans_name: str = (
                _pdf_display_basename(ans_file.name)
                if ans_file is not None
                else st.session_state.get("ebsi_answer_name", "answer.pdf")
            )

            if st.session_state.get("ebsi_problem_bytes") and prob_file is None:
                st.caption(f"🔗 EBSi에서 가져온 문제지 대기 중: `{active_prob_name}`")
            if st.session_state.get("ebsi_answer_bytes") and ans_file is None:
                st.caption(f"🔗 EBSi에서 가져온 해설지 대기 중: `{active_ans_name}`")

            # 업로드 즉시 fitz + re 기반 0.05초 로컬 파싱 (API 호출 없음, file.name + file.size 변경 감지 시 강제 덮어쓰기)
            file_sig = (
                f"{active_prob_name}:{prob_file.size}"
                if prob_file is not None
                else (f"{active_prob_name}:{len(active_prob_bytes)}" if active_prob_bytes else "")
            )

            if not file_sig:
                st.session_state["tab1_last_loaded_sig"] = ""
                st.session_state["tab1_meta_status"] = ""
            elif file_sig != st.session_state.get("tab1_last_loaded_sig"):
                parsed_local = {
                    "year": "",
                    "month": "",
                    "grade": "",
                    "subject": "",
                    "exam_title": "",
                }
                has_p1_text = False
                ai_verified = False
                if active_prob_bytes:
                    with st.spinner("🔍 1차 텍스트 파싱 + 2차 AI 비전 표지 교차 검증 중..."):
                        parsed_local, has_p1_text, ai_verified = _detect_single_exam_cover(
                            active_prob_bytes,
                            active_prob_name,
                            ctx.upload_gemini_key,
                        )

                st.session_state["meta_year"] = parsed_local.get("year", "")
                st.session_state["meta_month"] = parsed_local.get("month", "")
                st.session_state["meta_grade"] = parsed_local.get("grade", "")
                st.session_state["meta_subject"] = parsed_local.get("subject", "")
                st.session_state["meta_title"] = parsed_local.get("exam_title", "")
                st.session_state["tab1_subject_cat"] = infer_subject_category(
                    parsed_local.get("subject", ""),
                    exam_title=parsed_local.get("exam_title", "") or active_prob_name,
                )
                st.session_state["tab1_last_loaded_sig"] = file_sig

                if ai_verified and parsed_local.get("exam_title"):
                    st.session_state["tab1_meta_status"] = "verified_ai"
                elif not has_p1_text and not parsed_local.get("exam_title"):
                    st.session_state["tab1_meta_status"] = "scanned"
                elif parsed_local.get("exam_title"):
                    st.session_state["tab1_meta_status"] = "parsed"
                else:
                    st.session_state["tab1_meta_status"] = "partial_empty"

            meta_status = st.session_state.get("tab1_meta_status", "")
            if file_sig:
                if meta_status == "verified_ai":
                    st.success(
                        f"🎯 1차 텍스트 + 2차 AI 비전 표지 교차 검증 완료: **{st.session_state.get('meta_title', '')}** (필요 시 아래에서 수정 가능)"
                    )
                elif meta_status == "parsed":
                    st.success(
                        f"⚡ 1페이지 즉시 자동 인식 완료: **{st.session_state.get('meta_title', '')}** (필요 시 아래에서 바로 수정하세요)"
                    )
                elif meta_status == "scanned":
                    st.info("🖼️ 텍스트가 없는 스캔 PDF입니다. 아래 [시행 년도 / 시행 월 / 학년 / 과목 / 회차명]을 직접 입력해 주세요.")
                elif meta_status == "partial_empty":
                    st.info("📄 1페이지 헤더 정보가 일부만 확인되어 빈칸으로 두었습니다. 아래 입력칸을 직접 채워 주세요.")

            # 분석 성공 여부와 관계없이 5개 입력칸(+대분류)은 항상 즉시 렌더링
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                exam_year_input = st.text_input("시행 년도", key="meta_year", placeholder="예: 2026")
            with col_m2:
                exam_month_input = st.text_input("시행 월", key="meta_month", placeholder="예: 9월")
            with col_m3:
                exam_grade_input = st.text_input("학년", key="meta_grade", placeholder="예: 고1")
            with col_m4:
                exam_subject_input = st.text_input(
                    "과목",
                    key="meta_subject",
                    placeholder="예: 통합사회",
                )

            col_meta1, col_meta2 = st.columns([1, 2])
            with col_meta1:
                subject_cat = st.selectbox(
                    "과목 (대분류)",
                    options=["과학", "수학", "사회", "한국사", "국어", "영어"],
                    key="tab1_subject_cat",
                )
            with col_meta2:
                exam_name_input = st.text_input(
                    "회차명 (자동 조합)",
                    key="meta_title",
                    placeholder="예: 2026년 9월 고1 통합사회",
                )

            db_snapshot = db_records
            existing_exam_names = {
                re.sub(r"\s+", " ", r.exam_name.strip())
                for r in db_snapshot
                if r.exam_name
            }
            existing_exam_ids = {
                _normalize_exam_identity(r.exam_name)
                for r in db_snapshot
                if r.exam_name
            }
            norm_input_exam = normalize_exam_title_year(exam_name_input.strip(), filename=active_prob_name)
            norm_input_id = _normalize_exam_identity(norm_input_exam)
            is_duplicate_exam = bool(
                norm_input_exam
                and (norm_input_exam in existing_exam_names or (norm_input_id and norm_input_id in existing_exam_ids))
            )
            if is_duplicate_exam:
                st.error(
                    f"🚫 **'{norm_input_exam}'** 모의고사는 이미 DB에 등록되어 있습니다. "
                    "중복 수집 방지를 위해 크롭 버튼이 비활성화되었습니다."
                )

            is_missing_answer_single = not bool(active_ans_bytes)
            if active_prob_bytes and is_missing_answer_single:
                st.warning(
                    "🚫 **해설지 PDF가 필요합니다.** 정답 정확도를 위해 해설지 PDF를 함께 업로드(또는 매칭)해야 1단계 크롭과 태깅을 진행할 수 있습니다."
                )

            parse_col1, parse_col2 = st.columns([1, 1])
            with parse_col1:
                run_crop_btn = st.button(
                    "✂️ 1단계: 2단 편집 분석 및 고해상도 문항 크롭",
                    type="primary",
                    width="stretch",
                    disabled=is_duplicate_exam or is_missing_answer_single,
                )
            with parse_col2:
                vision_fallback_chk = st.checkbox(
                    "스캔(이미지) PDF인 경우 Gemini Vision 좌표 탐지 보조 사용",
                    value=True,
                )

            if run_crop_btn and not is_duplicate_exam and not is_missing_answer_single:
                if not active_prob_bytes:
                    st.error("문제지 PDF를 업로드하거나 EBSi에서 먼저 추출해 주세요.")
                elif not active_ans_bytes:
                    st.error("해설지 PDF가 필요합니다. 해설지 PDF를 함께 업로드해 주세요.")
                elif not exam_name_input.strip():
                    st.error("시험 회차명을 입력해 주세요.")
                else:
                    vision_status_ph = st.empty()
                    vision_prog_ph = st.empty()
                    with st.spinner("PyMuPDF 2단 편집(좌→우 칼럼) 분석 및 220 DPI PNG 크롭 중..."):
                        from src.ui.tab1_tagging import _crop_single_exam_pdf

                        crops, total_p, _ = _crop_single_exam_pdf(
                            active_prob_bytes,
                            use_vision_fallback=bool(vision_fallback_chk),
                            api_key=ctx.upload_gemini_key,
                            subject_hint=f"{subject_cat} {exam_subject_input} {exam_name_input}",
                            status_cb=lambda m: vision_status_ph.info(m),
                            progress_cb=lambda v: vision_prog_ph.progress(v),
                        )

                    if not crops:
                        st.error("문항 번호를 탐지하지 못했습니다. PDF 형식을 확인하거나 Gemini API 키 설정 후 Vision 보조를 활성화해 주세요.")
                    else:
                        st.session_state["parsed_crops"] = crops
                        st.session_state["parsed_exam_meta"] = {
                            "exam_name": norm_input_exam or exam_name_input.strip(),
                            "subject_category": subject_cat,
                            "prob_bytes": active_prob_bytes,
                            "prob_name": active_prob_name,
                            "ans_bytes": active_ans_bytes,
                            "ans_name": active_ans_name,
                        }
                        st.session_state["stage2_exam_name"] = ""
                        st.session_state["stage2_saved_q_nums"] = []
                        st.session_state["stage2_failed_q_nums"] = []
                        st.session_state["stage2_failed_errors"] = []
                        st.session_state["stage2_sync_err"] = ""
                        st.session_state["trigger_retry_failed"] = False
                        found_nums = sorted({c.q_num for c in crops})
                        min_q, max_q = found_nums[0], found_nums[-1]
                        subj_txt = (exam_subject_input or "").strip()
                        if any(k in subj_txt for k in ("국어", "영어")):
                            expected_max = 45
                        elif any(k in subj_txt for k in ("수학", "미적분", "기하", "확률")):
                            expected_max = 30
                        elif any(k in subj_txt for k in ("사회", "과학", "물리", "화학", "생명", "지구", "한국사", "윤리", "지리", "경제", "정치", "역사")):
                            expected_max = 20
                        else:
                            expected_max = max_q

                        range_end = max(max_q, expected_max) if total_p >= 3 else max_q
                        missing_in_found = [n for n in range(1, max_q + 1) if n not in set(found_nums)]
                        missing_tail = [n for n in range(max_q + 1, range_end + 1)]

                        st.success(f"✅ 총 {len(crops)}개 문항 크롭 완료 ({min_q}번 ~ {max_q}번)")
                        if missing_in_found or missing_tail:
                            missing_all = missing_in_found + missing_tail
                            st.warning(
                                f"⚠️ 누락된 문항 번호 ({len(missing_all)}개, 1~{range_end}번 기준): "
                                + ", ".join(f"{n}번" for n in missing_all)
                            )

    parsed_crops: list[CroppedQuestion] = st.session_state.get("parsed_crops") or []
    if (not is_batch_queue_mode) and parsed_crops:
        meta = st.session_state["parsed_exam_meta"]
        st.markdown(f"### 2️⃣ 크롭 문항 확인 & 2022 원리·용어 분해 태깅 (`{meta['exam_name']}`)")

        crop_prev_exp = create_lazy_expander(f"🖼️ 크롭된 문항 미리보기 ({len(parsed_crops)}문항)", expanded=False, key="tab1_crop_preview_exp")
        with crop_prev_exp:
            if is_container_open(crop_prev_exp):
                seen_p_groups: dict[str, CroppedQuestion] = {}
                for cq in parsed_crops:
                    p_b = getattr(cq, "passage_png_bytes", b"") or b""
                    p_g = getattr(cq, "passage_group_id", "") or ""
                    if p_b and p_g and p_g not in seen_p_groups:
                        seen_p_groups[p_g] = cq

                if seen_p_groups:
                    st.markdown(f"**📖 분리된 공통 지문 이미지 ({len(seen_p_groups)}개 세트)**")
                    p_list = list(seen_p_groups.items())
                    for idx in range(0, len(p_list), 3):
                        p_cols = st.columns(3)
                        for col_i, (p_gid, p_cq) in enumerate(p_list[idx : idx + 3]):
                            with p_cols[col_i]:
                                st.image(
                                    p_cq.passage_png_bytes,
                                    caption=f"[{p_gid}번] 공통 지문 ({p_cq.passage_width}x{p_cq.passage_height}px)",
                                    width="stretch",
                                )
                    st.markdown("---")
                    st.markdown(f"**📝 문항별 크롭 이미지 (문제만 · {len(parsed_crops)}문항)**")

                cols_per_row = 3
                for idx in range(0, len(parsed_crops), cols_per_row):
                    row_cols = st.columns(cols_per_row)
                    for col_i, cq in enumerate(parsed_crops[idx : idx + cols_per_row]):
                        with row_cols[col_i]:
                            st.image(
                                cq.png_bytes,
                                caption=f"{cq.q_num}번 ({cq.page_num}쪽 · {cq.width}x{cq.height}px)",
                                width="stretch",
                            )

        q_nums_all = [c.q_num for c in parsed_crops]
        selected_q_nums = st.multiselect(
            "태깅 및 DB 저장할 문항 선택 (기본: 전체)",
            options=q_nums_all,
            default=q_nums_all,
        )

        def _run_stage2_extraction(target_q_nums: list[int], is_retry: bool = False) -> None:
            if not meta.get("ans_bytes"):
                st.error(
                    "🚫 해설지 PDF가 첨부되지 않아 태깅을 실행할 수 없습니다. "
                    "상단에서 해설지 PDF를 업로드한 뒤 1단계 크롭부터 다시 진행해 주세요."
                )
                return
            # 탭 1 업로드·태깅 전체는 CUSTOM 포함 모든 등급 서버 공용 API 키 사용
            verified_key, verified_key_label = resolve_active_gemini_api_key(
                ctx.current_user, ctx.creds.gemini_api_key, force_server_key=True
            )
            if not verified_key:
                st.error(
                    f"업로드·태깅용 API 키가 설정되지 않았습니다 ({verified_key_label}). "
                    "`.streamlit/secrets.toml`의 `GEMINI_API_KEY`를 확인해 주세요."
                )
                return
            if not _ensure_token_quota(ctx):
                return
            if not can_admin_shared_db(ctx.current_user):
                if not is_retry:
                    norm_new_id = _normalize_exam_identity(meta["exam_name"])
                    dup_exists = any(
                        r.exam_name == meta["exam_name"]
                        or (norm_new_id and _normalize_exam_identity(r.exam_name) == norm_new_id)
                        for r in ctx.storage.load_db_readonly()
                    )
                    if dup_exists:
                        st.error(
                            f"🚫 이미 공용 기출 DB에 등록된 회차입니다 (`{meta['exam_name']}`). "
                            "기존 회차 덮어쓰기 및 재태깅은 최고 관리자(ADMIN) 전용입니다."
                        )
                        return
                    ok_up, _, up_msg = check_daily_shared_upload_quota(ctx.storage, ctx.current_user.username, 1)
                    if not ok_up:
                        st.error(up_msg)
                        return
            if not target_q_nums:
                st.warning("태깅할 문항을 최소 1개 이상 선택해 주세요.")
                return

            tagging_model_label = MODEL_LABELS.get(EXAM_TAGGING_MODEL, EXAM_TAGGING_MODEL)
            sync_notice = st.info(
                "💾 전체 PDF를 한 번에 태깅한 뒤 서버 영구 디스크(`./exam_db`)에 일괄 저장합니다."
            )
            status_placeholder = st.empty()
            log_placeholder = st.empty()
            live_logs: list[str] = []

            def _push_status(msg: str) -> None:
                status_placeholder.info(msg)

            def _append_log(line: str) -> None:
                live_logs.append(line)
                log_placeholder.code("\n".join(live_logs[-15:]), language="text")

            mode_label = "누락/실패 문항 재분석" if is_retry else "전체 PDF 일괄 태깅"
            _push_status(f"🔑 API 키 검증 완료 ({verified_key_label}) · {mode_label} 준비 중 (모델: {tagging_model_label})...")
            prog_bar = st.progress(0.05)

            from src.ui.tab1_tagging import _tag_and_save_single_exam

            res = _tag_and_save_single_exam(
                storage=ctx.storage,
                api_key=verified_key,
                meta=meta,
                parsed_crops=parsed_crops,
                target_q_nums=target_q_nums,
                is_retry=is_retry,
                push_status=_push_status,
                append_log=_append_log,
                on_progress=prog_bar.progress,
                on_warning=st.warning,
            )
            newly_saved = res["newly_saved"]
            failed_q_nums = res["failed_q_nums"]
            errors = res["errors"]
            sync_err = res["sync_err"]

            sync_notice.empty()
            status_placeholder.empty()

            if not is_retry and newly_saved:
                record_exam_upload_for_user(ctx.storage, ctx.current_user.username)
                if not can_admin_shared_db(ctx.current_user):
                    consume_daily_shared_upload_quota(ctx.storage, ctx.current_user.username, 1)

            prev_saved = (
                st.session_state.get("stage2_saved_q_nums", [])
                if is_retry and st.session_state.get("stage2_exam_name") == meta["exam_name"]
                else []
            )
            merged_saved = sorted(set(prev_saved) | {r.q_num for r in newly_saved})
            st.session_state["stage2_exam_name"] = meta["exam_name"]
            st.session_state["stage2_saved_q_nums"] = merged_saved
            st.session_state["stage2_failed_q_nums"] = sorted(set(failed_q_nums))
            st.session_state["stage2_failed_errors"] = errors
            st.session_state["stage2_sync_err"] = sync_err or ""

        if st.button("🧠 2단계: Gemini 원리·용어 분리 태깅 및 영구 저장", type="primary"):
            _run_stage2_extraction(selected_q_nums, is_retry=False)
        elif st.session_state.pop("trigger_retry_failed", False):
            pending_retry = [
                n for n in st.session_state.get("stage2_failed_q_nums", []) if isinstance(n, int)
            ]
            if pending_retry and st.session_state.get("stage2_exam_name") == meta["exam_name"]:
                _run_stage2_extraction(pending_retry, is_retry=True)

        if st.session_state.get("stage2_exam_name") == meta["exam_name"]:
            saved_q_nums = st.session_state.get("stage2_saved_q_nums") or []
            failed_q_nums_state = st.session_state.get("stage2_failed_q_nums") or []
            failed_errors_state = st.session_state.get("stage2_failed_errors") or []
            sync_err_state = st.session_state.get("stage2_sync_err") or ""

            if saved_q_nums:
                st.success(
                    f"🎉 `{meta['exam_name']}` 총 {len(saved_q_nums)}개 문항 태깅 및 영구 저장 완료! (로컬 ./exam_db 저장)"
                )
            if sync_err_state:
                st.warning(sync_err_state)
            for err in failed_errors_state:
                st.error(err)

            if failed_q_nums_state:
                preview_nums = ", ".join(f"{n}번" for n in failed_q_nums_state[:8])
                suffix = " ..." if len(failed_q_nums_state) > 8 else ""
                st.button(
                    f"🔄 실패한 {len(failed_q_nums_state)}개 문항만 다시 분석하기 ({preview_nums}{suffix})",
                    type="primary",
                    key="retry_failed_questions_btn",
                    on_click=lambda: st.session_state.update({"trigger_retry_failed": True}),
                )

            if saved_q_nums:
                saved_set = set(saved_q_nums)
                saved_records = sorted(
                    [
                        r
                        for r in ctx.storage.load_db_readonly()
                        if r.exam_name == meta["exam_name"] and r.q_num in saved_set
                    ],
                    key=lambda r: r.q_num,
                )
                for rec in saved_records:
                    review_badge = "⚠️ 검토 필요 · " if getattr(rec, "needs_review", False) else "✅ "
                    if rec.subject_category == "영어":
                        sec_lbl = getattr(rec, "english_section", "") or ("듣기" if rec.q_num <= 17 else "독해")
                        type_lbl = getattr(rec, "english_question_type", "") or rec.unit_name
                        exp_hdr = f"{review_badge}{rec.q_num}번 · [영어 · {sec_lbl}] {type_lbl} (정답: {rec.answer})"
                    else:
                        exp_hdr = (
                            f"{review_badge}{rec.q_num}번 · [{rec.curriculum_subject}] {rec.unit_name} "
                            f"(고{rec.min_grade} · 정답: {rec.answer})"
                        )
                    rec_exp = create_lazy_expander(exp_hdr, expanded=False, key=f"tab1_stage2_saved_exp_{rec.id}")
                    with rec_exp:
                        if is_container_open(rec_exp):
                            c_img, c_info = st.columns([1, 1.3])
                            with c_img:
                                b = ctx.storage.get_crop_bytes(
                                    rec.image_filename,
                                    subject_category=rec.subject_category,
                                    exam_name=rec.exam_name,
                                )
                                if b:
                                    st.image(b, width="stretch")
                            with c_info:
                                _render_question_tag_details(
                                    rec,
                                    storage=ctx.storage,
                                    key_prefix="stage2_saved",
                                    allow_edit=can_admin_shared_db(ctx.current_user),
                                    allow_delete=False,
                                )


    render_tab1_db_explorer(ctx, db_records=db_records)
