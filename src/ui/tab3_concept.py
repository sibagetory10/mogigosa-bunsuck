from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any

import streamlit as st

from src.auth import TIER_LABELS, can_access_shared_concept_db, resolve_model_for_user
from src.config import CONCEPT_DB_BUILD_MODEL, EXAM_TAGGING_MODEL, MODEL_LABELS
from src.models import ConceptBookRecord, ConceptUnit
from src.ui.common import (
    _ensure_token_quota,
    _usage_scope,
    create_lazy_expander,
    is_container_open,
    sanitize_id_part,
)
from src.ui.context import AppContext
from src.ui_styles import render_chips


def _render_ncic_curriculum_manager(key_prefix: str = "ncic") -> None:
    """2022 개정 교육과정(NCIC 고시 별책 5·7·8·9·14 PDF) 기준표 현황 조회 및 성취기준 누적 등록 UI."""
    ncic_exp = create_lazy_expander(
        "🧭 2022 개정 교육과정 공식 기준표(NCIC 고시 PDF) 관리 · 성취기준 보강",
        expanded=False,
        key=f"{key_prefix}_ncic_mgr_exp",
    )
    with ncic_exp:
        if not is_container_open(ncic_exp):
            return
        from src.curriculum_2022 import (
            clear_ncic_standards_for_category,
            fetch_and_sync_moe_2022_curriculum,
            get_catalog_summary_stats,
            merge_ncic_standards_to_catalog,
            parse_ncic_curriculum_pdf,
        )

        stats = get_catalog_summary_stats()
        st.caption(
            "기본적으로 2022 교육부 고시(제2022-33호) 공식 과목 편제표가 내장되어 있으며, "
            "국가교육과정정보센터(ncic.re.kr)의 [별책 5(국어) / 별책 7(사회·한국사) / 별책 8(수학) / 별책 9(과학) / 별책 14(영어)] PDF를 올리면 "
            "공식 성취기준 코드가 기준표에 누적되어 태깅 할루시네이션 검증이 한층 엄격해집니다."
        )
        r1 = st.columns(3)
        r1[0].metric("국어 기준표", f"{stats['국어']['subjects']}과목", f"성취기준 {stats['국어']['standards']}개")
        r1[1].metric("영어 기준표", f"{stats['영어']['subjects']}과목", f"성취기준 {stats['영어']['standards']}개")
        r1[2].metric("수학 기준표", f"{stats['수학']['subjects']}과목", f"성취기준 {stats['수학']['standards']}개")
        r2 = st.columns(3)
        r2[0].metric("사회 기준표", f"{stats['사회']['subjects']}과목", f"성취기준 {stats['사회']['standards']}개")
        r2[1].metric("과학 기준표", f"{stats['과학']['subjects']}과목", f"성취기준 {stats['과학']['standards']}개")
        r2[2].metric("한국사 기준표", f"{stats['한국사']['subjects']}과목", f"성취기준 {stats['한국사']['standards']}개")

        auto_c1, auto_c2 = st.columns([2.2, 1.4])
        with auto_c1:
            auto_cats = st.multiselect(
                "웹에서 자동 불러올 교과 (교육부 고시 제2022-33호 공식 원문)",
                options=["국어", "영어", "수학", "사회", "과학", "한국사"],
                default=["국어", "영어", "수학", "사회", "과학", "한국사"],
                key=f"{key_prefix}_moe_auto_cats",
            )
        with auto_c2:
            st.markdown("<div style='height: 1.65rem;'></div>", unsafe_allow_html=True)
            if st.button(
                "🌐 교육부 공식 원문 웹에서 자동 불러오기",
                key=f"{key_prefix}_moe_auto_fetch_btn",
                type="primary",
                disabled=not auto_cats,
                use_container_width=True,
            ):
                with st.spinner("교육부(moe.go.kr) 고시 제2022-33호 원문 불러오는 중 (지연 시 내장 공식 원문 자동 적용)..."):
                    try:
                        auto_res = fetch_and_sync_moe_2022_curriculum(target_categories=auto_cats)
                        src_note = (
                            " (웹 지연/차단으로 내장 교육부 공식 원문 시드 적용)"
                            if auto_res.get("source") == "seed_fallback"
                            else ""
                        )
                        st.success(
                            f"✅ 웹 자동 불러오기 완료!{src_note} 총 {auto_res['extracted_total']}개 추출 "
                            f"(신규 {auto_res['added']}개 추가 · 기존 {auto_res['updated']}개 갱신)"
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"❌ 교육부 공식 원문 불러오기 실패: {exc}")

        nc1, nc2 = st.columns([2, 1])
        with nc1:
            ncic_pdf = st.file_uploader(
                "NCIC 2022 교육과정 고시 PDF 업로드 (별책 5 국어 / 7 사회·한국사 / 8 수학 / 9 과학 / 14 영어)",
                type=["pdf"],
                key=f"{key_prefix}_ncic_pdf",
            )
        with nc2:
            ncic_cat_hint = st.selectbox(
                "기본 교과 대분류",
                options=["국어", "영어", "수학", "사회", "과학", "한국사"],
                key=f"{key_prefix}_ncic_cat",
            )
        if ncic_pdf is not None and st.button("📥 NCIC 고시 PDF 성취기준 추출·반영", key=f"{key_prefix}_ncic_apply"):
            with st.spinner("NCIC 고시 원문 PDF에서 2022 성취기준 코드 추출 중..."):
                extracted = parse_ncic_curriculum_pdf(
                    ncic_pdf.getvalue(),
                    subject_category_hint=ncic_cat_hint,
                )
            if not extracted:
                st.warning("PDF에서 `[10...]` / `[12...]` 형식의 2022 성취기준 코드를 찾지 못했습니다. 텍스트 레이어가 있는 공식 NCIC 원문 PDF인지 확인해 주세요.")
            else:
                res = merge_ncic_standards_to_catalog(extracted)
                st.success(
                    f"✅ 성취기준 {len(extracted)}개 추출 완료! (신규 {res['added']}개 추가 · 기존 {res['updated']}개 갱신)"
                )
                st.rerun()

        st.divider()
        st.caption("잘못된 PDF를 올렸거나 특정 교과 성취기준을 새로 다시 올리고 싶을 때 아래에서 해당 교과의 성취기준 목록만 비울 수 있습니다. (기본 과목 편제표는 유지됩니다)")
        rc1, rc2, rc3 = st.columns([1.2, 1.2, 1.4])
        with rc1:
            reset_cat = st.selectbox(
                "초기화할 교과",
                options=["국어", "영어", "수학", "사회", "과학", "한국사"],
                format_func=lambda c: f"{c} ({stats[c]['standards']}개)",
                key=f"{key_prefix}_ncic_reset_cat",
            )
        with rc2:
            confirm_reset = st.checkbox(
                "초기화 확인",
                value=False,
                key=f"{key_prefix}_ncic_reset_confirm",
            )
        with rc3:
            reset_disabled = (stats[reset_cat]["standards"] == 0) or (not confirm_reset)
            if st.button(
                f"🗑️ {reset_cat} 성취기준 초기화",
                key=f"{key_prefix}_ncic_reset_btn",
                disabled=reset_disabled,
                use_container_width=True,
            ):
                removed = clear_ncic_standards_for_category(reset_cat)
                st.success(f"🗑️ {reset_cat} 성취기준 {removed}개를 초기화했습니다.")
                st.rerun()



def _render_concept_book_builder_section(ctx: AppContext, key_prefix: str = "tab3") -> None:
    """관리자(ADMIN) 및 Pro(VIP) 등급용 개념서·교과서 PDF 단원별 공용 DB 구축 & 관리 UI."""
    st.subheader("📚 개념서·교과서 PDF 단원별 공용 DB 구축 (관리자 · Pro 전용)")
    st.caption(
        "지정한 개념 페이지 구간만 PyMuPDF로 추출해 목차·정답지·문제 풀이를 제외하고, "
        "사이드바에서 선택한 Gemini 모델로 단원별 핵심 개념과 허용 용어·단위를 구조화하여 "
        "로컬(`./concept_db`) 및 Google Drive 공용 폴더에 자동 동기화합니다."
    )

    if not can_access_shared_concept_db(ctx.current_user):
        st.warning(
            "🔒 **공용 개념서·교과서 DB 구축 및 공유 기능은 Pro(VIP) · ADMIN 등급 전용입니다.**\n\n"
            f"현재 로그인 등급: **{TIER_LABELS.get(ctx.current_user.tier, ctx.current_user.tier)} ({ctx.current_user.tier})** — "
            "관리자에게 Pro(VIP) 승급을 요청하시거나, [🎯 탭 2]의 **개별 개념서 PDF 직접 업로드** 기능을 이용해 주세요."
        )
        return

    with st.container(border=True):
        st.markdown("#### ➕ 새 개념서·교과서 PDF 업로드 및 개념 구간 추출")
        cb_pdf_file = st.file_uploader(
            "교재 / 개념서 PDF 파일 업로드 (올리는 즉시 표지 자동 인식)",
            type=["pdf"],
            key=f"{key_prefix}_pdf_uploader",
        )
        cb_pdf_bytes: bytes | None = cb_pdf_file.getvalue() if cb_pdf_file is not None else None
        if cb_pdf_bytes is not None:
            from src.gemini import (
                detect_concept_pages_with_ai,
                extract_concept_book_cover_metadata,
                extract_concept_book_units,
            )
            from src.pdf_parser import (
                extract_concept_pages_by_ranges,
                extract_first_page_for_metadata,
                extract_pdf_page_catalog,
                get_pdf_page_count,
                parse_page_ranges_spec,
                render_pdf_page_png,
            )

        cb_sig = f"{cb_pdf_file.name}:{cb_pdf_file.size}" if cb_pdf_file is not None else ""

        if not cb_sig:
            st.session_state[f"{key_prefix}_last_cb_sig"] = ""
            st.session_state[f"{key_prefix}_cb_meta_status"] = ""
        elif cb_sig != st.session_state.get(f"{key_prefix}_last_cb_sig"):
            auto_title = cb_pdf_file.name.rsplit(".", 1)[0].replace("_", " ").strip()
            auto_subj_grade = ""
            if cb_pdf_bytes and ctx.active_gemini_key and _ensure_token_quota(ctx, show_error=False):
                try:
                    with st.spinner("🔍 개념서 표지(교재명 · 과목 및 학년) 자동 인식 중..."):
                        p1_txt, p1_png = extract_first_page_for_metadata(cb_pdf_bytes, dpi=140)
                        if p1_png is None:
                            p1_png = render_pdf_page_png(cb_pdf_bytes, 1, dpi=140)
                        with _usage_scope(ctx):
                            cover_meta = extract_concept_book_cover_metadata(
                                api_key=ctx.active_gemini_key,
                                model_name=EXAM_TAGGING_MODEL,
                                first_page_text=p1_txt,
                                first_page_png_bytes=p1_png,
                                filename=cb_pdf_file.name,
                            )
                        if cover_meta.book_title:
                            auto_title = cover_meta.book_title
                        if cover_meta.subject_and_grade:
                            auto_subj_grade = cover_meta.subject_and_grade
                except Exception:
                    pass
            st.session_state[f"{key_prefix}_book_title"] = auto_title
            st.session_state[f"{key_prefix}_subject_grade"] = auto_subj_grade
            st.session_state[f"{key_prefix}_last_cb_sig"] = cb_sig
            st.session_state[f"{key_prefix}_cb_meta_status"] = (
                "parsed" if (auto_title and auto_subj_grade) else "partial"
            )

        if cb_sig:
            cb_meta_st = st.session_state.get(f"{key_prefix}_cb_meta_status", "")
            if cb_meta_st == "parsed":
                st.success(
                    f"🎯 표지 자동 인식 완료: **{st.session_state.get(f'{key_prefix}_book_title', '')}** "
                    f"({st.session_state.get(f'{key_prefix}_subject_grade', '')}) — 필요 시 아래에서 수정하세요"
                )
            elif cb_meta_st == "partial":
                st.info("📄 파일명/표지 일부를 채웠습니다. 빈칸이나 수정이 필요한 부분을 확인해 주세요.")

        cb_col1, cb_col2 = st.columns(2)
        with cb_col1:
            book_title_in = st.text_input(
                "교재명",
                placeholder="예: 완자 통합과학, 개념원리 공통수학1",
                key=f"{key_prefix}_book_title",
            )
        with cb_col2:
            subject_grade_in = st.text_input(
                "과목 및 학년",
                placeholder="예: 고1 통합과학",
                key=f"{key_prefix}_subject_grade",
            )

        extract_mode = st.radio(
            "개념 페이지 추출 방식",
            options=[
                "🤖 자동 추출 (AI가 개념 부분 자동 판별 · 단원 정리)",
                "✍️ 수동 추출 (추출할 개념 페이지 범위 직접 지정)",
            ],
            horizontal=True,
            key=f"{key_prefix}_extract_mode",
        )
        is_auto_mode = extract_mode.startswith("🤖")

        page_ranges_in = ""
        if not is_auto_mode:
            page_ranges_in = st.text_input(
                "추출할 개념 페이지 범위 입력 (쉼표 `,`와 하이픈 `-`으로 개념 설명 구간만 여러 개 지정 가능)",
                placeholder="예: 12-25, 34-48, 60-75",
                key=f"{key_prefix}_page_ranges",
            )
        else:
            st.info(
                "🤖 **AI 자동 추출 모드**: 페이지 범위를 직접 입력하지 않아도 "
                "Gemini 3.5 Flash-Lite가 먼저 목차를 분석한 뒤 본문과 대조해 **표지·목차·문제 풀이·정답지를 제외하고 "
                "순수 개념 설명 구간만 빠짐없이 판별**하여 단원별로 구조화합니다."
            )

        parsed_intervals: list[tuple[int, int]] = []
        if cb_pdf_bytes is not None:
            total_cb_pages = get_pdf_page_count(cb_pdf_bytes)
            st.caption(f"📄 업로드된 PDF: `{cb_pdf_file.name}` (전체 **{total_cb_pages}쪽**)")
            if not is_auto_mode and page_ranges_in.strip():
                try:
                    parsed_intervals = parse_page_ranges_spec(page_ranges_in, total_cb_pages)
                    total_sel_pages = sum(e - s + 1 for s, e in parsed_intervals)
                    interval_str = ", ".join(
                        f"p.{s}~p.{e}" if s != e else f"p.{s}" for s, e in parsed_intervals
                    )
                    st.info(
                        f"✅ 파싱된 개념 구간: **{interval_str}** (총 {len(parsed_intervals)}개 구간 · {total_sel_pages}쪽)"
                    )
                    with st.expander("👀 첫 구간 시작/끝 페이지 미리보기", expanded=False):
                        first_s, first_e = parsed_intervals[0][0], parsed_intervals[-1][1]
                        pv1, pv2 = st.columns(2)
                        with pv1:
                            st.image(
                                render_pdf_page_png(cb_pdf_bytes, first_s, dpi=130),
                                caption=f"첫 시작 페이지 (p.{first_s})",
                                width="stretch",
                            )
                        with pv2:
                            if first_e != first_s:
                                st.image(
                                    render_pdf_page_png(cb_pdf_bytes, first_e, dpi=130),
                                    caption=f"마지막 끝 페이지 (p.{first_e})",
                                    width="stretch",
                                )
                except ValueError as ve:
                    st.error(str(ve))

        def _finalize_and_save_concept_book(
            intervals_to_extract: list[tuple[int, int]],
            spec_label: str,
            status_box: Any,
        ) -> None:
            status_box.info(
                "2단계: PyMuPDF로 선택 구간에서 목차·정답지·문제 풀이를 제외하고 순수 개념 텍스트와 실제 페이지 번호를 매핑 중..."
            )
            sliced_bytes, mapped_text, stats = extract_concept_pages_by_ranges(
                cb_pdf_bytes, intervals_to_extract
            )
            skipped_info = stats.get("skipped_pages") or []
            if skipped_info:
                skip_summary = ", ".join(
                    f"p.{sp['page']}({sp['reason']})" for sp in skipped_info[:10]
                )
                st.caption(f"🧹 불필요 페이지 자동 제외 ({len(skipped_info)}쪽): {skip_summary}")

            db_build_model, db_model_notice = resolve_model_for_user(
                ctx.current_user, CONCEPT_DB_BUILD_MODEL
            )
            model_disp = MODEL_LABELS.get(db_build_model, db_build_model)
            status_box.info(
                f"3단계: {model_disp} 모델로 단원별 핵심 개념·허용 용어·페이지 범위를 구조화하고 공용 DB에 동기화 중..."
                + (f" ({db_model_notice})" if db_model_notice else "")
            )
            with _usage_scope(ctx):
                extracted_book = extract_concept_book_units(
                    api_key=ctx.active_gemini_key,
                    model_name=db_build_model,
                    book_title=book_title_in.strip(),
                    subject_and_grade=subject_grade_in.strip(),
                    page_ranges_display=stats["ranges_display"],
                    sliced_pdf_bytes=sliced_bytes,
                    mapped_concept_text=mapped_text,
                    original_pdf_bytes=cb_pdf_bytes,
                    kept_pages=stats.get("kept_pages"),
                    status_callback=lambda m: status_box.info(m),
                )
            grade_match = re.search(r"(고\s*[123])", subject_grade_in)
            grade_str = (
                grade_match.group(1).replace(" ", "")
                if grade_match
                else (extracted_book.grade or "고1")
            )
            book_id = f"{sanitize_id_part(book_title_in)}__{sanitize_id_part(subject_grade_in)}"
            record = ConceptBookRecord(
                id=book_id,
                book_title=book_title_in.strip(),
                subject=subject_grade_in.strip(),
                grade=grade_str,
                page_ranges_spec=spec_label,
                units=extracted_book.units,
                created_by=ctx.current_user.username,
                created_at=datetime.now().isoformat(timespec="seconds"),
            )
            save_err = ctx.storage.save_concept_book(record)
            status_box.empty()
            if save_err:
                st.warning(save_err)
            else:
                st.success(
                    f"🎉 공용 개념서 DB 저장 완료! (`{record.book_title}` · 총 {len(record.units)}개 단원 구조화 · 로컬 ./concept_db 저장 완료)"
                )

        if is_auto_mode:
            auto_b1, auto_b2 = st.columns([1.6, 1])
            with auto_b1:
                auto_build_btn = st.button(
                    "🤖 AI 개념 구간 자동 판별 & 단원별 공용 DB 저장",
                    type="primary",
                    width="stretch",
                    key=f"{key_prefix}_auto_build_btn",
                )
            with auto_b2:
                auto_preview_btn = st.button(
                    "🔍 AI 개념 구간만 먼저 미리보기",
                    width="stretch",
                    key=f"{key_prefix}_auto_preview_btn",
                )

            preview_state_key = f"{key_prefix}_auto_detect_preview"
            if cb_pdf_bytes is not None:
                cur_auto_sig = hashlib.sha256(cb_pdf_bytes).hexdigest()[:12]
                prev_auto = st.session_state.get(preview_state_key)
                if prev_auto and prev_auto.get("pdf_sig") != cur_auto_sig:
                    st.session_state.pop(preview_state_key, None)

            if auto_preview_btn or auto_build_btn:
                if not cb_pdf_bytes:
                    st.error("교재/개념서 PDF 파일을 먼저 업로드해 주세요.")
                elif not ctx.active_gemini_key:
                    st.error(f"사용 가능한 Gemini API 키({ctx.active_key_label})가 설정되지 않았습니다.")
                elif not _ensure_token_quota(ctx):
                    pass
                elif auto_build_btn and not book_title_in.strip():
                    st.error("교재명을 입력해 주세요. (예: 완자 통합과학)")
                elif auto_build_btn and not subject_grade_in.strip():
                    st.error("과목 및 학년을 입력해 주세요. (예: 고1 통합과학)")
                else:
                    status_box = st.empty()
                    try:
                        with _usage_scope(ctx):
                            cur_auto_sig = hashlib.sha256(cb_pdf_bytes).hexdigest()[:12]
                            cached_prev = st.session_state.get(preview_state_key)
                            if (
                                auto_build_btn
                                and cached_prev
                                and cached_prev.get("pdf_sig") == cur_auto_sig
                                and cached_prev.get("intervals")
                            ):
                                detected_out = cached_prev["detected_out"]
                                detected_intervals = cached_prev["intervals"]
                            else:
                                status_box.info("1단계: PyMuPDF로 PDF 전체 페이지 구조 스캔 중...")
                                catalog, has_text_layer = extract_pdf_page_catalog(cb_pdf_bytes)
                                detect_model, _ = resolve_model_for_user(
                                    ctx.current_user, CONCEPT_DB_BUILD_MODEL
                                )
                                detected_out, detected_intervals = detect_concept_pages_with_ai(
                                    api_key=ctx.active_gemini_key,
                                    model_name=detect_model,
                                    pdf_bytes=cb_pdf_bytes,
                                    page_catalog=catalog,
                                    has_text_layer=has_text_layer,
                                    book_title=book_title_in.strip(),
                                    subject_and_grade=subject_grade_in.strip(),
                                    status_callback=lambda m: status_box.info(m),
                                )
                                st.session_state[preview_state_key] = {
                                    "pdf_sig": cur_auto_sig,
                                    "detected_out": detected_out,
                                    "intervals": detected_intervals,
                                }

                            failed_chunks_list = list(getattr(detected_out, "failed_chunks", []) or [])
                            if failed_chunks_list:
                                st.warning(
                                    "⚠️ **일부 PDF 페이지 구간 판별에 실패했습니다** (자동 재시도 후 누락 방지를 위해 표시):\n- "
                                    + "\n- ".join(failed_chunks_list)
                                    + "\n\n해당 구간이 필요하다면 **[✍️ 수동 추출]**에서 직접 범위를 추가해 보완해 주세요."
                                )

                            if not detected_intervals:
                                status_box.empty()
                                st.warning(
                                    "AI가 순수 개념 설명 구간을 찾지 못했습니다. [✍️ 수동 추출]로 전환해 페이지 범위를 직접 지정해 주세요."
                                )
                            elif auto_preview_btn:
                                status_box.empty()
                                st.success(
                                    f"🔍 AI 개념 구간 판별 완료! (총 {len(detected_intervals)}개 구간 · "
                                    f"{sum(e - s + 1 for s, e in detected_intervals)}쪽 감지)"
                                )
                            else:
                                spec_auto = ", ".join(
                                    f"{s}-{e}" if s != e else f"{s}" for s, e in detected_intervals
                                )
                                _finalize_and_save_concept_book(
                                    intervals_to_extract=detected_intervals,
                                    spec_label=f"{spec_auto} (AI 자동 추출)",
                                    status_box=status_box,
                                )
                    except Exception as exc:
                        status_box.empty()
                        st.error(f"AI 자동 추출 실패: {exc}")

            saved_preview = st.session_state.get(preview_state_key)
            if saved_preview and (
                saved_preview.get("intervals")
                or getattr(saved_preview.get("detected_out"), "failed_chunks", None)
            ):
                det_out = saved_preview["detected_out"]
                det_ints = saved_preview.get("intervals") or []
                det_failed = list(getattr(det_out, "failed_chunks", []) or [])
                det_spec = ", ".join(f"p.{s}~p.{e}" if s != e else f"p.{s}" for s, e in det_ints)
                exp_label = (
                    f"🤖 AI가 자동 판별한 개념 구간 ({det_spec} · {sum(e - s + 1 for s, e in det_ints)}쪽)"
                    if det_ints
                    else "⚠️ AI 개념 구간 자동 판별 결과 (일부 구간 실패)"
                )
                with st.expander(exp_label, expanded=True):
                    if det_failed:
                        st.warning(
                            "⚠️ **자동 판별에 실패한 청크 구간:** " + " / ".join(det_failed)
                        )
                    if getattr(det_out, "toc_units", None):
                        st.caption(
                            f"📑 **목차 분석 기준 단원 ({len(det_out.toc_units)}개):** "
                            + " · ".join(det_out.toc_units)
                        )
                    for r_item in det_out.detected_ranges:
                        st.write(
                            f"- **{r_item.unit_hint}**: `p.{r_item.start_page}~p.{r_item.end_page}`"
                            + (f" — {r_item.reason}" if r_item.reason else "")
                        )
                    if det_out.excluded_summary:
                        st.caption(f"🧹 AI 제외 판단 구간: {det_out.excluded_summary}")
        else:
            build_db_btn = st.button(
                "🧠 PyMuPDF 개념 구간 추출 & Gemini 단원별 공용 DB 저장",
                type="primary",
                width="stretch",
                key=f"{key_prefix}_build_btn",
            )

            if build_db_btn:
                if not cb_pdf_bytes:
                    st.error("교재/개념서 PDF 파일을 먼저 업로드해 주세요.")
                elif not book_title_in.strip():
                    st.error("교재명을 입력해 주세요. (예: 완자 통합과학)")
                elif not subject_grade_in.strip():
                    st.error("과목 및 학년을 입력해 주세요. (예: 고1 통합과학)")
                elif not page_ranges_in.strip():
                    st.error("추출할 개념 페이지 범위를 입력해 주세요. (예: 12-25, 34-48, 60-75)")
                elif not ctx.active_gemini_key:
                    st.error(f"사용 가능한 Gemini API 키({ctx.active_key_label})가 설정되지 않았습니다.")
                elif not _ensure_token_quota(ctx):
                    pass
                else:
                    try:
                        total_cb_pages = get_pdf_page_count(cb_pdf_bytes)
                        intervals = parse_page_ranges_spec(page_ranges_in, total_cb_pages)
                    except ValueError as ve:
                        st.error(str(ve))
                        intervals = []

                    if intervals:
                        status_box = st.empty()
                        try:
                            _finalize_and_save_concept_book(
                                intervals_to_extract=intervals,
                                spec_label=f"{page_ranges_in.strip()} (수동 추출)",
                                status_box=status_box,
                            )
                        except Exception as exc:
                            status_box.empty()
                            st.error(f"개념서 단원별 구조화 추출 실패: {exc}")

    # 등록된 공용 개념서 DB 목록 조회 및 관리
    st.markdown("---")
    st.markdown("#### 🗂️ 등록된 공용 개념서·교과서 DB 목록 (`./concept_db` & Google Drive)")
    saved_books = ctx.storage.load_concept_books()
    if not saved_books:
        st.info("아직 등록된 공용 개념서 DB가 없습니다. 위에서 개념서 PDF를 업로드해 첫 공용 DB를 구축해 보세요.")
    else:
        for b_idx, book in enumerate(saved_books):
            with st.container(border=True):
                b_col1, b_col2, b_col3 = st.columns([3, 1.2, 0.8])
                with b_col1:
                    st.markdown(f"##### 📗 {book.book_title} (`{book.subject}`)")
                    st.caption(
                        f"단원 수: **{len(book.units)}개** · 추출 페이지 구간: `{book.page_ranges_spec}` · "
                        f"등록자: @{book.created_by or 'admin'}"
                    )
                with b_col2:
                    schema_dict = {
                        "book_title": book.book_title,
                        "subject": book.subject,
                        "grade": book.grade,
                        "units": [u.model_dump() for u in book.units],
                    }
                    st.download_button(
                        "📦 단원별 JSON 다운로드",
                        data=json.dumps(schema_dict, ensure_ascii=False, indent=2).encode("utf-8"),
                        file_name=f"{sanitize_id_part(book.book_title)}_concept_db.json",
                        mime="application/json",
                        key=f"{key_prefix}_dl_cb_{b_idx}_{book.id}",
                        width="stretch",
                    )
                with b_col3:
                    if st.button(
                        "🗑️ 삭제",
                        key=f"{key_prefix}_del_cb_{b_idx}_{book.id}",
                        width="stretch",
                    ):
                        del_err = ctx.storage.delete_concept_book(book.id)
                        if del_err:
                            st.warning(del_err)
                        st.rerun()

                unit_exp = create_lazy_expander(
                    f"📋 `{book.book_title}` 단원별 개념 및 허용 용어 상세 보기 ({len(book.units)}개 단원)",
                    expanded=False,
                    key=f"{key_prefix}_cb_units_exp_{b_idx}_{book.id}",
                )
                with unit_exp:
                    if is_container_open(unit_exp):
                        for u_idx, unit in enumerate(book.units, start=1):
                            st.markdown(f"**{u_idx}. {unit.unit_name}** (`{unit.page_range}`)")
                            uc1, uc2 = st.columns(2)
                            with uc1:
                                st.caption("핵심 원리 및 정의 요약 (`core_concepts`)")
                                render_chips(unit.core_concepts, variant="blue", limit=60)
                            with uc2:
                                st.caption("허용 용어·공식·단위 (`allowed_terms_and_units`)")
                                render_chips(unit.allowed_terms_and_units, variant="green", limit=80)
                            st.markdown("---")




render_ncic_curriculum_manager = _render_ncic_curriculum_manager
render_concept_book_builder_section = _render_concept_book_builder_section


def render_tab3(ctx: AppContext) -> None:
    render_ncic_curriculum_manager(key_prefix="tab3")
    render_concept_book_builder_section(ctx, key_prefix="tab3")
