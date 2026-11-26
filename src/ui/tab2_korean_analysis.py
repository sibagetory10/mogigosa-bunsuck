from __future__ import annotations

import concurrent.futures
import re
from typing import Any

import streamlit as st

from src.auth import CREDIT_COST_KOREAN_ANALYSIS, resolve_active_gemini_api_key
from src.config import EXAM_TAGGING_MODEL, MODEL_LABELS
from src.ui.common import (
    _usage_scope,
    format_credit_balance_inline,
    settle_credits,
    try_reserve_credits,
)
from src.ui.context import AppContext


def _execute_korean_analysis_jobs(
    ctx: AppContext,
    items: list[dict[str, Any]],
    include_diagrams: bool = True,
    include_passage: bool = False,
    force_regenerate: bool = False,
) -> None:
    """선택된 국어 지문 목록에 대해 정액 크레딧(세트당 10C · 캐시 풀 동일) 선예치·자동정산 하에 분석지 PDF를 생성한다."""
    if not items:
        st.warning("분석지를 생성할 국어 지문을 최소 1개 이상 선택해 주세요.")
        return

    reserve_credits = len(items) * CREDIT_COST_KOREAN_ANALYSIS
    ok_res, reservation = try_reserve_credits(
        ctx,
        reserve_credits,
        f"국어 지문 분석지 ({len(items)}세트)",
    )
    if not ok_res or reservation is None:
        return

    actual_credits = 0
    succeeded_an_sets = 0
    try:
        from src.gemini.korean import (
            generate_korean_passage_analysis,
            generate_korean_visual_diagram,
        )
        from src.gemini.usage import submit_with_usage_context
        from src.pdf.common import merge_korean_analysis_pdfs
        from src.pdf.korean_analysis_pdf import (
            _is_korean_poetry_item,
            generate_korean_passage_analysis_pdf,
            generate_korean_passage_only_pdf,
        )

        prepared_slots: list[dict[str, Any]] = []
        missing_ai_indices: list[int] = []
        missing_diag_indices: list[int] = []

        for idx_s, it in enumerate(items):
            p_txt = str(it.get("passage_text") or "").strip()
            gid = str(it.get("group_id") or "").strip()
            kor_sec = str(it.get("kor_sec") or "독서").strip()
            kor_sub = str(it.get("kor_sub") or "").strip()
            curr_subj = str(it.get("curr_subj") or kor_sec).strip()

            c_key = ctx.storage.build_korean_analysis_key(p_txt, gid, kor_sec)
            cached_an = None if force_regenerate else ctx.storage.get_cached_korean_analysis(c_key)
            cached_pdf = (
                None
                if force_regenerate
                else (ctx.storage.get_korean_analysis_pdf_bytes(c_key) if include_diagrams else None)
            )
            prepared_slots.append(
                {
                    "item": it,
                    "curr_subj": curr_subj,
                    "kor_sec": kor_sec,
                    "kor_sub": kor_sub,
                    "cache_key": c_key,
                    "analysis": cached_an,
                    "pdf_bytes": cached_pdf,
                }
            )
            if cached_an is None:
                missing_ai_indices.append(idx_s)
            elif (
                include_diagrams
                and kor_sec in ("독서", "문학")
                and getattr(cached_an, "visual_diagram", None) is None
            ):
                missing_diag_indices.append(idx_s)
                prepared_slots[idx_s]["pdf_bytes"] = None

        pdf_hit_cnt = sum(1 for sl in prepared_slots if sl["pdf_bytes"] is not None)
        json_to_pdf_cnt = 0
        ai_gen_cnt = 0

        if missing_ai_indices or missing_diag_indices:
            kor_api_key, kor_key_label = resolve_active_gemini_api_key(
                ctx.current_user,
                ctx.creds.gemini_api_key,
                model_name=EXAM_TAGGING_MODEL,
            )
            if not kor_api_key:
                st.error(f"국어 지문 분석용 Gemini API 키({kor_key_label})가 설정되지 않았습니다.")
                return

            need_total = len(missing_ai_indices) + len(missing_diag_indices)
            with _usage_scope(ctx):
                with st.spinner(
                    f"📑 {MODEL_LABELS.get(EXAM_TAGGING_MODEL, EXAM_TAGGING_MODEL)}로 "
                    f"국어 지문 분석지·도식 생성 {need_total}건 병렬 진행 중 "
                    f"(DB 로드 {len(items) - len(missing_ai_indices)}세트)..."
                ):
                    if missing_ai_indices:
                        def _run_ai_slot(slot_i: int):
                            slot = prepared_slots[slot_i]
                            it = slot["item"]
                            an_out = generate_korean_passage_analysis(
                                api_key=kor_api_key,
                                model_name=EXAM_TAGGING_MODEL,
                                exam_name=str(it.get("exam_name") or ""),
                                group_id=str(it.get("group_id") or ""),
                                passage_text=str(it.get("passage_text") or ""),
                                korean_section=slot["kor_sec"],
                                korean_subgenre=slot["kor_sub"],
                                passage_topic=str(it.get("passage_topic") or ""),
                                enable_proofread=False,
                                enable_web_search=True,
                            )
                            return slot_i, an_out

                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=min(4, len(missing_ai_indices))
                        ) as executor:
                            fut_map = {
                                submit_with_usage_context(executor, _run_ai_slot, si): si
                                for si in missing_ai_indices
                            }
                            for fut in concurrent.futures.as_completed(fut_map):
                                si = fut_map[fut]
                                slot = prepared_slots[si]
                                it = slot["item"]
                                try:
                                    _, an_out = fut.result()
                                    slot["analysis"] = an_out
                                    ai_gen_cnt += 1
                                    ctx.storage.save_cached_korean_analysis(
                                        slot["cache_key"],
                                        an_out,
                                        exam_name=str(it.get("exam_name") or ""),
                                        group_id=str(it.get("group_id") or ""),
                                        korean_section=slot["kor_sec"],
                                        korean_subgenre=slot["kor_sub"],
                                    )
                                except Exception as exc:
                                    st.error(f"[{it.get('q_label') or it.get('exam_name') or '국어 지문'}] 지문 분석 실패: {exc}")

                    valid_diag_indices = [
                        si for si in missing_diag_indices if prepared_slots[si]["analysis"] is not None
                    ]
                    if valid_diag_indices:
                        def _run_diag_slot(slot_i: int):
                            slot = prepared_slots[slot_i]
                            it = slot["item"]
                            cur_an = slot["analysis"]
                            diag_out = generate_korean_visual_diagram(
                                api_key=kor_api_key,
                                model_name=EXAM_TAGGING_MODEL,
                                passage_text=str(it.get("passage_text") or ""),
                                title=cur_an.title,
                                core_theme=cur_an.core_theme,
                                korean_section=slot["kor_sec"],
                            )
                            return slot_i, diag_out

                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=min(4, len(valid_diag_indices))
                        ) as executor:
                            fut_map = {
                                submit_with_usage_context(executor, _run_diag_slot, si): si
                                for si in valid_diag_indices
                            }
                            for fut in concurrent.futures.as_completed(fut_map):
                                si = fut_map[fut]
                                slot = prepared_slots[si]
                                it = slot["item"]
                                cur_an = slot["analysis"]
                                try:
                                    _, diag_out = fut.result()
                                    updated_an = cur_an.model_copy(update={"visual_diagram": diag_out})
                                    slot["analysis"] = updated_an
                                    ctx.storage.save_cached_korean_analysis(
                                        slot["cache_key"],
                                        updated_an,
                                        exam_name=str(it.get("exam_name") or ""),
                                        group_id=str(it.get("group_id") or ""),
                                        korean_section=slot["kor_sec"],
                                        korean_subgenre=slot["kor_sub"],
                                    )
                                except Exception as exc:
                                    st.warning(f"[{it.get('q_label') or it.get('exam_name') or '국어 지문'}] 이해 도식 생성 건너뜀: {exc}")

        analyses_list: list[dict[str, Any]] = []
        pdf_parts: list[bytes] = []
        missing_ai_set = set(missing_ai_indices)
        for idx_sl, slot in enumerate(prepared_slots):
            if slot["analysis"] is None and slot["pdf_bytes"] is None:
                continue
            it = slot["item"]
            single_item = {
                "exam_name": str(it.get("exam_name") or ""),
                "group_id": str(it.get("group_id") or ""),
                "q_label": str(it.get("q_label") or ""),
                "is_direct_input": bool(it.get("is_direct_input")),
                "section_label": slot["curr_subj"],
                "subgenre": slot["kor_sub"],
                "passage_text": str(it.get("passage_text") or ""),
                "analysis": slot["analysis"],
            }
            is_poetry_slot = _is_korean_poetry_item(
                slot["curr_subj"],
                slot["kor_sub"],
                str(it.get("passage_text") or ""),
                slot["analysis"],
            )
            if slot["pdf_bytes"] is None:
                try:
                    single_pdf = generate_korean_passage_analysis_pdf(
                        [single_item],
                        include_diagrams=include_diagrams,
                        include_passage=False,
                    )
                    if include_diagrams:
                        ctx.storage.save_korean_analysis_pdf_bytes(slot["cache_key"], single_pdf)
                    slot["pdf_bytes"] = single_pdf
                    if idx_sl not in missing_ai_set:
                        json_to_pdf_cnt += 1
                except Exception as exc:
                    st.error(f"[{it.get('q_label') or it.get('exam_name') or '국어 지문'}] 지문 분석 PDF 생성 실패: {exc}")
                    continue

            passage_pdf_bytes = None
            if include_passage and not is_poetry_slot:
                passage_pdf_bytes = (
                    None if force_regenerate else ctx.storage.get_korean_passage_pdf_bytes(slot["cache_key"])
                )
                if passage_pdf_bytes is None and str(it.get("passage_text") or "").strip():
                    try:
                        passage_pdf_bytes = generate_korean_passage_only_pdf(single_item)
                        ctx.storage.save_korean_passage_pdf_bytes(slot["cache_key"], passage_pdf_bytes)
                    except Exception as exc_p:
                        st.warning(f"[{it.get('q_label') or it.get('exam_name') or '국어 지문'}] 원문 PDF 생성 건너뜀: {exc_p}")

            pdf_parts.append(slot["pdf_bytes"])
            succeeded_an_sets += 1
            if include_passage and passage_pdf_bytes:
                pdf_parts.append(passage_pdf_bytes)
            if slot["analysis"] is not None:
                analyses_list.append(single_item)

        if pdf_parts:
            try:
                merged_pdf = merge_korean_analysis_pdfs(pdf_parts)
                st.session_state["kor_analysis_results"] = analyses_list
                st.session_state["kor_analysis_pdf"] = merged_pdf
                actual_credits = succeeded_an_sets * CREDIT_COST_KOREAN_ANALYSIS
            except Exception as exc:
                st.error(f"국어 지문 분석지 PDF 병합 실패: {exc}")
                actual_credits = 0
    finally:
        settle_credits(
            ctx,
            reservation,
            actual_credits,
            detail_note=f"국어 지문 분석지 ({succeeded_an_sets}/{len(items)}세트 성공 · {actual_credits:,}C 확정)",
        )

    if actual_credits > 0:
        detail_notes: list[str] = [f"저장 PDF 즉시 이어붙이기 {pdf_hit_cnt}세트"]
        if json_to_pdf_cnt > 0:
            detail_notes.append(f"기존 분석 JSON→PDF 저장 {json_to_pdf_cnt}세트")
        if ai_gen_cnt > 0:
            detail_notes.append(f"신규 AI 분석+PDF 저장 {ai_gen_cnt}세트")
        mode_note = "원문 지문 포함 ON" if include_passage else "원문 지문 제외 OFF"
        bal_msg = format_credit_balance_inline(
            ctx.storage,
            ctx.current_user.username if ctx.current_user is not None else "",
            charged=actual_credits,
        )
        st.success(
            f"📑 총 {len(analyses_list)}개 국어 지문 분석지 합본 PDF가 생성되었습니다! "
            f"({mode_note} · {' · '.join(detail_notes)}){bal_msg}"
        )


def _render_korean_analysis_results_section() -> None:
    """세션에 생성된 국어 지문 분석지 웹 미리보기 및 합본 PDF 다운로드 버튼을 렌더링한다."""
    saved_analyses: list[dict[str, Any]] = st.session_state.get("kor_analysis_results") or []
    saved_analysis_pdf: bytes | None = st.session_state.get("kor_analysis_pdf")
    if not (saved_analyses and saved_analysis_pdf):
        return

    st.markdown("---")
    st.markdown("#### 📑 생성된 국어 정밀 지문 분석지 (수능·학원형)")
    with st.container(border=True):
        dl_c1, dl_c2 = st.columns([2.2, 1.0])
        with dl_c1:
            st.write(
                f"총 **{len(saved_analyses)}개** 지문 세트 분석 완료 "
                "(핵심 주제 · 전개 흐름 · 문단별 요약 · 개념 정의 · 출제 포인트 포함)"
            )
        with dl_c2:
            st.download_button(
                "📥 국어 지문 분석지 PDF 다운로드",
                data=saved_analysis_pdf,
                file_name="국어_정밀_지문_분석지.pdf",
                mime="application/pdf",
                type="primary",
                width="stretch",
                key="kor_analysis_pdf_dl_btn",
            )

        for idx_a, an_item in enumerate(saved_analyses, start=1):
            an = an_item["analysis"]
            is_direct = bool(an_item.get("is_direct_input")) or not str(an_item.get("group_id") or "").strip()
            src_hdr = (
                f"{an_item['exam_name']}"
                if is_direct
                else f"{an_item['exam_name']} · {an_item['group_id']}번"
            )
            with st.expander(
                f"{idx_a}. [{src_hdr}] {an.title} ({an_item['section_label']})",
                expanded=(idx_a == 1),
            ):
                st.info(f"**📌 핵심 주제:** {an.core_theme}")
                st.success(f"**🧭 전개 흐름:**\n\n{an.structure_flow}")
                _v_diag = getattr(an, "visual_diagram", None)
                if _v_diag is not None and (
                    getattr(_v_diag, "nodes", None)
                    or getattr(_v_diag, "graph_spec", None)
                    or getattr(_v_diag, "emotion_points", None)
                ):
                    try:
                        from src.diagram_renderer import render_korean_diagram_png

                        st.image(
                            render_korean_diagram_png(_v_diag),
                            caption=f"🖼️ {_v_diag.title}",
                            width="stretch",
                        )
                    except Exception:
                        pass
                if an.paragraph_summaries:
                    st.markdown("**📌 문단 · 연별 요약**")
                    for ps in an.paragraph_summaries:
                        kp = f" *(포인트: {ps.key_point})*" if ps.key_point else ""
                        st.write(f"- **[{ps.label}]** {ps.summary}{kp}")
                if an.marker_and_vocab_notes:
                    st.markdown("**🔖 지문 핵심 개념 정의**")
                    for mv in an.marker_and_vocab_notes:
                        st.write(f"- **`{mv.term}`**: {mv.meaning}")
                if an.area_specific_notes:
                    st.markdown(f"**📖 영역별 핵심 정리 ({an_item['section_label']})**")
                    for note in an.area_specific_notes:
                        st.write(f"- {note}")
                if an.exam_checkpoints:
                    st.markdown("**⚠️ 출제 예상 포인트 및 오답 주의사항**")
                    for cp in an.exam_checkpoints:
                        st.write(f"- {cp}")
                if getattr(an, "poetry_underline_notes", None):
                    st.markdown("**✍️ 원문 시구 밑줄 해설**")
                    for idx_p, pn in enumerate(an.poetry_underline_notes, start=1):
                        st.write(f"- **({idx_p}) <u>{pn.quote}</u>**: {pn.explanation}", unsafe_allow_html=True)


def _render_korean_analysis_submode(ctx: AppContext) -> None:
    """국어 서브모드 2: 국어 지문 정밀 분석지 PDF 생성 전용 UI."""
    st.caption(
        "💡 선택한 국어 지문의 **핵심 주제 · 논리 전개 흐름 · 문단/연별 요약 · 이해용 시각 도식 · 출제 포인트**가 담긴 정밀 분석지 PDF를 생성합니다."
    )

    an_tab_db, an_tab_direct = st.tabs(
        [
            "📂 기출 DB 국어 지문에서 선택 (회차·영역별 검색)",
            "✍️ 새 국어 지문 직접 붙여넣기",
        ]
    )

    with an_tab_db:
        from src.ui.tab2_korean import (
            _filter_korean_db_sets,
            _render_korean_passage_preview_expander,
        )

        has_db_recs, filtered_sets = _filter_korean_db_sets(ctx, key_prefix="tab2_kor_an")
        if has_db_recs:
            if not filtered_sets:
                st.warning("조건에 맞는 국어 지문 세트가 없습니다. 필터를 조정해 주세요.")
            else:
                st.caption(
                    f"총 **{len(filtered_sets)}개** 국어 지문 세트가 조회되었습니다. "
                    "분석지를 만들 지문을 체크하세요. (`📑 분석지PDF DB` 배지가 붙은 지문은 재호출 없이 즉시 병합됩니다.)"
                )
                with st.form("tab2_kor_db_analysis_form", border=True):
                    opt_c1, opt_c2, opt_c3 = st.columns(3)
                    with opt_c1:
                        include_kor_db_diagrams = st.checkbox(
                            "🎨 지문 이해용 시각 도식 그림 포함",
                            value=True,
                            key="tab2_kor_an_db_include_diagrams",
                        )
                    with opt_c2:
                        include_kor_db_passage = st.checkbox(
                            "📄 원문 지문 전체 포함",
                            value=False,
                            key="tab2_kor_an_db_include_passage",
                        )
                    with opt_c3:
                        force_regen_db = st.checkbox(
                            "🔄 기존 캐시 무시하고 새로 분석지 생성",
                            value=False,
                            key="tab2_kor_an_db_force_regen",
                        )

                    st.markdown("---")
                    st.markdown("**✅ 분석지를 생성할 국어 기출 지문 세트 선택:**")
                    checked_an_sets: list[dict[str, Any]] = []
                    for s_idx, s_entry in enumerate(filtered_sets):
                        q_nums = [r.q_num for r in s_entry["records"]]
                        sec_tag = s_entry["curr_subj"]
                        topic_tag = (s_entry.get("passage_topic") or "").strip()
                        p_preview = re.sub(r"\s+", " ", (s_entry.get("passage_text") or "").strip())[:115]
                        c_key_chk = ctx.storage.build_korean_analysis_key(
                            s_entry["passage_text"],
                            s_entry["group_id"],
                            s_entry["kor_sec"],
                        )
                        pdf_status_tag = " · 📑 분석지PDF DB" if ctx.storage.has_korean_analysis_pdf(c_key_chk) else ""
                        row_label = (
                            f"[{s_entry['exam_name']} {s_entry['group_id']}번] 영역: {sec_tag} · {len(q_nums)}문항{pdf_status_tag}"
                            + (f" · 주제: {topic_tag}" if topic_tag else "")
                            + f" — \"{p_preview}...\""
                        )
                        if st.checkbox(
                            row_label,
                            value=False,
                            key=f"tab2_kor_an_chk_{s_entry['exam_name']}_{s_entry['group_id']}_{s_idx}",
                        ):
                            checked_an_sets.append(s_entry)

                    run_kor_db_analysis_btn = st.form_submit_button(
                        f"📑 선택한 기출 국어 지문 분석지 PDF 생성 (세트당 {CREDIT_COST_KOREAN_ANALYSIS}C · 캐시 풀 동일)",
                        type="primary",
                        width="stretch",
                    )

                _render_korean_passage_preview_expander(filtered_sets)

                if run_kor_db_analysis_btn:
                    if not checked_an_sets:
                        st.error("분석지를 생성할 국어 지문 세트를 최소 1개 이상 체크해 주세요.")
                    else:
                        an_items = [
                            {
                                "exam_name": s_entry["exam_name"],
                                "group_id": s_entry["group_id"],
                                "q_label": f"{s_entry['exam_name']} [{s_entry['group_id']}번]",
                                "curr_subj": s_entry["curr_subj"],
                                "kor_sec": s_entry["kor_sec"],
                                "kor_sub": s_entry["kor_sub"],
                                "passage_topic": s_entry.get("passage_topic", ""),
                                "passage_text": s_entry["passage_text"],
                                "is_direct_input": False,
                            }
                            for s_entry in checked_an_sets
                        ]
                        _execute_korean_analysis_jobs(
                            ctx,
                            items=an_items,
                            include_diagrams=include_kor_db_diagrams,
                            include_passage=include_kor_db_passage,
                            force_regenerate=force_regen_db,
                        )

    with an_tab_direct:
        with st.form("tab2_kor_direct_analysis_form", border=True):
            st.markdown("**✍️ 분석할 국어 지문을 붙여넣으세요:**")
            d_c1, d_c2, d_c3 = st.columns([1.4, 1.0, 1.0])
            with d_c1:
                an_src_name = st.text_input(
                    "출처 표기 (분석지 상단 표시용)",
                    value="직접 입력 국어 지문",
                    key="tab2_kor_an_direct_src",
                )
            with d_c2:
                an_kor_sec = st.selectbox(
                    "영역 선택",
                    options=["독서", "문학", "화법과 작문", "언어와 매체"],
                    index=0,
                    key="tab2_kor_an_direct_sec",
                )
            with d_c3:
                an_kor_sub = st.text_input(
                    "세부 갈래 (선택)",
                    value="",
                    placeholder="예: 과학·기술, 현대시, 고전시가...",
                    key="tab2_kor_an_direct_subgenre",
                )

            an_passage_raw = st.text_area(
                "국어 지문 원문 입력 (여러 개의 지문을 한 번에 넣으려면 지문 사이에 '---' 줄을 넣어 구분하세요)",
                height=220,
                placeholder="분석할 국어 지문 원문을 붙여넣으세요...",
                key="tab2_kor_an_direct_text",
            )
            d_opt_c1, d_opt_c2, d_opt_c3 = st.columns(3)
            with d_opt_c1:
                include_kor_direct_diagrams = st.checkbox(
                    "🎨 지문 이해용 시각 도식 그림 포함",
                    value=True,
                    key="tab2_kor_an_direct_include_diagrams",
                )
            with d_opt_c2:
                include_kor_direct_passage = st.checkbox(
                    "📄 원문 지문 전체 포함",
                    value=False,
                    key="tab2_kor_an_direct_include_passage",
                )
            with d_opt_c3:
                force_regen_direct = st.checkbox(
                    "🔄 기존 캐시 무시하고 새로 분석지 생성",
                    value=False,
                    key="tab2_kor_an_direct_force_regen",
                )
            run_kor_direct_analysis_btn = st.form_submit_button(
                f"📑 직접 입력한 국어 지문 분석지 PDF 생성 (세트당 {CREDIT_COST_KOREAN_ANALYSIS}C · 캐시 풀 동일)",
                type="primary",
                width="stretch",
            )

        if run_kor_direct_analysis_btn:
            raw_chunks = [
                ch.strip()
                for ch in re.split(r"\n\s*-{3,}\s*\n", (an_passage_raw or "").strip())
                if ch.strip()
            ]
            if not raw_chunks:
                st.error("분석지를 생성할 국어 지문 텍스트를 입력해 주세요.")
            else:
                base_src = (an_src_name or "직접 입력 국어 지문").strip()
                clean_sub = (an_kor_sub or "").strip()
                curr_subj_lbl = f"{an_kor_sec}({clean_sub})" if clean_sub else an_kor_sec
                direct_an_items = [
                    {
                        "exam_name": f"{base_src} #{c_idx}" if len(raw_chunks) > 1 else base_src,
                        "group_id": "",
                        "q_label": f"{base_src} #{c_idx}" if len(raw_chunks) > 1 else base_src,
                        "curr_subj": curr_subj_lbl,
                        "kor_sec": an_kor_sec,
                        "kor_sub": clean_sub,
                        "passage_topic": "",
                        "passage_text": chunk_txt,
                        "is_direct_input": True,
                    }
                    for c_idx, chunk_txt in enumerate(raw_chunks, start=1)
                ]
                _execute_korean_analysis_jobs(
                    ctx,
                    items=direct_an_items,
                    include_diagrams=include_kor_direct_diagrams,
                    include_passage=include_kor_direct_passage,
                    force_regenerate=force_regen_direct,
                )

    _render_korean_analysis_results_section()
