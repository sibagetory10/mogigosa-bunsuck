from __future__ import annotations

import concurrent.futures
import html as _html
import re
from typing import Any

import streamlit as st

from src.auth import resolve_active_gemini_api_key
from src.config import FLASH_LITE_MODEL, MODEL_LABELS
from src.models import (
    CURRENT_ENGLISH_ANALYSIS_VERSION,
    QuestionRecord,
    normalize_english_group_id,
)
from src.ui.common import _ensure_token_quota, _usage_scope
from src.ui.context import AppContext

_ENG_CIRCLED_NUMS = (
    "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩",
    "⑪", "⑫", "⑬", "⑭", "⑮",
)


def _format_eng_badge(num: int) -> str:
    if 1 <= num <= len(_ENG_CIRCLED_NUMS):
        return _ENG_CIRCLED_NUMS[num - 1]
    return f"({num})"


def _execute_english_analysis_jobs(
    ctx: AppContext,
    items: list[dict[str, Any]],
    include_diagrams: bool = True,
    force_regenerate: bool = False,
) -> None:
    """선택된 영어 지문 목록에 대해 DB 캐시 조회/병렬 AI(주제·도식) 분석/지문별 PDF 저장/합본 PDF 생성을 수행한다."""
    if not items:
        st.warning("분석지를 생성할 영어 지문을 최소 1개 이상 선택해 주세요.")
        return
    from src.gemini import (
        generate_english_passage_analysis,
        generate_english_visual_diagram,
        submit_with_usage_context,
    )
    from src.pdf.common import merge_korean_analysis_pdfs
    from src.pdf.english_analysis_pdf import generate_english_passage_analysis_pdf
    from src.pdf_parser.english_source_extractor import (
        build_english_sentence_items,
        extract_english_passage_from_problem_source,
        extract_english_question_translation_from_pdf,
    )

    # 동일 시험지 PDF 중복 디스크 로드 방지를 위해 사전 캐싱
    pdf_bytes_cache: dict[tuple[str, str], bytes | None] = {}

    def _get_cached_pdf_bytes(exam_name: str, filename: str) -> bytes | None:
        fn = (filename or "").strip()
        if not fn:
            return None
        key = ((exam_name or "").strip(), fn)
        if key not in pdf_bytes_cache:
            pdf_bytes_cache[key] = ctx.storage.get_pdf_bytes(
                fn,
                subject_category="영어",
                exam_name=(exam_name or "").strip(),
            )
        return pdf_bytes_cache[key]

    prepared_slots: list[dict[str, Any]] = []
    missing_ai_indices: list[int] = []
    missing_diag_indices: list[int] = []

    for idx_s, it in enumerate(items):
        p_txt = str(it.get("passage_text") or "").strip()
        q_lbl = str(it.get("q_label") or "").strip()
        direct_kor = str(it.get("direct_korean_translation") or "").strip()
        c_key = ctx.storage.build_english_analysis_key(p_txt, q_lbl, extra_key=direct_kor)

        src_pdf_b = _get_cached_pdf_bytes(str(it.get("exam_name") or ""), str(it.get("source_pdf") or ""))
        ans_pdf_b = _get_cached_pdf_bytes(str(it.get("exam_name") or ""), str(it.get("answer_pdf") or ""))
        q_num = int(it.get("q_num") or 0)
        p_gid = str(it.get("passage_group_id") or "").strip()

        cached_an = None if force_regenerate else ctx.storage.get_cached_english_analysis(c_key)
        cached_pdf = (
            None
            if force_regenerate
            else (ctx.storage.get_english_analysis_pdf_bytes(c_key) if include_diagrams else None)
        )

        if cached_an is not None:
            is_stale_ver = int(getattr(cached_an, "analysis_version", 1) or 1) < CURRENT_ENGLISH_ANALYSIS_VERSION
            if is_stale_ver:
                cached_an = None
                cached_pdf = None

        # v5 캐시가 있으나 당시 해설지 PDF 미보유로 translation_source=='none'이었던 경우 AI 호출 없이 3번 섹션만 보강
        if (
            cached_an is not None
            and getattr(cached_an, "translation_source", "") == "none"
            and ans_pdf_b is not None
            and q_num > 0
        ):
            new_kor = extract_english_question_translation_from_pdf(
                answer_pdf_bytes=ans_pdf_b,
                q_num=q_num,
                passage_group_id=p_gid,
                orig_question_type=str(it.get("orig_type") or ""),
                answer=str(it.get("answer") or ""),
            )
            if new_kor:
                clean_p, fns = extract_english_passage_from_problem_source(
                    passage_text=p_txt,
                    question_text=str(it.get("question_text") or ""),
                    source_pdf_bytes=src_pdf_b,
                    q_num=q_num,
                    passage_group_id=p_gid,
                    orig_question_type=str(it.get("orig_type") or ""),
                    answer=str(it.get("answer") or ""),
                )
                new_items = build_english_sentence_items(clean_p or p_txt, new_kor)
                cached_an = cached_an.model_copy(
                    update={
                        "sentence_analyses": new_items,
                        "footnotes": fns or cached_an.footnotes,
                        "translation_source": "answer_pdf",
                    }
                )
                cached_pdf = None
                ctx.storage.save_cached_english_analysis(
                    c_key,
                    cached_an,
                    exam_name=it.get("exam_name", ""),
                    q_label=q_lbl,
                    question_type=it.get("orig_type", ""),
                )

        prepared_slots.append(
            {
                "item": it,
                "cache_key": c_key,
                "analysis": cached_an,
                "pdf_bytes": cached_pdf,
                "source_pdf_bytes": src_pdf_b,
                "answer_pdf_bytes": ans_pdf_b,
                "q_num": q_num,
                "passage_group_id": p_gid,
                "direct_korean_translation": direct_kor,
            }
        )
        if cached_an is None:
            missing_ai_indices.append(idx_s)
        elif include_diagrams and getattr(cached_an, "visual_diagram", None) is None:
            missing_diag_indices.append(idx_s)
            prepared_slots[idx_s]["pdf_bytes"] = None

    pdf_hit_cnt = sum(1 for sl in prepared_slots if sl["pdf_bytes"] is not None)
    json_to_pdf_cnt = 0
    ai_gen_cnt = 0

    if missing_ai_indices or missing_diag_indices:
        if not _ensure_token_quota(ctx):
            return
        an_model = FLASH_LITE_MODEL
        an_api_key, an_key_label = resolve_active_gemini_api_key(
            ctx.current_user, ctx.creds.gemini_api_key, model_name=an_model
        )
        if not an_api_key:
            st.error(f"영어 지문 분석용 Gemini API 키({an_key_label})가 설정되지 않았습니다.")
            return

        need_total = len(missing_ai_indices) + len(missing_diag_indices)
        with _usage_scope(ctx):
            with st.spinner(
                f"📑 {MODEL_LABELS.get(an_model, an_model)}로 "
                f"영어 지문 주제·논리흐름·도식 생성 {need_total}건 병렬 진행 중 "
                f"(DB 로드 {len(items) - len(missing_ai_indices)}건)..."
            ):
                if missing_ai_indices:
                    def _run_ai_slot(slot_i: int):
                        slot = prepared_slots[slot_i]
                        it = slot["item"]
                        an_out = generate_english_passage_analysis(
                            api_key=an_api_key,
                            model_name=an_model,
                            passage_text=it["passage_text"],
                            source_label=it["q_label"],
                            orig_question_type=it.get("orig_type", ""),
                            passage_topic=it.get("passage_topic", ""),
                            answer=it.get("answer", ""),
                            choices=it.get("choices", []),
                            explanation=it.get("explanation", ""),
                            question_text=it.get("question_text", ""),
                            source_pdf_bytes=slot["source_pdf_bytes"],
                            answer_pdf_bytes=slot["answer_pdf_bytes"],
                            q_num=slot["q_num"],
                            passage_group_id=slot["passage_group_id"],
                            direct_korean_translation=slot["direct_korean_translation"],
                            include_diagrams=include_diagrams,
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
                                ctx.storage.save_cached_english_analysis(
                                    slot["cache_key"],
                                    an_out,
                                    exam_name=it.get("exam_name", ""),
                                    q_label=it.get("q_label", ""),
                                    question_type=it.get("orig_type", ""),
                                )
                            except Exception as exc:
                                st.error(f"[{it.get('q_label', '영어 지문')}] 지문 분석 실패: {exc}")

                valid_diag_indices = [
                    si for si in missing_diag_indices if prepared_slots[si]["analysis"] is not None
                ]
                if valid_diag_indices:
                    def _run_diag_slot(slot_i: int):
                        slot = prepared_slots[slot_i]
                        it = slot["item"]
                        cur_an = slot["analysis"]
                        diag_out = generate_english_visual_diagram(
                            api_key=an_api_key,
                            model_name=an_model,
                            passage_text=it["passage_text"],
                            title=cur_an.title,
                            core_theme=cur_an.core_theme,
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
                                ctx.storage.save_cached_english_analysis(
                                    slot["cache_key"],
                                    updated_an,
                                    exam_name=it.get("exam_name", ""),
                                    q_label=it.get("q_label", ""),
                                    question_type=it.get("orig_type", ""),
                                )
                            except Exception as exc:
                                st.warning(f"[{it.get('q_label', '영어 지문')}] 이해 도식 생성 건너뜀: {exc}")

    analyses_list: list[dict[str, Any]] = []
    pdf_parts: list[bytes] = []
    missing_ai_set = set(missing_ai_indices)
    for idx_sl, slot in enumerate(prepared_slots):
        if slot["analysis"] is None and slot["pdf_bytes"] is None:
            continue
        it = slot["item"]
        single_item = {
            "exam_name": it.get("exam_name", ""),
            "q_label": it.get("q_label", ""),
            "orig_type": it.get("orig_type", "영어 독해"),
            "passage_text": it.get("passage_text", ""),
            "analysis": slot["analysis"],
        }
        if slot["pdf_bytes"] is None:
            try:
                single_pdf = generate_english_passage_analysis_pdf(
                    [single_item],
                    include_diagrams=include_diagrams,
                )
                if include_diagrams:
                    ctx.storage.save_english_analysis_pdf_bytes(slot["cache_key"], single_pdf)
                slot["pdf_bytes"] = single_pdf
                if idx_sl not in missing_ai_set:
                    json_to_pdf_cnt += 1
            except Exception as exc:
                st.error(f"[{it.get('q_label', '영어 지문')}] 분석 PDF 생성 실패: {exc}")
                continue
        pdf_parts.append(slot["pdf_bytes"])
        if slot["analysis"] is not None:
            analyses_list.append(single_item)

    if pdf_parts:
        try:
            merged_pdf = merge_korean_analysis_pdfs(pdf_parts)
            st.session_state["eng_analysis_results"] = analyses_list
            st.session_state["eng_analysis_pdf"] = merged_pdf
            detail_notes: list[str] = [f"저장 PDF 즉시 이어붙이기 {pdf_hit_cnt}건"]
            if json_to_pdf_cnt > 0:
                detail_notes.append(f"기존 분석 JSON→PDF 저장 {json_to_pdf_cnt}건")
            if ai_gen_cnt > 0:
                detail_notes.append(f"신규 분석+PDF 저장 {ai_gen_cnt}건")
            st.success(
                f"📑 총 {len(analyses_list)}개 영어 지문 분석지 합본 PDF가 생성되었습니다! "
                f"({' · '.join(detail_notes)})"
            )
        except Exception as exc:
            st.error(f"영어 지문 분석지 PDF 병합 실패: {exc}")


def _render_english_analysis_results_section() -> None:
    """세션에 생성된 영어 지문 분석지 웹 미리보기 및 합본 PDF 다운로드 버튼을 렌더링한다."""
    saved_analyses: list[dict[str, Any]] = st.session_state.get("eng_analysis_results") or []
    saved_pdf: bytes | None = st.session_state.get("eng_analysis_pdf")
    if not (saved_analyses and saved_pdf):
        return
    from src.diagram_renderer import render_korean_diagram_png

    st.markdown("---")
    st.markdown("#### 📑 생성된 영어 지문 분석지 (핵심 주제 · 논리 흐름 · 도식 · 문장별 원문/해설지 해석)")
    with st.container(border=True):
        dl_c1, dl_c2 = st.columns([2.2, 1.0])
        with dl_c1:
            st.write(
                f"총 **{len(saved_analyses)}개** 영어 지문 분석 완료 "
                "(핵심 주제 · 논리 전개 흐름 · 시각 도식 · 문장별 원문/해설지 해석)"
            )
        with dl_c2:
            st.download_button(
                "📥 영어 지문 분석지 PDF 다운로드",
                data=saved_pdf,
                file_name="영어_지문_분석지.pdf",
                mime="application/pdf",
                type="primary",
                width="stretch",
                key="eng_analysis_pdf_dl_btn",
            )

        for idx_a, an_item in enumerate(saved_analyses, start=1):
            an = an_item["analysis"]
            q_lbl = an_item.get("q_label") or an_item.get("exam_name") or "영어 지문"
            orig_t = an_item.get("orig_type") or "독해"

            with st.expander(
                f"{idx_a}. [{q_lbl}] {an.title} ({orig_t})",
                expanded=(idx_a == 1),
            ):
                st.info(f"**📌 핵심 주제:** {an.core_theme}")
                st.success(f"**🧭 논리 전개 흐름:**\n\n{an.structure_flow}")
                _fn_list = [str(f).strip() for f in (getattr(an, "footnotes", None) or []) if str(f).strip()]
                if _fn_list:
                    st.caption("📝 **문제지 공식 각주:** " + " · ".join(_fn_list))
                _v_diag = getattr(an, "visual_diagram", None)
                if _v_diag is not None and (
                    getattr(_v_diag, "nodes", None)
                    or getattr(_v_diag, "graph_spec", None)
                    or getattr(_v_diag, "emotion_points", None)
                ):
                    try:
                        st.image(
                            render_korean_diagram_png(_v_diag),
                            caption=f"🖼️ {_v_diag.title}",
                            width="stretch",
                        )
                    except Exception:
                        pass

                sent_list = list(getattr(an, "sentence_analyses", None) or [])
                if sent_list:
                    st.markdown("**🔍 3. 문장별 정밀 구문 분석** (저장된 문제지 원문 · 해설지 해석 그대로 1:1 매칭)")
                    if getattr(an, "translation_source", "") == "none":
                        st.caption("ℹ️ 연결된 해설지 해석이 없어 영어 원문 문장만 표시됩니다.")
                    for s_i, sa in enumerate(sent_list):
                        s_num = int(getattr(sa, "sentence_num", s_i + 1) or (s_i + 1))
                        n_badge = _format_eng_badge(s_num)
                        orig_eng = (
                            getattr(sa, "original_english", "")
                            or getattr(sa, "chunked_english", "")
                            or ""
                        ).strip()
                        kor_txt = (getattr(sa, "korean_translation", "") or "").strip()
                        kor_block = (
                            f"<div style='color:#334155;font-size:0.95em;line-height:1.5;margin-top:4px;'>"
                            f"{_html.escape(kor_txt)}</div>"
                            if kor_txt
                            else ""
                        )
                        st.markdown(
                            "<div style='border:1px solid #E2E8F0;background:#FFFFFF;border-radius:6px;"
                            "padding:10px 12px;margin-bottom:8px;'>"
                            f"<div style='font-weight:600;color:#0F172A;line-height:1.6;'>"
                            f"<strong style='color:#1E3A8A;margin-right:6px;'>{n_badge}</strong>"
                            f"{_html.escape(orig_eng)}</div>"
                            f"{kor_block}"
                            "</div>",
                            unsafe_allow_html=True,
                        )


def _render_english_analysis_submode(ctx: AppContext) -> None:
    """영어 서브모드 2: 영어 지문 분석지 PDF 생성 전용 UI."""
    st.caption(
        "💡 선택한 영어 지문의 **핵심 주제 · 논리 전개 흐름 · 이해용 시각 도식 · 문장별 원문/해설지 해석(100% 원문·해설지 PDF 복붙)** 분석지 PDF를 생성합니다."
    )

    an_tab_db, an_tab_direct = st.tabs(
        [
            "📂 기출 DB 영어 지문에서 선택 (회차·원래 유형별 검색)",
            "✍️ 새 영어 지문 직접 붙여넣기",
        ]
    )

    with an_tab_db:
        from src.ui.tab2_english import (
            _filter_english_db_passages,
            _render_english_passage_preview_expander,
        )

        has_db_recs, filtered_passages = _filter_english_db_passages(ctx, key_prefix="tab2_eng_an")
        if has_db_recs:
            if not filtered_passages:
                st.warning("조건에 맞는 영어 독해 지문이 없습니다. 필터를 완화하거나 상단 제외 문항 포함 체크박스를 켜 보세요.")
            else:
                st.caption(
                    f"총 **{len(filtered_passages)}개** 영어 독해 지문이 조회되었습니다. "
                    "분석지를 만들 지문을 체크하세요. (`📑 분석지PDF DB` 배지가 붙은 지문은 재호출 없이 즉시 병합됩니다.)"
                )
                with st.form("tab2_eng_db_analysis_form", border=True):
                    opt_c1, opt_c2 = st.columns([1.0, 1.3])
                    with opt_c1:
                        include_eng_db_diagrams = st.checkbox(
                            "🎨 지문 이해용 시각 도식 그림 포함",
                            value=True,
                            key="tab2_eng_an_db_include_diagrams",
                        )
                    with opt_c2:
                        force_regen_db = st.checkbox(
                            "🔄 기존 캐시 무시하고 새로 분석지 생성",
                            value=False,
                            key="tab2_eng_an_db_force_regen",
                        )
                    st.markdown("---")
                    st.markdown("**✅ 분석지를 생성할 영어 기출 지문 선택:**")
                    checked_an_rows: list[tuple[QuestionRecord, str]] = []
                    for p_idx, (rec_p, q_label) in enumerate(filtered_passages):
                        r_orig_t = (getattr(rec_p, "english_question_type", "") or rec_p.unit_name).strip()
                        r_topic = (getattr(rec_p, "passage_topic", "") or "").strip()
                        p_preview = re.sub(r"\s+", " ", (rec_p.passage_text or "").strip())[:115]
                        c_key_chk = ctx.storage.build_english_analysis_key(rec_p.passage_text, q_label)
                        pdf_status_tag = " · 📑 분석지PDF DB" if ctx.storage.has_english_analysis_pdf(c_key_chk) else ""
                        row_label = (
                            f"[{q_label}] 원래 유형: {r_orig_t}{pdf_status_tag}"
                            + (f" · 주제: {r_topic}" if r_topic else "")
                            + f" — \"{p_preview}...\""
                        )
                        if st.checkbox(
                            row_label,
                            value=False,
                            key=f"tab2_eng_an_chk_{rec_p.id}_{p_idx}",
                        ):
                            checked_an_rows.append((rec_p, q_label))

                    run_eng_db_analysis_btn = st.form_submit_button(
                        "📑 선택한 기출 영어 지문 분석지 PDF 생성",
                        type="primary",
                        width="stretch",
                    )

                _render_english_passage_preview_expander(filtered_passages)

                if run_eng_db_analysis_btn:
                    if not checked_an_rows:
                        st.error("분석지를 생성할 영어 지문을 최소 1개 이상 체크해 주세요.")
                    else:
                        an_items = [
                            {
                                "exam_name": rec_p.exam_name,
                                "q_label": q_label,
                                "q_num": int(rec_p.q_num),
                                "passage_group_id": normalize_english_group_id(rec_p.q_num),
                                "orig_type": (getattr(rec_p, "english_question_type", "") or rec_p.unit_name).strip(),
                                "passage_topic": (getattr(rec_p, "passage_topic", "") or "").strip(),
                                "passage_text": rec_p.passage_text,
                                "question_text": (getattr(rec_p, "question_text", "") or "").strip(),
                                "answer": (getattr(rec_p, "answer", "") or "").strip(),
                                "choices": list(getattr(rec_p, "choices", None) or []),
                                "explanation": (
                                    getattr(rec_p, "official_explanation", "")
                                    or getattr(rec_p, "explanation", "")
                                    or ""
                                ).strip(),
                                "source_pdf": (getattr(rec_p, "source_pdf", "") or "").strip(),
                                "answer_pdf": (getattr(rec_p, "answer_pdf", "") or "").strip(),
                            }
                            for rec_p, q_label in checked_an_rows
                        ]
                        _execute_english_analysis_jobs(
                            ctx,
                            items=an_items,
                            include_diagrams=include_eng_db_diagrams,
                            force_regenerate=force_regen_db,
                        )

    with an_tab_direct:
        with st.form("tab2_eng_direct_analysis_form", border=True):
            st.markdown("**✍️ 분석할 영어 지문을 붙여넣으세요:**")
            an_src_name = st.text_input(
                "출처 표기 (분석지 상단 표시용)",
                value="직접 입력 영어 지문",
                key="tab2_eng_an_direct_src",
            )
            an_passage_raw = st.text_area(
                "영어 지문 원문 입력 (여러 개의 지문을 한 번에 넣으려면 지문 사이에 '---' 줄을 넣어 구분하세요)",
                height=200,
                placeholder="Paste your English reading passage here...",
                key="tab2_eng_an_direct_text",
            )
            an_kor_raw = st.text_area(
                "한국어 해석 원문 붙여넣기 (선택 — 해설지 해석을 붙여넣으면 AI 없이 문장별 자동 매칭, 여러 지문은 '---' 구분)",
                height=140,
                placeholder="(선택) 해설지의 [해석] 본문을 그대로 붙여넣으세요...",
                key="tab2_eng_an_direct_kor",
            )
            d_opt_c1, d_opt_c2 = st.columns([1.0, 1.3])
            with d_opt_c1:
                include_eng_direct_diagrams = st.checkbox(
                    "🎨 지문 이해용 시각 도식 그림 포함",
                    value=True,
                    key="tab2_eng_an_direct_include_diagrams",
                )
            with d_opt_c2:
                force_regen_direct = st.checkbox(
                    "🔄 기존 캐시 무시하고 새로 분석지 생성",
                    value=False,
                    key="tab2_eng_an_direct_force_regen",
                )
            run_eng_direct_analysis_btn = st.form_submit_button(
                "📑 직접 입력한 영어 지문 분석지 PDF 생성",
                type="primary",
                width="stretch",
            )

        if run_eng_direct_analysis_btn:
            raw_chunks = [
                ch.strip()
                for ch in re.split(r"\n\s*-{3,}\s*\n", (an_passage_raw or "").strip())
                if ch.strip()
            ]
            kor_chunks = (
                [
                    ch.strip()
                    for ch in re.split(r"\n\s*-{3,}\s*\n", (an_kor_raw or "").strip())
                    if ch.strip()
                ]
                if (an_kor_raw or "").strip()
                else []
            )
            if not raw_chunks:
                st.error("분석지를 생성할 영어 지문 텍스트를 입력해 주세요.")
            else:
                base_src = (an_src_name or "직접 입력 영어 지문").strip()
                direct_an_items = [
                    {
                        "exam_name": f"{base_src} #{c_idx}" if len(raw_chunks) > 1 else base_src,
                        "q_label": f"{base_src} #{c_idx}" if len(raw_chunks) > 1 else base_src,
                        "orig_type": "직접 입력 지문",
                        "passage_topic": "",
                        "passage_text": chunk_txt,
                        "direct_korean_translation": kor_chunks[c_idx - 1] if (c_idx - 1) < len(kor_chunks) else "",
                    }
                    for c_idx, chunk_txt in enumerate(raw_chunks, start=1)
                ]
                _execute_english_analysis_jobs(
                    ctx,
                    items=direct_an_items,
                    include_diagrams=include_eng_direct_diagrams,
                    force_regenerate=force_regen_direct,
                )

    _render_english_analysis_results_section()
