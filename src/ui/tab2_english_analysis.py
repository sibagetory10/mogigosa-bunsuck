from __future__ import annotations

import concurrent.futures
import html as _html
import re
from datetime import datetime
from typing import Any

import streamlit as st

from src.auth import (
    TIER_LABELS,
    format_token_count,
    get_max_batch_questions,
    get_monthly_token_limit,
    get_remaining_monthly_tokens,
    get_user_by_username,
    resolve_active_gemini_api_key,
)
from src.config import EXAM_TAGGING_MODEL, FLASH_LITE_MODEL, MODEL_LABELS
from src.models import CURRENT_ENGLISH_ANALYSIS_VERSION, QuestionRecord
from src.text_formatter import normalize_english_korean_translation
from src.ui.common import _ensure_token_quota, _usage_scope
from src.ui.context import AppContext
from src.ui_styles import render_chips

_ENG_CIRCLED_NUMS = ("①", "②", "③", "④", "⑤")


def _execute_english_analysis_jobs(
    ctx: AppContext,
    items: list[dict[str, Any]],
    include_diagrams: bool = True,
    force_regenerate: bool = False,
) -> None:
    """선택된 영어 지문 목록에 대해 DB 캐시 조회/병렬 AI 분석/지문별 PDF 저장/합본 PDF 생성을 수행한다."""
    if not items:
        st.warning("분석지를 생성할 영어 지문을 최소 1개 이상 선택해 주세요.")
        return
    from src.gemini import (
        _needs_english_passage_restoration,
        generate_english_passage_analysis,
        generate_english_visual_diagram,
        submit_with_usage_context,
    )
    from src.pdf.common import merge_korean_analysis_pdfs
    from src.pdf.english_analysis_pdf import generate_english_passage_analysis_pdf

    prepared_slots: list[dict[str, Any]] = []
    missing_ai_indices: list[int] = []
    missing_diag_indices: list[int] = []

    for idx_s, it in enumerate(items):
        p_txt = str(it.get("passage_text") or "").strip()
        q_lbl = str(it.get("q_label") or "").strip()
        c_key = ctx.storage.build_english_analysis_key(p_txt, q_lbl)
        cached_an = None if force_regenerate else ctx.storage.get_cached_english_analysis(c_key)
        cached_pdf = (
            None
            if force_regenerate
            else (ctx.storage.get_english_analysis_pdf_bytes(c_key) if include_diagrams else None)
        )
        if cached_an is not None:
            is_stale_ver = int(getattr(cached_an, "analysis_version", 1) or 1) < CURRENT_ENGLISH_ANALYSIS_VERSION
            needs_legacy_restore = (
                not getattr(cached_an, "is_source_restored", False)
                and _needs_english_passage_restoration(p_txt, str(it.get("orig_type", "") or ""))
            )
            if is_stale_ver or needs_legacy_restore:
                # 구버전(v1: 원문 무손실 검증·기출 포인트 미적용) 캐시는 최신 엔진으로 자동 재분석
                cached_an = None
                cached_pdf = None
        if cached_an is not None and getattr(cached_an, "sentence_analyses", None):
            _dirty_kor = False
            for _sa in cached_an.sentence_analyses:
                _orig_kor = (getattr(_sa, "korean_translation", "") or "").strip()
                _norm_kor = normalize_english_korean_translation(_orig_kor)
                if _norm_kor != _orig_kor:
                    _sa.korean_translation = _norm_kor
                    _dirty_kor = True
            if _dirty_kor:
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
                f"영어 정밀 지문 분석지·도식 생성 {need_total}건 병렬 진행 중 "
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
                detail_notes.append(f"신규 AI 분석+PDF 저장 {ai_gen_cnt}건")
            st.success(
                f"📑 총 {len(analyses_list)}개 영어 지문 분석지 합본 PDF가 생성되었습니다! "
                f"({' · '.join(detail_notes)})"
            )
        except Exception as exc:
            st.error(f"영어 지문 분석지 PDF 병합 실패: {exc}")


def _render_english_analysis_results_section() -> None:
    """세션에 생성된 영어 지문 정밀 분석지 웹 미리보기 및 합본 PDF 다운로드 버튼을 렌더링한다."""
    saved_analyses: list[dict[str, Any]] = st.session_state.get("eng_analysis_results") or []
    saved_pdf: bytes | None = st.session_state.get("eng_analysis_pdf")
    if not (saved_analyses and saved_pdf):
        return
    from src.diagram_renderer import render_korean_diagram_png
    from src.gemini.english_validators import (
        _enforce_english_analysis_token_guard,
        _resolve_english_exam_point_title,
        _split_english_sentences,
    )
    from src.pdf.english_analysis_pdf import _format_chunked_english_html

    st.markdown("---")
    st.markdown("#### 📑 생성된 영어 정밀 지문 분석지 (주제 · 논리 흐름 · 도식 · 문장별 구문독해)")
    with st.container(border=True):
        dl_c1, dl_c2 = st.columns([2.2, 1.0])
        with dl_c1:
            st.write(
                f"총 **{len(saved_analyses)}개** 영어 지문 정밀 분석 완료 "
                "(핵심 주제 · 논리 전개 흐름 · 시각 도식 · 문장별 `/`·`[ ]`·`( )` 구문독해 · 주제문 ★ 표시)"
            )
        with dl_c2:
            st.download_button(
                "📥 영어 지문 분석지 PDF 다운로드",
                data=saved_pdf,
                file_name="영어_정밀_지문_분석지.pdf",
                mime="application/pdf",
                type="primary",
                width="stretch",
                key="eng_analysis_pdf_dl_btn",
            )

        for idx_a, an_item in enumerate(saved_analyses, start=1):
            an = an_item["analysis"]
            q_lbl = an_item.get("q_label") or an_item.get("exam_name") or "영어 지문"
            orig_t = an_item.get("orig_type") or "독해"
            src_p = str(an_item.get("passage_text") or "").strip()
            src_s = _split_english_sentences(src_p) if src_p else None
            if src_s and len(src_s) == len(getattr(an, "sentence_analyses", None) or []):
                an = _enforce_english_analysis_token_guard(an, src_s)
            else:
                an = _enforce_english_analysis_token_guard(an, None)

            with st.expander(
                f"{idx_a}. [{q_lbl}] {an.title} ({orig_t})",
                expanded=(idx_a == 1),
            ):
                st.info(f"**📌 핵심 주제:** {an.core_theme}")
                st.success(f"**🧭 논리 전개 흐름:**\n\n{an.structure_flow}")
                _ep_note = (getattr(an, "exam_point_note", "") or "").strip()
                if _ep_note:
                    _ep_title = (getattr(an, "exam_point_title", "") or "").strip() or _resolve_english_exam_point_title(
                        orig_t, _ep_note
                    )
                    st.warning(f"**🎯 {_ep_title}:**\n\n{_ep_note}")
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
                    st.markdown("**🔍 문장별 정밀 구문 분석** (`/` 끊어읽기 · `[절]` · `(수식어구)` · `★ 주제문` · `기출 포인트`)")
                    st.markdown(
                        "<style>"
                        ".eng-slash{color:#DC2626;font-weight:700;}"
                        ".eng-bracket{color:#1D4ED8;font-weight:700;}"
                        ".eng-paren{color:#0F766E;font-weight:700;}"
                        ".eng-blank-hl{background:#E0E7FF;border-bottom:2px solid #4F46E5;padding:0 2px;border-radius:3px;}"
                        "</style>",
                        unsafe_allow_html=True,
                    )
                    is_non_blank_t = any(
                        k in orig_t
                        for k in ("삽입", "순서", "무관", "어법", "어휘", "요약", "주제", "제목", "요지", "주장", "목적", "심경", "함축")
                    )
                    for s_i, sa in enumerate(sent_list):
                        s_num = int(getattr(sa, "sentence_num", s_i + 1) or (s_i + 1))
                        n_badge = _ENG_CIRCLED_NUMS[s_num - 1] if 1 <= s_num <= len(_ENG_CIRCLED_NUMS) else f"({s_num})"
                        is_topic = bool(getattr(sa, "is_topic_sentence", False))
                        raw_bp = (getattr(sa, "original_blank_phrase", "") or "").strip()
                        blank_phrase = "" if is_non_blank_t else raw_bp
                        is_inserted = bool(getattr(sa, "is_inserted_sentence", False)) and ("삽입" in orig_t or not orig_t)
                        badges: list[str] = []
                        if is_topic:
                            badges.append(
                                "<span style='background:#FEF3C7;color:#B45309;border:1px solid #F59E0B;"
                                "padding:1px 6px;border-radius:4px;font-size:0.8em;font-weight:700;'>"
                                "★ 주제문 (Topic Sentence)</span>"
                            )
                        if blank_phrase:
                            badges.append(
                                "<span style='background:#EEF2FF;color:#3730A3;border:1px solid #818CF8;"
                                "padding:1px 6px;border-radius:4px;font-size:0.8em;font-weight:700;'>"
                                "📌 기출 빈칸 원위치</span>"
                            )
                        if is_inserted:
                            badges.append(
                                "<span style='background:#ECFDF5;color:#065F46;border:1px solid #34D399;"
                                "padding:1px 6px;border-radius:4px;font-size:0.8em;font-weight:700;'>"
                                "📥 기출 삽입 문장</span>"
                            )
                        badges_html = "".join(badges)
                        border_style = (
                            "border:1.5px solid #F59E0B;border-left:4px solid #D97706;background:#FFFBEB;"
                            if is_topic
                            else "border:1px solid #E2E8F0;background:#FFFFFF;"
                        )
                        chunk_html = _format_chunked_english_html(
                            getattr(sa, "chunked_english", ""),
                            blank_phrase=blank_phrase,
                        )
                        kor_txt = _html.escape(normalize_english_korean_translation(getattr(sa, "korean_translation", "") or ""))
                        st.markdown(
                            f"<div style='{border_style}border-radius:6px;padding:10px 12px;margin-bottom:8px;'>"
                            f"<div style='display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:4px;'>"
                            f"<strong style='color:#1E3A8A;'>{n_badge}</strong>{badges_html}</div>"
                            f"<div style='font-weight:600;color:#0F172A;line-height:1.6;margin-bottom:4px;'>{chunk_html}</div>"
                            f"<div style='color:#334155;font-size:0.95em;line-height:1.5;'>{kor_txt}</div>"
                            "</div>",
                            unsafe_allow_html=True,
                        )
def _render_english_analysis_submode(ctx: AppContext) -> None:
    """영어 서브모드 2: 영어 지문 정밀 분석지 PDF 생성 전용 UI."""
    st.caption(
        "💡 선택한 영어 지문의 **핵심 주제 · 논리 전개 흐름 · 이해용 시각 도식 · 문장별 `/` 끊어읽기 및 `[절]`/`(수식어구)` 구문독해(주제문 ★ 표시)** 분석지 PDF를 생성합니다."
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
                            "🔄 기존 캐시 무시하고 최신 정밀 검증 엔진으로 재분석",
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
            st.markdown("**✍️ 정밀 분석할 영어 지문을 붙여넣으세요:**")
            an_src_name = st.text_input(
                "출처 표기 (분석지 상단 표시용)",
                value="직접 입력 영어 지문",
                key="tab2_eng_an_direct_src",
            )
            an_passage_raw = st.text_area(
                "영어 지문 원문 입력 (여러 개의 지문을 한 번에 넣으려면 지문 사이에 '---' 줄을 넣어 구분하세요)",
                height=230,
                placeholder="Paste your English reading passage here...",
                key="tab2_eng_an_direct_text",
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
                    "🔄 기존 캐시 무시하고 새로 정밀 분석",
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
