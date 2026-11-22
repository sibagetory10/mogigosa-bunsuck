from __future__ import annotations

import concurrent.futures
import hashlib
import html as _html
import re
from datetime import datetime
from typing import Any

import streamlit as st

from src.auth import (
    TIER_LABELS,
    consume_extraction_quota,
    format_token_count,
    get_max_batch_questions,
    get_monthly_token_limit,
    get_remaining_monthly_tokens,
    get_user_by_username,
    resolve_active_gemini_api_key,
)
from src.config import EXAM_TAGGING_MODEL, FLASH_LITE_MODEL, MODEL_LABELS
from src.diagram_renderer import render_korean_diagram_png
from src.gemini import (
    _needs_english_passage_restoration,
    generate_english_passage_analysis,
    generate_english_transformed_question,
    generate_english_visual_diagram,
    submit_with_usage_context,
)
from src.korean_pdf_generator import (
    _format_chunked_english_html,
    generate_english_passage_analysis_pdf,
    merge_korean_analysis_pdfs,
)
from src.models import (
    CURRENT_ENGLISH_ANALYSIS_VERSION,
    ENGLISH_DEFAULT_EXCLUDED_Q_NUMS,
    ENGLISH_TRANSFORM_TYPES,
    QuestionRecord,
    RewrittenQuestion,
    normalize_english_group_id,
)
from src.text_formatter import normalize_english_korean_translation
from src.ui.common import _ensure_token_quota, _usage_scope
from src.ui.context import AppContext
from src.ui_styles import render_chips


def _resolve_tab2_english_rewrite_model(key_suffix: str = "eng") -> tuple[str, bool]:
    """탭 2 영어 지문 변형 모델을 전 등급 동일하게 초경량 Gemini 3.5 Flash-Lite로 통일한다."""
    rw_model = EXAM_TAGGING_MODEL
    use_precision_mode = False
    st.session_state["active_rewrite_model"] = rw_model
    st.session_state["active_eng_precision_mode"] = False
    return rw_model, use_precision_mode


def _execute_english_transform_jobs(
    ctx: AppContext,
    jobs: list[dict[str, Any]],
    rewrite_model: str,
    source_summary_label: str,
    use_precision_mode: bool = False,
) -> None:
    """(영어 지문, 목표 수능 유형) 작업 목록을 Gemini 3.5 Flash-Lite 단일 모델 및 등급별 월간 토큰 한도 하에 실행한다."""
    if not jobs:
        st.warning("변형할 영어 지문과 목표 수능 유형을 최소 1개 이상 선택해 주세요.")
        return

    if not _ensure_token_quota(ctx):
        return

    user_max_jobs = get_max_batch_questions(ctx.current_user)
    if user_max_jobs is not None and len(jobs) > user_max_jobs:
        st.warning(
            f"⚠️ 현재 등급({TIER_LABELS.get(ctx.current_user.tier, ctx.current_user.tier)})은 1회 실행당 최대 **{user_max_jobs}문항**까지만 생성됩니다. "
            f"(요청 {len(jobs)}건 중 앞의 {user_max_jobs}문항만 실행)"
        )
        jobs = jobs[:user_max_jobs]

    rewrite_model = FLASH_LITE_MODEL
    use_precision_mode = False

    rewrite_api_key, rewrite_key_label = resolve_active_gemini_api_key(
        ctx.current_user, ctx.creds.gemini_api_key, model_name=rewrite_model
    )
    if not rewrite_api_key:
        st.error(f"영어 지문 변형용 Gemini API 키({rewrite_key_label})가 설정되지 않았습니다.")
        return

    total_jobs = len(jobs)
    prog = st.progress(0.0)
    stat = st.empty()
    n_workers = max(1, min(total_jobs, 8))
    stat.info(f"🔤 영어 지문 수능형 변형 {total_jobs}문항 병렬 출제 중 (최대 {n_workers}스레드)...")

    results_by_idx: dict[int, RewrittenQuestion] = {}
    errors_by_idx: dict[int, str] = {}

    with _usage_scope(ctx):
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_map = {
                submit_with_usage_context(
                    executor,
                    generate_english_transformed_question,
                    api_key=rewrite_api_key,
                    model_name=rewrite_model,
                    source_passage=job["passage_text"],
                    target_type=job["target_type"],
                    difficulty=job["difficulty"],
                    blank_unit=job["blank_unit"],
                    vocab_style=job["vocab_style"],
                    source_label=job["source_label"],
                    use_precision_mode=use_precision_mode,
                    storage=ctx.storage,
                    use_cache=True,
                    username=ctx.current_user.username if ctx.current_user is not None else "",
                ): idx
                for idx, job in enumerate(jobs)
            }
            done_cnt = 0
            for fut in concurrent.futures.as_completed(future_map):
                idx = future_map[fut]
                done_cnt += 1
                try:
                    results_by_idx[idx] = fut.result()
                except Exception as exc:
                    errors_by_idx[idx] = str(exc)
                prog.progress(done_cnt / total_jobs)
                stat.info(f"🔤 영어 지문 수능형 변형 출제 중 ({done_cnt}/{total_jobs})...")

    prog.progress(1.0)
    stat.empty()

    for idx, err_str in sorted(errors_by_idx.items()):
        j_info = jobs[idx]
        st.error(f"❌ [{j_info['source_label']} → {j_info['target_type']}] 변형 실패: {err_str}")

    assembled: list[dict[str, Any]] = []
    for idx, job in enumerate(jobs):
        rw_item = results_by_idx.get(idx)
        if rw_item is None:
            continue
        base_q: QuestionRecord = job["base_question"]
        unique_q = base_q.model_copy(
            update={
                "id": f"{base_q.id}__eng_{idx + 1:02d}",
                "subject_category": "영어",
                "curriculum_subject": "영어",
                "unit_name": rw_item.target_question_type or job["target_type"],
            }
        )
        assembled.append(
            {
                "question": unique_q,
                "judgment": None,
                "crop_png_bytes": None,
                "rewritten": rw_item,
                "concept_reference": f"📚 {job['source_label']}",
                "english_meta": {
                    "source_passage": job["passage_text"],
                    "target_type": job["target_type"],
                    "difficulty": job["difficulty"],
                    "blank_unit": job["blank_unit"],
                    "vocab_style": job["vocab_style"],
                    "source_label": job["source_label"],
                    "use_precision_mode": use_precision_mode,
                    "rewrite_model": rewrite_model,
                },
            }
        )

    if not assembled:
        st.warning("생성된 영어 변형 문항이 없습니다.")
        return

    consume_extraction_quota(ctx.storage, ctx.current_user.username, len(assembled))

    st.session_state["matched_items"] = assembled
    st.session_state["match_diagnostics"] = []
    st.session_state["concept_source_label"] = source_summary_label
    st.session_state["generated_exam_pdf"] = None
    st.session_state["user_answers"] = {}
    st.session_state["graded"] = False
    st.session_state["ai_explanations"] = {}
    st.session_state["ai_chats"] = {}

    rem_msg = ""
    fresh_u = get_user_by_username(ctx.storage, ctx.current_user.username)
    if fresh_u is not None:
        m_lim = get_monthly_token_limit(fresh_u)
        m_rem = get_remaining_monthly_tokens(fresh_u)
        if m_lim is not None and m_rem is not None:
            rem_msg = f" | 이번 달 잔여 토큰: {m_rem:,} / {format_token_count(m_lim)}"

    st.success(f"🎯 총 **{len(assembled)}문항**의 수능형 영어 변형 문제 출제가 완료되었습니다! 아래 모드 A(시험지 PDF) 또는 모드 B(웹 바로 풀기)에서 확인하세요.{rem_msg}")


_ENG_CIRCLED_NUMS = (
    "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩",
    "⑪", "⑫", "⑬", "⑭", "⑮", "⑯", "⑰", "⑱", "⑲", "⑳",
)


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
            with st.expander(
                f"{idx_a}. [{q_lbl}] {an.title} ({orig_t})",
                expanded=(idx_a == 1),
            ):
                st.info(f"**📌 핵심 주제:** {an.core_theme}")
                st.success(f"**🧭 논리 전개 흐름:**\n\n{an.structure_flow}")
                _ep_note = (getattr(an, "exam_point_note", "") or "").strip()
                if _ep_note:
                    st.warning(f"**🎯 기출 포인트 (빈칸 원위치 · 무관한 문장):**\n\n{_ep_note}")
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
                    st.markdown("**🔍 문장별 정밀 구문 분석** (`/` 끊어읽기 · `[절]` · `(수식어구)` · `★ 주제문` · `기출 빈칸`)")
                    st.markdown(
                        "<style>"
                        ".eng-slash{color:#DC2626;font-weight:700;}"
                        ".eng-bracket{color:#1D4ED8;font-weight:700;}"
                        ".eng-paren{color:#0F766E;font-weight:700;}"
                        ".eng-blank-hl{background:#E0E7FF;border-bottom:2px solid #4F46E5;padding:0 2px;border-radius:3px;}"
                        "</style>",
                        unsafe_allow_html=True,
                    )
                    for s_i, sa in enumerate(sent_list):
                        s_num = int(getattr(sa, "sentence_num", s_i + 1) or (s_i + 1))
                        n_badge = _ENG_CIRCLED_NUMS[s_num - 1] if 1 <= s_num <= len(_ENG_CIRCLED_NUMS) else f"({s_num})"
                        is_topic = bool(getattr(sa, "is_topic_sentence", False))
                        blank_phrase = (getattr(sa, "original_blank_phrase", "") or "").strip()
                        topic_html = (
                            " <span style='background:#FEF3C7;color:#B45309;border:1px solid #F59E0B;"
                            "padding:1px 6px;border-radius:4px;font-size:0.8em;font-weight:700;'>"
                            "★ 주제문 (Topic Sentence)</span>"
                            if is_topic
                            else ""
                        )
                        blank_html = (
                            " <span style='background:#EEF2FF;color:#3730A3;border:1px solid #818CF8;"
                            "padding:1px 6px;border-radius:4px;font-size:0.8em;font-weight:700;'>"
                            "📌 기출 빈칸 원위치</span>"
                            if blank_phrase
                            else ""
                        )
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
                            f"<div style='margin-bottom:4px;'><strong style='color:#1E3A8A;'>{n_badge}</strong>{topic_html}{blank_html}</div>"
                            f"<div style='font-weight:600;color:#0F172A;line-height:1.6;margin-bottom:4px;'>{chunk_html}</div>"
                            f"<div style='color:#334155;font-size:0.95em;line-height:1.5;'>{kor_txt}</div>"
                            "</div>",
                            unsafe_allow_html=True,
                        )


def _filter_english_db_passages(
    ctx: AppContext,
    key_prefix: str,
) -> tuple[bool, list[tuple[QuestionRecord, str]]]:
    """기출 DB 영어 독해 지문을 조회하고 회차·원래 유형·키워드·제외문항 필터 UI를 렌더링한다."""
    all_db = ctx.storage.load_db()
    eng_reading_recs = [
        r
        for r in all_db
        if r.subject_category == "영어"
        and (getattr(r, "english_section", "") == "독해" or r.q_num >= 18)
        and bool((getattr(r, "passage_text", "") or "").strip())
    ]

    if not eng_reading_recs:
        st.info(
            "아직 DB에 저장된 **영어 독해(18~45번) 지문**이 없습니다. "
            "**[📥 탭 1]**에서 영어 모의고사 문제지+해설지 PDF를 먼저 수집·추출하거나, "
            "우측 **[✍️ 새 영어 지문 직접 붙여넣기]** 탭을 이용해 주세요."
        )
        return False, []

    f_c1, f_c2, f_c3 = st.columns([1.2, 1.1, 1.3])
    avail_eng_exams = sorted({r.exam_name for r in eng_reading_recs}, reverse=True)
    avail_eng_types = sorted(
        {
            (getattr(r, "english_question_type", "") or r.unit_name).strip()
            for r in eng_reading_recs
            if (getattr(r, "english_question_type", "") or r.unit_name).strip()
        }
    )
    with f_c1:
        sel_eng_exam = st.selectbox(
            "모의고사 회차 필터",
            options=["전체"] + avail_eng_exams,
            key=f"{key_prefix}_filter_exam",
        )
    with f_c2:
        sel_eng_orig_type = st.selectbox(
            "원래 문항 유형 필터",
            options=["전체"] + avail_eng_types,
            key=f"{key_prefix}_filter_type",
        )
    with f_c3:
        eng_kw = st.text_input(
            "주제·어휘·지문 키워드 검색",
            placeholder="예: artificial intelligence, climate, 감정...",
            key=f"{key_prefix}_filter_kw",
        )

    include_excluded_q = st.checkbox(
        "목적·심경·도표·내용일치·실용문·장문(18, 19, 25, 26, 27, 28, 43, 44, 45번) 지문도 목록에 포함",
        value=False,
        key=f"{key_prefix}_include_excluded_q",
    )

    # 필터링 및 세트 지문(41~42, 43~45) 중복 묶음 처리
    filtered_passages: list[tuple[QuestionRecord, str]] = []
    seen_group_keys: set[tuple[str, str]] = set()
    kw_low = (eng_kw or "").strip().lower()

    for r in sorted(eng_reading_recs, key=lambda x: (x.exam_name, x.q_num)):
        if not include_excluded_q and r.q_num in ENGLISH_DEFAULT_EXCLUDED_Q_NUMS:
            continue
        if sel_eng_exam != "전체" and r.exam_name != sel_eng_exam:
            continue
        r_orig_t = (getattr(r, "english_question_type", "") or r.unit_name).strip()
        if sel_eng_orig_type != "전체" and r_orig_t != sel_eng_orig_type:
            continue
        if kw_low:
            blob = " ".join(
                [
                    r.exam_name,
                    r_orig_t,
                    getattr(r, "passage_topic", ""),
                    " ".join(getattr(r, "key_vocabulary", []) or []),
                    getattr(r, "passage_text", ""),
                ]
            ).lower()
            if kw_low not in blob:
                continue

        gid = normalize_english_group_id(r.q_num)
        if gid:
            g_key = (r.exam_name, gid)
            if g_key in seen_group_keys:
                continue
            seen_group_keys.add(g_key)
            q_label = f"{r.exam_name} {gid}번"
        else:
            q_label = f"{r.exam_name} {r.q_num}번"
        filtered_passages.append((r, q_label))

    return True, filtered_passages


def _render_english_passage_preview_expander(filtered_passages: list[tuple[QuestionRecord, str]]) -> None:
    """조회된 영어 기출 지문 원문·어휘 미리보기 Expander를 렌더링한다."""
    with st.expander("📖 조회된 기출 영어 지문 원문·어휘 미리보기", expanded=False):
        for rec_p, q_label in filtered_passages[:30]:
            r_orig_t = (getattr(rec_p, "english_question_type", "") or rec_p.unit_name).strip()
            r_topic = (getattr(rec_p, "passage_topic", "") or "-").strip()
            st.markdown(f"**📌 {q_label}** (`{r_orig_t}` · 주제: {r_topic})")
            if getattr(rec_p, "key_vocabulary", None):
                render_chips(rec_p.key_vocabulary, variant="amber", limit=10)
            st.markdown(
                f"<div class='eng-given-box' style='white-space:pre-wrap;'>{_html.escape(rec_p.passage_text)}</div>",
                unsafe_allow_html=True,
            )


def _render_english_transform_submode(ctx: AppContext) -> None:
    """영어 서브모드 1: 수능 독해 11개 유형 변형문제 생성 UI."""
    rewrite_model, use_precision_mode = _resolve_tab2_english_rewrite_model(key_suffix="eng")
    user_max_jobs = get_max_batch_questions(ctx.current_user)
    max_limit_badge = f"1회 최대 **{user_max_jobs}문항**" if user_max_jobs is not None else "1회 문항 수 **제한 없음**"

    eng_tab_db, eng_tab_direct = st.tabs(
        [
            "📂 기출 DB 영어 지문에서 선택 (회차·원래 유형별 검색)",
            "✍️ 새 영어 지문 직접 붙여넣기",
        ]
    )

    with eng_tab_db:
        has_db_recs, filtered_passages = _filter_english_db_passages(ctx, key_prefix="tab2_eng_tr")
        if has_db_recs:
            assign_mode = st.radio(
                "목표 변형 유형 지정 방식",
                options=[
                    "⚡ 일괄 유형 적용 (선택 지문 전체에 동일 유형 적용)",
                    "🎯 지문별 개별 유형 지정 (지문마다 원하는 유형 복수 선택)",
                ],
                horizontal=True,
                key="tab2_eng_assign_mode",
            )
            is_per_passage_mode = assign_mode.startswith("🎯")

            if not filtered_passages:
                st.warning("조건에 맞는 영어 독해 지문이 없습니다. 필터를 완화하거나 상단 제외 문항 포함 체크박스를 켜 보세요.")
            else:
                st.caption(
                    f"총 **{len(filtered_passages)}개** 영어 독해 지문이 조회되었습니다. "
                    f"(과금 방지를 위해 기본 미선택 상태이며, 현재 등급은 {max_limit_badge} 생성 가능합니다.)"
                )
                with st.form("tab2_eng_db_transform_form", border=True):
                    cap_txt = f"최대 {user_max_jobs}건" if user_max_jobs is not None else "문항 수 제한 없음"
                    st.caption(
                        f"📊 예상 AI 호출: N건 ({cap_txt}) — "
                        "변형하려는 지문만 직접 체크해 주세요 (전체 선택 비활성화)."
                    )
                    cfg_c1, cfg_c2, cfg_c3 = st.columns(3)
                    with cfg_c1:
                        eng_difficulty = st.selectbox(
                            "변형 목표 난이도",
                            options=["고3 평가원·수능 수준", "고1·고2 학평 수준"],
                            index=0,
                            key="tab2_eng_db_diff",
                        )
                    with cfg_c2:
                        eng_blank_unit = st.selectbox(
                            "빈칸 추론 선택 시 빈칸 단위",
                            options=["구 (3~6단어)", "단어 (1~2단어)", "절 (주어+동사 포함)"],
                            index=0,
                            key="tab2_eng_db_blank_unit",
                        )
                    with cfg_c3:
                        eng_vocab_style = st.selectbox(
                            "문맥상 어휘 선택 시 출제 형식",
                            options=["밑줄형 (①~⑤)", "선택형 ((A)(B)(C) 네모)"],
                            index=0,
                            key="tab2_eng_db_vocab_style",
                        )

                    bulk_target_types: list[str] = []
                    if not is_per_passage_mode:
                        bulk_target_types = st.multiselect(
                            "🎯 선택한 모든 지문에 적용할 목표 수능 독해 유형 (1개~여러 개 복수 선택 가능)",
                            options=list(ENGLISH_TRANSFORM_TYPES),
                            default=["빈칸 추론"],
                            key="tab2_eng_bulk_types",
                        )

                    st.markdown("---")
                    hdr_limit = f"1회 최대 {user_max_jobs}문항" if user_max_jobs is not None else "원하는 만큼 선택 가능"
                    st.markdown(f"**✅ 변형할 영어 기출 지문 선택 (모두 기본 미선택 · {hdr_limit}):**")

                    checked_rows: list[tuple[QuestionRecord, str, list[str]]] = []
                    for p_idx, (rec_p, q_label) in enumerate(filtered_passages):
                        r_orig_t = (getattr(rec_p, "english_question_type", "") or rec_p.unit_name).strip()
                        r_topic = (getattr(rec_p, "passage_topic", "") or "").strip()
                        p_preview = re.sub(r"\s+", " ", (rec_p.passage_text or "").strip())[:115]
                        row_label = (
                            f"[{q_label}] 원래 유형: {r_orig_t}"
                            + (f" · 주제: {r_topic}" if r_topic else "")
                            + f" — \"{p_preview}...\""
                        )
                        if is_per_passage_mode:
                            rc1, rc2 = st.columns([1.8, 1.2])
                            with rc1:
                                is_chk = st.checkbox(
                                    row_label,
                                    value=False,
                                    key=f"tab2_eng_p_chk_{rec_p.id}_{p_idx}",
                                )
                            with rc2:
                                row_types = st.multiselect(
                                    f"목표 유형 ({q_label})",
                                    options=list(ENGLISH_TRANSFORM_TYPES),
                                    default=["빈칸 추론"],
                                    key=f"tab2_eng_p_types_{rec_p.id}_{p_idx}",
                                    label_visibility="collapsed",
                                )
                        else:
                            is_chk = st.checkbox(
                                row_label,
                                value=False,
                                key=f"tab2_eng_p_chk_{rec_p.id}_{p_idx}",
                            )
                            row_types = bulk_target_types

                        if is_chk:
                            checked_rows.append((rec_p, q_label, row_types))

                    est_db_calls = sum(
                        len(r_types if is_per_passage_mode else bulk_target_types)
                        for _, _, r_types in checked_rows
                    )
                    if est_db_calls > 0:
                        eff_db_calls = min(est_db_calls, user_max_jobs) if user_max_jobs is not None else est_db_calls
                        lim_note = f"(최대 {user_max_jobs}건)" if user_max_jobs is not None else "(제한 없음)"
                        st.info(
                            f"📊 **예상 AI 호출: {eff_db_calls}건 {lim_note}** · 요청 {est_db_calls}건 "
                            "(상세 구문/오답 해설은 [✨ AI 자동 해설] 클릭 시에만 지연 호출)"
                        )
                    else:
                        lim_note = f"최대 {user_max_jobs}건" if user_max_jobs is not None else "제한 없음"
                        st.info(
                            f"📊 **예상 AI 호출: N건 ({lim_note})** · "
                            "체크한 지문 × 선택 유형 수만큼 호출됩니다."
                        )

                    run_eng_db_btn = st.form_submit_button(
                        "🚀 선택한 기출 영어 지문 수능형 변형 문제 생성",
                        type="primary",
                        width="stretch",
                    )

                _render_english_passage_preview_expander(filtered_passages)

                if run_eng_db_btn:
                    if not checked_rows:
                        st.error("변형할 영어 지문을 최소 1개 이상 체크해 주세요.")
                    else:
                        jobs: list[dict[str, Any]] = []
                        for rec_p, q_label, r_types in checked_rows:
                            eff_types = r_types if is_per_passage_mode else bulk_target_types
                            for t_type in eff_types:
                                jobs.append(
                                    {
                                        "base_question": rec_p,
                                        "passage_text": rec_p.passage_text,
                                        "target_type": t_type,
                                        "difficulty": eng_difficulty,
                                        "blank_unit": eng_blank_unit,
                                        "vocab_style": eng_vocab_style,
                                        "source_label": f"출처: {q_label}",
                                    }
                                )
                        if not jobs:
                            st.error("적용할 목표 수능 변형 유형을 최소 1개 이상 선택해 주세요.")
                        else:
                            if user_max_jobs is not None and len(jobs) > user_max_jobs:
                                st.warning(
                                    f"⚠️ 현재 등급은 1회 실행당 최대 {user_max_jobs}문항까지만 생성됩니다. "
                                    f"(선택 {len(jobs)}건 중 앞의 {user_max_jobs}문항만 실행)"
                                )
                                jobs = jobs[:user_max_jobs]
                            used_exams = sorted({j["base_question"].exam_name for j in jobs})
                            summary_lbl = (
                                f"{used_exams[0]} 외 {len(used_exams) - 1}개 회차 ({len(jobs)}문항)"
                                if len(used_exams) > 1
                                else f"{used_exams[0]} ({len(jobs)}문항)"
                            )
                            _execute_english_transform_jobs(
                                ctx,
                                jobs=jobs,
                                rewrite_model=rewrite_model,
                                source_summary_label=summary_lbl,
                                use_precision_mode=use_precision_mode,
                            )

    with eng_tab_direct:
        with st.form("tab2_eng_direct_transform_form", border=True):
            st.markdown("**✍️ 변형할 새 영어 지문을 붙여넣고 원하는 수능 독해 유형을 선택하세요:**")
            d_c1, d_c2 = st.columns([1.2, 1.8])
            with d_c1:
                direct_src_name = st.text_input(
                    "출처 표기 (시험지·해설 표시용)",
                    value="직접 입력 영어 지문",
                    key="tab2_eng_direct_src",
                )
            with d_c2:
                direct_types = st.multiselect(
                    "🎯 생성할 수능 독해 변형 유형 (복수 선택 시 해당 유형들을 한 번에 생성)",
                    options=list(ENGLISH_TRANSFORM_TYPES),
                    default=["빈칸 추론"],
                    key="tab2_eng_direct_types",
                )

            d_opt1, d_opt2, d_opt3 = st.columns(3)
            with d_opt1:
                d_difficulty = st.selectbox(
                    "변형 목표 난이도",
                    options=["고3 평가원·수능 수준", "고1·고2 학평 수준"],
                    index=0,
                    key="tab2_eng_direct_diff",
                )
            with d_opt2:
                d_blank_unit = st.selectbox(
                    "빈칸 추론 선택 시 빈칸 단위",
                    options=["구 (3~6단어)", "단어 (1~2단어)", "절 (주어+동사 포함)"],
                    index=0,
                    key="tab2_eng_direct_blank_unit",
                )
            with d_opt3:
                d_vocab_style = st.selectbox(
                    "문맥상 어휘 선택 시 출제 형식",
                    options=["밑줄형 (①~⑤)", "선택형 ((A)(B)(C) 네모)"],
                    index=0,
                    key="tab2_eng_direct_vocab_style",
                )

            direct_passage_raw = st.text_area(
                "영어 지문 원문 입력 (여러 개의 지문을 한 번에 넣으려면 지문 사이에 '---' 줄을 넣어 구분하세요)",
                height=210,
                placeholder="Paste your English reading passage here...",
                key="tab2_eng_direct_text",
            )
            direct_chunk_cnt = len(
                [ch for ch in re.split(r"\n\s*-{3,}\s*\n", (direct_passage_raw or "").strip()) if ch.strip()]
            ) or 1
            raw_direct_calls = direct_chunk_cnt * len(direct_types)
            est_direct_calls = min(raw_direct_calls, user_max_jobs) if user_max_jobs is not None else raw_direct_calls
            lim_note_d = f"(최대 {user_max_jobs}건)" if user_max_jobs is not None else "(제한 없음)"
            st.info(
                f"📊 **예상 AI 호출: {est_direct_calls}건 {lim_note_d}** · "
                f"선택 유형 {len(direct_types)}개 × 지문 수"
            )
            run_eng_direct_btn = st.form_submit_button(
                "🚀 직접 입력한 영어 지문으로 수능형 변형 문제 생성",
                type="primary",
                width="stretch",
            )

        if run_eng_direct_btn:
            raw_chunks = [
                ch.strip()
                for ch in re.split(r"\n\s*-{3,}\s*\n", (direct_passage_raw or "").strip())
                if ch.strip()
            ]
            if not raw_chunks:
                st.error("변형할 영어 지문 텍스트를 입력해 주세요.")
            elif not direct_types:
                st.error("생성할 목표 수능 독해 유형을 최소 1개 이상 선택해 주세요.")
            else:
                base_src = (direct_src_name or "직접 입력 영어 지문").strip()
                direct_jobs: list[dict[str, Any]] = []
                now_str = datetime.now().isoformat(timespec="seconds")
                for c_idx, chunk_txt in enumerate(raw_chunks, start=1):
                    src_title = f"{base_src} #{c_idx}" if len(raw_chunks) > 1 else base_src
                    virtual_q = QuestionRecord(
                        id=f"direct_eng_{hashlib.md5(chunk_txt.encode('utf-8')).hexdigest()[:8]}_{c_idx}",
                        exam_name=src_title,
                        q_num=c_idx,
                        subject_category="영어",
                        curriculum_subject="영어",
                        unit_name="직접 입력 지문",
                        min_grade=1,
                        core_principles=["영어 독해 변형"],
                        required_terms=[],
                        is_single_concept=True,
                        answer="1",
                        official_explanation="",
                        question_text=chunk_txt,
                        english_section="독해",
                        english_question_type="직접 입력",
                        passage_text=chunk_txt,
                        created_at=now_str,
                    )
                    for t_type in direct_types:
                        direct_jobs.append(
                            {
                                "base_question": virtual_q,
                                "passage_text": chunk_txt,
                                "target_type": t_type,
                                "difficulty": d_difficulty,
                                "blank_unit": d_blank_unit,
                                "vocab_style": d_vocab_style,
                                "source_label": f"출처: {src_title}",
                            }
                        )
                if user_max_jobs is not None and len(direct_jobs) > user_max_jobs:
                    st.warning(
                        f"⚠️ 현재 등급은 1회 실행당 최대 {user_max_jobs}문항까지만 생성됩니다. "
                        f"(요청 {len(direct_jobs)}건 중 앞의 {user_max_jobs}문항만 실행)"
                    )
                    direct_jobs = direct_jobs[:user_max_jobs]
                _execute_english_transform_jobs(
                    ctx,
                    jobs=direct_jobs,
                    rewrite_model=rewrite_model,
                    source_summary_label=base_src,
                    use_precision_mode=use_precision_mode,
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


def _render_tab2_english_transform_section(ctx: AppContext) -> None:
    """탭 2 영어 과목 전용: [변형문제 생성]과 [영어 지문 분석지 PDF 생성] 작업 모드로 분리해 렌더링한다."""
    st.subheader("1️⃣ 영어 작업 모드 선택")
    eng_sub_mode = st.radio(
        "수행할 영어 작업을 선택하세요",
        options=[
            "🚀 수능 독해 유형 변형문제 생성",
            "📑 영어 지문 정밀 분석지 PDF 생성",
        ],
        horizontal=True,
        key="tab2_eng_sub_mode",
    )
    if eng_sub_mode.startswith("📑"):
        _render_english_analysis_submode(ctx)
    else:
        _render_english_transform_submode(ctx)


render_tab2_english_transform_section = _render_tab2_english_transform_section
