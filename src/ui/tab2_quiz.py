from __future__ import annotations

import html as _html
from typing import Any

import streamlit as st

from src.auth import (
    CREDIT_COST_CONCEPT_REWRITE,
    CREDIT_COST_ENGLISH_VARIANT,
    MAX_CHAT_TURNS_PER_QUESTION,
    consume_daily_free_ai_quota,
    refund_daily_free_ai_quota,
    resolve_active_gemini_api_key,
)
from src.config import EXAM_TAGGING_MODEL
from src.gemini.usage import ForbiddenTermRemainingError
from src.models import (
    ConceptWhitelist,
    QuestionRecord,
    RewrittenQuestion,
    resolve_question_concept_reference,
)
from src.text_formatter import format_chemical_scripts, format_explanation_text, has_visual_reference
from src.ui.common import (
    CIRCLED_MAP,
    _auto_save_current_exam,
    _infer_exam_subject,
    _record_english_seen_for_user,
    _render_rewritten_card_html,
    _usage_scope,
    is_multiple_choice_question,
    normalize_answer,
    settle_credits,
    try_reserve_credits,
)
from src.ui.context import AppContext
from src.ui_styles import render_chips


@st.fragment
def _render_mode_b_solve_fragment(
    ctx: AppContext,
    matched_items: list[dict[str, Any]],
    whitelist: ConceptWhitelist | None,
    active_gemini_key: str,
    active_key_label: str,
    selected_model: str,
    rewrite_model: str | None = None,
) -> None:
    """모드 B(웹에서 바로 풀기 & 자동 채점 + AI 자동 해설/채팅 + 문항별 개별 재변환) 전용 프래그먼트."""
    st.markdown("#### ✏️ 실전 풀이 & 자동 채점 + 개념서 기반 AI 자동 해설")

    # 채점 완료 시 상단 성적표 카드 UI
    if st.session_state.get("graded"):
        correct_cnt = 0
        ox_pills: list[str] = []
        for idx_g, item in enumerate(matched_items, start=1):
            q: QuestionRecord = item["question"]
            rw: RewrittenQuestion | None = item.get("rewritten")
            truth = rw.preserved_answer if rw else q.answer
            user_val = st.session_state["user_answers"].get(q.id, "")
            is_ok = normalize_answer(user_val) == normalize_answer(truth) and normalize_answer(user_val) != ""
            if is_ok:
                correct_cnt += 1
                ox_pills.append(f"<span class='score-ox-pill score-ox-ok'>Q{idx_g:02d} ⭕</span>")
            else:
                ox_pills.append(f"<span class='score-ox-pill score-ox-no'>Q{idx_g:02d} ❌</span>")
        total_q = len(matched_items)
        score_pct = round((correct_cnt / max(1, total_q)) * 100, 1)
        st.markdown(
            f"""
            <div class="score-report-card">
              <div class="score-report-top">
                <div class="score-big">📊 채점 성적표: {score_pct}점</div>
                <div style="font-size:0.95rem; font-weight:700; color:#334155;">
                  총 {total_q}문항 중 <span style="color:#16A34A;">{correct_cnt}문항 정답</span> ·
                  <span style="color:#DC2626;">{total_q - correct_cnt}문항 오답</span>
                </div>
              </div>
              <div class="score-bar-track"><div class="score-bar-fill" style="width:{score_pct}%;"></div></div>
              <div class="score-ox-row">{''.join(ox_pills)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for seq_num, item in enumerate(matched_items, start=1):
        q = item["question"]
        j = item["judgment"]
        crop_b = item.get("crop_png_bytes")
        rw = item.get("rewritten")
        truth_ans = rw.preserved_answer if rw else q.answer
        concept_ref_str = (
            item.get("concept_reference")
            or resolve_question_concept_reference(
                q, j, whitelist, fallback_source_label=st.session_state.get("concept_source_label", "")
            )
        )

        with st.container(border=True):
            is_eng_item = (
                q.subject_category == "영어"
                or getattr(rw, "subject_domain", "") == "영어"
                or bool(item.get("english_meta"))
            )
            eng_target_type = (
                (getattr(rw, "target_question_type", "") if rw is not None else "")
                or (item.get("english_meta") or {}).get("target_type", "")
                or getattr(q, "english_question_type", "")
                or q.unit_name
            )
            eng_source_caption = (
                f"출처: {q.exam_name}"
                if "직접 입력" in (q.exam_name or "")
                else f"출처: {q.exam_name} {q.q_num}번"
            )

            # 카드 상단 헤더 + 번호 배지
            hdr_col1, hdr_col2 = st.columns([3.2, 1])
            with hdr_col1:
                header_title = f"영어 · {eng_target_type}" if is_eng_item else f"{q.curriculum_subject} · {q.unit_name}"
                st.markdown(
                    f"""
                    <div class="q-badge-row">
                      <span class="q-num-pill">Q{seq_num:02d}</span>
                      <span style="font-size:0.9rem; font-weight:700; color:#1E293B;">
                        {_html.escape(header_title)}
                      </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if is_eng_item:
                    st.caption(eng_source_caption)
                else:
                    st.caption(
                        f"출처: {q.exam_name} {q.q_num}번 · 본질 원리: {', '.join(q.core_principles)}"
                    )
            with hdr_col2:
                if rw is not None:
                    badge_txt = (
                        f"✨ 수능형 변형 · {_html.escape(eng_target_type)}"
                        if is_eng_item
                        else "✨ 개념서 기준 개조 문항"
                    )
                    st.markdown(
                        f"<div style='text-align:right;'><span class='chip chip-blue'>{badge_txt}</span></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div style='text-align:right;'><span class='chip chip-green'>📌 원문 그대로 출제</span></div>",
                        unsafe_allow_html=True,
                    )

            # 문제 본문 표시
            if rw is not None:
                st.markdown(_render_rewritten_card_html(rw), unsafe_allow_html=True)

                if not is_eng_item:
                    vis_term_note = (getattr(rw, "visual_term_note", "") or "").strip()
                    needs_visual_display = bool(crop_b) and (
                        has_visual_reference(rw.rewritten_stem)
                        or has_visual_reference(" ".join(rw.rewritten_choices or []))
                        or has_visual_reference(getattr(q, "question_text", ""))
                        or bool(vis_term_note)
                    )

                    if needs_visual_display and crop_b:
                        st.image(
                            crop_b,
                            caption=f"📊 원본 그림·도표 참고 ({q.exam_name} {q.q_num}번)",
                            width="stretch",
                        )
                    if vis_term_note:
                        st.markdown(
                            f"<div class='visual-term-note'>📌 <b>[그림 용어 안내]</b> {_html.escape(format_chemical_scripts(vis_term_note))}</div>",
                            unsafe_allow_html=True,
                        )

                    if rw.replaced_terms_map:
                        render_chips(rw.replaced_terms_map, variant="purple")

                    exp_title = (
                        "🔍 개조 검증 & 치환 내역 보기"
                        if needs_visual_display
                        else "🖼️ 원본 문제 이미지 & 개조 검증 내역 보기"
                    )
                    with st.expander(exp_title, expanded=False):
                        if crop_b and not needs_visual_display:
                            st.image(crop_b, caption=f"원본 ({q.exam_name} {q.q_num}번)", width="stretch")
                        st.info(f"**자가 검증:** {rw.self_verification_note}")
                else:
                    if rw.replaced_terms_map:
                        render_chips(rw.replaced_terms_map, variant="purple")
            else:
                if crop_b and not is_eng_item:
                    st.image(crop_b, width="stretch")
                else:
                    st.warning("문항 이미지를 불러올 수 없습니다.")

            # 문항 카드 하단: 참고 개념서 단원/페이지 또는 영어 기출 출처 표기
            st.markdown(
                f"<div style='background:#EFF6FF; border-left:4px solid #2563EB; padding:8px 12px; "
                f"border-radius:6px; margin:8px 0; font-size:0.9rem; font-weight:600; color:#1E3A8A;'>"
                f"{_html.escape(concept_ref_str)}</div>",
                unsafe_allow_html=True,
            )

            # 답안 입력 UI (객관식 1~5 터치 버튼 또는 주관식 입력)
            st.markdown("**내 답안 선택 / 입력:**")
            current_ans = st.session_state["user_answers"].get(q.id, "")
            is_mc = True if is_eng_item else is_multiple_choice_question(q, rw)

            if is_mc:
                st.markdown("<div class='omr-row'>", unsafe_allow_html=True)
                omr_cols = st.columns(5)
                for opt_num in ("1", "2", "3", "4", "5"):
                    with omr_cols[int(opt_num) - 1]:
                        selected = normalize_answer(current_ans) == opt_num
                        if st.button(
                            f"{CIRCLED_MAP[opt_num]} ({opt_num}번)",
                            key=f"omr_{q.id}_{opt_num}",
                            type="primary" if selected else "secondary",
                            width="stretch",
                        ):
                            st.session_state["user_answers"][q.id] = opt_num
                            st.rerun(scope="fragment")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                typed = st.text_input(
                    "주관식 답안 입력",
                    value=current_ans,
                    key=f"short_{q.id}",
                    placeholder="정답을 입력하세요",
                )
                st.session_state["user_answers"][q.id] = typed

            # 채점 상태 표시
            if st.session_state.get("graded"):
                is_correct = (
                    normalize_answer(st.session_state["user_answers"].get(q.id, ""))
                    == normalize_answer(truth_ans)
                    and normalize_answer(st.session_state["user_answers"].get(q.id, "")) != ""
                )
                disp_truth = CIRCLED_MAP.get(normalize_answer(truth_ans), truth_ans)
                disp_user = CIRCLED_MAP.get(normalize_answer(current_ans), current_ans) or "미입력"
                if is_correct:
                    st.success(f"⭕ **정답입니다!** (내 답: {disp_user} / 정답: {disp_truth})")
                else:
                    st.error(f"❌ **오답입니다.** (내 답: {disp_user} / 정답: {disp_truth})")

                with st.expander("📖 빠른 정답 해설 보기", expanded=not is_correct):
                    if rw is not None:
                        st.markdown(format_explanation_text(rw.pdf_grounded_explanation, for_markdown=True))
                    else:
                        st.markdown(format_explanation_text(q.official_explanation, for_markdown=True))
                    if is_eng_item:
                        st.caption(
                            "💡 토큰 절약을 위해 생성 단계에서는 1~2줄 핵심 정답 근거만 수록됩니다. "
                            "구문 분석 및 ①~⑤ 오답 분석은 아래 [✨ AI 자동 해설] 버튼을 클릭해 생성하세요."
                        )
                    st.caption(concept_ref_str)

            # ✨ AI 자동 해설 버튼 & 🔄 이 문제 다시 변환 버튼
            regen_cost = CREDIT_COST_ENGLISH_VARIANT if is_eng_item else CREDIT_COST_CONCEPT_REWRITE
            uname_quiz = ctx.current_user.username if ctx.current_user is not None else ""
            ai_col1, ai_col2, _ = st.columns([1.15, 1.35, 1.3])
            with ai_col1:
                if st.button("✨ AI 자동 해설 (0C)", key=f"ai_btn_{q.id}", width="stretch"):
                    if not active_gemini_key:
                        st.error(f"사용 가능한 Gemini API 키({active_key_label})가 설정되지 않았습니다.")
                    elif is_eng_item:
                        ok_free, free_msg = consume_daily_free_ai_quota(
                            ctx.storage, uname_quiz, "AI 자동 해설"
                        )
                        if not ok_free:
                            st.error(free_msg)
                        else:
                            with st.spinner("영어 구문·논리·오답 분석 해설 생성 중..."):
                                try:
                                    from src.gemini import generate_english_explanation

                                    with _usage_scope(ctx):
                                        exp_md = generate_english_explanation(
                                            api_key=active_gemini_key,
                                            model_name=selected_model,
                                            question=q,
                                            rewritten=rw,
                                            user_answer=st.session_state["user_answers"].get(q.id, ""),
                                            source_reference=concept_ref_str,
                                        )
                                    st.session_state["ai_explanations"][q.id] = exp_md
                                except Exception as exc:
                                    refund_daily_free_ai_quota(ctx.storage, uname_quiz)
                                    st.error(f"영어 AI 해설 생성 실패: {exc}")
                    elif whitelist is None:
                        st.error("개념서 화이트리스트가 필요합니다.")
                    else:
                        ok_free, free_msg = consume_daily_free_ai_quota(
                            ctx.storage, uname_quiz, "AI 자동 해설"
                        )
                        if not ok_free:
                            st.error(free_msg)
                        else:
                            with st.spinner("개념서 범위 내 용어만 사용하여 맞춤 해설 작성 중..."):
                                try:
                                    from src.gemini import generate_pdf_grounded_explanation

                                    with _usage_scope(ctx):
                                        exp_md = generate_pdf_grounded_explanation(
                                            api_key=active_gemini_key,
                                            model_name=selected_model,
                                            question=q,
                                            crop_png_bytes=crop_b,
                                            whitelist=whitelist,
                                            user_answer=st.session_state["user_answers"].get(q.id, ""),
                                            rewritten=rw,
                                            sliced_text=st.session_state.get("concept_sliced_text", ""),
                                            concept_reference=concept_ref_str,
                                        )
                                    st.session_state["ai_explanations"][q.id] = exp_md
                                except Exception as exc:
                                    refund_daily_free_ai_quota(ctx.storage, uname_quiz)
                                    st.error(f"AI 해설 생성 실패: {exc}")

            with ai_col2:
                if st.button(f"🔄 이 문제 다시 변환 ({regen_cost}C)", key=f"re_rw_btn_{q.id}", width="stretch"):
                    if not active_gemini_key:
                        st.error(f"사용 가능한 Gemini API 키({active_key_label})가 설정되지 않았습니다.")
                    elif not is_eng_item and whitelist is None:
                        st.error("개념서 화이트리스트가 필요합니다.")
                    else:
                        ok_regen_res, regen_reservation = try_reserve_credits(
                            ctx,
                            regen_cost,
                            "개별 문항 재변환",
                        )
                        if ok_regen_res and regen_reservation is not None:
                            actual_regen_c = 0
                            regen_succeeded = False
                            try:
                                if is_eng_item:
                                    eng_meta = item.get("english_meta") or {}
                                    src_passage = (
                                        eng_meta.get("source_passage")
                                        or getattr(q, "passage_text", "")
                                        or getattr(q, "question_text", "")
                                    )
                                    eng_rw_model = EXAM_TAGGING_MODEL
                                    eng_precision = False
                                    eng_rw_key, _ = resolve_active_gemini_api_key(
                                        ctx.current_user,
                                        ctx.creds.gemini_api_key,
                                        model_name=eng_rw_model,
                                    )
                                    eff_eng_key = eng_rw_key or active_gemini_key
                                    with st.spinner(f"'{eng_target_type}' 유형으로 이 지문을 새롭게 다시 변형 중..."):
                                        from src.gemini import generate_english_transformed_question

                                        src_pdf_b = None
                                        if getattr(q, "source_pdf", "") and hasattr(ctx.storage, "get_pdf_bytes"):
                                            src_pdf_b = ctx.storage.get_pdf_bytes(
                                                q.source_pdf,
                                                subject_category="영어",
                                                exam_name=getattr(q, "exam_name", ""),
                                            )
                                        with _usage_scope(ctx):
                                            new_rw = generate_english_transformed_question(
                                                api_key=eff_eng_key,
                                                model_name=eng_rw_model,
                                                source_passage=src_passage,
                                                target_type=eng_meta.get("target_type") or eng_target_type,
                                                difficulty=eng_meta.get("difficulty") or "고3 평가원·수능 수준",
                                                blank_unit=eng_meta.get("blank_unit") or "구 (3~6단어)",
                                                vocab_style=eng_meta.get("vocab_style") or "밑줄형 (①~⑤)",
                                                source_label=eng_meta.get("source_label") or eng_source_caption,
                                                previous_attempt=rw,
                                                use_precision_mode=eng_precision,
                                                storage=ctx.storage,
                                                use_cache=False,
                                                username=uname_quiz,
                                                base_question=q,
                                                source_pdf_bytes=src_pdf_b,
                                            )
                                        st.session_state["matched_items"][seq_num - 1]["rewritten"] = new_rw
                                        st.session_state["ai_explanations"].pop(q.id, None)
                                        st.session_state["ai_chats"].pop(q.id, None)
                                        st.session_state["generated_exam_pdf"] = None
                                        st.session_state["generated_exam_pdf_sig"] = ""
                                        actual_regen_c = regen_cost
                                        regen_succeeded = True
                                else:
                                    with st.spinner("이전 변환의 미흡 용어를 보완해 이 문항만 다시 변환 중..."):
                                        target_oos = list(
                                            getattr(j, "out_of_scope_terms", [])
                                            or getattr(q, "advanced_terms", [])
                                            or []
                                        )
                                        math_rw_model = EXAM_TAGGING_MODEL
                                        from src.gemini import rewrite_question_for_pdf

                                        with _usage_scope(ctx):
                                            new_rw = rewrite_question_for_pdf(
                                                api_key=active_gemini_key,
                                                model_name=math_rw_model,
                                                question=q,
                                                crop_png_bytes=crop_b,
                                                whitelist=whitelist,
                                                out_of_scope_terms=target_oos,
                                                sliced_text=st.session_state.get("concept_sliced_text", ""),
                                                previous_attempt=rw,
                                            )
                                        st.session_state["matched_items"][seq_num - 1]["rewritten"] = new_rw
                                        st.session_state["ai_explanations"].pop(q.id, None)
                                        st.session_state["ai_chats"].pop(q.id, None)
                                        st.session_state["generated_exam_pdf"] = None
                                        st.session_state["generated_exam_pdf_sig"] = ""
                                        actual_regen_c = regen_cost
                                        regen_succeeded = True
                            except ForbiddenTermRemainingError as fexc:
                                actual_regen_c = 0
                                st.warning(f"⛔ 재변환 후에도 금지 용어가 남아 반영하지 않았습니다 (크레딧 자동 환불): {fexc}")
                            except Exception as exc:
                                actual_regen_c = 0
                                st.error(f"문항 재변환 실패 (크레딧 자동 환불): {exc}")
                            finally:
                                settle_credits(
                                    ctx,
                                    regen_reservation,
                                    actual_regen_c,
                                    detail_note=f"개별 문항 재변환 ({actual_regen_c}C)",
                                )
                            if regen_succeeded:
                                st.rerun(scope="fragment")

            if q.id in st.session_state["ai_explanations"]:
                st.info("✨ **수능 영어 구문·논리·오답 AI 맞춤 해설**" if is_eng_item else "✨ **개념서 범위 기준 AI 맞춤 해설**")
                st.markdown(st.session_state["ai_explanations"][q.id])
                if concept_ref_str not in st.session_state["ai_explanations"][q.id]:
                    st.markdown(f"**{_html.escape(concept_ref_str)}**")

                # 문항별 추가 질문 채팅 (문항당 최대 5턴 · 일일 무료 한도 적용)
                q_chat_history: list[dict[str, str]] = st.session_state["ai_chats"].get(q.id, [])
                user_turns = sum(1 for m in q_chat_history if m.get("role") == "user")
                st.markdown(f"💬 **추가 질문 채팅 (1:1 질의응답 · {user_turns}/{MAX_CHAT_TURNS_PER_QUESTION}턴 · 0C 무료)**")
                for msg in q_chat_history:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                if user_turns >= MAX_CHAT_TURNS_PER_QUESTION:
                    st.info(f"💡 문항당 추가 질문은 최대 {MAX_CHAT_TURNS_PER_QUESTION}턴까지 지원됩니다.")
                else:
                    with st.form(key=f"chat_form_{q.id}", clear_on_submit=True):
                        followup_q = st.text_input(
                            "이 문제에 대해 궁금한 점을 질문하세요",
                            placeholder=(
                                "예: 3번 선지가 왜 오답인지, 또는 첫 문장 구문 구조를 설명해 주세요."
                                if is_eng_item
                                else "예: 왜 이 식에서 질량비가 이렇게 적용되나요?"
                            ),
                            label_visibility="collapsed",
                        )
                        send_chat = st.form_submit_button("질문 보내기")

                    if send_chat and followup_q.strip():
                        if not active_gemini_key or (not is_eng_item and whitelist is None):
                            st.error(f"사용 가능한 Gemini API 키({active_key_label})가 필요합니다.")
                        else:
                            ok_chat, chat_msg = consume_daily_free_ai_quota(
                                ctx.storage, uname_quiz, "1:1 질문 채팅"
                            )
                            if not ok_chat:
                                st.error(chat_msg)
                            else:
                                q_chat_history.append({"role": "user", "content": followup_q.strip()})
                                with st.spinner("답변 생성 중..."):
                                    try:
                                        with _usage_scope(ctx):
                                            if is_eng_item:
                                                from src.gemini import chat_followup_with_english

                                                reply = chat_followup_with_english(
                                                    api_key=active_gemini_key,
                                                    model_name=selected_model,
                                                    question=q,
                                                    rewritten=rw,
                                                    chat_history=q_chat_history[-10:-1],
                                                    user_query=followup_q.strip(),
                                                )
                                            else:
                                                from src.gemini import chat_followup_with_pdf

                                                reply = chat_followup_with_pdf(
                                                    api_key=active_gemini_key,
                                                    model_name=selected_model,
                                                    question=q,
                                                    crop_png_bytes=crop_b,
                                                    whitelist=whitelist,
                                                    chat_history=q_chat_history[-10:-1],
                                                    user_question=followup_q.strip(),
                                                    rewritten=rw,
                                                    sliced_text=st.session_state.get("concept_sliced_text", ""),
                                                )
                                        q_chat_history.append({"role": "assistant", "content": reply})
                                        st.session_state["ai_chats"][q.id] = q_chat_history
                                        st.rerun(scope="fragment")
                                    except Exception as exc:
                                        q_chat_history.pop()
                                        refund_daily_free_ai_quota(ctx.storage, uname_quiz)
                                        st.error(f"답변 생성 오류: {exc}")

    # 하단 채점하기 버튼
    grade_c1, grade_c2 = st.columns([2, 1])
    with grade_c1:
        if st.button("📝 채점하기", type="primary", width="stretch"):
            st.session_state["graded"] = True
            _record_english_seen_for_user(ctx, matched_items)
            correct_cnt = 0
            for item_g in matched_items:
                q_g: QuestionRecord = item_g["question"]
                rw_g: RewrittenQuestion | None = item_g.get("rewritten")
                truth_g = rw_g.preserved_answer if rw_g else q_g.answer
                user_val_g = st.session_state["user_answers"].get(q_g.id, "")
                if normalize_answer(user_val_g) == normalize_answer(truth_g) and normalize_answer(user_val_g) != "":
                    correct_cnt += 1
            total_g = len(matched_items)
            subj_g = _infer_exam_subject(matched_items)
            _auto_save_current_exam(ctx, 
                matched_items,
                title=f"{subj_g} 실전 풀이 시험지 ({total_g}문항)",
                score_info={
                    "correct": correct_cnt,
                    "total": total_g,
                    "percent": round((correct_cnt / max(1, total_g)) * 100, 1),
                },
            )
            st.rerun(scope="fragment")
    with grade_c2:
        if st.button("🔄 답안 초기화", width="stretch"):
            st.session_state["user_answers"] = {}
            st.session_state["graded"] = False
            st.rerun(scope="fragment")




render_mode_b_solve_fragment = _render_mode_b_solve_fragment
