from __future__ import annotations

import concurrent.futures
import hashlib
import html as _html
import json
import random
import re
from typing import Any

import streamlit as st

from src.auth import (
    CREDIT_COST_CONCEPT_MATCH_ORIG,
    CREDIT_COST_CONCEPT_REWRITE,
    CREDIT_COST_DIRECT_CONCEPT_PDF,
    TIER_LABELS,
    can_access_shared_concept_db,
    can_use_premium_features,
    consume_extraction_quota,
    get_max_batch_questions,
    get_total_available_credits,
    get_user_by_username,
    resolve_active_gemini_api_key,
)
from src.config import CONCEPT_SCOPE_MODEL, EXAM_TAGGING_MODEL
from src.gemini.usage import ForbiddenTermRemainingError
from src.models import (
    ConceptUnit,
    ConceptWhitelist,
    QuestionRecord,
    RewrittenQuestion,
    build_whitelist_from_concept_units,
    resolve_question_concept_reference,
)
from src.text_formatter import has_visual_reference
from src.ui.common import (
    _auto_save_current_exam,
    _build_exam_sig,
    _ensure_token_quota,
    _record_english_seen_for_user,
    _render_rewritten_card_html,
    _usage_scope,
    create_lazy_expander,
    create_lazy_tabs,
    ensure_credit_quota,
    format_credit_balance_inline,
    is_container_open,
    sanitize_id_part,
    settle_credits,
    try_reserve_credits,
)
from src.ui.context import AppContext
from src.ui_styles import render_chips


def _render_tab2_concept_source_tabs(ctx: AppContext) -> None:
    st.subheader("1️⃣ 기준 개념서 범위 설정 (공용 개념서 DB 선택 또는 PDF 직접 업로드)")

    source_tab_db, source_tab_pdf = create_lazy_tabs(
        [
            "📚 공용 개념서·교과서 DB에서 단원 선택 (Pro · ADMIN 전용)",
            "📄 개념서·교과서 PDF 1회성 직접 업로드",
        ],
        key="tab2_concept_src_tabs",
    )

    with source_tab_db:
        if is_container_open(source_tab_db):
            _render_tab2_shared_concept_db_panel(ctx)

    with source_tab_pdf:
        if is_container_open(source_tab_pdf):
            _render_tab2_direct_pdf_panel(ctx)


def _render_tab2_shared_concept_db_panel(ctx: AppContext) -> None:
        has_concept_db_access = can_access_shared_concept_db(ctx.current_user)
        shared_books = ctx.storage.load_concept_books()

        if not has_concept_db_access:
            st.warning(
                "🔒 **공용 개념서·교과서 DB 불러오기는 Pro(VIP) · ADMIN 등급 전용 기능입니다.**\n\n"
                f"현재 등급(**{TIER_LABELS.get(ctx.current_user.tier, ctx.current_user.tier)}**)에서는 잠금 처리되어 있으며, "
                "우측 **[📄 개념서·교과서 PDF 1회성 직접 업로드]** 탭에서 PDF를 직접 올려 이용하실 수 있습니다."
            )
            locked_options = (
                [f"🔒 {b.book_title} ({b.subject})" for b in shared_books]
                if shared_books
                else ["🔒 등록된 공용 개념서 DB (Pro 이상 전용)"]
            )
            st.selectbox(
                "공용 개념서·교과서 선택 (잠금)",
                options=locked_options,
                disabled=True,
                key="locked_shared_book_select",
            )
        else:
            if not shared_books:
                st.info(
                    "아직 Google Drive / `./concept_db`에 저장된 공용 개념서 DB가 없습니다. "
                    "상단 **[📚 탭 3: 공용 개념서·교과서 DB 구축]** 또는 아래 패널에서 교재 PDF를 먼저 등록해 주세요."
                )
            else:
                book_labels = [
                    f"{b.book_title} ({b.subject} · {len(b.units)}개 단원)" for b in shared_books
                ]
                sel_book_idx = st.selectbox(
                    "📖 공용 개념서·교과서 DB 선택",
                    options=list(range(len(shared_books))),
                    format_func=lambda idx: book_labels[idx],
                    key="tab2_shared_book_idx",
                )
                chosen_book = shared_books[sel_book_idx]
                st.caption(
                    f"선택 교재: **{chosen_book.book_title}** | 과목/학년: `{chosen_book.subject}` | "
                    f"추출 페이지 구간: `{chosen_book.page_ranges_spec}`"
                )

                all_unit_key = f"sel_all_units_{chosen_book.id}"
                unit_chk_keys = [f"unit_chk_{chosen_book.id}_{u_i}" for u_i in range(len(chosen_book.units))]
                if all_unit_key not in st.session_state:
                    st.session_state[all_unit_key] = False
                for uk in unit_chk_keys:
                    if uk not in st.session_state:
                        st.session_state[uk] = False

                def _on_toggle_all_units(
                    _all_key: str = all_unit_key,
                    _unit_keys: list[str] = unit_chk_keys,
                ) -> None:
                    target_val = bool(st.session_state.get(_all_key, False))
                    for _uk in _unit_keys:
                        st.session_state[_uk] = target_val

                def _on_toggle_single_unit(
                    _all_key: str = all_unit_key,
                    _unit_keys: list[str] = unit_chk_keys,
                ) -> None:
                    st.session_state[_all_key] = bool(_unit_keys) and all(
                        bool(st.session_state.get(_uk, False)) for _uk in _unit_keys
                    )

                with st.container(border=True):
                    st.markdown("**✅ 출제 및 개조 기준으로 사용할 단원을 체크한 뒤 아래 적용 버튼을 누르세요:**")
                    st.checkbox(
                        "전체 단원 선택 / 해제",
                        key=all_unit_key,
                        on_change=_on_toggle_all_units,
                    )
                    selected_units_list: list[ConceptUnit] = []
                    for u_i, unit_obj in enumerate(chosen_book.units):
                        checked = st.checkbox(
                            f"{unit_obj.unit_name} ({unit_obj.page_range}) — 핵심개념 {len(unit_obj.core_concepts)}개 · 허용용어 {len(unit_obj.allowed_terms_and_units)}개",
                            key=unit_chk_keys[u_i],
                            on_change=_on_toggle_single_unit,
                        )
                        if checked:
                            selected_units_list.append(unit_obj)

                    st.caption(f"현재 선택된 단원: **{len(selected_units_list)} / {len(chosen_book.units)}개**")
                    apply_units_btn = st.button(
                        "📚 선택한 단원으로 지식 화이트리스트 즉시 적용",
                        type="primary",
                        width="stretch",
                        key=f"apply_units_btn_{chosen_book.id}",
                    )

                if apply_units_btn:
                    effective_units = selected_units_list
                    if not effective_units:
                        st.error("최소 1개 이상의 단원을 체크해 주세요.")
                    else:
                        wl_from_db, ctx_text = build_whitelist_from_concept_units(
                            book_title=chosen_book.book_title,
                            subject=chosen_book.subject,
                            grade=chosen_book.grade,
                            selected_units=effective_units,
                        )
                        unit_names_short = ", ".join(
                            f"{u.unit_name}({u.page_range})" for u in effective_units[:3]
                        )
                        if len(effective_units) > 3:
                            unit_names_short += f" 외 {len(effective_units) - 3}개 단원"
                        st.session_state["concept_whitelist"] = wl_from_db
                        st.session_state["concept_sliced_text"] = ctx_text
                        st.session_state["concept_source_label"] = (
                            f"{chosen_book.book_title} - {unit_names_short}"
                        )
                        st.session_state["matched_items"] = []
                        st.session_state["generated_exam_pdf"] = None
                        st.success(
                            f"✅ 공용 개념서 `{chosen_book.book_title}`의 {len(effective_units)}개 단원이 출제·개조 기준으로 적용되었습니다!"
                        )

            cb_inline_exp = create_lazy_expander(
                "➕ 새 개념서·교과서 PDF를 공용 DB에 바로 추가하기",
                expanded=not bool(shared_books),
                key="tab2_inline_cb_exp",
            )
            with cb_inline_exp:
                if is_container_open(cb_inline_exp):
                    from src.ui.tab3_concept import render_concept_book_builder_section

                    render_concept_book_builder_section(ctx, key_prefix="tab2_inline")


def _render_tab2_direct_pdf_panel(ctx: AppContext) -> None:
    concept_file = st.file_uploader(
        "통합과학·통합사회·한국사·공통수학 등 개념서/교과서 PDF 업로드",
        type=["pdf"],
        key="tab2_concept_pdf",
    )

    concept_pdf_bytes: bytes | None = concept_file.getvalue() if concept_file is not None else None
    if concept_pdf_bytes is not None:
        from src.pdf_parser import (
            extract_concept_pages_by_ranges,
            get_pdf_page_count,
            parse_page_ranges_spec,
            render_pdf_page_png,
        )

        cur_pdf_hash = hashlib.sha256(concept_pdf_bytes).hexdigest()[:16]
        if st.session_state["concept_pdf_hash"] and st.session_state["concept_pdf_hash"] != cur_pdf_hash:
            st.session_state["concept_whitelist"] = None
            st.session_state["matched_items"] = []
            st.session_state["generated_exam_pdf"] = None
        st.session_state["concept_pdf_hash"] = cur_pdf_hash

        total_pages = get_pdf_page_count(concept_pdf_bytes)
        default_book_name = re.sub(r"\.pdf$", "", concept_file.name, flags=re.IGNORECASE)
        up_meta_c1, up_meta_c2 = st.columns([1, 1.4])
        with up_meta_c1:
            direct_book_title = st.text_input(
                "교재명 (결과 출처 표기용)",
                value=default_book_name,
                placeholder="예: 완자 통합과학",
                key="tab2_direct_book_title",
            )
        with up_meta_c2:
            multi_range_str = st.text_input(
                "개념 페이지 범위 입력 (예: 12-25, 34-48 처럼 여러 구간 지정 가능)",
                value=f"1-{min(total_pages, 5)}",
                placeholder="예: 12-25, 34-48, 60-75",
                key="tab2_direct_page_ranges",
            )

        parsed_direct_ranges: list[tuple[int, int]] = []
        try:
            parsed_direct_ranges = parse_page_ranges_spec(multi_range_str, total_pages)
            range_disp = ", ".join(
                f"p.{s}~p.{e}" if s != e else f"p.{s}" for s, e in parsed_direct_ranges
            )
            sel_cnt = sum(e - s + 1 for s, e in parsed_direct_ranges)
            st.info(f"전체 {total_pages}쪽 중 **{range_disp}** (총 {sel_cnt}쪽) 선택됨")
        except ValueError as ve:
            st.error(str(ve))

        if parsed_direct_ranges:
            first_page_num = parsed_direct_ranges[0][0]
            last_page_num = parsed_direct_ranges[-1][1]
            with st.expander("👀 선택한 범위 첫 페이지(및 끝 페이지) 미리보기", expanded=True):
                prev_c1, prev_c2 = st.columns(2)
                with prev_c1:
                    st.image(
                        render_pdf_page_png(concept_pdf_bytes, first_page_num, dpi=140),
                        caption=f"시작 페이지 ({first_page_num}쪽)",
                        width="stretch",
                    )
                with prev_c2:
                    if last_page_num != first_page_num:
                        st.image(
                            render_pdf_page_png(concept_pdf_bytes, last_page_num, dpi=140),
                            caption=f"끝 페이지 ({last_page_num}쪽)",
                            width="stretch",
                        )
                    else:
                        st.caption("단일 페이지가 선택되었습니다.")

            if st.button(
                f"🔍 1단계: 업로드된 PDF 지식 화이트리스트 추출 ({CREDIT_COST_DIRECT_CONCEPT_PDF}C)",
                type="primary",
            ):
                scope_api_key, scope_key_label = resolve_active_gemini_api_key(
                    ctx.current_user, ctx.creds.gemini_api_key, model_name=CONCEPT_SCOPE_MODEL
                )
                if not scope_api_key:
                    st.error(f"사용 가능한 Gemini API 키({scope_key_label})가 설정되지 않았습니다.")
                else:
                    ok_pdf_res, pdf_reservation = try_reserve_credits(
                        ctx,
                        CREDIT_COST_DIRECT_CONCEPT_PDF,
                        "개인 개념서 PDF 화이트리스트 추출",
                    )
                    if ok_pdf_res and pdf_reservation is not None:
                        actual_pdf_c = 0
                        try:
                            with st.spinner(f"PDF ({range_disp})의 순수 개념 본문·도표·용어를 정독해 화이트리스트를 구축 중..."):
                                sliced_bytes, sliced_text, slice_stats = extract_concept_pages_by_ranges(
                                    concept_pdf_bytes, parsed_direct_ranges
                                )
                                from src.gemini import extract_concept_whitelist

                                with _usage_scope(ctx):
                                    wl = extract_concept_whitelist(
                                        api_key=scope_api_key,
                                        model_name=CONCEPT_SCOPE_MODEL,
                                        sliced_pdf_bytes=sliced_bytes,
                                        sliced_text=sliced_text,
                                        start_page=first_page_num,
                                        end_page=last_page_num,
                                        book_title=direct_book_title.strip() or default_book_name,
                                        range_display=slice_stats["ranges_display"],
                                    )
                                st.session_state["concept_whitelist"] = wl
                                st.session_state["concept_sliced_text"] = sliced_text
                                st.session_state["concept_source_label"] = (
                                    f"{direct_book_title.strip() or concept_file.name} ({slice_stats['ranges_display']})"
                                )
                                st.session_state["matched_items"] = []
                                st.session_state["generated_exam_pdf"] = None
                                actual_pdf_c = CREDIT_COST_DIRECT_CONCEPT_PDF
                        except Exception as exc:
                            st.error(f"화이트리스트 추출 실패: {exc}")
                            actual_pdf_c = 0
                        finally:
                            settle_credits(
                                ctx,
                                pdf_reservation,
                                actual_pdf_c,
                                detail_note="개인 개념서 PDF 화이트리스트 추출",
                            )
                        if actual_pdf_c > 0:
                            bal_msg = format_credit_balance_inline(
                                ctx.storage,
                                ctx.current_user.username if ctx.current_user is not None else "",
                                charged=actual_pdf_c,
                            )
                            st.success(f"✅ PDF 지식 화이트리스트 추출이 완료되었습니다!{bal_msg}")



def _render_tab2_korean_transform_section(ctx: AppContext) -> None:
    """[탭 2 - 국어 모드] src/ui/tab2_korean.py로 분리된 국어 세트 변형·분석 UI를 호출한다."""
    from src.ui.tab2_korean import render_tab2_korean_transform_section

    render_tab2_korean_transform_section(
        storage=ctx.storage,
        current_user=ctx.current_user,
        creds=ctx.creds,
        ensure_token_quota_fn=lambda: _ensure_token_quota(ctx),
        usage_scope_fn=lambda: _usage_scope(ctx),
    )


def render_tab2(ctx: AppContext) -> None:
    tab2_domain_mode = st.radio(
        "작업 과목 모드 선택",
        options=[
            "📚 수학 · 사회 · 과학 · 한국사 (개념서 범위 추출 & 상위 용어 개조)",
            "📖 국어 (지문 분석지 PDF & 변형문제 세트 생성)",
            "🔤 영어 (지문 분석지 PDF & 수능 독해 유형 변형)",
        ],
        horizontal=True,
        key="tab2_domain_mode",
    )
    is_tab2_english_mode = tab2_domain_mode.startswith("🔤")
    is_tab2_korean_mode = tab2_domain_mode.startswith("📖")
    is_tab2_concept_mode = not is_tab2_english_mode and not is_tab2_korean_mode
    if is_tab2_english_mode:
        from src.ui.tab2_english import _render_tab2_english_transform_section

        _render_tab2_english_transform_section(ctx)
    elif is_tab2_korean_mode:
        _render_tab2_korean_transform_section(ctx)
    else:
        _render_tab2_concept_source_tabs(ctx)

    # 추출된 화이트리스트 표시
    whitelist: ConceptWhitelist | None = st.session_state.get("concept_whitelist")
    if is_tab2_concept_mode and (whitelist is not None):
        st.markdown("### 📋 현재 적용된 개념서 지식 화이트리스트 (이 범위만 학습한 것으로 간주)")
        with st.container(border=True):
            book_badge = f"**기준 교재:** `{whitelist.book_title}` | " if whitelist.book_title else ""
            st.write(f"{book_badge}**추정 과목:** `{whitelist.subject_guess}` | **범위 요약:** {whitelist.pdf_summary}")
            if whitelist.units:
                unit_badges = [f"{u.unit_name} ({u.page_range})" for u in whitelist.units]
                st.caption("📖 **적용 단원 및 페이지:** " + " · ".join(unit_badges))
            w_c1, w_c2 = st.columns(2)
            with w_c1:
                st.markdown(f"**이 범위에 실제로 설명된 개념·원리 (`pdf_concepts`, {len(whitelist.pdf_concepts)}개)**")
                render_chips(whitelist.pdf_concepts, variant="blue", limit=100)
            with w_c2:
                st.markdown(f"**이 범위에 실제로 등장하는 허용 용어 (`pdf_vocabulary`, {len(whitelist.pdf_vocabulary)}개)**")
                render_chips(whitelist.pdf_vocabulary, variant="green", limit=120)

            with st.expander("✏️ 필요 시 허용 개념/용어 수동 보완하기"):
                edited_concepts = st.text_area(
                    "허용 개념 (쉼표 또는 줄바꿈 구분)",
                    value="\n".join(whitelist.pdf_concepts),
                    height=100,
                )
                edited_vocab = st.text_area(
                    "허용 용어 (쉼표 또는 줄바꿈 구분)",
                    value=", ".join(whitelist.pdf_vocabulary),
                    height=100,
                )
                if st.button("화이트리스트 수정 적용"):
                    whitelist.pdf_concepts = [
                        s.strip() for s in re.split(r"[\n,]+", edited_concepts) if s.strip()
                    ]
                    whitelist.pdf_vocabulary = [
                        s.strip() for s in re.split(r"[\n,]+", edited_vocab) if s.strip()
                    ]
                    st.session_state["concept_whitelist"] = whitelist
                    st.success("화이트리스트를 업데이트했습니다.")

        # -------------------------------------------------------------------
        # 2단계 & 3단계: 기출 매칭·필터링 & 상위 과목 문제 자동 개조
        # -------------------------------------------------------------------
        st.markdown("---")
        st.subheader("2️⃣~3️⃣ 엄격한 범위 내 기출 매칭 & 상위 문제 자동 개조")

        is_premium = can_use_premium_features(ctx.current_user)
        match_model = EXAM_TAGGING_MODEL
        rewrite_model = EXAM_TAGGING_MODEL
        st.session_state["active_rewrite_model"] = rewrite_model
        tab2_max_batch = get_max_batch_questions(ctx.current_user)

        with st.container(border=True):
            with st.form("tab2_extract_options_form", border=False):
                opt_c1, opt_c2, opt_c3 = st.columns(3)
                with opt_c1:
                    filter_subject = st.selectbox("과목 필터", ["전체", "수학", "사회", "과학", "한국사"], index=0)
                with opt_c2:
                    filter_grade = st.selectbox("학년 수준 필터", ["전체", "고1 이하", "고2 이하"], index=0)
                with opt_c3:
                    count_mode = st.radio("추출 문항 수", ["전체", "N문제 지정"], horizontal=True)
                    max_allowed_n = tab2_max_batch or 100
                    default_n = min(5, max_allowed_n)
                    target_n = int(
                        st.number_input(
                            "지정 문항 수 (N · 'N문제 지정' 선택 시 적용)",
                            min_value=1,
                            max_value=max_allowed_n,
                            value=default_n,
                            step=1,
                        )
                    )

                chk_c1, chk_c2 = st.columns(2)
                with chk_c1:
                    only_single = st.checkbox("단일 개념 문항만 보기 (`is_single_concept`)", value=False)
                with chk_c2:
                    auto_rewrite_mode = st.checkbox(
                        "✨ 상위 과목 문제 자동 개조 모드 (PDF-Grounded Rewriter)",
                        value=True,
                        help="풀이 원리는 PDF로 해결되지만 범위 외 상위 용어(mol, 엔탈피, 한계효용 등)가 포함된 문항을 PDF 내부 용어만으로 자동 개조합니다.",
                    )

                run_extract_btn = st.form_submit_button(
                    f"🚀 기출 매칭 & 개념서 맞춤 문제 추출·개조 실행 (원문 {CREDIT_COST_CONCEPT_MATCH_ORIG}C · 개조 {CREDIT_COST_CONCEPT_REWRITE}C / 문항)",
                    type="primary",
                )

            if run_extract_btn:
                db_list = ctx.storage.load_db_readonly()
                match_api_key, match_key_label = resolve_active_gemini_api_key(
                    ctx.current_user, ctx.creds.gemini_api_key, model_name=match_model
                )
                rewrite_api_key, rewrite_key_label = resolve_active_gemini_api_key(
                    ctx.current_user, ctx.creds.gemini_api_key, model_name=rewrite_model
                )
                if not match_api_key:
                    st.error(f"사용 가능한 Gemini API 키({match_key_label})가 설정되지 않았습니다.")
                elif not ensure_credit_quota(ctx, CREDIT_COST_CONCEPT_MATCH_ORIG, "수·사·과·한 문제 추출·개조"):
                    pass
                elif not db_list:
                    st.warning("저장된 기출 DB가 비어 있습니다. 먼저 [탭 1]에서 모의고사를 수집·태깅해 주세요.")
                else:
                    # 1차 메타데이터 필터링 (비영어·비국어 과목 전용 · 과거 '사회'로 저장된 한국사 문항 호환 포함)
                    candidates: list[QuestionRecord] = []
                    for r in db_list:
                        if r.subject_category in ("영어", "국어"):
                            continue
                        is_history_rec = (
                            r.subject_category == "한국사"
                            or "한국사" in (r.curriculum_subject or "")
                            or "한국사" in (r.exam_name or "")
                        )
                        eff_subj_cat = "한국사" if is_history_rec else r.subject_category
                        if filter_subject != "전체" and eff_subj_cat != filter_subject:
                            continue
                        if filter_grade == "고1 이하" and r.min_grade > 1:
                            continue
                        if filter_grade == "고2 이하" and r.min_grade > 2:
                            continue
                        if only_single and not r.is_single_concept:
                            continue
                        candidates.append(r)

                    if not candidates:
                        st.warning("선택한 과목/학년/단일개념 조건에 맞는 후보 문항이 DB에 없습니다.")
                    else:
                        with _usage_scope(ctx):
                            from src.gemini import (
                                match_questions_to_whitelist,
                                rewrite_question_for_pdf,
                                submit_with_usage_context,
                            )

                            with st.spinner(f"후보 {len(candidates)}개 문항의 핵심 원리를 개념서 범위와 대조 중..."):
                                try:
                                    judgments = match_questions_to_whitelist(
                                        api_key=match_api_key,
                                        model_name=match_model,
                                        candidates=candidates,
                                        whitelist=whitelist,
                                    )
                                except Exception as exc:
                                    st.error(f"원리 매칭 중 오류가 발생했습니다: {exc}")
                                    judgments = []

                            j_map = {j.question_id: j for j in judgments}
                            diagnostics = []
                            selected_pairs: list[tuple[QuestionRecord, Any]] = []
                            skipped_needs_rewrite = 0

                            for q in candidates:
                                j = j_map.get(q.id)
                                if not j:
                                    continue
                                diagnostics.append((q, j))
                                if not j.is_solvable_by_pdf_only:
                                    continue
                                if j.needs_rewriting and not auto_rewrite_mode:
                                    skipped_needs_rewrite += 1
                                    continue
                                selected_pairs.append((q, j))

                            if count_mode == "N문제 지정":
                                selected_pairs = selected_pairs[:target_n]
                            if tab2_max_batch is not None and len(selected_pairs) > tab2_max_batch:
                                st.warning(
                                    f"⚠️ 현재 등급은 1회 실행당 최대 {tab2_max_batch}문항까지만 추출·개조됩니다. "
                                    f"(매칭 {len(selected_pairs)}문항 중 앞의 {tab2_max_batch}문항만 실행)"
                                )
                                selected_pairs = selected_pairs[:tab2_max_batch]

                            st.session_state["match_diagnostics"] = diagnostics

                            if not selected_pairs:
                                st.warning(
                                    "선택한 개념서 범위의 개념만으로 풀 수 있는 문항이 없습니다."
                                    + (
                                        f" (단, '상위 과목 문제 자동 개조 모드'를 켜면 원리가 일치하는 {skipped_needs_rewrite}문항을 개조해 출제할 수 있습니다.)"
                                        if skipped_needs_rewrite > 0
                                        else ""
                                    )
                                )
                                st.session_state["matched_items"] = []
                            else:
                                fresh_u_now = get_user_by_username(ctx.storage, ctx.current_user.username)
                                avail_now = get_total_available_credits(fresh_u_now)
                                planned_c = sum(
                                    CREDIT_COST_CONCEPT_REWRITE if (j.needs_rewriting and auto_rewrite_mode) else CREDIT_COST_CONCEPT_MATCH_ORIG
                                    for _, j in selected_pairs
                                )
                                if avail_now is not None and planned_c > avail_now:
                                    affordable_pairs: list[tuple[QuestionRecord, Any]] = []
                                    acc_cost = 0
                                    for q_cand, j_cand in selected_pairs:
                                        c_step = (
                                            CREDIT_COST_CONCEPT_REWRITE
                                            if (j_cand.needs_rewriting and auto_rewrite_mode)
                                            else CREDIT_COST_CONCEPT_MATCH_ORIG
                                        )
                                        if acc_cost + c_step <= avail_now:
                                            affordable_pairs.append((q_cand, j_cand))
                                            acc_cost += c_step
                                    if not affordable_pairs:
                                        st.error(
                                            f"🚫 적합한 문항({len(selected_pairs)}건)을 출제하기 위한 크레딧이 부족합니다 "
                                            f"(현재 가용 {avail_now:,}C)."
                                        )
                                        selected_pairs = []
                                    else:
                                        st.warning(
                                            f"💡 현재 가용 크레딧({avail_now:,}C) 범위 내에서 앞의 **{len(affordable_pairs)}문항**"
                                            f"(예상 {acc_cost:,}C)만 자동 선별하여 진행합니다. "
                                            f"(전체 {len(selected_pairs)}문항 필요량 {planned_c:,}C)"
                                        )
                                        selected_pairs = affordable_pairs
                                        planned_c = acc_cost

                            if selected_pairs:
                                ok_ext_res, ext_reservation = try_reserve_credits(
                                    ctx,
                                    planned_c,
                                    f"수·사·과·한 문제 추출·개조 ({len(selected_pairs)}문항)",
                                )
                                if ok_ext_res and ext_reservation is not None:
                                    actual_ext_c = 0
                                    try:
                                        assembled: list[dict[str, Any]] = []
                                        prog = st.progress(0.0)
                                        stat = st.empty()
                                        sliced_text_ctx = st.session_state.get("concept_sliced_text", "")
                                        fallback_source_ctx = st.session_state.get("concept_source_label", "")

                                        prepared_items: list[tuple[QuestionRecord, Any, bytes | None, str]] = []
                                        rewrite_indices: list[int] = []
                                        for idx, (q, j) in enumerate(selected_pairs):
                                            crop_bytes = ctx.storage.get_crop_bytes(
                                                q.image_filename,
                                                subject_category=q.subject_category,
                                                exam_name=q.exam_name,
                                            )
                                            concept_ref = resolve_question_concept_reference(
                                                question=q,
                                                judgment=j,
                                                whitelist=whitelist,
                                                fallback_source_label=fallback_source_ctx,
                                            )
                                            prepared_items.append((q, j, crop_bytes, concept_ref))
                                            if j.needs_rewriting and auto_rewrite_mode:
                                                rewrite_indices.append(idx)

                                        rewritten_results: dict[int, RewrittenQuestion] = {}
                                        failed_indices: set[int] = set()
                                        if rewrite_indices:
                                            if not rewrite_api_key:
                                                st.error(
                                                    f"상위 문제 개조용 API 키가 설정되지 않았습니다 ({rewrite_key_label})."
                                                )
                                                failed_indices.update(rewrite_indices)
                                            else:
                                                total_rw = len(rewrite_indices)
                                                stat.info(f"상위 문제 {total_rw}문항 병렬 자동 개조 중 (최대 4스레드)...")
                                                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                                                    future_map = {}
                                                    for idx in rewrite_indices:
                                                        q, j, crop_bytes, _ = prepared_items[idx]
                                                        fut = submit_with_usage_context(
                                                            executor,
                                                            rewrite_question_for_pdf,
                                                            api_key=rewrite_api_key,
                                                            model_name=rewrite_model,
                                                            question=q,
                                                            crop_png_bytes=crop_bytes,
                                                            whitelist=whitelist,
                                                            out_of_scope_terms=j.out_of_scope_terms,
                                                            sliced_text=sliced_text_ctx,
                                                        )
                                                        future_map[fut] = idx

                                                    done_cnt = 0
                                                    for fut in concurrent.futures.as_completed(future_map):
                                                        idx = future_map[fut]
                                                        q_done = prepared_items[idx][0]
                                                        done_cnt += 1
                                                        try:
                                                            rewritten_results[idx] = fut.result()
                                                        except ForbiddenTermRemainingError as fexc:
                                                            failed_indices.add(idx)
                                                            st.warning(f"⛔ {q_done.id} 금지 용어 잔존으로 제외: {fexc}")
                                                        except Exception as exc:
                                                            failed_indices.add(idx)
                                                            st.error(f"{q_done.id} 개조 실패 (원본 제외): {exc}")
                                                        prog.progress(done_cnt / total_rw)
                                                        stat.info(f"상위 문제 병렬 자동 개조 중 ({done_cnt}/{total_rw})...")

                                        for idx, (q, j, crop_bytes, concept_ref) in enumerate(prepared_items):
                                            if idx in failed_indices:
                                                continue
                                            assembled.append(
                                                {
                                                    "question": q,
                                                    "judgment": j,
                                                    "crop_png_bytes": crop_bytes,
                                                    "rewritten": rewritten_results.get(idx),
                                                    "concept_reference": concept_ref,
                                                }
                                            )

                                        prog.progress(1.0)
                                        stat.empty()
                                        consume_extraction_quota(
                                            ctx.storage, ctx.current_user.username, len(assembled)
                                        )
                                        rewritten_cnt = sum(1 for it in assembled if it["rewritten"] is not None)
                                        orig_cnt = len(assembled) - rewritten_cnt
                                        actual_ext_c = (
                                            orig_cnt * CREDIT_COST_CONCEPT_MATCH_ORIG
                                            + rewritten_cnt * CREDIT_COST_CONCEPT_REWRITE
                                        )
                                        st.session_state["matched_items"] = assembled
                                        st.session_state["generated_exam_pdf"] = None
                                        st.session_state["user_answers"] = {}
                                        st.session_state["graded"] = False
                                        st.session_state["ai_explanations"] = {}
                                        st.session_state["ai_chats"] = {}
                                        st.session_state["_tab2_last_extract_summary"] = (
                                            len(assembled),
                                            orig_cnt,
                                            rewritten_cnt,
                                            actual_ext_c,
                                        )
                                    finally:
                                        settle_credits(
                                            ctx,
                                            ext_reservation,
                                            actual_ext_c,
                                            detail_note=f"수·사·과·한 추출·개조 ({len(assembled)}문항 확정 · {actual_ext_c:,}C)",
                                        )

                        last_sum = st.session_state.pop("_tab2_last_extract_summary", None)
                        if last_sum is not None:
                            tot_c, orig_c, rw_c, charged_c = last_sum
                            rem_msg = format_credit_balance_inline(
                                ctx.storage,
                                ctx.current_user.username if ctx.current_user is not None else "",
                                charged=charged_c,
                            )
                            st.success(
                                f"🎯 총 {tot_c}문항 추출 완료! "
                                f"(원문 그대로 {orig_c}문항 · 개념서 기준 개조 {rw_c}문항){rem_msg}"
                            )

    # -----------------------------------------------------------------------
    # 결과 활용 2가지 모드 (모드 A: PDF 다운로드 / 모드 B: 웹에서 풀기 + AI 해설)
    # -----------------------------------------------------------------------
    matched_items: list[dict[str, Any]] = st.session_state.get("matched_items") or []
    if is_tab2_concept_mode and st.session_state.get("match_diagnostics"):
        with st.expander("🔎 후보 문항 매칭/제외 판정 근거 상세 보기", expanded=False):
            for q, j in st.session_state["match_diagnostics"]:
                icon = "🟢" if j.is_solvable_by_pdf_only and not j.needs_rewriting else ("🔵" if j.is_solvable_by_pdf_only else "⚪")
                status_lbl = (
                    "원문 그대로 매칭"
                    if (j.is_solvable_by_pdf_only and not j.needs_rewriting)
                    else ("원리 일치 → 상위 용어 개조 대상" if j.is_solvable_by_pdf_only else "범위 외 원리 포함 (제외)")
                )
                ref_preview = ""
                if j.is_solvable_by_pdf_only:
                    ref_preview = " | " + resolve_question_concept_reference(
                        q, j, whitelist, fallback_source_label=st.session_state.get("concept_source_label", "")
                    )
                st.write(
                    f"{icon} **{q.exam_name} {q.q_num}번** [{status_lbl}] — {j.matching_reason}"
                    + (f" | 범위 외 용어: `{', '.join(j.out_of_scope_terms)}`" if j.out_of_scope_terms else "")
                    + ref_preview
                )

    is_eng_analysis_submode = is_tab2_english_mode and str(
        st.session_state.get("tab2_eng_sub_mode", "")
    ).startswith("📑")
    if matched_items and not is_eng_analysis_submode:
        st.markdown("---")
        if len(matched_items) >= 2:
            shuf_c1, shuf_c2, shuf_c3 = st.columns([1.15, 1.05, 2.3])
            with shuf_c1:
                if st.button("🔀 문항 순서 섞기", key="tab2_shuffle_matched_items", width="stretch"):
                    has_kor_sets = any(
                        getattr(it["question"], "subject_category", "") == "국어"
                        or getattr(it.get("rewritten"), "subject_domain", "") == "국어"
                        or bool(it.get("korean_meta"))
                        for it in matched_items
                    )
                    if has_kor_sets:
                        # 국어 세트 문항은 지문이 깨지지 않도록 세트 묶음 단위로 순서 섞기
                        blocks_by_key: list[list[dict[str, Any]]] = []
                        key_to_idx: dict[str, int] = {}
                        for it in matched_items:
                            rw_i = it.get("rewritten")
                            km_i = it.get("korean_meta") or {}
                            q_i = it["question"]
                            g_k = (
                                (getattr(rw_i, "passage_group_key", "") if rw_i is not None else "")
                                or f"{q_i.exam_name}::{km_i.get('group_id') or getattr(q_i, 'passage_group_id', '') or q_i.id}"
                            )
                            if g_k not in key_to_idx:
                                key_to_idx[g_k] = len(blocks_by_key)
                                blocks_by_key.append([])
                            blocks_by_key[key_to_idx[g_k]].append(it)
                        random.shuffle(blocks_by_key)
                        shuffled_items = [it for blk in blocks_by_key for it in blk]
                    else:
                        shuffled_items = list(matched_items)
                        random.shuffle(shuffled_items)
                    st.session_state["matched_items"] = shuffled_items
                    st.session_state["generated_exam_pdf"] = None
                    st.session_state["generated_exam_pdf_sig"] = ""
                    st.rerun()
            with shuf_c2:
                if st.button("🔢 번호순 정렬", key="tab2_sort_matched_items", width="stretch"):
                    sorted_items = sorted(
                        matched_items,
                        key=lambda it: (it["question"].exam_name, it["question"].q_num, it["question"].id),
                    )
                    st.session_state["matched_items"] = sorted_items
                    st.session_state["generated_exam_pdf"] = None
                    st.session_state["generated_exam_pdf_sig"] = ""
                    st.rerun()
            with shuf_c3:
                order_preview = " → ".join(
                    f"{idx_o}. {getattr(it.get('rewritten'), 'target_question_type', '') or it['question'].unit_name}({it['question'].q_num}번)"
                    for idx_o, it in enumerate(matched_items[:6], start=1)
                )
                if len(matched_items) > 6:
                    order_preview += f" … (총 {len(matched_items)}문항)"
                st.caption(f"📋 현재 출제 순서: {order_preview}")

        mode_tab_a, mode_tab_b = st.tabs(
            [
                "📄 모드 A: 맞춤형 시험지 PDF 생성 & 다운로드",
                "✏️ 모드 B: 웹에서 바로 풀기 & 자동 채점 + AI 자동 해설",
            ]
        )

        # ----- 모드 A: 맞춤형 시험지 PDF 생성 & 다운로드 -----
        with mode_tab_a:
            st.markdown("#### 📄 모의고사 스타일 맞춤형 시험지 PDF 제작")
            if not can_use_premium_features(ctx.current_user):
                st.warning(
                    "🔒 **맞춤형 시험지 PDF 병합 및 다운로드는 CUSTOM · PRO · ADMIN 등급 전용 기능입니다.**\n\n"
                    "사이드바에서 **개인 API 키 등록(CUSTOM)** 시 즉시 이용 가능하며, 무료 회원은 우측 **[모드 B: 웹에서 바로 풀기]**를 계속 이용하실 수 있습니다."
                )
            else:
                with st.container(border=True):
                    is_all_eng_matched = bool(matched_items) and all(
                        it["question"].subject_category == "영어"
                        or getattr(it.get("rewritten"), "subject_domain", "") == "영어"
                        or bool(it.get("english_meta"))
                        for it in matched_items
                    )
                    is_all_kor_matched = bool(matched_items) and all(
                        it["question"].subject_category == "국어"
                        or getattr(it.get("rewritten"), "subject_domain", "") == "국어"
                        or bool(it.get("korean_meta"))
                        for it in matched_items
                    )
                    default_pdf_title = (
                        "수능·모의고사 영어 지문 맞춤형 변형 모의고사"
                        if is_all_eng_matched
                        else (
                            "수능·모의고사 국어(독서·문학) 지문 세트 변형 모의고사"
                            if is_all_kor_matched
                            else "2022 개정 교육과정 개념서 맞춤형 기출 모의고사"
                        )
                    )
                    with st.form("tab2_mode_a_pdf_form", border=False):
                        pdf_title = st.text_input(
                            "시험지 제목",
                            value=default_pdf_title,
                        )
                        pa_c1, pa_c2 = st.columns(2)
                        with pa_c1:
                            layout_choice = st.radio(
                                "시험지 레이아웃",
                                options=["2단 모의고사 스타일", "1단 태블릿/인쇄 스타일"],
                                horizontal=True,
                            )
                        with pa_c2:
                            if is_all_eng_matched or is_all_kor_matched:
                                subj_lbl = "영어" if is_all_eng_matched else "국어"
                                st.caption(f"📌 {subj_lbl} 변형 시험지는 원본 이미지를 제외하고 기출 출처(회차·번호)만 표기됩니다.")
                                include_orig_fig = False
                            else:
                                include_orig_fig = st.checkbox(
                                    "개조 문항 아래 참고용 원본 도표/이미지 함께 수록",
                                    value=True,
                                )

                        build_pdf_btn = st.form_submit_button("🛠️ 맞춤형 시험지 PDF 생성하기", type="primary")

                    items_sig = hashlib.sha256(
                        json.dumps(
                            [
                                (
                                    it["question"].id,
                                    getattr(it.get("rewritten"), "rewritten_stem", ""),
                                    getattr(it.get("rewritten"), "rewritten_choices", []),
                                    pdf_title.strip(),
                                    layout_choice,
                                    include_orig_fig,
                                )
                                for it in matched_items
                            ],
                            ensure_ascii=False,
                        ).encode("utf-8")
                    ).hexdigest()[:16]

                    need_auto_build_pdf = (
                        not st.session_state.get("generated_exam_pdf")
                        or st.session_state.get("generated_exam_pdf_sig", "") != items_sig
                    )
                    if build_pdf_btn or need_auto_build_pdf:
                        with st.spinner("PDF 조판 및 뒷장 정답·해설 페이지 생성 중..."):
                            try:
                                from src.pdf.exam_pdf import generate_custom_exam_pdf

                                eff_source_lbl = st.session_state.get("concept_source_label", "업로드 개념서 PDF")
                                if is_all_kor_matched and (not eff_source_lbl or "업로드 개념서" in eff_source_lbl):
                                    kor_exams = list(
                                        dict.fromkeys(
                                            str(getattr(it["question"], "exam_name", "") or "").strip()
                                            for it in matched_items
                                            if getattr(it["question"], "exam_name", "")
                                        )
                                    )
                                    eff_source_lbl = ", ".join(kor_exams) if kor_exams else "국어 기출 지문 세트"

                                pdf_bytes = generate_custom_exam_pdf(
                                    items=matched_items,
                                    exam_title=pdf_title.strip() or "맞춤형 기출 모의고사",
                                    source_label=eff_source_lbl,
                                    whitelist=whitelist,
                                    two_column=(layout_choice == "2단 모의고사 스타일"),
                                    include_original_for_rewritten=include_orig_fig,
                                )
                                st.session_state["generated_exam_pdf"] = pdf_bytes
                                st.session_state["generated_exam_pdf_sig"] = items_sig
                                _record_english_seen_for_user(ctx, matched_items)
                                _auto_save_current_exam(
                                    ctx,
                                    matched_items,
                                    title=pdf_title.strip() or "맞춤형 기출 모의고사",
                                    pdf_bytes=pdf_bytes,
                                    sig=_build_exam_sig(matched_items),
                                )
                                if build_pdf_btn:
                                    st.success("✅ 시험지 PDF(뒷장 정답·개념서 맞춤 해설 포함)가 생성되고 '나의 시험지'에 자동 저장되었습니다!")
                            except Exception as pdf_exc:
                                st.error(f"PDF 생성 중 오류가 발생했습니다: {pdf_exc}")

                    if (
                        st.session_state.get("generated_exam_pdf")
                        and st.session_state.get("generated_exam_pdf_sig", items_sig) == items_sig
                    ):
                        dl_bytes = st.session_state["generated_exam_pdf"]
                        st.download_button(
                            label=f"⬇️ 맞춤형 시험지 PDF 다운로드 ({len(dl_bytes) // 1024} KB)",
                            data=dl_bytes,
                            file_name=f"{sanitize_id_part(pdf_title)}.pdf",
                            mime="application/pdf",
                            type="primary",
                            width="stretch",
                        )

        # ----- 모드 B: 웹에서 바로 풀기 & 자동 채점 + AI 자동 해설 -----
        with mode_tab_b:
            from src.ui.tab2_quiz import _render_mode_b_solve_fragment

            _render_mode_b_solve_fragment(ctx, 
                matched_items=matched_items,
                whitelist=whitelist,
                active_gemini_key=ctx.active_gemini_key,
                active_key_label=ctx.active_key_label,
                selected_model=ctx.selected_model,
                rewrite_model=st.session_state.get("active_rewrite_model") or ctx.selected_model,
            )

