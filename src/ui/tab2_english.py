from __future__ import annotations

import concurrent.futures
import hashlib
import html as _html
import re
from datetime import datetime
from typing import Any

import streamlit as st

from src.auth import (
    CREDIT_COST_ENGLISH_VARIANT,
    TIER_LABELS,
    can_use_premium_features,
    consume_extraction_quota,
    get_max_batch_questions,
    resolve_active_gemini_api_key,
)
from src.config import EXAM_TAGGING_MODEL, FLASH_LITE_MODEL, MODEL_LABELS
from src.models import (
    ENGLISH_DEFAULT_EXCLUDED_Q_NUMS,
    ENGLISH_TRANSFORM_TYPES,
    QuestionRecord,
    RewrittenQuestion,
    normalize_english_group_id,
)
from src.ui.common import (
    _auto_save_current_exam,
    _build_exam_sig,
    _record_english_seen_for_user,
    _usage_scope,
    format_credit_balance_inline,
    settle_credits,
    try_reserve_credits,
)
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
    """(영어 지문, 목표 수능 유형) 작업 목록을 정액 크레딧(문항당 2C · 풀 동일) 선예치·자동정산 하에 실행한다."""
    if not jobs:
        st.warning("변형할 영어 지문과 목표 수능 유형을 최소 1개 이상 선택해 주세요.")
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

    reserve_credits = len(jobs) * CREDIT_COST_ENGLISH_VARIANT
    ok_res, reservation = try_reserve_credits(
        ctx,
        reserve_credits,
        f"영어 변형 출제 ({len(jobs)}문항)",
    )
    if not ok_res or reservation is None:
        return

    from src.english_v3.pipeline import (
        _needs_english_passage_restoration,
        generate_batch_questions_v3,
        prepare_clean_english_source_passage,
    )

    total_jobs = len(jobs)
    prog = st.progress(0.0)
    stat = st.empty()
    n_workers = max(1, min(total_jobs, 4))
    tier_hint = "Flex 50% 할인" if total_jobs >= 3 else "Standard 즉시 응답"
    stat.info(f"🔤 영어 지문 수능형 변형 v3 {total_jobs}문항 출제 중 ({tier_hint}, 최대 {n_workers}스레드)...")

    pdf_bytes_cache: dict[tuple[str, str], bytes | None] = {}

    def _get_job_source_pdf_bytes(base_q: QuestionRecord | None) -> bytes | None:
        if base_q is None:
            return None
        fn = (getattr(base_q, "source_pdf", "") or "").strip()
        if not fn or not hasattr(ctx.storage, "get_pdf_bytes"):
            return None
        key = ((getattr(base_q, "exam_name", "") or "").strip(), fn)
        if key not in pdf_bytes_cache:
            pdf_bytes_cache[key] = ctx.storage.get_pdf_bytes(
                fn,
                subject_category="영어",
                exam_name=key[0],
            )
        return pdf_bytes_cache[key]

    results_by_idx: dict[int, Any] = {}
    errors_by_idx: dict[int, str] = {}
    assembled: list[dict[str, Any]] = []
    actual_credits = 0
    try:
        # PyMuPDF 멀티스레드 경합/데드락 방지: 메인 스레드에서 1회만 원문 정화·복원 후 워커 스레드에는 정화된 텍스트만 전달
        cleaned_passages_cache: dict[str, str] = {}
        prepared_jobs: list[dict[str, Any]] = []
        with _usage_scope(ctx):
            for job in jobs:
                raw_p = str(job.get("passage_text") or "").strip()
                base_q = job.get("base_question")
                cache_k = f"{getattr(base_q, 'id', '')}||{hashlib.sha256(raw_p.encode('utf-8')).hexdigest()[:16]}"
                if cache_k not in cleaned_passages_cache:
                    orig_t = (
                        (getattr(base_q, "english_question_type", "") if base_q is not None else "")
                        or (getattr(base_q, "unit_name", "") if base_q is not None else "")
                    )
                    needs_pdf = len(raw_p) < 80 or _needs_english_passage_restoration(raw_p, orig_t)
                    src_pdf_b = _get_job_source_pdf_bytes(base_q) if needs_pdf else None
                    cleaned_passages_cache[cache_k] = prepare_clean_english_source_passage(
                        api_key=rewrite_api_key,
                        model_name=rewrite_model,
                        source_passage=raw_p,
                        base_question=base_q,
                        source_pdf_bytes=src_pdf_b,
                    ) or raw_p
                p_job = dict(job)
                p_job["passage_text"] = cleaned_passages_cache[cache_k]
                p_job["key_vocabulary"] = list(getattr(base_q, "key_vocabulary", []) or []) if base_q is not None else []
                p_job["base_question"] = None
                prepared_jobs.append(p_job)

            def _on_v3_progress(done_cnt: int, tot_cnt: int) -> None:
                prog.progress(min(1.0, done_cnt / max(1, tot_cnt)))
                stat.info(f"🔤 영어 지문 수능형 변형 v3 출제·검수 중 ({done_cnt}/{tot_cnt})...")

            results_by_idx, errors_by_idx = generate_batch_questions_v3(
                api_key=rewrite_api_key,
                jobs=prepared_jobs,
                storage=ctx.storage,
                username=ctx.current_user.username if ctx.current_user is not None else "",
                max_workers=n_workers,
                progress_callback=_on_v3_progress,
            )

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
        actual_credits = len(assembled) * CREDIT_COST_ENGLISH_VARIANT
    finally:
        settle_credits(
            ctx,
            reservation,
            actual_credits,
            detail_note=f"영어 변형 출제 ({len(assembled)}/{total_jobs}문항 성공 · {actual_credits:,}C 확정)",
        )

    prog.progress(1.0)
    stat.empty()

    for idx, err_str in sorted(errors_by_idx.items()):
        j_info = jobs[idx]
        st.error(f"❌ [{j_info['source_label']} → {j_info['target_type']}] 변형 실패: {err_str}")

    if not assembled:
        st.warning("생성된 영어 변형 문항이 없습니다. (선예치 크레딧 전액 자동 환불됨)")
        return

    consume_extraction_quota(ctx.storage, ctx.current_user.username, len(assembled))

    st.session_state["matched_items"] = assembled
    st.session_state["match_diagnostics"] = []
    st.session_state["concept_source_label"] = source_summary_label
    st.session_state["generated_exam_pdf"] = None
    st.session_state["generated_exam_pdf_sig"] = ""
    st.session_state["user_answers"] = {}
    st.session_state["graded"] = False
    st.session_state["ai_explanations"] = {}
    st.session_state["ai_chats"] = {}

    if can_use_premium_features(ctx.current_user):
        try:
            import json
            from src.pdf.exam_pdf import generate_custom_exam_pdf

            default_title = "수능·모의고사 영어 지문 맞춤형 변형 모의고사"
            default_layout = "2단 모의고사 스타일"
            items_sig = hashlib.sha256(
                json.dumps(
                    [
                        (
                            it["question"].id,
                            getattr(it.get("rewritten"), "rewritten_stem", ""),
                            getattr(it.get("rewritten"), "rewritten_choices", []),
                            default_title,
                            default_layout,
                            False,
                        )
                        for it in assembled
                    ],
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()[:16]
            pre_pdf_bytes = generate_custom_exam_pdf(
                items=assembled,
                exam_title=default_title,
                source_label=source_summary_label or "영어 기출 지문",
                whitelist=None,
                two_column=True,
                include_original_for_rewritten=False,
            )
            st.session_state["generated_exam_pdf"] = pre_pdf_bytes
            st.session_state["generated_exam_pdf_sig"] = items_sig
            _record_english_seen_for_user(ctx, assembled)
            _auto_save_current_exam(
                ctx,
                assembled,
                title=default_title,
                pdf_bytes=pre_pdf_bytes,
                sig=_build_exam_sig(assembled),
            )
        except Exception:
            pass

    rem_msg = format_credit_balance_inline(
        ctx.storage,
        ctx.current_user.username if ctx.current_user is not None else "",
        charged=actual_credits,
    )
    st.success(f"🎯 총 **{len(assembled)}문항**의 수능형 영어 변형 문제 출제가 완료되었습니다! 아래 모드 A(시험지 PDF) 또는 모드 B(웹 바로 풀기)에서 확인하세요.{rem_msg}")
def _filter_english_db_passages(
    ctx: AppContext,
    key_prefix: str,
) -> tuple[bool, list[tuple[QuestionRecord, str]]]:
    """기출 DB 영어 독해 지문을 조회하고 회차·원래 유형·키워드·제외문항 필터 UI를 렌더링한다."""
    all_db = ctx.storage.load_db_readonly()
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
                        f"🚀 선택한 기출 영어 지문 수능형 변형 문제 생성 (문항당 {CREDIT_COST_ENGLISH_VARIANT}C · 캐시 풀 동일)",
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
                f"🚀 직접 입력한 영어 지문으로 수능형 변형 문제 생성 (문항당 {CREDIT_COST_ENGLISH_VARIANT}C · 캐시 풀 동일)",
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
        from src.ui.tab2_english_analysis import _render_english_analysis_submode

        _render_english_analysis_submode(ctx)
    else:
        _render_english_transform_submode(ctx)


render_tab2_english_transform_section = _render_tab2_english_transform_section
