from __future__ import annotations

import concurrent.futures
import hashlib
import html as _html
import json
import re
from datetime import datetime
from typing import Any

import streamlit as st

from src.auth import (
    CREDIT_COST_KOREAN_SET_TRANSFORM,
    TIER_LABELS,
    can_use_premium_features,
    consume_extraction_quota,
    get_max_batch_questions,
    resolve_active_gemini_api_key,
)
from src.config import EXAM_TAGGING_MODEL, MODEL_LABELS
from src.models import QuestionRecord, normalize_korean_group_id
from src.ui.common import (
    _auto_save_current_exam,
    _build_exam_sig,
    _record_korean_seen_for_user,
    _usage_scope,
    format_credit_balance_inline,
    settle_credits,
    try_reserve_credits,
)
from src.ui.context import AppContext
from src.ui_styles import render_chips

_KOR_SECTIONS = ["독서", "문학", "화법과 작문", "언어와 매체"]


def _classify_kor_sec(cs: str) -> str:
    txt = (cs or "").strip()
    if "문학" in txt:
        return "문학"
    if "화법" in txt or "작문" in txt:
        return "화법과 작문"
    if "언어" in txt or "매체" in txt or "문법" in txt:
        return "언어와 매체"
    return "독서"


def _extract_kor_subgenre(cs: str) -> str:
    m = re.search(r"\(([^)]+)\)", cs or "")
    return m.group(1).strip() if m else ""


def _filter_korean_db_sets(
    ctx: AppContext,
    key_prefix: str,
) -> tuple[bool, list[dict[str, Any]]]:
    """기출 DB 국어 지문 세트를 조회하고 회차·영역·키워드 필터 UI를 렌더링한다."""
    all_db = ctx.storage.load_db_readonly()
    kor_recs = [
        r
        for r in all_db
        if r.subject_category == "국어"
        and bool((getattr(r, "passage_text", "") or "").strip())
    ]

    if not kor_recs:
        st.info(
            "아직 DB에 저장된 **국어 지문 세트**가 없습니다. "
            "**[📥 탭 1]**에서 국어 모의고사 문제지+해설지 PDF를 먼저 수집·추출하거나, "
            "우측 **[✍️ 새 국어 지문 직접 붙여넣기]** 탭을 이용해 주세요."
        )
        return False, []

    sets_map: dict[tuple[str, str], dict[str, Any]] = {}
    for r in sorted(kor_recs, key=lambda x: (x.exam_name, x.q_num)):
        gid = normalize_korean_group_id(getattr(r, "passage_group_id", ""), r.q_num)
        key = (r.exam_name, gid)
        curr_subj = (r.curriculum_subject or "독서").strip()
        entry = sets_map.setdefault(
            key,
            {
                "exam_name": r.exam_name,
                "group_id": gid,
                "records": [],
                "passage_text": "",
                "passage_topic": "",
                "curr_subj": curr_subj,
                "kor_sec": _classify_kor_sec(curr_subj),
                "kor_sub": _extract_kor_subgenre(curr_subj),
                "key_vocabulary": [],
            },
        )
        entry["records"].append(r)
        p_txt = (getattr(r, "passage_text", "") or "").strip()
        if len(p_txt) > len(entry["passage_text"]):
            entry["passage_text"] = p_txt
        t_txt = (getattr(r, "passage_topic", "") or "").strip()
        if len(t_txt) > len(entry["passage_topic"]):
            entry["passage_topic"] = t_txt
        for kw in (getattr(r, "key_vocabulary", None) or []):
            kw_s = str(kw).strip()
            if kw_s and kw_s not in entry["key_vocabulary"]:
                entry["key_vocabulary"].append(kw_s)

    set_entries = sorted(
        sets_map.values(),
        key=lambda e: (e["exam_name"], min(x.q_num for x in e["records"])),
    )

    f_c1, f_c2, f_c3 = st.columns([1.2, 1.1, 1.3])
    avail_kor_exams = sorted({e["exam_name"] for e in set_entries}, reverse=True)
    with f_c1:
        sel_kor_exam = st.selectbox(
            "모의고사 회차 필터",
            options=["전체"] + avail_kor_exams,
            key=f"{key_prefix}_filter_exam",
        )
    with f_c2:
        sel_kor_sec = st.selectbox(
            "영역 필터",
            options=["전체"] + _KOR_SECTIONS,
            key=f"{key_prefix}_filter_sec",
        )
    with f_c3:
        kor_kw = st.text_input(
            "주제·작품명·본문 키워드 검색",
            placeholder="예: 박완서, 현대시, 논리실증주의, 법...",
            key=f"{key_prefix}_filter_kw",
        )

    kw_low = (kor_kw or "").strip().lower()
    filtered_sets: list[dict[str, Any]] = []
    for e in set_entries:
        if sel_kor_exam != "전체" and e["exam_name"] != sel_kor_exam:
            continue
        if sel_kor_sec != "전체" and e["kor_sec"] != sel_kor_sec:
            continue
        if kw_low:
            blob = " ".join(
                [
                    e["exam_name"],
                    e["curr_subj"],
                    e["passage_topic"],
                    " ".join(e.get("key_vocabulary") or []),
                    e["passage_text"],
                ]
            ).lower()
            if kw_low not in blob:
                continue
        filtered_sets.append(e)

    return True, filtered_sets


def _render_korean_passage_preview_expander(filtered_sets: list[dict[str, Any]]) -> None:
    """조회된 국어 기출 지문 원문·어휘 미리보기 Expander를 렌더링한다."""
    with st.expander("📖 조회된 기출 국어 지문 원문·어휘 미리보기", expanded=False):
        for s_entry in filtered_sets[:30]:
            q_label = f"{s_entry['exam_name']} {s_entry['group_id']}번"
            sec_tag = s_entry["curr_subj"]
            topic_tag = (s_entry.get("passage_topic") or "-").strip()
            st.markdown(f"**📌 {q_label}** (`{sec_tag}` · 주제: {topic_tag})")
            if s_entry.get("key_vocabulary"):
                render_chips(s_entry["key_vocabulary"], variant="amber", limit=10)
            st.markdown(
                f"<div class='eng-given-box' style='white-space:pre-wrap;'>{_html.escape(s_entry['passage_text'])}</div>",
                unsafe_allow_html=True,
            )


def _execute_korean_transform_jobs(
    ctx: AppContext,
    jobs: list[dict[str, Any]],
    source_summary_label: str,
    force_regenerate: bool = False,
) -> None:
    """국어 지문 세트 변형 작업 목록을 정액 크레딧(세트당 10C · 풀 동일) 선예치·자동정산 하에 실행한다."""
    if not jobs:
        st.warning("변형할 국어 지문 세트를 최소 1개 이상 선택해 주세요.")
        return

    user_max_jobs = get_max_batch_questions(ctx.current_user)
    if user_max_jobs is not None:
        adjusted_jobs: list[dict[str, Any]] = []
        acc_q = 0
        trimmed = False
        for job in jobs:
            j_copy = dict(job)
            recs = list(j_copy["records"])
            if j_copy.get("is_direct_input") and len(recs) > user_max_jobs and not adjusted_jobs:
                recs = recs[:user_max_jobs]
                j_copy["records"] = recs
                j_copy["group_id"] = f"1~{len(recs)}"
                trimmed = True
            if acc_q + len(recs) <= user_max_jobs or not adjusted_jobs:
                adjusted_jobs.append(j_copy)
                acc_q += len(recs)
            else:
                trimmed = True
                break
        if trimmed:
            total_req_q = sum(len(j["records"]) for j in jobs)
            st.warning(
                f"⚠️ 현재 등급({TIER_LABELS.get(ctx.current_user.tier, ctx.current_user.tier)})은 1회 실행당 최대 **{user_max_jobs}문항**까지만 생성됩니다. "
                f"(요청 총 {total_req_q}문항 → 최대 {user_max_jobs}문항 범위 내로 실행)"
            )
            jobs = adjusted_jobs

    uname_kor = ctx.current_user.username if ctx.current_user is not None else ""
    reserve_credits = len(jobs) * CREDIT_COST_KOREAN_SET_TRANSFORM
    ok_res, reservation = try_reserve_credits(
        ctx,
        reserve_credits,
        f"국어 지문 세트 변형 ({len(jobs)}세트)",
    )
    if not ok_res or reservation is None:
        return

    assembled: list[dict[str, Any]] = []
    succeeded_sets = 0
    cache_hit_cnt = 0
    ai_gen_cnt = 0
    actual_credits = 0
    seen_variant_ids: list[str] = []
    try:
        from src.gemini.korean import (
            balance_korean_answer_distribution,
            generate_korean_set_reconstruction,
        )
        from src.gemini.usage import submit_with_usage_context

        slot_results: dict[int, tuple[str, str, list[Any], str]] = {}
        miss_indices: list[int] = []
        for idx_c, job in enumerate(jobs):
            kor_sec = str(job.get("kor_sec") or _classify_kor_sec(job.get("curr_subj", "")))
            kor_sub = str(job.get("kor_sub") or _extract_kor_subgenre(job.get("curr_subj", "")))
            ordered_recs = sorted(job["records"], key=lambda r: r.q_num)

            cached_hit = None
            if not force_regenerate:
                c_key = ctx.storage.build_korean_pool_key(
                    job["passage_text"],
                    group_id=job["group_id"],
                    korean_section=kor_sec,
                    question_count=len(ordered_recs),
                    korean_subgenre=kor_sub,
                )
                cached_hit = ctx.storage.get_cached_korean_set(
                    c_key,
                    expected_count=len(ordered_recs),
                    username=uname_kor,
                )
            if cached_hit is not None:
                cached_rws, hit_vid = cached_hit
                group_key = f"{ordered_recs[0].exam_name}::{job['group_id'] or ordered_recs[0].q_num}"
                hit_pairs = [
                    (
                        ordered_recs[i],
                        cached_rws[i].model_copy(
                            update={
                                "original_question_id": ordered_recs[i].id,
                                "passage_group_key": group_key,
                            }
                        ),
                    )
                    for i in range(len(ordered_recs))
                ]
                slot_results[idx_c] = (kor_sec, kor_sub, hit_pairs, hit_vid)
                cache_hit_cnt += 1
            else:
                miss_indices.append(idx_c)

        if miss_indices:
            kor_api_key, kor_key_label = resolve_active_gemini_api_key(
                ctx.current_user, ctx.creds.gemini_api_key, model_name=EXAM_TAGGING_MODEL
            )
            if not kor_api_key:
                st.error(f"국어 세트 변형용 Gemini API 키({kor_key_label})가 설정되지 않았습니다.")
            else:
                with _usage_scope(ctx):
                    n_workers = max(1, min(3, len(miss_indices)))
                    with st.spinner(
                        f"🤖 {MODEL_LABELS.get(EXAM_TAGGING_MODEL, EXAM_TAGGING_MODEL)}로 "
                        f"국어 지문 세트 {len(miss_indices)}개 신규 생성·검수 중 "
                        f"(DB 풀 즉시 로드 {cache_hit_cnt}세트)..."
                    ):
                        def _run_kor_set(set_idx: int):
                            j_item = jobs[set_idx]
                            k_sec = str(j_item.get("kor_sec") or _classify_kor_sec(j_item.get("curr_subj", "")))
                            k_sub = str(j_item.get("kor_sub") or _extract_kor_subgenre(j_item.get("curr_subj", "")))
                            pairs = generate_korean_set_reconstruction(
                                api_key=kor_api_key,
                                model_name=EXAM_TAGGING_MODEL,
                                source_records=j_item["records"],
                                passage_text=j_item["passage_text"],
                                korean_section=k_sec,
                                korean_subgenre=k_sub,
                                passage_topic=j_item.get("passage_topic", ""),
                                storage=ctx.storage,
                                use_cache=False,
                                username=uname_kor,
                            )
                            c_k = ctx.storage.build_korean_pool_key(
                                j_item["passage_text"],
                                group_id=j_item["group_id"],
                                korean_section=k_sec,
                                question_count=len(pairs),
                                korean_subgenre=k_sub,
                            )
                            vid = ctx.storage.build_korean_set_variant_id(
                                c_k, [rw.rewritten_stem for _, rw in pairs]
                            )
                            return set_idx, k_sec, k_sub, pairs, vid

                        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                            fut_map = {
                                submit_with_usage_context(executor, _run_kor_set, idx_m): idx_m
                                for idx_m in miss_indices
                            }
                            for fut in concurrent.futures.as_completed(fut_map):
                                idx_m = fut_map[fut]
                                j_item = jobs[idx_m]
                                try:
                                    _, k_sec, k_sub, pairs, vid = fut.result()
                                    if pairs:
                                        slot_results[idx_m] = (k_sec, k_sub, pairs, vid)
                                        ai_gen_cnt += 1
                                except Exception as exc:
                                    err_tag = (
                                        j_item["exam_name"]
                                        if j_item.get("is_direct_input")
                                        else f"{j_item['exam_name']} {j_item['group_id']}번"
                                    )
                                    st.error(f"❌ [{err_tag}] 세트 재구성 실패: {exc}")

        combined_pairs: list[tuple[Any, Any]] = []
        pair_metas: list[tuple[dict[str, Any], str, str, str]] = []
        for idx_c, job in enumerate(jobs):
            res_t = slot_results.get(idx_c)
            if not res_t:
                continue
            k_sec, k_sub, pairs, vid = res_t
            succeeded_sets += 1
            if vid:
                seen_variant_ids.append(vid)
            for rec, rw in pairs:
                combined_pairs.append((rec, rw))
                pair_metas.append((job, k_sec, k_sub, vid))

        if combined_pairs:
            balanced_pairs = balance_korean_answer_distribution(combined_pairs)
            for (rec, rw), (job, k_sec, k_sub, vid) in zip(balanced_pairs, pair_metas):
                is_direct = bool(job.get("is_direct_input"))
                ref_lbl = (
                    f"📚 출처: {job['exam_name']}"
                    if is_direct
                    else f"출처: {job['exam_name']} [{job['group_id']}번]"
                )
                assembled.append(
                    {
                        "question": rec,
                        "judgment": None,
                        "crop_png_bytes": None,
                        "rewritten": rw,
                        "concept_reference": ref_lbl,
                        "korean_meta": {
                            "section": k_sec,
                            "subgenre": k_sub,
                            "group_id": job["group_id"],
                            "passage_text": getattr(rw, "shared_passage", "") or job["passage_text"],
                            "variant_id": vid,
                            "is_direct_input": is_direct,
                        },
                    }
                )
        if uname_kor and seen_variant_ids:
            ctx.storage.mark_user_seen_korean_sets(uname_kor, seen_variant_ids)
        actual_credits = succeeded_sets * CREDIT_COST_KOREAN_SET_TRANSFORM
    finally:
        settle_credits(
            ctx,
            reservation,
            actual_credits,
            detail_note=f"국어 세트 변형 ({succeeded_sets}/{len(jobs)}세트 성공 · {actual_credits:,}C 확정)",
        )

    if not assembled:
        st.warning("생성된 국어 변형 문항이 없습니다. (선예치 크레딧 전액 자동 환불됨)")
        return

    consume_extraction_quota(ctx.storage, uname_kor, len(assembled))

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
            from src.pdf.exam_pdf import generate_custom_exam_pdf

            default_title = "수능·모의고사 국어 지문 세트 맞춤형 변형 모의고사"
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
                source_label=source_summary_label or "국어 기출 지문 세트",
                whitelist=None,
                two_column=True,
                include_original_for_rewritten=False,
            )
            st.session_state["generated_exam_pdf"] = pre_pdf_bytes
            st.session_state["generated_exam_pdf_sig"] = items_sig
            _record_korean_seen_for_user(ctx, assembled)
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
        uname_kor,
        charged=actual_credits,
    )
    pool_detail = f"DB 풀 즉시 로드 {cache_hit_cnt}세트 · 신규 AI 생성·DB 저장 {ai_gen_cnt}세트"
    st.success(
        f"🎯 총 **{succeeded_sets}개 지문 세트({len(assembled)}문항)**의 수능형 국어 변형 문제 출제가 완료되었습니다! "
        f"({pool_detail}) 아래 모드 A(시험지 PDF) 또는 모드 B(웹 바로 풀기)에서 확인하세요.{rem_msg}"
    )


def _render_korean_transform_submode(ctx: AppContext) -> None:
    """국어 서브모드 1: 수능 국어 지문 세트 자유 유형 변형문제 생성 UI."""
    user_max_jobs = get_max_batch_questions(ctx.current_user)
    max_limit_badge = (
        f"1회 최대 **{user_max_jobs}문항**"
        if user_max_jobs is not None
        else "1회 문항 수 **제한 없음**"
    )

    kor_tab_db, kor_tab_direct = st.tabs(
        [
            "📂 기출 DB 국어 지문에서 선택 (회차·영역별 검색)",
            "✍️ 새 국어 지문 직접 붙여넣기",
        ]
    )

    with kor_tab_db:
        has_db_recs, filtered_sets = _filter_korean_db_sets(ctx, key_prefix="tab2_kor_tr")
        if has_db_recs:
            if not filtered_sets:
                st.warning("조건에 맞는 국어 지문 세트가 없습니다. 필터를 조정해 주세요.")
            else:
                st.caption(
                    f"총 **{len(filtered_sets)}개** 국어 지문 세트가 조회되었습니다. "
                    f"(과금 방지를 위해 기본 미선택 상태이며, 현재 등급은 {max_limit_badge} 생성 가능합니다.)"
                )
                with st.form("tab2_kor_db_transform_form", border=True):
                    cap_txt = f"최대 {user_max_jobs}문항" if user_max_jobs is not None else "문항 수 제한 없음"
                    force_regen_db = st.checkbox(
                        "🔄 기존 변형 풀 무시하고 새로 문항 출제",
                        value=False,
                        key="tab2_kor_db_force_regen",
                    )

                    st.markdown("---")
                    hdr_limit = f"1회 최대 {user_max_jobs}문항" if user_max_jobs is not None else "원하는 만큼 선택 가능"
                    st.markdown(f"**✅ 변형할 국어 기출 지문 세트 선택 (모두 기본 미선택 · {hdr_limit}):**")

                    checked_sets: list[dict[str, Any]] = []
                    for s_idx, s_entry in enumerate(filtered_sets):
                        q_nums = [r.q_num for r in s_entry["records"]]
                        sec_tag = s_entry["curr_subj"]
                        topic_tag = (s_entry.get("passage_topic") or "").strip()
                        p_preview = re.sub(r"\s+", " ", (s_entry.get("passage_text") or "").strip())[:115]
                        q_pool_key_chk = ctx.storage.build_korean_pool_key(
                            s_entry["passage_text"],
                            group_id=s_entry["group_id"],
                            korean_section=s_entry["kor_sec"],
                            question_count=len(q_nums),
                            korean_subgenre=s_entry["kor_sub"],
                        )
                        pool_cnt_chk = ctx.storage.get_korean_pool_set_count(q_pool_key_chk)
                        pool_status_tag = f" · 🎯 변형DB {pool_cnt_chk}세트" if pool_cnt_chk > 0 else ""
                        row_label = (
                            f"[{s_entry['exam_name']} {s_entry['group_id']}번] 영역: {sec_tag} · {len(q_nums)}문항{pool_status_tag}"
                            + (f" · 주제: {topic_tag}" if topic_tag else "")
                            + f" — \"{p_preview}...\""
                        )
                        if st.checkbox(
                            row_label,
                            value=False,
                            key=f"tab2_kor_set_chk_{s_entry['exam_name']}_{s_entry['group_id']}_{s_idx}",
                        ):
                            checked_sets.append(s_entry)

                    est_sets = len(checked_sets)
                    est_qs = sum(len(s["records"]) for s in checked_sets)
                    if est_sets > 0:
                        eff_qs = min(est_qs, user_max_jobs) if user_max_jobs is not None else est_qs
                        st.info(
                            f"📊 **예상 AI 호출: {est_sets}세트 (총 {eff_qs}문항 · {cap_txt})** · "
                            f"요청 {est_qs}문항 (세트당 {CREDIT_COST_KOREAN_SET_TRANSFORM}C · 자유 유형 2-Gate 자동 재구성)"
                        )
                    else:
                        st.info(
                            f"📊 **예상 AI 호출: N세트 ({cap_txt})** · "
                            "체크한 세트 수만큼 호출됩니다 (세트당 자유 유형 자동 재구성)."
                        )

                    run_kor_db_btn = st.form_submit_button(
                        f"🚀 선택한 기출 국어 지문 세트 변형문제 생성 (세트당 {CREDIT_COST_KOREAN_SET_TRANSFORM}C · 캐시 풀 동일)",
                        type="primary",
                        width="stretch",
                    )

                _render_korean_passage_preview_expander(filtered_sets)

                if run_kor_db_btn:
                    if not checked_sets:
                        st.error("변형할 국어 지문 세트를 최소 1개 이상 체크해 주세요.")
                    else:
                        jobs = [
                            {
                                "exam_name": s_entry["exam_name"],
                                "group_id": s_entry["group_id"],
                                "records": s_entry["records"],
                                "passage_text": s_entry["passage_text"],
                                "passage_topic": s_entry.get("passage_topic", ""),
                                "curr_subj": s_entry["curr_subj"],
                                "kor_sec": s_entry["kor_sec"],
                                "kor_sub": s_entry["kor_sub"],
                                "is_direct_input": False,
                            }
                            for s_entry in checked_sets
                        ]
                        used_exams = list(dict.fromkeys(j["exam_name"] for j in jobs if j.get("exam_name")))
                        summary_lbl = (
                            f"{used_exams[0]} 외 {len(used_exams) - 1}개 회차 ({len(jobs)}세트)"
                            if len(used_exams) > 1
                            else (f"{used_exams[0]} ({len(jobs)}세트)" if used_exams else "국어 기출 지문 세트")
                        )
                        _execute_korean_transform_jobs(
                            ctx,
                            jobs=jobs,
                            source_summary_label=summary_lbl,
                            force_regenerate=force_regen_db,
                        )

    with kor_tab_direct:
        with st.form("tab2_kor_direct_transform_form", border=True):
            st.markdown("**✍️ 변형할 새 국어 지문을 붙여넣고 영역과 세트 문항 수를 선택하세요:**")
            d_c1, d_c2, d_c3, d_c4 = st.columns([1.3, 0.9, 1.0, 0.8])
            with d_c1:
                direct_src_name = st.text_input(
                    "출처 표기 (시험지·해설 표시용)",
                    value="직접 입력 국어 지문",
                    key="tab2_kor_direct_src",
                )
            with d_c2:
                direct_sec = st.selectbox(
                    "영역 선택",
                    options=_KOR_SECTIONS,
                    index=0,
                    key="tab2_kor_direct_sec",
                )
            with d_c3:
                direct_sub = st.text_input(
                    "세부 갈래 (선택)",
                    value="",
                    placeholder="예: 사회·경제, 현대시, 고전소설...",
                    key="tab2_kor_direct_subgenre",
                )
            with d_c4:
                direct_q_cnt = int(
                    st.selectbox(
                        "세트당 생성 문항 수",
                        options=[3, 4, 5, 6],
                        index=1,
                        format_func=lambda n: f"{n}문항",
                        key="tab2_kor_direct_q_cnt",
                    )
                )

            direct_force_regen = st.checkbox(
                "🔄 기존 변형 풀 무시하고 새로 문항 출제",
                value=False,
                key="tab2_kor_direct_force_regen",
            )
            direct_passage_raw = st.text_area(
                "국어 지문 원문 입력 (여러 개의 지문을 한 번에 넣으려면 지문 사이에 '---' 줄을 넣어 구분하세요)",
                height=220,
                placeholder="변형할 국어 지문 원문을 붙여넣으세요...",
                key="tab2_kor_direct_text",
            )
            direct_chunk_cnt = (
                len([ch for ch in re.split(r"\n\s*-{3,}\s*\n", (direct_passage_raw or "").strip()) if ch.strip()])
                or 1
            )
            raw_direct_qs = direct_chunk_cnt * direct_q_cnt
            eff_direct_qs = min(raw_direct_qs, user_max_jobs) if user_max_jobs is not None else raw_direct_qs
            lim_note_d = f"최대 {user_max_jobs}문항" if user_max_jobs is not None else "문항 수 제한 없음"
            st.info(
                f"📊 **예상 AI 호출: {direct_chunk_cnt}세트 (총 {eff_direct_qs}문항 · {lim_note_d})** · "
                f"세트당 {direct_q_cnt}문항 자유 유형 2-Gate 자동 재구성"
            )
            run_kor_direct_btn = st.form_submit_button(
                f"🚀 직접 입력한 국어 지문으로 수능형 변형 문제 생성 (세트당 {CREDIT_COST_KOREAN_SET_TRANSFORM}C · 캐시 풀 동일)",
                type="primary",
                width="stretch",
            )

        if run_kor_direct_btn:
            raw_chunks = [
                ch.strip()
                for ch in re.split(r"\n\s*-{3,}\s*\n", (direct_passage_raw or "").strip())
                if ch.strip()
            ]
            if not raw_chunks:
                st.error("변형할 국어 지문 텍스트를 입력해 주세요.")
            else:
                base_src = (direct_src_name or "직접 입력 국어 지문").strip()
                clean_sub = (direct_sub or "").strip()
                curr_subj_lbl = f"{direct_sec}({clean_sub})" if clean_sub else direct_sec
                now_str = datetime.now().isoformat(timespec="seconds")
                direct_jobs: list[dict[str, Any]] = []
                for c_idx, chunk_txt in enumerate(raw_chunks, start=1):
                    src_title = f"{base_src} #{c_idx}" if len(raw_chunks) > 1 else base_src
                    chunk_hash = hashlib.md5(chunk_txt.encode("utf-8")).hexdigest()[:8]
                    gid = f"1~{direct_q_cnt}"
                    virtual_recs = [
                        QuestionRecord(
                            id=f"direct_kor_{chunk_hash}_{c_idx}_{q_i}",
                            exam_name=src_title,
                            q_num=q_i,
                            subject_category="국어",
                            curriculum_subject=curr_subj_lbl,
                            unit_name="직접 입력 지문",
                            min_grade=1,
                            core_principles=["국어 지문 세트 변형"],
                            required_terms=[],
                            is_single_concept=False,
                            answer="1",
                            official_explanation="",
                            question_text=chunk_txt,
                            passage_group_id=gid,
                            passage_text=chunk_txt,
                            created_at=now_str,
                        )
                        for q_i in range(1, direct_q_cnt + 1)
                    ]
                    direct_jobs.append(
                        {
                            "exam_name": src_title,
                            "group_id": gid,
                            "records": virtual_recs,
                            "passage_text": chunk_txt,
                            "passage_topic": "",
                            "curr_subj": curr_subj_lbl,
                            "kor_sec": direct_sec,
                            "kor_sub": clean_sub,
                            "is_direct_input": True,
                        }
                    )
                _execute_korean_transform_jobs(
                    ctx,
                    jobs=direct_jobs,
                    source_summary_label=base_src,
                    force_regenerate=direct_force_regen,
                )


def _render_tab2_korean_transform_section(ctx: AppContext) -> None:
    """탭 2 국어 과목 전용: [변형문제 생성]과 [국어 지문 분석지 PDF 생성] 작업 모드로 분리해 렌더링한다."""
    st.subheader("1️⃣ 국어 작업 모드 선택")
    kor_sub_mode = st.radio(
        "수행할 국어 작업을 선택하세요",
        options=[
            "🚀 국어 지문 세트 변형문제 생성",
            "📑 국어 지문 정밀 분석지 PDF 생성",
        ],
        horizontal=True,
        key="tab2_kor_sub_mode",
    )
    if kor_sub_mode.startswith("📑"):
        from src.ui.tab2_korean_analysis import _render_korean_analysis_submode

        _render_korean_analysis_submode(ctx)
    else:
        _render_korean_transform_submode(ctx)


def render_tab2_korean_transform_section(
    ctx: AppContext | None = None,
    **_: Any,
) -> None:
    """하위 호환 래퍼: AppContext를 받아 국어 작업 모드 섹션을 렌더링한다."""
    if ctx is not None:
        _render_tab2_korean_transform_section(ctx)
