from __future__ import annotations

import concurrent.futures
import html
import re
from typing import Any, Callable

import streamlit as st

from src.auth import (
    CREDIT_COST_KOREAN_ANALYSIS,
    CREDIT_COST_KOREAN_SET_TRANSFORM,
    acquire_credit_reservation,
    get_max_batch_questions,
    resolve_active_gemini_api_key,
    settle_credit_reservation,
)
from src.config import (
    EXAM_TAGGING_MODEL,
    MODEL_LABELS,
)
from src.gemini import submit_with_usage_context
from src.models import normalize_korean_group_id
from src.ui.common import format_credit_balance_inline


def render_tab2_korean_transform_section(
    *,
    storage: Any,
    current_user: Any,
    creds: Any,
    ensure_token_quota_fn: Callable[[], bool],
    usage_scope_fn: Callable[[], Any],
) -> None:
    """[탭 2 - 국어 모드] DB에 저장된 국어 독서·문학 지문 세트를 선택해 세트 전체 문항을 자유 유형으로 재구성한다."""
    st.subheader("1️⃣ 국어(독서 · 문학) 기출 지문 세트 선택 & 전체 문항 재구성")
    max_batch_q = get_max_batch_questions(current_user)
    user_max_jobs = None if max_batch_q is None else max_batch_q
    max_limit_badge = "제한 없음" if user_max_jobs is None else f"1회 최대 **{user_max_jobs}문항**까지"
    st.caption(
        f"💡 **[탭 1]**에서 수집한 국어 전 영역(**독서 · 문학 · 화법과 작문 · 언어와 매체**) 지문/문항 세트를 선택하면, "
        f"해당 세트의 문항 수만큼 AI가 핵심 출제 포인트를 바탕으로 **자유 유형 세트 문항을 일괄 재창작**합니다. ({max_limit_badge})"
    )

    all_db = storage.load_db_readonly()
    kor_recs = [
        r
        for r in all_db
        if r.subject_category == "국어"
        and bool((getattr(r, "passage_text", "") or "").strip())
    ]

    if not kor_recs:
        st.info(
            "아직 DB에 저장된 **국어(독서·문학) 지문 세트**가 없습니다. "
            "**[📥 탭 1]**에서 국어 모의고사 문제지+해설지 PDF를 먼저 수집·추출해 주세요."
        )
        return

    # (exam_name, passage_group_id) 단위로 세트 묶기
    sets_map: dict[tuple[str, str], dict[str, Any]] = {}
    for r in sorted(kor_recs, key=lambda x: (x.exam_name, x.q_num)):
        gid = normalize_korean_group_id(getattr(r, "passage_group_id", ""), r.q_num)
        key = (r.exam_name, gid)
        entry = sets_map.setdefault(
            key,
            {
                "exam_name": r.exam_name,
                "group_id": gid,
                "records": [],
                "passage_text": "",
                "passage_topic": "",
                "curr_subj": r.curriculum_subject or "독서",
            },
        )
        entry["records"].append(r)
        p_txt = (getattr(r, "passage_text", "") or "").strip()
        if len(p_txt) > len(entry["passage_text"]):
            entry["passage_text"] = p_txt
        t_txt = (getattr(r, "passage_topic", "") or "").strip()
        if len(t_txt) > len(entry["passage_topic"]):
            entry["passage_topic"] = t_txt

    set_entries = sorted(
        sets_map.values(),
        key=lambda e: (e["exam_name"], min(x.q_num for x in e["records"])),
    )

    fc1, fc2, fc3 = st.columns([1.3, 1.1, 1.3])
    avail_kor_exams = sorted({e["exam_name"] for e in set_entries}, reverse=True)
    with fc1:
        sel_kor_exams = st.multiselect(
            "모의고사 회차 필터 (복수 선택)",
            options=avail_kor_exams,
            default=[],
            placeholder="전체 회차",
            key="tab2_kor_filter_exams",
        )
    with fc2:
        sel_kor_secs = st.multiselect(
            "영역 필터 (복수 선택)",
            options=["독서", "문학", "화법과 작문", "언어와 매체"],
            default=[],
            placeholder="전체 영역",
            key="tab2_kor_filter_secs",
        )
    with fc3:
        kor_kw = st.text_input(
            "주제·작품명·본문 키워드 검색",
            placeholder="예: 박완서, 현대시, 논리실증주의, 법...",
            key="tab2_kor_filter_kw",
        )

    def _classify_kor_sec(cs: str) -> str:
        if "문학" in cs:
            return "문학"
        if "화법" in cs or "작문" in cs:
            return "화법과 작문"
        if "언어" in cs or "매체" in cs or "문법" in cs:
            return "언어와 매체"
        return "독서"

    kw_low = (kor_kw or "").strip().lower()
    filtered_sets: list[dict[str, Any]] = []
    for e in set_entries:
        if sel_kor_exams and e["exam_name"] not in sel_kor_exams:
            continue
        sec_label = _classify_kor_sec(e["curr_subj"])
        if sel_kor_secs and sec_label not in sel_kor_secs:
            continue
        if kw_low:
            blob = " ".join(
                [e["exam_name"], e["curr_subj"], e["passage_topic"], e["passage_text"]]
            ).lower()
            if kw_low not in blob:
                continue
        filtered_sets.append(e)

    if not filtered_sets:
        st.warning("조건에 맞는 국어 지문 세트가 없습니다. 필터를 조정해 주세요.")
        return

    st.caption(f"총 **{len(filtered_sets)}개** 국어 지문 세트가 조회되었습니다. 재구성할 세트를 체크한 뒤 실행 버튼을 누르세요.")

    with st.form("tab2_kor_set_transform_form", border=True):
        selected_set_indices: list[int] = []
        for idx, s_entry in enumerate(filtered_sets):
            q_nums = [r.q_num for r in s_entry["records"]]
            sec_tag = s_entry["curr_subj"]
            topic_tag = s_entry["passage_topic"] or "주제 미상"
            kor_sec_chk = _classify_kor_sec(sec_tag)
            c_key_chk = storage.build_korean_analysis_key(
                s_entry["passage_text"],
                s_entry["group_id"],
                kor_sec_chk,
            )
            pdf_status_tag = " · 📑 분석지PDF DB" if storage.has_korean_analysis_pdf(c_key_chk) else ""
            set_label = f"[{s_entry['exam_name']}] [{s_entry['group_id']}번] ({sec_tag} · {len(q_nums)}문항{pdf_status_tag}) — {topic_tag}"
            is_chk = st.checkbox(
                set_label,
                value=False,
                key=f"tab2_kor_set_chk_{s_entry['exam_name']}_{s_entry['group_id']}_{idx}",
            )
            if is_chk:
                selected_set_indices.append(idx)
            with st.expander(f"📄 지문 미리보기: {s_entry['exam_name']} [{s_entry['group_id']}번]", expanded=False):
                st.text(s_entry["passage_text"][:1500] + ("..." if len(s_entry["passage_text"]) > 1500 else ""))

        opt_c1, opt_c2 = st.columns(2)
        with opt_c1:
            include_kor_diagrams = st.checkbox(
                "🎨 비문학·문학 이해용 도식 그림 포함",
                value=True,
                key="tab2_kor_include_diagrams",
            )
        with opt_c2:
            include_kor_passage = st.checkbox(
                "📄 원문 지문 전체 포함 (ON/OFF)",
                value=False,
                key="tab2_kor_include_passage",
            )
        opt_c3, opt_c4 = st.columns(2)
        with opt_c3:
            enable_kor_web_search = st.checkbox(
                "🌐 문학 웹검색 정밀 분석 (EBS·백과 우선)",
                value=True,
                key="tab2_kor_enable_web_search",
            )
        with opt_c4:
            enable_kor_proofread = st.checkbox(
                "🔍 2차 원문 대조 교정 (느림)",
                value=False,
                key="tab2_kor_enable_proofread",
            )
        kb_c1, kb_c2 = st.columns(2)
        with kb_c1:
            run_kor_btn = st.form_submit_button(
                f"🚀 선택한 국어 지문 세트 변형문제 생성 (세트당 {CREDIT_COST_KOREAN_SET_TRANSFORM}C)",
                type="primary",
                width="stretch",
            )
        with kb_c2:
            run_kor_analysis_btn = st.form_submit_button(
                f"📑 선택한 지문 분석지 PDF 생성 (세트당 {CREDIT_COST_KOREAN_ANALYSIS}C · 캐시 동일)",
                type="secondary",
                width="stretch",
            )

    if run_kor_btn:
        if not selected_set_indices:
            st.error("최소 1개 이상의 국어 지문 세트를 체크해 주세요.")
            return

        chosen_sets = [filtered_sets[i] for i in selected_set_indices]
        total_est_q = sum(len(s["records"]) for s in chosen_sets)
        if user_max_jobs is not None and total_est_q > user_max_jobs:
            st.warning(
                f"⚠️ 현재 등급은 1회 실행당 최대 {user_max_jobs}문항까지만 생성됩니다. "
                f"(선택 세트 총 {total_est_q}문항 → 앞의 세트부터 최대 {user_max_jobs}문항 범위 내로 제한)"
            )
            trimmed_sets: list[dict[str, Any]] = []
            acc = 0
            for s in chosen_sets:
                if acc + len(s["records"]) <= user_max_jobs or not trimmed_sets:
                    trimmed_sets.append(s)
                    acc += len(s["records"])
                else:
                    break
            chosen_sets = trimmed_sets

        kor_api_key, kor_key_label = resolve_active_gemini_api_key(
            current_user, creds.gemini_api_key, model_name=EXAM_TAGGING_MODEL
        )
        if not kor_api_key:
            st.error(f"사용 가능한 Gemini API 키({kor_key_label})가 설정되지 않았습니다.")
            return

        uname_kor = current_user.username if current_user is not None else ""
        reserve_c = len(chosen_sets) * CREDIT_COST_KOREAN_SET_TRANSFORM
        ok_res, reservation, res_msg = acquire_credit_reservation(
            storage,
            uname_kor,
            reserve_credits=reserve_c,
            reason=f"국어 지문 세트 변형 ({len(chosen_sets)}세트)",
        )
        if not ok_res or reservation is None:
            if res_msg:
                st.error(res_msg)
            return

        new_matched: list[dict[str, Any]] = []
        succeeded_sets = 0
        actual_c = 0
        try:
            with usage_scope_fn():
                with st.spinner(f"🤖 {MODEL_LABELS.get(EXAM_TAGGING_MODEL, EXAM_TAGGING_MODEL)}로 국어 지문 세트({len(chosen_sets)}개 세트) 변형문제 병렬 생성·검증 중..."):
                    from src.gemini.korean import (
                        balance_korean_answer_distribution,
                        generate_korean_set_reconstruction,
                    )

                    def _run_kor_set(set_idx: int, s_entry: dict[str, Any]):
                        curr_subj = s_entry["curr_subj"]
                        kor_sec = _classify_kor_sec(curr_subj)
                        m_sub = re.search(r"\(([^)]+)\)", curr_subj)
                        kor_sub = m_sub.group(1) if m_sub else ""
                        pairs = generate_korean_set_reconstruction(
                            api_key=kor_api_key,
                            model_name=EXAM_TAGGING_MODEL,
                            source_records=s_entry["records"],
                            passage_text=s_entry["passage_text"],
                            korean_section=kor_sec,
                            korean_subgenre=kor_sub,
                            passage_topic=s_entry["passage_topic"],
                        )
                        return set_idx, kor_sec, kor_sub, pairs

                    slot_results: dict[int, tuple[str, str, list[Any]]] = {}
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=min(3, len(chosen_sets))
                    ) as executor:
                        fut_map = {
                            submit_with_usage_context(executor, _run_kor_set, idx_c, s_entry): (idx_c, s_entry)
                            for idx_c, s_entry in enumerate(chosen_sets)
                        }
                        for fut in concurrent.futures.as_completed(fut_map):
                            idx_c, s_entry = fut_map[fut]
                            try:
                                _, kor_sec, kor_sub, pairs = fut.result()
                                if pairs:
                                    slot_results[idx_c] = (kor_sec, kor_sub, pairs)
                            except Exception as exc:
                                st.error(f"[{s_entry['exam_name']} {s_entry['group_id']}번] 세트 재구성 실패: {exc}")

                    combined_pairs: list[tuple[Any, Any]] = []
                    pair_metas: list[tuple[dict[str, Any], str, str]] = []
                    for idx_c, s_entry in enumerate(chosen_sets):
                        res_t = slot_results.get(idx_c)
                        if not res_t:
                            continue
                        kor_sec, kor_sub, pairs = res_t
                        succeeded_sets += 1
                        for rec, rw in pairs:
                            combined_pairs.append((rec, rw))
                            pair_metas.append((s_entry, kor_sec, kor_sub))

                    if combined_pairs:
                        balanced_pairs = balance_korean_answer_distribution(combined_pairs)
                        for (rec, rw), (s_entry, kor_sec, kor_sub) in zip(balanced_pairs, pair_metas):
                            new_matched.append(
                                {
                                    "question": rec,
                                    "judgment": None,
                                    "crop_png_bytes": None,
                                    "rewritten": rw,
                                    "concept_reference": f"출처: {s_entry['exam_name']} [{s_entry['group_id']}번]",
                                    "korean_meta": {
                                        "section": kor_sec,
                                        "subgenre": kor_sub,
                                        "group_id": s_entry["group_id"],
                                        "passage_text": getattr(rw, "shared_passage", "") or s_entry["passage_text"],
                                    },
                                }
                            )
            actual_c = succeeded_sets * CREDIT_COST_KOREAN_SET_TRANSFORM
        finally:
            settle_credit_reservation(
                storage,
                reservation,
                actual_credits=actual_c,
                detail_note=f"국어 세트 변형 ({succeeded_sets}/{len(chosen_sets)}세트 성공 · {actual_c:,}C 확정)",
            )

        if new_matched:
            exam_labels = list(dict.fromkeys(s["exam_name"] for s in chosen_sets if s.get("exam_name")))
            st.session_state["matched_items"] = new_matched
            st.session_state["concept_source_label"] = ", ".join(exam_labels) if exam_labels else "국어 기출 지문 세트"
            st.session_state["generated_exam_pdf"] = None
            st.session_state["generated_exam_pdf_sig"] = ""
            st.session_state["user_answers"] = {}
            st.session_state["graded"] = False
            st.session_state["ai_explanations"] = {}
            st.session_state["ai_chats"] = {}
            bal_msg = format_credit_balance_inline(storage, uname_kor, charged=actual_c)
            st.success(
                f"🎯 총 {succeeded_sets}개 지문 세트({len(new_matched)}문항) 변형문제 생성 및 2-Gate 검수 완료! 아래 [모드 A] PDF 또는 [모드 B] 웹 풀기를 이용해 주세요.{bal_msg}"
            )

    if run_kor_analysis_btn:
        if not selected_set_indices:
            st.error("분석지를 생성할 국어 지문 세트를 최소 1개 이상 체크해 주세요.")
            return

        chosen_sets = [filtered_sets[i] for i in selected_set_indices]
        uname_kor = current_user.username if current_user is not None else ""
        reserve_c_an = len(chosen_sets) * CREDIT_COST_KOREAN_ANALYSIS
        ok_res_an, reservation_an, res_msg_an = acquire_credit_reservation(
            storage,
            uname_kor,
            reserve_credits=reserve_c_an,
            reason=f"국어 지문 분석지 ({len(chosen_sets)}세트)",
        )
        if not ok_res_an or reservation_an is None:
            if res_msg_an:
                st.error(res_msg_an)
            return

        actual_c_an = 0
        succeeded_an_sets = 0
        try:
            prepared_slots: list[dict[str, Any]] = []
            missing_ai_indices: list[int] = []
            missing_diag_indices: list[int] = []
            for idx_s, s_entry in enumerate(chosen_sets):
                curr_subj = s_entry["curr_subj"]
                kor_sec = _classify_kor_sec(curr_subj)
                m_sub = re.search(r"\(([^)]+)\)", curr_subj)
                kor_sub = m_sub.group(1) if m_sub else ""
                c_key = storage.build_korean_analysis_key(
                    s_entry["passage_text"],
                    s_entry["group_id"],
                    kor_sec,
                )
                cached_an = storage.get_cached_korean_analysis(c_key)
                cached_pdf = storage.get_korean_analysis_pdf_bytes(c_key) if include_kor_diagrams else None
                prepared_slots.append(
                    {
                        "entry": s_entry,
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
                    include_kor_diagrams
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
                    current_user, creds.gemini_api_key, model_name=EXAM_TAGGING_MODEL
                )
                if not kor_api_key:
                    st.error(f"사용 가능한 Gemini API 키({kor_key_label})가 설정되지 않았습니다.")
                    return

                with usage_scope_fn():
                    need_total = len(missing_ai_indices) + len(missing_diag_indices)
                    mode_lbl = "2차 교정 포함" if enable_kor_proofread else "고속 병렬"
                    with st.spinner(
                        f"📑 {MODEL_LABELS.get(EXAM_TAGGING_MODEL, EXAM_TAGGING_MODEL)}로 "
                        f"국어 지문 분석지·도식 생성 ({mode_lbl}) {need_total}건 병렬 진행 중 "
                        f"(DB 로드 {len(chosen_sets) - len(missing_ai_indices)}세트)..."
                    ):
                        if missing_ai_indices:
                            def _run_ai_slot(slot_i: int):
                                from src.gemini.korean import generate_korean_passage_analysis

                                slot = prepared_slots[slot_i]
                                s_entry = slot["entry"]
                                an_out = generate_korean_passage_analysis(
                                    api_key=kor_api_key,
                                    model_name=EXAM_TAGGING_MODEL,
                                    exam_name=s_entry["exam_name"],
                                    group_id=s_entry["group_id"],
                                    passage_text=s_entry["passage_text"],
                                    korean_section=slot["kor_sec"],
                                    korean_subgenre=slot["kor_sub"],
                                    passage_topic=s_entry["passage_topic"],
                                    enable_proofread=enable_kor_proofread,
                                    enable_web_search=enable_kor_web_search,
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
                                    s_entry = slot["entry"]
                                    try:
                                        _, an_out = fut.result()
                                        slot["analysis"] = an_out
                                        ai_gen_cnt += 1
                                        storage.save_cached_korean_analysis(
                                            slot["cache_key"],
                                            an_out,
                                            exam_name=s_entry["exam_name"],
                                            group_id=s_entry["group_id"],
                                            korean_section=slot["kor_sec"],
                                            korean_subgenre=slot["kor_sub"],
                                        )
                                    except Exception as exc:
                                        st.error(f"[{s_entry['exam_name']} {s_entry['group_id']}번] 지문 분석 실패: {exc}")

                        valid_diag_indices = [
                            si for si in missing_diag_indices if prepared_slots[si]["analysis"] is not None
                        ]
                        if valid_diag_indices:
                            def _run_diag_slot(slot_i: int):
                                from src.gemini.korean import generate_korean_visual_diagram

                                slot = prepared_slots[slot_i]
                                s_entry = slot["entry"]
                                cur_an = slot["analysis"]
                                diag_out = generate_korean_visual_diagram(
                                    api_key=kor_api_key,
                                    model_name=EXAM_TAGGING_MODEL,
                                    passage_text=s_entry["passage_text"],
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
                                    s_entry = slot["entry"]
                                    cur_an = slot["analysis"]
                                    try:
                                        _, diag_out = fut.result()
                                        updated_an = cur_an.model_copy(update={"visual_diagram": diag_out})
                                        slot["analysis"] = updated_an
                                        storage.save_cached_korean_analysis(
                                            slot["cache_key"],
                                            updated_an,
                                            exam_name=s_entry["exam_name"],
                                            group_id=s_entry["group_id"],
                                            korean_section=slot["kor_sec"],
                                            korean_subgenre=slot["kor_sub"],
                                        )
                                    except Exception as exc:
                                        st.warning(f"[{s_entry['exam_name']} {s_entry['group_id']}번] 이해 도식 생성 건너뜀: {exc}")

            analyses_list: list[dict[str, Any]] = []
            pdf_parts: list[bytes] = []
            if prepared_slots:
                from src.pdf.korean_analysis_pdf import (
                    _is_korean_poetry_item,
                    generate_korean_passage_analysis_pdf,
                    generate_korean_passage_only_pdf,
                )

            for slot in prepared_slots:
                if slot["analysis"] is None and slot["pdf_bytes"] is None:
                    continue
                s_entry = slot["entry"]
                single_item = {
                    "exam_name": s_entry["exam_name"],
                    "group_id": s_entry["group_id"],
                    "section_label": slot["curr_subj"],
                    "subgenre": slot["kor_sub"],
                    "passage_text": s_entry["passage_text"],
                    "analysis": slot["analysis"],
                }
                is_poetry_slot = _is_korean_poetry_item(
                    slot["curr_subj"], slot["kor_sub"], s_entry["passage_text"], slot["analysis"]
                )
                if slot["pdf_bytes"] is None:
                    try:
                        single_pdf = generate_korean_passage_analysis_pdf(
                            [single_item],
                            include_diagrams=include_kor_diagrams,
                            include_passage=False,
                        )
                        if include_kor_diagrams:
                            storage.save_korean_analysis_pdf_bytes(slot["cache_key"], single_pdf)
                        slot["pdf_bytes"] = single_pdf
                        if slot not in [prepared_slots[i] for i in missing_ai_indices]:
                            json_to_pdf_cnt += 1
                    except Exception as exc:
                        st.error(f"[{s_entry['exam_name']} {s_entry['group_id']}번] 지문 분석 PDF 생성 실패: {exc}")
                        continue

                # 시 갈래는 분석 PDF 자체에 밑줄 원문이 항상 포함되므로 원문 전용 PDF 중복 병합 제외
                passage_pdf_bytes = None
                if include_kor_passage and not is_poetry_slot:
                    passage_pdf_bytes = storage.get_korean_passage_pdf_bytes(slot["cache_key"])
                    if passage_pdf_bytes is None and (s_entry.get("passage_text") or "").strip():
                        try:
                            passage_pdf_bytes = generate_korean_passage_only_pdf(single_item)
                            storage.save_korean_passage_pdf_bytes(slot["cache_key"], passage_pdf_bytes)
                        except Exception as exc_p:
                            st.warning(f"[{s_entry['exam_name']} {s_entry['group_id']}번] 원문 PDF 생성 건너뜀: {exc_p}")

                pdf_parts.append(slot["pdf_bytes"])
                succeeded_an_sets += 1
                if include_kor_passage and passage_pdf_bytes:
                    pdf_parts.append(passage_pdf_bytes)
                if slot["analysis"] is not None:
                    analyses_list.append(single_item)

            if pdf_parts:
                try:
                    from src.pdf.common import merge_korean_analysis_pdfs

                    pdf_bytes = merge_korean_analysis_pdfs(pdf_parts)
                    st.session_state["kor_analysis_results"] = analyses_list
                    st.session_state["kor_analysis_pdf"] = pdf_bytes
                    actual_c_an = succeeded_an_sets * CREDIT_COST_KOREAN_ANALYSIS
                except Exception as exc:
                    st.error(f"분석지 PDF 병합 실패: {exc}")
                    actual_c_an = 0
        finally:
            settle_credit_reservation(
                storage,
                reservation_an,
                actual_credits=actual_c_an,
                detail_note=f"국어 지문 분석지 ({succeeded_an_sets}/{len(chosen_sets)}세트 성공 · {actual_c_an:,}C 확정)",
            )

        if actual_c_an > 0:
            detail_notes: list[str] = [f"저장 PDF 즉시 이어붙이기 {pdf_hit_cnt}세트"]
            if json_to_pdf_cnt > 0:
                detail_notes.append(f"기존 분석 JSON→PDF 저장 {json_to_pdf_cnt}세트")
            if ai_gen_cnt > 0:
                detail_notes.append(f"신규 AI 분석+PDF 저장 {ai_gen_cnt}세트")
            mode_note = "원문 지문 포함 ON" if include_kor_passage else "원문 지문 제외 OFF"
            bal_msg = format_credit_balance_inline(storage, uname_kor, charged=actual_c_an)
            st.success(
                f"📑 총 {len(analyses_list)}개 국어 지문 분석지 합본 PDF가 생성되었습니다! ({mode_note} · "
                f"{' · '.join(detail_notes)}){bal_msg}"
            )

    saved_analyses: list[dict[str, Any]] = st.session_state.get("kor_analysis_results") or []
    saved_analysis_pdf: bytes | None = st.session_state.get("kor_analysis_pdf")
    if saved_analyses and saved_analysis_pdf:
        st.markdown("#### 📑 생성된 국어 정밀 지문 분석지 (수능·학원형)")
        with st.container(border=True):
            dl_c1, dl_c2 = st.columns([2.2, 1.0])
            with dl_c1:
                st.write(f"총 **{len(saved_analyses)}개** 지문 세트 분석 완료 (핵심 주제 · 전개 흐름 · 문단별 요약 · 표지/어휘 · 출제 포인트 포함)")
            with dl_c2:
                st.download_button(
                    "📥 지문 분석지 PDF 다운로드",
                    data=saved_analysis_pdf,
                    file_name="국어_정밀_지문_분석지.pdf",
                    mime="application/pdf",
                    type="primary",
                    width="stretch",
                    key="kor_analysis_pdf_dl_btn",
                )
            for idx_a, an_item in enumerate(saved_analyses, start=1):
                an = an_item["analysis"]
                with st.expander(
                    f"{idx_a}. [{an_item['exam_name']} · {an_item['group_id']}번] {an.title} ({an_item['section_label']})",
                    expanded=(idx_a == 1),
                ):
                    st.info(f"**핵심 주제:** {an.core_theme}")
                    st.success(f"**전개 흐름:** {an.structure_flow}")
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


