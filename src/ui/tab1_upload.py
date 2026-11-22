from __future__ import annotations

import hashlib
import html as _html
import os
import re
import time
from typing import Any

import streamlit as st

from src.auth import (
    consume_extraction_quota,
    get_monthly_token_limit,
    record_exam_upload_for_user,
    resolve_active_gemini_api_key,
)
from src.config import EXAM_TAGGING_MODEL, MODEL_LABELS
from src.curriculum_2022 import get_official_subjects_for_category
from src.ebsi_fetcher import attempt_ebsi_pdf_fetch
from src.models import (
    QuestionRecord,
    infer_subject_category,
    normalize_english_group_id,
    normalize_exam_title_year,
    normalize_korean_group_id,
)
from src.pdf_parser import CroppedQuestion, extract_exam_metadata_fast, render_pdf_page_png
from src.storage import StorageManager
from src.ui.common import (
    _build_cached_exam_exports,
    _ensure_token_quota,
    _usage_scope,
    sanitize_id_part,
)
from src.ui.context import AppContext
from src.ui.tab1_tagging import (
    _compute_pairing_signature,
    _compute_pairing_widget_hash,
    _crop_single_exam_pdf,
    _detect_single_exam_cover,
    _invalidate_auto_pair_cache,
    _pdf_display_basename,
    _resolve_auto_answer_matches,
    _resolve_mixed_pdf_split_with_ai,
    _tag_and_save_single_exam,
)
from src.ui_styles import render_chips


def _render_question_tag_details(
    rec: QuestionRecord,
    storage: StorageManager,
    key_prefix: str,
    allow_delete: bool = False,
) -> None:
    """문항 1건의 2022 원리/용어 분리 결과(또는 영어 전용 지문·유형·어휘)·인라인 교정 UI를 공통 렌더링한다."""
    if rec.subject_category == "영어":
        eng_sec = getattr(rec, "english_section", "") or ("듣기" if rec.q_num <= 17 else "독해")
        eng_type = getattr(rec, "english_question_type", "") or rec.unit_name
        grp_id = normalize_english_group_id(rec.q_num) if rec.subject_category == "영어" else getattr(rec, "passage_group_id", "")
        topic = getattr(rec, "passage_topic", "") or "-"
        vocab_list = list(getattr(rec, "key_vocabulary", []) or rec.required_terms or [])
        passage_txt = (getattr(rec, "passage_text", "") or "").strip()

        badge_bits = [f"**영역:** `{eng_sec}`", f"**원래 유형:** `{eng_type}`"]
        if grp_id:
            badge_bits.append(f"**세트 그룹:** `[{grp_id}]`")
        st.caption(" · ".join(badge_bits))
        st.write(f"**지문 주제·소재 (`passage_topic`):** {topic}")
        if vocab_list:
            st.markdown("**핵심 어휘 (`key_vocabulary`)**")
            render_chips(vocab_list, variant="amber")
        if passage_txt:
            with st.expander("📖 추출된 영어 지문 원문 (`passage_text`) 보기", expanded=False):
                st.markdown(
                    f"<div class='eng-given-box' style='white-space:pre-wrap;'>{_html.escape(passage_txt)}</div>",
                    unsafe_allow_html=True,
                )
        st.write(f"**공식 해설 요약:** {rec.official_explanation}")

        with st.expander("✏️ 영어 유형·주제·지문 원문·정답 빠른 수정", expanded=False):
            ec1, ec2, ec3 = st.columns([1.2, 1.8, 0.8])
            with ec1:
                new_eng_type = st.text_input(
                    "원래 문항 유형",
                    value=eng_type,
                    key=f"{key_prefix}_edit_eng_type_{rec.id}",
                )
            with ec2:
                new_topic = st.text_input(
                    "지문 주제·소재",
                    value="" if topic == "-" else topic,
                    key=f"{key_prefix}_edit_eng_topic_{rec.id}",
                )
            with ec3:
                new_ans = st.text_input(
                    "정답",
                    value=rec.answer,
                    key=f"{key_prefix}_edit_eng_ans_{rec.id}",
                )
            new_passage = st.text_area(
                "영어 지문 원문 (passage_text)",
                value=passage_txt,
                height=130,
                key=f"{key_prefix}_edit_eng_passage_{rec.id}",
            )
            if st.button("💾 영어 문항 수정 저장", key=f"{key_prefix}_save_eng_edit_{rec.id}"):
                all_recs = storage.load_db()
                for r in all_recs:
                    if r.id == rec.id:
                        r.english_question_type = new_eng_type.strip() or r.english_question_type
                        r.unit_name = r.english_question_type or r.unit_name
                        r.passage_topic = new_topic.strip()
                        r.core_principles = [r.passage_topic] if r.passage_topic else [r.unit_name]
                        r.answer = new_ans.strip() or r.answer
                        r.passage_text = new_passage.strip()
                        break
                storage.save_db(all_recs, target_exams={rec.exam_name})
                _build_cached_exam_exports.clear()
                st.success("영어 문항 정보를 저장했습니다.")
                st.rerun()

        if allow_delete:
            if st.button("이 문항 삭제", key=f"{key_prefix}_del_{rec.id}"):
                storage.delete_question(rec.id)
                _build_cached_exam_exports.clear()
                st.rerun()
        return

    if rec.subject_category == "국어":
        kor_sub = rec.curriculum_subject or "독서"
        kor_qtype = getattr(rec, "english_question_type", "") or rec.unit_name
        grp_id = normalize_korean_group_id(getattr(rec, "passage_group_id", ""), rec.q_num)
        topic = getattr(rec, "passage_topic", "") or "-"
        vocab_list = list(getattr(rec, "key_vocabulary", []) or rec.required_terms or [])
        passage_txt = (getattr(rec, "passage_text", "") or "").strip()

        badge_bits = [f"**영역/갈래:** `{kor_sub}`", f"**원래 유형:** `{kor_qtype}`"]
        if grp_id:
            badge_bits.append(f"**지문 세트:** `[{grp_id}]`")
        st.caption(" · ".join(badge_bits))
        st.write(f"**지문 주제·작품 (`passage_topic`):** {topic}")
        if vocab_list:
            st.markdown("**핵심 어휘·개념 (`key_vocabulary`)**")
            render_chips(vocab_list, variant="amber")
        p_img_fn = (getattr(rec, "passage_image_filename", "") or "").strip()
        if p_img_fn:
            p_img_bytes = storage.get_crop_bytes(
                p_img_fn,
                subject_category=rec.subject_category,
                exam_name=rec.exam_name,
            )
            if p_img_bytes:
                with st.expander(f"🖼️ 분리된 공통 지문 이미지 (`{grp_id or p_img_fn}`) 보기", expanded=False):
                    st.image(p_img_bytes, caption=f"[{grp_id}] 공통 지문", width="stretch")
        if passage_txt:
            with st.expander("📖 추출된 국어 지문 원문 (`passage_text`) 보기", expanded=False):
                st.markdown(
                    f"<div class='eng-given-box' style='white-space:pre-wrap;'>{_html.escape(passage_txt)}</div>",
                    unsafe_allow_html=True,
                )
        st.write(f"**공식 해설 요약:** {rec.official_explanation}")

        with st.expander("✏️ 국어 갈래·유형·주제·지문 원문·정답 빠른 수정", expanded=False):
            kc1, kc2, kc3 = st.columns([1.2, 1.8, 0.8])
            with kc1:
                new_kor_type = st.text_input(
                    "원래 문항 유형",
                    value=kor_qtype,
                    key=f"{key_prefix}_edit_kor_type_{rec.id}",
                )
            with kc2:
                new_kor_topic = st.text_input(
                    "지문 주제·작품",
                    value="" if topic == "-" else topic,
                    key=f"{key_prefix}_edit_kor_topic_{rec.id}",
                )
            with kc3:
                new_kor_ans = st.text_input(
                    "정답",
                    value=rec.answer,
                    key=f"{key_prefix}_edit_kor_ans_{rec.id}",
                )
            new_kor_passage = st.text_area(
                "국어 지문 원문 (passage_text)",
                value=passage_txt,
                height=140,
                key=f"{key_prefix}_edit_kor_passage_{rec.id}",
            )
            if st.button("💾 국어 문항 수정 저장", key=f"{key_prefix}_save_kor_edit_{rec.id}"):
                all_recs = storage.load_db()
                for r in all_recs:
                    if r.id == rec.id:
                        r.english_question_type = new_kor_type.strip() or r.english_question_type
                        r.unit_name = r.english_question_type or r.unit_name
                        r.passage_topic = new_kor_topic.strip()
                        r.core_principles = [r.passage_topic] if r.passage_topic else [r.unit_name]
                        r.answer = new_kor_ans.strip() or r.answer
                        r.passage_text = new_kor_passage.strip()
                        break
                storage.save_db(all_recs, target_exams={rec.exam_name})
                _build_cached_exam_exports.clear()
                st.success("국어 문항 정보를 저장했습니다.")
                st.rerun()

        if allow_delete:
            if st.button("이 문항 삭제", key=f"{key_prefix}_del_{rec.id}"):
                storage.delete_question(rec.id)
                _build_cached_exam_exports.clear()
                st.rerun()
        return

    if getattr(rec, "needs_review", False):
        notes = getattr(rec, "review_notes", []) or []
        note_txt = " / ".join(notes) if notes else "2022 개정 기준표와 불일치 의심 항목입니다."
        st.warning(f"⚠️ **검토 필요 (할루시네이션·모순 점검):** {note_txt}")

    std_code = (getattr(rec, "standard_code", "") or "").strip()
    std_codes = [c for c in (getattr(rec, "standard_codes", []) or []) if c]
    if std_code and std_code not in std_codes:
        std_codes = [std_code, *std_codes]
    grade_ev = (getattr(rec, "grade_evidence", "") or "").strip()
    if std_codes or grade_ev:
        meta_bits: list[str] = []
        if std_codes:
            meta_bits.append("**성취기준:** " + ", ".join(f"`[{c}]`" for c in std_codes))
        if grade_ev:
            meta_bits.append(f"**학년 판정 근거:** {grade_ev}")
        st.caption(" · ".join(meta_bits))

    st.markdown("**본질적 풀이 원리 (`core_principles`)**")
    render_chips(rec.core_principles, variant="blue")

    adv_terms = list(getattr(rec, "advanced_terms", []) or [])
    basic_terms = list(getattr(rec, "basic_terms", []) or [])
    if adv_terms or basic_terms:
        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown("**상위 전문 용어 (`advanced_terms`)**")
            render_chips(adv_terms, variant="amber")
        with tc2:
            st.markdown(f"**해당 학년 기본 용어 (`basic_terms` · 고{rec.min_grade})**")
            render_chips(basic_terms, variant="green")
    else:
        st.markdown("**표면 전문 용어·단위 (`required_terms`)**")
        render_chips(rec.required_terms, variant="amber")

    st.write(f"**단일 개념 여부:** {'예 (단일 원리)' if rec.is_single_concept else '아니오 (복합 원리)'}")
    st.write(f"**공식 해설 요약:** {rec.official_explanation}")

    with st.expander("✏️ 2022 과목·최저 학년·성취기준 빠른 수정", expanded=False):
        official_subjs = [s["name"] for s in get_official_subjects_for_category(rec.subject_category)]
        if rec.curriculum_subject and rec.curriculum_subject not in official_subjs:
            official_subjs = [rec.curriculum_subject] + official_subjs

        ec1, ec2, ec3 = st.columns([1.4, 1.4, 0.8])
        with ec1:
            new_subj = st.selectbox(
                "2022 세부 과목",
                options=official_subjs or [rec.curriculum_subject],
                index=(official_subjs.index(rec.curriculum_subject) if rec.curriculum_subject in official_subjs else 0),
                key=f"{key_prefix}_edit_subj_{rec.id}",
            )
        with ec2:
            new_unit = st.text_input(
                "단원명",
                value=rec.unit_name,
                key=f"{key_prefix}_edit_unit_{rec.id}",
            )
        with ec3:
            new_grade = st.selectbox(
                "최저 학년",
                options=[1, 2, 3],
                index=max(0, min(2, int(rec.min_grade) - 1)),
                format_func=lambda g: f"고{g}",
                key=f"{key_prefix}_edit_grade_{rec.id}",
            )
        ec4, ec5 = st.columns([1.2, 1.8])
        with ec4:
            new_std_str = st.text_input(
                "성취기준 코드 (복수 시 쉼표 구분)",
                value=", ".join(std_codes) if std_codes else std_code,
                key=f"{key_prefix}_edit_std_{rec.id}",
            )
        with ec5:
            new_ev = st.text_input(
                "학년 판정 근거",
                value=grade_ev,
                key=f"{key_prefix}_edit_ev_{rec.id}",
            )
        if st.button("💾 수정 저장 (검토 완료 처리)", key=f"{key_prefix}_save_edit_{rec.id}"):
            parsed_stds = [
                re.sub(r"[\[\]\s]+", "", tok)
                for tok in new_std_str.split(",")
                if re.sub(r"[\[\]\s]+", "", tok)
            ]
            dedup_stds = list(dict.fromkeys(parsed_stds))
            all_recs = storage.load_db()
            for r in all_recs:
                if r.id == rec.id:
                    r.curriculum_subject = new_subj
                    r.unit_name = new_unit.strip() or r.unit_name
                    r.min_grade = int(new_grade)
                    r.standard_code = dedup_stds[0] if dedup_stds else ""
                    r.standard_codes = dedup_stds
                    r.grade_evidence = new_ev.strip()
                    r.needs_review = False
                    r.review_notes = []
                    break
            storage.save_db(all_recs, target_exams={rec.exam_name})
            _build_cached_exam_exports.clear()
            st.success("수정 사항을 저장했습니다.")
            st.rerun()

    if allow_delete:
        if st.button("이 문항 삭제", key=f"{key_prefix}_del_{rec.id}"):
            storage.delete_question(rec.id)
            _build_cached_exam_exports.clear()
            st.rerun()



def _normalize_exam_identity(name: str) -> str:
    """회차명 표기 차이(09월/9월, 공백, 로마숫자 등)를 흡수해 동일 기출 판별용 키를 생성한다."""
    s = (name or "").strip()
    if not s:
        return ""
    s_norm = (
        s.replace("Ⅰ", "1")
        .replace("Ⅱ", "2")
        .replace("Ⅲ", "3")
        .replace("(", " ")
        .replace(")", " ")
    )
    year_m = re.search(r"(20\d{2})", s_norm)
    month_m = re.search(r"(?<!\d)(0?[1-9]|1[0-2])\s*월", s_norm)
    grade_m = re.search(r"(고\s*[123])", s_norm)

    year_tok = year_m.group(1) if year_m else ""
    month_tok = str(int(month_m.group(1))) if month_m else ("11" if "수능" in s_norm else "")
    grade_tok = re.sub(r"\s+", "", grade_m.group(1)) if grade_m else ("고3" if "수능" in s_norm else "")

    if year_tok and month_tok and grade_tok:
        tail = s_norm
        if year_m:
            tail = tail.replace(year_m.group(0), " ")
        if month_m:
            tail = tail.replace(month_m.group(0), " ")
        if grade_m:
            tail = tail.replace(grade_m.group(0), " ")
        for noise in ("대학수학능력시험", "전국연합학력평가", "모의평가", "학력평가", "모의고사", "학년도", "수능", "학평", "모평", "영역", "년"):
            tail = tail.replace(noise, " ")
        subj_tok = re.sub(r"[^가-힣a-zA-Z0-9]+", "", tail).lower()
        return f"{year_tok}|{month_tok}|{grade_tok}|{subj_tok}"

    return re.sub(r"[^가-힣a-zA-Z0-9]+", "", s_norm).lower()


@st.cache_data(show_spinner=False)
def _build_recommended_exam_catalog() -> list[dict[str, str]]:
    """최근 3개년(2024~2026) 고1·고2·고3 주요 학평/모평/수능 + 과목별 추천 기출 정적 카탈로그를 생성한다."""
    years = ["2026", "2025", "2024"]
    grade_specs: list[tuple[str, list[str], list[tuple[str, str]]]] = [
        (
            "고1",
            ["3월", "6월", "9월", "10월"],
            [("국어", "국어"), ("영어", "영어"), ("수학", "수학"), ("한국사", "한국사"), ("과학", "통합과학"), ("사회", "통합사회")],
        ),
        (
            "고2",
            ["3월", "6월", "9월", "10월"],
            [
                ("국어", "국어"),
                ("영어", "영어"),
                ("수학", "수학"),
                ("한국사", "한국사"),
                ("과학", "물리학Ⅰ"),
                ("과학", "화학Ⅰ"),
                ("과학", "생명과학Ⅰ"),
                ("과학", "지구과학Ⅰ"),
            ],
        ),
        (
            "고3",
            ["3월", "6월", "9월", "10월", "11월(수능)"],
            [
                ("국어", "국어"),
                ("영어", "영어"),
                ("수학", "수학"),
                ("한국사", "한국사"),
                ("과학", "물리학Ⅰ"),
                ("과학", "화학Ⅰ"),
                ("과학", "생명과학Ⅰ"),
                ("과학", "지구과학Ⅰ"),
            ],
        ),
    ]
    catalog: list[dict[str, str]] = []
    for y in years:
        for grade, months, subjects in grade_specs:
            for m in months:
                for subj_cat, subj_name in subjects:
                    ex_name = f"{y}년 {m} {grade} {subj_name}"
                    catalog.append(
                        {
                            "year": y,
                            "month": m,
                            "grade": grade,
                            "subject_cat": subj_cat,
                            "subject_name": subj_name,
                            "exam_name": ex_name,
                            "norm_id": _normalize_exam_identity(ex_name),
                        }
                    )
    return catalog


def _render_exam_coverage_board(ctx: AppContext) -> None:
    """탭 1 상단에 이미 등록된 기출 목록과 추천 미등록 기출 목록을 나란히 표시한다."""
    db_records = ctx.storage.load_db()
    registered_map: dict[str, dict[str, Any]] = {}
    registered_ids: set[str] = set()
    for rec in db_records:
        en = re.sub(r"\s+", " ", (rec.exam_name or "").strip())
        if not en:
            continue
        norm_id = _normalize_exam_identity(en)
        if norm_id:
            registered_ids.add(norm_id)
        inferred_cat = infer_subject_category(rec.curriculum_subject or en, exam_title=en)
        eff_cat = "한국사" if inferred_cat == "한국사" else (rec.subject_category or inferred_cat)
        entry = registered_map.setdefault(
            en,
            {
                "회차명": en,
                "과목": eff_cat,
                "문항 수": 0,
            },
        )
        entry["문항 수"] += 1

    catalog = _build_recommended_exam_catalog()
    missing_all = [
        item
        for item in catalog
        if (item.get("norm_id") or _normalize_exam_identity(item["exam_name"])) not in registered_ids
    ]

    with st.expander(
        f"📋 기출 DB 등록 현황 & 추천 미등록 기출 보기 (등록 {len(registered_map)}회차 · 추천 미등록 {len(missing_all)}건)",
        expanded=True,
    ):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            f_subj = st.selectbox(
                "과목 필터",
                options=["전체", "국어", "영어", "수학", "한국사", "과학", "사회"],
                key="cov_board_subj",
            )
        with fc2:
            f_grade = st.selectbox(
                "학년 필터",
                options=["전체", "고1", "고2", "고3"],
                key="cov_board_grade",
            )
        with fc3:
            f_year = st.selectbox(
                "연도 필터",
                options=["전체", "2026", "2025", "2024"],
                key="cov_board_year",
            )

        reg_rows = [
            row
            for row in sorted(registered_map.values(), key=lambda x: x["회차명"], reverse=True)
            if (f_subj == "전체" or row["과목"] == f_subj)
            and (f_grade == "전체" or f_grade in row["회차명"])
            and (f_year == "전체" or f_year in row["회차명"])
        ]
        miss_rows = [
            {
                "연월": f"{it['year']}년 {it['month']}",
                "학년": it["grade"],
                "과목": it["subject_name"],
                "추천 회차명": it["exam_name"],
            }
            for it in missing_all
            if (f_subj == "전체" or it["subject_cat"] == f_subj)
            and (f_grade == "전체" or it["grade"] == f_grade)
            and (f_year == "전체" or it["year"] == f_year)
        ]

        col_reg, col_miss = st.columns(2)
        with col_reg:
            st.markdown(f"**✅ 이미 등록된 기출 ({len(reg_rows)}회차)**")
            if reg_rows:
                st.dataframe(reg_rows, hide_index=True, width="stretch", height=240)
            else:
                st.info("해당 조건으로 등록된 기출이 아직 없습니다.")
        with col_miss:
            st.markdown(f"**📌 추천 미등록 기출 ({len(miss_rows)}건)**")
            if miss_rows:
                st.dataframe(miss_rows, hide_index=True, width="stretch", height=240)
            else:
                st.success("해당 조건의 추천 기출이 모두 등록되어 있습니다!")



def render_tab1(ctx: AppContext) -> None:
    st.subheader("1️⃣ 기출 PDF 수집 및 고해상도 자동 크롭 (250 DPI)")
    _render_exam_coverage_board(ctx)

    collect_mode = st.radio(
        "수집 방식 선택",
        options=["📤 수동 업로드 (기본)", "🔗 EBSi 링크 수집 (보조)"],
        horizontal=True,
    )

    if collect_mode == "🔗 EBSi 링크 수집 (보조)":
        with st.container(border=True):
            ebsi_url = st.text_input(
                "EBSi 기출 문제지 URL 입력",
                placeholder="https://www.ebsi.co.kr/...",
            )
            if st.button("🌐 EBSi 링크에서 공개 PDF 추출 시도", width="content"):
                with st.spinner("EBSi 페이지에서 공개 PDF 링크 확인 중..."):
                    ebsi_res = attempt_ebsi_pdf_fetch(ebsi_url)
                if ebsi_res.success and ebsi_res.problem_pdf_bytes:
                    st.session_state["ebsi_problem_bytes"] = ebsi_res.problem_pdf_bytes
                    st.session_state["ebsi_problem_name"] = ebsi_res.problem_filename
                    if ebsi_res.answer_pdf_bytes:
                        st.session_state["ebsi_answer_bytes"] = ebsi_res.answer_pdf_bytes
                        st.session_state["ebsi_answer_name"] = ebsi_res.answer_filename
                    st.success(ebsi_res.message)
                else:
                    st.warning(ebsi_res.message)
                    st.info("👉 아래 [수동 업로드] 영역에 내려받은 문제지·해설지 PDF를 직접 올려주세요.")

    with st.container(border=True):
        upload_mode_label = st.radio(
            "업로드 방식 선택",
            options=["📄 파일 선택 (1개 또는 여러 개)", "📁 폴더 통째로 업로드 (하위 폴더 포함)"],
            horizontal=True,
            key="tab1_upload_mode",
            help="여러 개의 PDF를 드래그/다중 선택하거나, 모의고사 폴더 자체를 선택해 한 번에 대기열에 넣을 수 있습니다. 분석·태깅·저장은 PDF 1개씩 순차 진행됩니다.",
        )
        is_dir_upload = "폴더" in upload_mode_label
        accept_mode: bool | str = "directory" if is_dir_upload else True
        unified_up_label = (
            "📁 모의고사 폴더 통째로 업로드 (문제지·해설지 구분 없이 한 번에 선택 — AI 자동 분류 및 표지 매칭)"
            if is_dir_upload
            else "📄 모의고사 PDF 한 번에 업로드 (문제지 + 해설지를 구분 없이 모두 선택 — AI 자동 분류 및 표지 매칭)"
        )

        raw_all_uploads = st.file_uploader(
            unified_up_label,
            type=["pdf"],
            accept_multiple_files=accept_mode,
            key="tab1_unified_pdf_dir" if is_dir_upload else "tab1_unified_pdf_files",
        )

        raw_all_files = sorted(
            [f for f in (raw_all_uploads or []) if f.name.lower().endswith(".pdf")],
            key=lambda f: f.name,
        )

        prob_files: list[Any] = []
        ans_files: list[Any] = []

        if len(raw_all_files) == 1:
            prob_files = [raw_all_files[0]]
        elif len(raw_all_files) >= 2:
            split_prob_idxs, split_ans_idxs, split_source = _resolve_mixed_pdf_split_with_ai(
                raw_all_files, ctx.upload_gemini_key
            )
            prob_files = [raw_all_files[i] for i in split_prob_idxs]
            ans_files = [raw_all_files[i] for i in split_ans_idxs]

            # 만약 AI/파일명 분류에서 문제지가 0개이거나 해설지가 0개인 예외 상황 안내
            split_badge = "🤖 Gemini AI 표지 판별" if split_source == "ai" else "📁 파일명 판별"
            sc_info, sc_btn = st.columns([3.2, 1.0])
            with sc_info:
                st.caption(
                    f"{split_badge}: 총 {len(raw_all_files)}개 PDF → "
                    f"**문제지 {len(prob_files)}개** · **해설지 {len(ans_files)}개** 자동 분리 완료"
                )
            with sc_btn:
                if st.button("🔄 AI 표지 다시 분류", key="tab1_resplit_btn", width="stretch"):
                    raw_sig = (
                        tuple((f.name, int(getattr(f, "size", 0))) for f in raw_all_files),
                        bool(ctx.upload_gemini_key),
                    )
                    st.session_state.get("tab1_ai_split_cache", {}).pop(raw_sig, None)
                    st.rerun()

            if not prob_files and ans_files:
                st.warning("⚠️ 업로드된 파일이 모두 해설지로 분류되었습니다. 문제지 PDF도 함께 선택해 주세요.")

        is_batch_queue_mode = len(prob_files) >= 2

        if is_batch_queue_mode:
            auto_matches, match_reasons, match_source, match_err = _resolve_auto_answer_matches(
                prob_files,
                ans_files,
                ctx.upload_gemini_key,
            )
            matched_cnt = sum(1 for m in auto_matches if m is not None)
            pair_sig = _compute_pairing_signature(prob_files, ans_files)
            pair_ver = int(st.session_state.get("tab1_ai_pair_ver", 0))
            pair_hash = _compute_pairing_widget_hash(pair_sig, pair_ver)

            if match_source == "ai":
                pair_badge = f"🤖 Gemini 3.5 Flash-Lite 해설지 자동 매칭 **{matched_cnt}/{len(prob_files)}개**"
            elif ans_files:
                pair_badge = f"해설지 자동 매칭 **{matched_cnt}/{len(prob_files)}개**"
            else:
                pair_badge = "해설지 미업로드"

            st.info(
                f"📦 **총 {len(prob_files)}개 문제지 PDF**가 대기열에 등록되었습니다 ({pair_badge}). "
                "시작 버튼을 누르면 **PDF 1개씩 순서대로** 표지 인식 → 문항 크롭 → Gemini 원리·용어 태깅 → DB/Drive 저장을 실행합니다."
            )
            if match_source == "fallback" and match_err and ans_files:
                st.warning(f"⚠️ AI 자동 매칭 일시 오류로 파일명 기준 매칭을 적용했습니다: `{match_err}`")

            ans_option_indices: list[int | None] = [None] + list(range(len(ans_files)))
            selected_ans_indices: list[int | None] = []
            with st.expander(f"📋 대기열 목록 및 해설지 1:1 매칭 확인 ({len(prob_files)}개)", expanded=len(prob_files) <= 12):
                if ans_files:
                    hdr_l, hdr_r = st.columns([2.2, 1.0])
                    with hdr_l:
                        st.caption("🤖 AI가 파일명과 1페이지 내용을 대조해 해설지를 자동 지정했습니다. 필요 시 드롭다운에서 수동 변경할 수 있습니다.")
                    with hdr_r:
                        if st.button(
                            "🔄 AI(3.5 Flash-Lite) 다시 매칭",
                            key=f"batch_rematch_{pair_hash}",
                            width="stretch",
                        ):
                            _invalidate_auto_pair_cache(pair_sig, ctx.upload_gemini_key)
                            st.rerun()

                for q_idx, pf in enumerate(prob_files):
                    c_left, c_right = st.columns([1.2, 1.0])
                    p_base = _pdf_display_basename(pf.name)
                    with c_left:
                        st.markdown(f"**{q_idx + 1}. `{p_base}`**")
                        if p_base != pf.name:
                            st.caption(f"경로: `{pf.name}`")
                    with c_right:
                        default_ans = auto_matches[q_idx]
                        default_opt_pos = (
                            ans_option_indices.index(default_ans)
                            if default_ans in ans_option_indices
                            else 0
                        )
                        chosen_ans_idx = st.selectbox(
                            f"해설지 #{q_idx + 1}",
                            options=ans_option_indices,
                            index=default_opt_pos,
                            format_func=lambda idx: (
                                "(해설지 없음)"
                                if idx is None
                                else _pdf_display_basename(ans_files[idx].name)
                            ),
                            key=f"batch_ans_map_{pair_hash}_{q_idx}_{pf.name}",
                            label_visibility="collapsed",
                        )
                        if match_reasons.get(q_idx):
                            st.caption(f"🤖 {match_reasons[q_idx]}")
                        selected_ans_indices.append(chosen_ans_idx)

            b_opt1, b_opt2 = st.columns(2)
            with b_opt1:
                batch_vision_chk = st.checkbox(
                    "스캔(이미지) PDF인 경우 Gemini Vision 좌표 탐지 보조 사용",
                    value=True,
                    key="tab1_batch_vision_chk",
                )
            with b_opt2:
                batch_skip_dup_chk = st.checkbox(
                    "이미 DB에 등록된 회차명 자동 건너뛰기 (중복 방지)",
                    value=True,
                    key="tab1_batch_skip_dup_chk",
                )

            matched_queue_count = sum(1 for x in selected_ans_indices if x is not None)
            unmatched_queue_count = len(prob_files) - matched_queue_count
            if matched_queue_count == 0:
                st.error(
                    "🚫 해설지가 매칭된 문제지가 없습니다. 해설지 PDF를 함께 업로드해야 태깅을 시작할 수 있습니다."
                )
            elif unmatched_queue_count > 0:
                st.warning(
                    f"⚠️ 해설지가 매칭되지 않은 문제지 **{unmatched_queue_count}개**는 자동 건너뛰기되며, "
                    f"해설지가 연결된 **{matched_queue_count}개**만 순차 태깅합니다."
                )

            batch_btn_label = (
                f"🚀 대기열 {len(prob_files)}개 PDF 순차 분석·저장 시작 (하나씩 처리)"
                if unmatched_queue_count == 0
                else f"🚀 해설지 매칭된 {matched_queue_count}/{len(prob_files)}개 PDF 순차 분석·저장 시작 (미매칭 {unmatched_queue_count}개 건너뜀)"
            )
            run_batch_btn = st.button(
                batch_btn_label,
                type="primary",
                width="stretch",
                key="tab1_run_batch_btn",
                disabled=(matched_queue_count == 0),
            )

            if run_batch_btn and matched_queue_count > 0:
                verified_key, verified_key_label = resolve_active_gemini_api_key(
                    ctx.current_user, ctx.creds.gemini_api_key, force_server_key=True
                )
                if not verified_key:
                    st.error(
                        f"서버 업로드·태깅용 API 키가 설정되지 않았습니다 ({verified_key_label}). "
                        "`.streamlit/secrets.toml`의 `GEMINI_API_KEY`를 확인해 주세요."
                    )
                elif not _ensure_token_quota(ctx):
                    pass
                else:
                    db_batch_snapshot = ctx.storage.load_db()
                    existing_exam_names = {
                        re.sub(r"\s+", " ", r.exam_name.strip())
                        for r in db_batch_snapshot
                        if r.exam_name
                    }
                    existing_exam_ids = {
                        _normalize_exam_identity(r.exam_name)
                        for r in db_batch_snapshot
                        if r.exam_name
                    }
                    total_n = len(prob_files)
                    overall_prog = st.progress(0.0)
                    overall_status = st.empty()
                    item_prog = st.progress(0.0)
                    item_status = st.empty()
                    item_log = st.empty()
                    batch_summary: list[dict[str, Any]] = []
                    live_logs: list[str] = []

                    def _append_batch_log(msg: str) -> None:
                        live_logs.append(msg)
                        item_log.code("\n".join(live_logs[-18:]), language="text")

                    for idx, prob_f in enumerate(prob_files, start=1):
                        if not _ensure_token_quota(ctx):
                            _append_batch_log("🚫 월간 토큰 한도 도달로 남은 대기열을 중단했습니다.")
                            break
                        prob_base = _pdf_display_basename(prob_f.name)
                        matched_ans_i = selected_ans_indices[idx - 1]
                        ans_f = ans_files[matched_ans_i] if matched_ans_i is not None else None
                        ans_base = _pdf_display_basename(ans_f.name) if ans_f is not None else ""

                        if ans_f is None:
                            _append_batch_log(f"⏭️ [{idx}/{total_n}] {prob_base} — 해설지 미매칭 → 건너뜀")
                            batch_summary.append(
                                {
                                    "file": prob_base,
                                    "exam_name": os.path.splitext(prob_base)[0].replace("_", " ").strip(),
                                    "status": "skipped",
                                    "saved_count": 0,
                                    "message": "해설지 미첨부(건너뜀 · 해설지 필수)",
                                }
                            )
                            overall_prog.progress(idx / total_n)
                            continue

                        overall_status.info(
                            f"📦 **[{idx}/{total_n}]** `{prob_base}` 순차 처리 중... (해설지: `{ans_base}`)"
                        )
                        item_prog.progress(0.05)
                        _append_batch_log(f"--- [{idx}/{total_n}] {prob_base} 시작 ---")

                        try:
                            prob_bytes = prob_f.getvalue()
                            ans_bytes = ans_f.getvalue() if ans_f is not None else None

                            item_status.info("🔍 1단계: 표지 메타데이터 판독 중...")
                            parsed_local, _, _ = _detect_single_exam_cover(prob_bytes, prob_base, verified_key)
                            raw_title = (parsed_local.get("exam_title") or "").strip()
                            if not raw_title:
                                raw_title = os.path.splitext(prob_base)[0].replace("_", " ").strip()
                            norm_exam_name = normalize_exam_title_year(raw_title, filename=prob_base)
                            subj_cat = infer_subject_category(parsed_local.get("subject", ""), exam_title=norm_exam_name)
                            _append_batch_log(f"🏷️ 회차명: {norm_exam_name} ({subj_cat})")

                            norm_exam_id = _normalize_exam_identity(norm_exam_name)
                            if norm_exam_name in existing_exam_names or (norm_exam_id and norm_exam_id in existing_exam_ids):
                                _append_batch_log(f"🚫 이미 DB에 등록된 회차 → 업로드 차단: {norm_exam_name}")
                                batch_summary.append(
                                    {
                                        "file": prob_base,
                                        "exam_name": norm_exam_name,
                                        "status": "skipped",
                                        "saved_count": 0,
                                        "message": "이미 DB에 등록된 회차(중복 차단)",
                                    }
                                )
                                overall_prog.progress(idx / total_n)
                                continue

                            item_status.info("✂️ 2단계: 2단 편집 분석 및 문항 크롭 중...")
                            item_prog.progress(0.20)
                            crops, _, _ = _crop_single_exam_pdf(
                                prob_bytes,
                                use_vision_fallback=batch_vision_chk,
                                api_key=verified_key,
                                subject_hint=f"{subj_cat} {norm_exam_name}",
                                status_cb=lambda m: item_status.info(m),
                                progress_cb=lambda v: item_prog.progress(0.20 + 0.20 * v),
                            )
                            if not crops:
                                _append_batch_log(f"❌ 문항 번호 탐지 실패 → 건너뜀: {prob_base}")
                                batch_summary.append(
                                    {
                                        "file": prob_base,
                                        "exam_name": norm_exam_name,
                                        "status": "failed",
                                        "saved_count": 0,
                                        "message": "문항 번호를 탐지하지 못했습니다.",
                                    }
                                )
                                overall_prog.progress(idx / total_n)
                                continue

                            _append_batch_log(f"✂️ 크롭 완료: {len(crops)}문항")
                            res = _tag_and_save_single_exam(
                                storage=ctx.storage,
                                api_key=verified_key,
                                meta={
                                    "exam_name": norm_exam_name,
                                    "subject_category": subj_cat,
                                    "prob_bytes": prob_bytes,
                                    "prob_name": prob_base,
                                    "ans_bytes": ans_bytes,
                                    "ans_name": ans_base,
                                },
                                parsed_crops=crops,
                                target_q_nums=[c.q_num for c in crops],
                                is_retry=False,
                                push_status=lambda m: item_status.info(m),
                                append_log=_append_batch_log,
                                on_progress=lambda v: item_prog.progress(0.40 + 0.60 * v),
                                on_warning=lambda w: _append_batch_log(f"⚠️ {w}"),
                            )
                            newly_saved_items = res["newly_saved"]
                            if newly_saved_items:
                                existing_exam_names.add(norm_exam_name)
                                if norm_exam_id:
                                    existing_exam_ids.add(norm_exam_id)
                                record_exam_upload_for_user(ctx.storage, ctx.current_user.username)
                            failed_qs = res["failed_q_nums"]
                            sync_err = res["sync_err"]
                            msg_parts = [f"{len(newly_saved_items)}문항 저장 ({res['elapsed_s']:.1f}초)"]
                            if failed_qs:
                                msg_parts.append(f"누락 {len(failed_qs)}문항")
                            if sync_err:
                                msg_parts.append(f"동기화 경고: {sync_err}")
                            batch_summary.append(
                                {
                                    "file": prob_base,
                                    "exam_name": norm_exam_name,
                                    "status": "success" if newly_saved_items else "failed",
                                    "saved_count": len(newly_saved_items),
                                    "message": " · ".join(msg_parts),
                                }
                            )
                        except Exception as exc:
                            _append_batch_log(f"❌ 처리 오류 ({prob_base}): {exc}")
                            batch_summary.append(
                                {
                                    "file": prob_base,
                                    "exam_name": prob_base,
                                    "status": "failed",
                                    "saved_count": 0,
                                    "message": str(exc),
                                }
                            )

                        overall_prog.progress(idx / total_n)

                    overall_status.success(f"🏁 대기열 {total_n}개 PDF 순차 처리 완료!")
                    item_status.empty()
                    st.session_state["tab1_batch_summary"] = batch_summary

            saved_batch_summary = st.session_state.get("tab1_batch_summary") or []
            if saved_batch_summary:
                ok_cnt = sum(1 for r in saved_batch_summary if r["status"] == "success")
                skip_cnt = sum(1 for r in saved_batch_summary if r["status"] == "skipped")
                fail_cnt = sum(1 for r in saved_batch_summary if r["status"] == "failed")
                st.markdown(
                    f"#### 📊 순차 처리 결과 요약 — ✅ 완료 **{ok_cnt}건** · ⏭️ 중복 건너뜀 **{skip_cnt}건** · ❌ 실패 **{fail_cnt}건**"
                )
                for row in saved_batch_summary:
                    icon = "✅" if row["status"] == "success" else ("⏭️" if row["status"] == "skipped" else "❌")
                    st.write(f"{icon} **{row['exam_name']}** (`{row['file']}`) — {row['message']}")

        else:
            prob_file = prob_files[0] if prob_files else None
            ans_file = None
            if ans_files:
                if prob_file is not None and len(ans_files) > 1:
                    s_matches, s_reasons, s_source, s_err = _resolve_auto_answer_matches(
                        [prob_file],
                        ans_files,
                        ctx.upload_gemini_key,
                    )
                    s_pair_sig = _compute_pairing_signature([prob_file], ans_files)
                    s_pair_ver = int(st.session_state.get("tab1_ai_pair_ver", 0))
                    s_pair_hash = _compute_pairing_widget_hash(s_pair_sig, s_pair_ver)
                    s_options: list[int | None] = [None] + list(range(len(ans_files)))
                    s_default = s_matches[0]
                    s_pos = s_options.index(s_default) if s_default in s_options else 0

                    sc_l, sc_r = st.columns([2.2, 1.0])
                    with sc_l:
                        chosen_single_idx = st.selectbox(
                            "🤖 AI 자동 매칭 해설지 (3.5 Flash-Lite · 필요 시 변경 가능)",
                            options=s_options,
                            index=s_pos,
                            format_func=lambda idx: (
                                "(해설지 없음)"
                                if idx is None
                                else _pdf_display_basename(ans_files[idx].name)
                            ),
                            key=f"single_ans_map_{s_pair_hash}_{prob_file.name}",
                        )
                        if s_reasons.get(0):
                            st.caption(f"🤖 {s_reasons[0]}")
                        elif s_source == "fallback" and s_err:
                            st.caption(f"⚠️ AI 자동 매칭 일시 오류로 파일명 기준 선택됨: {s_err}")
                    with sc_r:
                        st.write("")
                        if st.button(
                            "🔄 AI 다시 매칭",
                            key=f"single_rematch_{s_pair_hash}",
                            width="stretch",
                        ):
                            _invalidate_auto_pair_cache(s_pair_sig, ctx.upload_gemini_key)
                            st.rerun()

                    ans_file = ans_files[chosen_single_idx] if chosen_single_idx is not None else None
                else:
                    ans_file = ans_files[0]

            active_prob_bytes: bytes | None = (
                prob_file.getvalue() if prob_file is not None else st.session_state.get("ebsi_problem_bytes")
            )
            active_prob_name: str = (
                _pdf_display_basename(prob_file.name)
                if prob_file is not None
                else st.session_state.get("ebsi_problem_name", "problem.pdf")
            )
            active_ans_bytes: bytes | None = (
                ans_file.getvalue() if ans_file is not None else st.session_state.get("ebsi_answer_bytes")
            )
            active_ans_name: str = (
                _pdf_display_basename(ans_file.name)
                if ans_file is not None
                else st.session_state.get("ebsi_answer_name", "answer.pdf")
            )

            if st.session_state.get("ebsi_problem_bytes") and prob_file is None:
                st.caption(f"🔗 EBSi에서 가져온 문제지 대기 중: `{active_prob_name}`")
            if st.session_state.get("ebsi_answer_bytes") and ans_file is None:
                st.caption(f"🔗 EBSi에서 가져온 해설지 대기 중: `{active_ans_name}`")

            # 업로드 즉시 fitz + re 기반 0.05초 로컬 파싱 (API 호출 없음, file.name + file.size 변경 감지 시 강제 덮어쓰기)
            file_sig = (
                f"{active_prob_name}:{prob_file.size}"
                if prob_file is not None
                else (f"{active_prob_name}:{len(active_prob_bytes)}" if active_prob_bytes else "")
            )

            if not file_sig:
                st.session_state["tab1_last_loaded_sig"] = ""
                st.session_state["tab1_meta_status"] = ""
            elif file_sig != st.session_state.get("tab1_last_loaded_sig"):
                parsed_local = {
                    "year": "",
                    "month": "",
                    "grade": "",
                    "subject": "",
                    "exam_title": "",
                }
                has_p1_text = False
                ai_verified = False
                if active_prob_bytes:
                    with st.spinner("🔍 1차 텍스트 파싱 + 2차 AI 비전 표지 교차 검증 중..."):
                        parsed_local, has_p1_text, ai_verified = _detect_single_exam_cover(
                            active_prob_bytes,
                            active_prob_name,
                            ctx.upload_gemini_key,
                        )

                st.session_state["meta_year"] = parsed_local.get("year", "")
                st.session_state["meta_month"] = parsed_local.get("month", "")
                st.session_state["meta_grade"] = parsed_local.get("grade", "")
                st.session_state["meta_subject"] = parsed_local.get("subject", "")
                st.session_state["meta_title"] = parsed_local.get("exam_title", "")
                st.session_state["tab1_subject_cat"] = infer_subject_category(
                    parsed_local.get("subject", ""),
                    exam_title=parsed_local.get("exam_title", "") or active_prob_name,
                )
                st.session_state["tab1_last_loaded_sig"] = file_sig

                if ai_verified and parsed_local.get("exam_title"):
                    st.session_state["tab1_meta_status"] = "verified_ai"
                elif not has_p1_text and not parsed_local.get("exam_title"):
                    st.session_state["tab1_meta_status"] = "scanned"
                elif parsed_local.get("exam_title"):
                    st.session_state["tab1_meta_status"] = "parsed"
                else:
                    st.session_state["tab1_meta_status"] = "partial_empty"

            meta_status = st.session_state.get("tab1_meta_status", "")
            if file_sig:
                if meta_status == "verified_ai":
                    st.success(
                        f"🎯 1차 텍스트 + 2차 AI 비전 표지 교차 검증 완료: **{st.session_state.get('meta_title', '')}** (필요 시 아래에서 수정 가능)"
                    )
                elif meta_status == "parsed":
                    st.success(
                        f"⚡ 1페이지 즉시 자동 인식 완료: **{st.session_state.get('meta_title', '')}** (필요 시 아래에서 바로 수정하세요)"
                    )
                elif meta_status == "scanned":
                    st.info("🖼️ 텍스트가 없는 스캔 PDF입니다. 아래 [시행 년도 / 시행 월 / 학년 / 과목 / 회차명]을 직접 입력해 주세요.")
                elif meta_status == "partial_empty":
                    st.info("📄 1페이지 헤더 정보가 일부만 확인되어 빈칸으로 두었습니다. 아래 입력칸을 직접 채워 주세요.")

            # 분석 성공 여부와 관계없이 5개 입력칸(+대분류)은 항상 즉시 렌더링
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                exam_year_input = st.text_input("시행 년도", key="meta_year", placeholder="예: 2026")
            with col_m2:
                exam_month_input = st.text_input("시행 월", key="meta_month", placeholder="예: 9월")
            with col_m3:
                exam_grade_input = st.text_input("학년", key="meta_grade", placeholder="예: 고1")
            with col_m4:
                exam_subject_input = st.text_input(
                    "과목",
                    key="meta_subject",
                    placeholder="예: 통합사회",
                )

            col_meta1, col_meta2 = st.columns([1, 2])
            with col_meta1:
                subject_cat = st.selectbox(
                    "과목 (대분류)",
                    options=["과학", "수학", "사회", "한국사", "국어", "영어"],
                    key="tab1_subject_cat",
                )
            with col_meta2:
                exam_name_input = st.text_input(
                    "회차명 (자동 조합)",
                    key="meta_title",
                    placeholder="예: 2026년 9월 고1 통합사회",
                )

            db_snapshot = ctx.storage.load_db()
            existing_exam_names = {
                re.sub(r"\s+", " ", r.exam_name.strip())
                for r in db_snapshot
                if r.exam_name
            }
            existing_exam_ids = {
                _normalize_exam_identity(r.exam_name)
                for r in db_snapshot
                if r.exam_name
            }
            norm_input_exam = normalize_exam_title_year(exam_name_input.strip(), filename=active_prob_name)
            norm_input_id = _normalize_exam_identity(norm_input_exam)
            is_duplicate_exam = bool(
                norm_input_exam
                and (norm_input_exam in existing_exam_names or (norm_input_id and norm_input_id in existing_exam_ids))
            )
            if is_duplicate_exam:
                st.error(
                    f"🚫 **'{norm_input_exam}'** 모의고사는 이미 DB에 등록되어 있습니다. "
                    "중복 수집 방지를 위해 크롭 버튼이 비활성화되었습니다."
                )

            is_missing_answer_single = not bool(active_ans_bytes)
            if active_prob_bytes and is_missing_answer_single:
                st.warning(
                    "🚫 **해설지 PDF가 필요합니다.** 정답 정확도를 위해 해설지 PDF를 함께 업로드(또는 매칭)해야 1단계 크롭과 태깅을 진행할 수 있습니다."
                )

            parse_col1, parse_col2 = st.columns([1, 1])
            with parse_col1:
                run_crop_btn = st.button(
                    "✂️ 1단계: 2단 편집 분석 및 고해상도 문항 크롭",
                    type="primary",
                    width="stretch",
                    disabled=is_duplicate_exam or is_missing_answer_single,
                )
            with parse_col2:
                vision_fallback_chk = st.checkbox(
                    "스캔(이미지) PDF인 경우 Gemini Vision 좌표 탐지 보조 사용",
                    value=True,
                )

            if run_crop_btn and not is_duplicate_exam and not is_missing_answer_single:
                if not active_prob_bytes:
                    st.error("문제지 PDF를 업로드하거나 EBSi에서 먼저 추출해 주세요.")
                elif not active_ans_bytes:
                    st.error("해설지 PDF가 필요합니다. 해설지 PDF를 함께 업로드해 주세요.")
                elif not exam_name_input.strip():
                    st.error("시험 회차명을 입력해 주세요.")
                else:
                    vision_status_ph = st.empty()
                    vision_prog_ph = st.empty()
                    with st.spinner("PyMuPDF 2단 편집(좌→우 칼럼) 분석 및 220 DPI PNG 크롭 중..."):
                        crops, total_p, _ = _crop_single_exam_pdf(
                            active_prob_bytes,
                            use_vision_fallback=bool(vision_fallback_chk),
                            api_key=ctx.upload_gemini_key,
                            subject_hint=f"{subject_cat} {exam_subject_input} {exam_name_input}",
                            status_cb=lambda m: vision_status_ph.info(m),
                            progress_cb=lambda v: vision_prog_ph.progress(v),
                        )

                    if not crops:
                        st.error("문항 번호를 탐지하지 못했습니다. PDF 형식을 확인하거나 Gemini API 키 설정 후 Vision 보조를 활성화해 주세요.")
                    else:
                        st.session_state["parsed_crops"] = crops
                        st.session_state["parsed_exam_meta"] = {
                            "exam_name": norm_input_exam or exam_name_input.strip(),
                            "subject_category": subject_cat,
                            "prob_bytes": active_prob_bytes,
                            "prob_name": active_prob_name,
                            "ans_bytes": active_ans_bytes,
                            "ans_name": active_ans_name,
                        }
                        st.session_state["stage2_exam_name"] = ""
                        st.session_state["stage2_saved_q_nums"] = []
                        st.session_state["stage2_failed_q_nums"] = []
                        st.session_state["stage2_failed_errors"] = []
                        st.session_state["stage2_sync_err"] = ""
                        st.session_state["trigger_retry_failed"] = False
                        found_nums = sorted({c.q_num for c in crops})
                        min_q, max_q = found_nums[0], found_nums[-1]
                        subj_txt = (exam_subject_input or "").strip()
                        if any(k in subj_txt for k in ("국어", "영어")):
                            expected_max = 45
                        elif any(k in subj_txt for k in ("수학", "미적분", "기하", "확률")):
                            expected_max = 30
                        elif any(k in subj_txt for k in ("사회", "과학", "물리", "화학", "생명", "지구", "한국사", "윤리", "지리", "경제", "정치", "역사")):
                            expected_max = 20
                        else:
                            expected_max = max_q

                        range_end = max(max_q, expected_max) if total_p >= 3 else max_q
                        missing_in_found = [n for n in range(1, max_q + 1) if n not in set(found_nums)]
                        missing_tail = [n for n in range(max_q + 1, range_end + 1)]

                        st.success(f"✅ 총 {len(crops)}개 문항 크롭 완료 ({min_q}번 ~ {max_q}번)")
                        if missing_in_found or missing_tail:
                            missing_all = missing_in_found + missing_tail
                            st.warning(
                                f"⚠️ 누락된 문항 번호 ({len(missing_all)}개, 1~{range_end}번 기준): "
                                + ", ".join(f"{n}번" for n in missing_all)
                            )

    parsed_crops: list[CroppedQuestion] = st.session_state.get("parsed_crops") or []
    if (not is_batch_queue_mode) and parsed_crops:
        meta = st.session_state["parsed_exam_meta"]
        st.markdown(f"### 2️⃣ 크롭 문항 확인 & 2022 원리·용어 분해 태깅 (`{meta['exam_name']}`)")

        with st.expander(f"🖼️ 크롭된 문항 미리보기 ({len(parsed_crops)}문항)", expanded=False):
            seen_p_groups: dict[str, CroppedQuestion] = {}
            for cq in parsed_crops:
                p_b = getattr(cq, "passage_png_bytes", b"") or b""
                p_g = getattr(cq, "passage_group_id", "") or ""
                if p_b and p_g and p_g not in seen_p_groups:
                    seen_p_groups[p_g] = cq

            if seen_p_groups:
                st.markdown(f"**📖 분리된 공통 지문 이미지 ({len(seen_p_groups)}개 세트)**")
                p_list = list(seen_p_groups.items())
                for idx in range(0, len(p_list), 3):
                    p_cols = st.columns(3)
                    for col_i, (p_gid, p_cq) in enumerate(p_list[idx : idx + 3]):
                        with p_cols[col_i]:
                            st.image(
                                p_cq.passage_png_bytes,
                                caption=f"[{p_gid}번] 공통 지문 ({p_cq.passage_width}x{p_cq.passage_height}px)",
                                width="stretch",
                            )
                st.markdown("---")
                st.markdown(f"**📝 문항별 크롭 이미지 (문제만 · {len(parsed_crops)}문항)**")

            cols_per_row = 3
            for idx in range(0, len(parsed_crops), cols_per_row):
                row_cols = st.columns(cols_per_row)
                for col_i, cq in enumerate(parsed_crops[idx : idx + cols_per_row]):
                    with row_cols[col_i]:
                        st.image(
                            cq.png_bytes,
                            caption=f"{cq.q_num}번 ({cq.page_num}쪽 · {cq.width}x{cq.height}px)",
                            width="stretch",
                        )

        q_nums_all = [c.q_num for c in parsed_crops]
        selected_q_nums = st.multiselect(
            "태깅 및 DB 저장할 문항 선택 (기본: 전체)",
            options=q_nums_all,
            default=q_nums_all,
        )

        def _run_stage2_extraction(target_q_nums: list[int], is_retry: bool = False) -> None:
            if not meta.get("ans_bytes"):
                st.error(
                    "🚫 해설지 PDF가 첨부되지 않아 태깅을 실행할 수 없습니다. "
                    "상단에서 해설지 PDF를 업로드한 뒤 1단계 크롭부터 다시 진행해 주세요."
                )
                return
            # 탭 1 업로드·태깅 전체는 CUSTOM 포함 모든 등급 서버 공용 API 키 사용
            verified_key, verified_key_label = resolve_active_gemini_api_key(
                ctx.current_user, ctx.creds.gemini_api_key, force_server_key=True
            )
            if not verified_key:
                st.error(
                    f"업로드·태깅용 API 키가 설정되지 않았습니다 ({verified_key_label}). "
                    "`.streamlit/secrets.toml`의 `GEMINI_API_KEY`를 확인해 주세요."
                )
                return
            if not _ensure_token_quota(ctx):
                return
            if not target_q_nums:
                st.warning("태깅할 문항을 최소 1개 이상 선택해 주세요.")
                return

            tagging_model_label = MODEL_LABELS.get(EXAM_TAGGING_MODEL, EXAM_TAGGING_MODEL)
            sync_notice = st.info(
                "💾 전체 PDF를 한 번에 태깅한 뒤 서버 영구 디스크(`./exam_db`)에 일괄 저장합니다."
            )
            status_placeholder = st.empty()
            log_placeholder = st.empty()
            live_logs: list[str] = []

            def _push_status(msg: str) -> None:
                status_placeholder.info(msg)

            def _append_log(line: str) -> None:
                live_logs.append(line)
                log_placeholder.code("\n".join(live_logs[-15:]), language="text")

            mode_label = "누락/실패 문항 재분석" if is_retry else "전체 PDF 일괄 태깅"
            _push_status(f"🔑 API 키 검증 완료 ({verified_key_label}) · {mode_label} 준비 중 (모델: {tagging_model_label})...")
            prog_bar = st.progress(0.05)

            res = _tag_and_save_single_exam(
                storage=ctx.storage,
                api_key=verified_key,
                meta=meta,
                parsed_crops=parsed_crops,
                target_q_nums=target_q_nums,
                is_retry=is_retry,
                push_status=_push_status,
                append_log=_append_log,
                on_progress=prog_bar.progress,
                on_warning=st.warning,
            )
            newly_saved = res["newly_saved"]
            failed_q_nums = res["failed_q_nums"]
            errors = res["errors"]
            sync_err = res["sync_err"]

            sync_notice.empty()
            status_placeholder.empty()

            if not is_retry and newly_saved:
                record_exam_upload_for_user(ctx.storage, ctx.current_user.username)

            prev_saved = (
                st.session_state.get("stage2_saved_q_nums", [])
                if is_retry and st.session_state.get("stage2_exam_name") == meta["exam_name"]
                else []
            )
            merged_saved = sorted(set(prev_saved) | {r.q_num for r in newly_saved})
            st.session_state["stage2_exam_name"] = meta["exam_name"]
            st.session_state["stage2_saved_q_nums"] = merged_saved
            st.session_state["stage2_failed_q_nums"] = sorted(set(failed_q_nums))
            st.session_state["stage2_failed_errors"] = errors
            st.session_state["stage2_sync_err"] = sync_err or ""

        if st.button("🧠 2단계: Gemini 원리·용어 분리 태깅 및 영구 저장", type="primary"):
            _run_stage2_extraction(selected_q_nums, is_retry=False)
        elif st.session_state.pop("trigger_retry_failed", False):
            pending_retry = [
                n for n in st.session_state.get("stage2_failed_q_nums", []) if isinstance(n, int)
            ]
            if pending_retry and st.session_state.get("stage2_exam_name") == meta["exam_name"]:
                _run_stage2_extraction(pending_retry, is_retry=True)

        if st.session_state.get("stage2_exam_name") == meta["exam_name"]:
            saved_q_nums = st.session_state.get("stage2_saved_q_nums") or []
            failed_q_nums_state = st.session_state.get("stage2_failed_q_nums") or []
            failed_errors_state = st.session_state.get("stage2_failed_errors") or []
            sync_err_state = st.session_state.get("stage2_sync_err") or ""

            if saved_q_nums:
                st.success(
                    f"🎉 `{meta['exam_name']}` 총 {len(saved_q_nums)}개 문항 태깅 및 영구 저장 완료! (로컬 ./exam_db 저장)"
                )
            if sync_err_state:
                st.warning(sync_err_state)
            for err in failed_errors_state:
                st.error(err)

            if failed_q_nums_state:
                preview_nums = ", ".join(f"{n}번" for n in failed_q_nums_state[:8])
                suffix = " ..." if len(failed_q_nums_state) > 8 else ""
                st.button(
                    f"🔄 실패한 {len(failed_q_nums_state)}개 문항만 다시 분석하기 ({preview_nums}{suffix})",
                    type="primary",
                    key="retry_failed_questions_btn",
                    on_click=lambda: st.session_state.update({"trigger_retry_failed": True}),
                )

            if saved_q_nums:
                saved_set = set(saved_q_nums)
                saved_records = sorted(
                    [
                        r
                        for r in ctx.storage.load_db()
                        if r.exam_name == meta["exam_name"] and r.q_num in saved_set
                    ],
                    key=lambda r: r.q_num,
                )
                for rec in saved_records:
                    review_badge = "⚠️ 검토 필요 · " if getattr(rec, "needs_review", False) else "✅ "
                    if rec.subject_category == "영어":
                        sec_lbl = getattr(rec, "english_section", "") or ("듣기" if rec.q_num <= 17 else "독해")
                        type_lbl = getattr(rec, "english_question_type", "") or rec.unit_name
                        exp_hdr = f"{review_badge}{rec.q_num}번 · [영어 · {sec_lbl}] {type_lbl} (정답: {rec.answer})"
                    else:
                        exp_hdr = (
                            f"{review_badge}{rec.q_num}번 · [{rec.curriculum_subject}] {rec.unit_name} "
                            f"(고{rec.min_grade} · 정답: {rec.answer})"
                        )
                    with st.expander(exp_hdr):
                        c_img, c_info = st.columns([1, 1.3])
                        with c_img:
                            b = ctx.storage.get_crop_bytes(
                                rec.image_filename,
                                subject_category=rec.subject_category,
                                exam_name=rec.exam_name,
                            )
                            if b:
                                st.image(b, width="stretch")
                        with c_info:
                            _render_question_tag_details(
                                rec,
                                storage=ctx.storage,
                                key_prefix="stage2_saved",
                                allow_delete=False,
                            )

    # 저장된 DB 조회 및 과목·회차별 관리
    st.markdown("---")
    st.subheader("3️⃣ 과목별 · 모의고사 회차별 기출 DB 탐색 & 파일 내보내기")
    st.caption(
        "💾 로컬 `./exam_db/과목별_기출DB/<과목>/<회차명>/` 폴더에 "
        "`기출정리.html`, `기출정리.md`, `문제목록.json`, 크롭 PNG, 원본 PDF가 자동 정리됩니다."
    )

    db_records = ctx.storage.load_db()

    if ctx.is_admin:
        with st.expander("🧹 관리자: 누적 기출 DB 전체 초기화 (회원 계정은 유지)", expanded=False):
            st.warning("서버(`./exam_db`)에 누적된 기출 문항·크롭 이미지·PDF 파일을 모두 삭제합니다. (`users.json` 회원 정보는 유지됩니다.)")
            confirm_wipe = st.checkbox("정말 모든 기출 DB 파일을 초기화합니다.", key="confirm_db_wipe")
            if st.button("🗑️ 기출 DB 전체 초기화 실행", disabled=not confirm_wipe, key="exec_db_wipe"):
                wipe_res = ctx.storage.reset_exam_database(delete_users=False)
                st.success(f"초기화 완료! (로컬 항목 {wipe_res['removed_local']}개 정리)")
                st.rerun()

    def _extract_exam_search_attrs(exam_name: str, subj_cat: str) -> dict[str, str]:
        ym = re.search(r"(20\d{2})", exam_name or "")
        mm = re.search(r"(\d{1,2})\s*월", exam_name or "")
        gm = re.search(r"고\s*([123])", exam_name or "")
        inferred_cat = infer_subject_category(exam_name or "", exam_title=exam_name or "")
        eff_subj = "한국사" if inferred_cat == "한국사" else (subj_cat or "")
        return {
            "year": f"{ym.group(1)}년" if ym else "",
            "month": f"{int(mm.group(1))}월" if mm else "",
            "grade": f"고{gm.group(1)}" if gm else "",
            "subject": eff_subj,
        }

    if not db_records:
        st.info("아직 저장된 기출 문항이 없습니다. 위에서 모의고사 PDF를 수집·태깅해 보세요.")
    else:
        exam_groups: dict[tuple[str, str], list[QuestionRecord]] = {}
        for rec in db_records:
            exam_groups.setdefault((rec.subject_category, rec.exam_name), []).append(rec)

        exam_attr_map: dict[tuple[str, str], dict[str, str]] = {
            k: _extract_exam_search_attrs(k[1], k[0]) for k in exam_groups
        }

        avail_years = sorted(
            {v["year"] for v in exam_attr_map.values() if v["year"]},
            reverse=True,
        )
        avail_months = sorted(
            {v["month"] for v in exam_attr_map.values() if v["month"]},
            key=lambda m: int(re.sub(r"[^0-9]", "", m) or 0),
        )
        avail_grades = ["고1", "고2", "고3"]
        avail_subjects = ["수학", "사회", "과학", "한국사", "국어", "영어"]

        with st.form("db_search_form", border=True):
            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                sel_year = st.selectbox(
                    "시행 년도",
                    options=["전체"] + avail_years,
                    key="db_search_year",
                )
            with sc2:
                sel_month = st.selectbox(
                    "시행 월",
                    options=["전체"] + avail_months,
                    key="db_search_month",
                )
            with sc3:
                sel_grade = st.selectbox(
                    "학년",
                    options=["전체"] + avail_grades,
                    key="db_search_grade",
                )
            with sc4:
                sel_subject = st.selectbox(
                    "과목",
                    options=["전체"] + avail_subjects,
                    key="db_search_subject",
                )

            btn_c1, btn_c2 = st.columns([3, 1])
            with btn_c1:
                search_clicked = st.form_submit_button(
                    "🔍 기출 DB 검색 (파일·문항 불러오기)",
                    type="primary",
                    width="stretch",
                )
            with btn_c2:
                reset_clicked = st.form_submit_button(
                    "🧹 검색 초기화",
                    width="stretch",
                )

        if search_clicked:
            st.session_state["db_search_submitted"] = True
            st.session_state["db_search_filters"] = {
                "year": sel_year,
                "month": sel_month,
                "grade": sel_grade,
                "subject": sel_subject,
            }
        elif reset_clicked:
            st.session_state["db_search_submitted"] = False
            st.session_state["db_search_filters"] = {}

        if not st.session_state.get("db_search_submitted"):
            st.info(
                "시행 년도 · 시행 월 · 학년 · 과목을 선택한 뒤 **'🔍 기출 DB 검색'**을 누르면 "
                "해당 모의고사 파일과 문항 목록을 불러옵니다."
            )
        else:
            flt = st.session_state.get("db_search_filters") or {}
            f_year = flt.get("year", "전체")
            f_month = flt.get("month", "전체")
            f_grade = flt.get("grade", "전체")
            f_subject = flt.get("subject", "전체")

            matched_keys: list[tuple[str, str]] = []
            for key_pair in sorted(exam_groups.keys(), key=lambda x: (x[0], x[1]), reverse=True):
                attrs = exam_attr_map[key_pair]
                if f_year != "전체" and attrs["year"] != f_year:
                    continue
                if f_month != "전체" and attrs["month"] != f_month:
                    continue
                if f_grade != "전체" and attrs["grade"] != f_grade:
                    continue
                if f_subject != "전체" and attrs["subject"] != f_subject:
                    continue
                matched_keys.append(key_pair)

            if not matched_keys:
                st.warning("선택한 조건(시행 년도·시행 월·학년·과목)에 일치하는 기출 회차가 없습니다.")
            else:
                can_retag_exam = bool(
                    ctx.current_user is not None and ctx.current_user.tier in ("PRO", "ULTRA", "ADMIN")
                )
                retag_notice = st.session_state.pop("tab1_retag_notice", None)
                if retag_notice:
                    prev_cnt = retag_notice.get("prev_count", retag_notice["saved"])
                    recov_cnt = retag_notice.get("recovered", 0)
                    st.success(
                        f"🔄 `{retag_notice['exam_name']}` 재크롭·다시 태깅 완료! "
                        f"(기존 {prev_cnt}문항 → 현재 {retag_notice['saved']}문항 · 신규 복구 {recov_cnt}문항 · {retag_notice['elapsed']:.1f}초)"
                    )
                    if retag_notice.get("failed"):
                        st.warning(
                            "⚠️ 재태깅 누락 문항: "
                            + ", ".join(f"{n}번" for n in retag_notice["failed"])
                        )
                    if retag_notice.get("sync_err"):
                        st.warning(retag_notice["sync_err"])

                st.success(f"검색 결과: 총 **{len(matched_keys)}개** 모의고사 회차를 불러왔습니다.")
                for subj_cat, exam_name in matched_keys:
                    exam_recs = sorted(exam_groups[(subj_cat, exam_name)], key=lambda r: r.q_num)
                    safe_fn_prefix = sanitize_id_part(exam_name)
                    exam_uid = f"{safe_fn_prefix}_{hashlib.sha1(f'{subj_cat}||{exam_name}'.encode('utf-8')).hexdigest()[:8]}"

                    with st.container(border=True):
                        retag_clicked = False
                        units_preview = sorted({r.unit_name for r in exam_recs if r.unit_name})
                        if can_retag_exam:
                            head_col1, head_col2, head_col3 = st.columns([2.5, 1.0, 0.9])
                            with head_col1:
                                st.markdown(f"#### 📘 {exam_name}")
                                st.caption(
                                    f"과목: **{subj_cat}** · 저장 문항: **{len(exam_recs)}문항** · 단원: {', '.join(units_preview[:4])}"
                                    + (" 등" if len(units_preview) > 4 else "")
                                )
                            with head_col2:
                                retag_clicked = st.button(
                                    "🔄 다시 태깅",
                                    key=f"retag_exam_search_{subj_cat}_{exam_uid}",
                                    width="stretch",
                                    help="PRO · ULTRA · ADMIN 전용 — 현재 2022 개정 기준표와 공용 개념서 DB 기준으로 이 회차 전체 문항을 다시 태깅합니다.",
                                )
                            with head_col3:
                                if st.button(
                                    "🗑️ 회차 삭제",
                                    key=f"del_exam_search_{subj_cat}_{exam_uid}",
                                    width="stretch",
                                ):
                                    cnt = ctx.storage.delete_exam(exam_name)
                                    _build_cached_exam_exports.clear()
                                    st.success(f"'{exam_name}' ({cnt}문항) 삭제 완료")
                                    st.rerun()
                        else:
                            head_col1, head_col2 = st.columns([3, 1])
                            with head_col1:
                                st.markdown(f"#### 📘 {exam_name}")
                                st.caption(
                                    f"과목: **{subj_cat}** · 저장 문항: **{len(exam_recs)}문항** · 단원: {', '.join(units_preview[:4])}"
                                    + (" 등" if len(units_preview) > 4 else "")
                                )
                            with head_col2:
                                if st.button(
                                    "🗑️ 회차 삭제",
                                    key=f"del_exam_search_{subj_cat}_{exam_uid}",
                                    width="stretch",
                                ):
                                    cnt = ctx.storage.delete_exam(exam_name)
                                    _build_cached_exam_exports.clear()
                                    st.success(f"'{exam_name}' ({cnt}문항) 삭제 완료")
                                    st.rerun()

                        if can_retag_exam:
                            with st.expander("✏️ 회차명 수정 (잘못 저장된 연도/과목명 교정)", expanded=False):
                                rn_c1, rn_c2 = st.columns([3.2, 1])
                                with rn_c1:
                                    edited_exam_name = st.text_input(
                                        "새 회차명",
                                        value=normalize_exam_title_year(exam_name),
                                        key=f"rename_input_{subj_cat}_{exam_uid}",
                                        label_visibility="collapsed",
                                    )
                                with rn_c2:
                                    if st.button("💾 회차명 저장", key=f"rename_btn_{subj_cat}_{exam_uid}", width="stretch"):
                                        new_en = re.sub(r"\s+", " ", (edited_exam_name or "").strip())
                                        if new_en and new_en != exam_name:
                                            rn_cnt = ctx.storage.rename_exam(exam_name, new_en)
                                            _build_cached_exam_exports.clear()
                                            st.success(f"회차명을 '{new_en}'(으)로 변경했습니다 ({rn_cnt}문항 반영).")
                                            st.rerun()
                                        else:
                                            st.info("변경할 회차명이 기존과 같습니다.")

                        if retag_clicked and can_retag_exam:
                            corrected_en = normalize_exam_title_year(exam_name)
                            if corrected_en and corrected_en != exam_name:
                                ctx.storage.rename_exam(exam_name, corrected_en)
                                exam_name = corrected_en
                            subj_cat = infer_subject_category("", exam_title=exam_name, default=subj_cat)
                            verified_key, verified_key_label = resolve_active_gemini_api_key(
                                ctx.current_user, ctx.creds.gemini_api_key, force_server_key=True
                            )
                            if not verified_key:
                                st.error(f"API 키가 설정되지 않았습니다 ({verified_key_label}).")
                            else:
                                retag_prob_name = (
                                    next((r.source_pdf for r in exam_recs if r.source_pdf), "")
                                    or f"{safe_fn_prefix}__문제지.pdf"
                                )
                                retag_ans_name = (
                                    next((r.answer_pdf for r in exam_recs if r.answer_pdf), "")
                                    or f"{safe_fn_prefix}__해설지.pdf"
                                )
                                retag_prob_bytes = ctx.storage.get_pdf_bytes(
                                    retag_prob_name,
                                    subject_category=subj_cat,
                                    exam_name=exam_name,
                                )
                                retag_ans_bytes = ctx.storage.get_pdf_bytes(
                                    retag_ans_name,
                                    subject_category=subj_cat,
                                    exam_name=exam_name,
                                )
                                if not retag_ans_bytes:
                                    st.error(
                                        f"🚫 **'{exam_name}'** 회차에 저장된 해설지 PDF가 없어 다시 태깅할 수 없습니다. "
                                        "기존 회차를 삭제한 뒤 문제지·해설지 PDF를 함께 다시 업로드해 주세요."
                                    )
                                elif not _ensure_token_quota(ctx):
                                    pass
                                else:
                                    saved_crops_by_q = {
                                        r.q_num: CroppedQuestion(
                                            q_num=r.q_num,
                                            page_num=1,
                                            png_bytes=(
                                                ctx.storage.get_crop_bytes(
                                                    r.image_filename,
                                                    subject_category=subj_cat,
                                                    exam_name=exam_name,
                                                )
                                                or b""
                                            )
                                            if r.image_filename
                                            else b"",
                                            width=0,
                                            height=0,
                                            extracted_text=r.question_text or "",
                                        )
                                        for r in exam_recs
                                    }

                                    retag_status = st.empty()
                                    retag_prog = st.progress(0.05)
                                    retag_log = st.empty()
                                    retag_logs: list[str] = []

                                    def _append_retag_log(msg: str) -> None:
                                        retag_logs.append(msg)
                                        retag_log.code("\n".join(retag_logs[-15:]), language="text")

                                    if retag_prob_bytes:
                                        retag_status.info(
                                            "✂️ 1단계: 원본 문제지 PDF에서 누락 문항 포함 전체 문항 재크롭 중 (PyMuPDF + Vision 누락 복구)..."
                                        )
                                        fresh_crops, _, _ = _crop_single_exam_pdf(
                                            retag_prob_bytes,
                                            use_vision_fallback=True,
                                            api_key=verified_key,
                                            subject_hint=f"{subj_cat} {exam_name}",
                                            status_cb=retag_status.info,
                                            progress_cb=lambda v: retag_prog.progress(0.05 + 0.35 * v),
                                        )
                                        merged_by_q: dict[int, CroppedQuestion] = {
                                            c.q_num: c for c in fresh_crops
                                        }
                                        for q_n, old_cq in saved_crops_by_q.items():
                                            if q_n not in merged_by_q and old_cq.png_bytes:
                                                merged_by_q[q_n] = old_cq
                                        retag_crops = sorted(merged_by_q.values(), key=lambda c: c.q_num)
                                        _append_retag_log(
                                            f"✂️ 원본 PDF 재크롭 완료: 기존 {len(exam_recs)}문항 → 확보 {len(retag_crops)}문항 "
                                            f"(신규 복구 {max(0, len(retag_crops) - len(exam_recs))}문항)"
                                        )
                                    else:
                                        retag_crops = sorted(saved_crops_by_q.values(), key=lambda c: c.q_num)

                                    if not retag_crops or (
                                        not retag_prob_bytes and all(not c.png_bytes for c in retag_crops)
                                    ):
                                        retag_status.empty()
                                        st.error(
                                            "원본 문제지 PDF와 저장된 크롭 이미지를 모두 찾을 수 없어 다시 태깅할 수 없습니다."
                                        )
                                    else:
                                        existing_q_set = {r.q_num for r in exam_recs}
                                        res = _tag_and_save_single_exam(
                                            storage=ctx.storage,
                                            api_key=verified_key,
                                            meta={
                                                "exam_name": exam_name,
                                                "subject_category": subj_cat,
                                                "prob_bytes": retag_prob_bytes,
                                                "prob_name": retag_prob_name if retag_prob_bytes else "",
                                                "ans_bytes": retag_ans_bytes,
                                                "ans_name": retag_ans_name if retag_ans_bytes else "",
                                                "source_pdf": (
                                                    retag_prob_name
                                                    if retag_prob_bytes
                                                    else next((r.source_pdf for r in exam_recs if r.source_pdf), "")
                                                ),
                                                "answer_pdf": (
                                                    retag_ans_name
                                                    if retag_ans_bytes
                                                    else next((r.answer_pdf for r in exam_recs if r.answer_pdf), "")
                                                ),
                                                "skip_crop_save": False,
                                            },
                                            parsed_crops=retag_crops,
                                            target_q_nums=[c.q_num for c in retag_crops],
                                            is_retry=True,
                                            push_status=retag_status.info,
                                            append_log=_append_retag_log,
                                            on_progress=lambda v: retag_prog.progress(0.40 + 0.60 * v),
                                            on_warning=st.warning,
                                        )
                                        saved_q_set = {r.q_num for r in res["newly_saved"]}
                                        st.session_state["tab1_retag_notice"] = {
                                            "exam_name": exam_name,
                                            "prev_count": len(exam_recs),
                                            "saved": len(res["newly_saved"]),
                                            "recovered": len(saved_q_set - existing_q_set),
                                            "failed": res["failed_q_nums"],
                                            "elapsed": res["elapsed_s"],
                                            "sync_err": res["sync_err"],
                                        }
                                        st.rerun()

                        # 회차별 내보내기·원본 PDF 및 문항 상세 이미지 지연 로드 토글 (검색 목록 초기 렌더링 속도 극대화)
                        tgl_c1, tgl_c2 = st.columns(2)
                        with tgl_c1:
                            open_exports = st.toggle(
                                "📥 내보내기 · 원본 PDF 다운로드 열기",
                                value=False,
                                key=f"tgl_exports_{subj_cat}_{exam_uid}",
                            )
                        with tgl_c2:
                            open_details = st.toggle(
                                f"📋 '{exam_name}' 문항 상세 · 이미지 열기 ({len(exam_recs)}문항)",
                                value=False,
                                key=f"tgl_details_{subj_cat}_{exam_uid}",
                            )

                        if open_exports:
                            records_sig = f"{len(exam_recs)}:{','.join(r.id for r in exam_recs)}"
                            html_bytes, md_bytes, ai_jsonl_bytes, json_bytes = _build_cached_exam_exports(
                                ctx.storage,
                                exam_name,
                                subj_cat,
                                records_sig,
                                exam_recs,
                            )
                            dl_c1, dl_c2, dl_c3, dl_c4, dl_c5, dl_c6 = st.columns(6)
                            with dl_c1:
                                st.download_button(
                                    "🌐 웹페이지(HTML)",
                                    data=html_bytes,
                                    file_name=f"{safe_fn_prefix}_기출정리.html",
                                    mime="text/html",
                                    key=f"dl_html_search_{subj_cat}_{exam_uid}",
                                    width="stretch",
                                )
                            with dl_c2:
                                st.download_button(
                                    "📝 정리본(MD)",
                                    data=md_bytes,
                                    file_name=f"{safe_fn_prefix}_기출정리.md",
                                    mime="text/markdown",
                                    key=f"dl_md_search_{subj_cat}_{exam_uid}",
                                    width="stretch",
                                )
                            with dl_c3:
                                st.download_button(
                                    "🤖 AI용(JSONL)",
                                    data=ai_jsonl_bytes,
                                    file_name=f"{safe_fn_prefix}_ai_index.jsonl",
                                    mime="application/x-ndjson",
                                    key=f"dl_jsonl_search_{subj_cat}_{exam_uid}",
                                    width="stretch",
                                )
                            with dl_c4:
                                st.download_button(
                                    "📦 JSON",
                                    data=json_bytes,
                                    file_name=f"{safe_fn_prefix}_문제목록.json",
                                    mime="application/json",
                                    key=f"dl_json_search_{subj_cat}_{exam_uid}",
                                    width="stretch",
                                )

                            prob_pdf_name = next((r.source_pdf for r in exam_recs if r.source_pdf), "")
                            ans_pdf_name = next((r.answer_pdf for r in exam_recs if r.answer_pdf), "")
                            with dl_c5:
                                prob_bytes = (
                                    ctx.storage.get_pdf_bytes(prob_pdf_name, subject_category=subj_cat, exam_name=exam_name)
                                    if prob_pdf_name
                                    else None
                                )
                                if prob_bytes:
                                    st.download_button(
                                        "📕 문제지 PDF",
                                        data=prob_bytes,
                                        file_name=prob_pdf_name,
                                        mime="application/pdf",
                                        key=f"dl_prob_search_{subj_cat}_{exam_uid}",
                                        width="stretch",
                                    )
                            with dl_c6:
                                ans_bytes = (
                                    ctx.storage.get_pdf_bytes(ans_pdf_name, subject_category=subj_cat, exam_name=exam_name)
                                    if ans_pdf_name
                                    else None
                                )
                                if ans_bytes:
                                    st.download_button(
                                        "📗 해설지 PDF",
                                        data=ans_bytes,
                                        file_name=ans_pdf_name,
                                        mime="application/pdf",
                                        key=f"dl_ans_search_{subj_cat}_{exam_uid}",
                                        width="stretch",
                                    )

                        if open_details:
                            with st.container(border=True):
                                for rec in exam_recs:
                                    warn_prefix = "⚠️ 검토 필요 · " if getattr(rec, "needs_review", False) else ""
                                    if rec.subject_category == "영어":
                                        sec_lbl = getattr(rec, "english_section", "") or ("듣기" if rec.q_num <= 17 else "독해")
                                        type_lbl = getattr(rec, "english_question_type", "") or rec.unit_name
                                        grp_lbl = f" · 세트 `[{rec.passage_group_id}]`" if getattr(rec, "passage_group_id", "") else ""
                                        st.markdown(
                                            f"{warn_prefix}**{rec.q_num}번** · `영어 ({sec_lbl})` · {type_lbl}{grp_lbl} "
                                            f"(정답 `{rec.answer}`)"
                                        )
                                    else:
                                        st.markdown(
                                            f"{warn_prefix}**{rec.q_num}번** · `{rec.curriculum_subject}` · {rec.unit_name} "
                                            f"(고{rec.min_grade}, {'단일 원리' if rec.is_single_concept else '복합 원리'}, 정답 `{rec.answer}`)"
                                        )
                                    col_a, col_b = st.columns([1, 1.3])
                                    with col_a:
                                        img_b = ctx.storage.get_crop_bytes(
                                            rec.image_filename,
                                            subject_category=rec.subject_category,
                                            exam_name=rec.exam_name,
                                        )
                                        if img_b:
                                            st.image(img_b, width="stretch")
                                    with col_b:
                                        _render_question_tag_details(
                                            rec,
                                            storage=ctx.storage,
                                            key_prefix="db_search",
                                            allow_delete=True,
                                        )
                                    st.markdown("---")

