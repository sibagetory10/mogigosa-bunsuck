from __future__ import annotations

import hashlib
import html as _html
import re
from typing import TYPE_CHECKING, Any, Sequence

import streamlit as st

from src.auth import can_admin_shared_db, resolve_active_gemini_api_key
from src.curriculum_2022 import get_official_subjects_for_category
from src.models import (
    QuestionRecord,
    infer_subject_category,
    normalize_english_group_id,
    normalize_exam_title_year,
    normalize_korean_group_id,
)
from src.storage import StorageManager
from src.ui.common import (
    _build_cached_exam_exports,
    _ensure_token_quota,
    sanitize_id_part,
)
from src.ui.context import AppContext
from src.ui_styles import render_chips

if TYPE_CHECKING:
    from src.pdf_parser import CroppedQuestion

def _render_question_tag_details(
    rec: QuestionRecord,
    storage: StorageManager,
    key_prefix: str,
    allow_delete: bool = False,
    allow_edit: bool = False,
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

        if allow_edit:
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

        if allow_edit:
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

    if allow_edit:
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


def _render_exam_coverage_board(ctx: AppContext, db_records: Sequence[QuestionRecord] | None = None) -> None:
    """탭 1 상단에 이미 등록된 기출 목록과 추천 미등록 기출 목록을 나란히 표시한다."""
    if db_records is None:
        db_records = ctx.storage.load_db_readonly()
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

def render_tab1_db_explorer(
    ctx: AppContext,
    db_records: Sequence[QuestionRecord] | None = None,
) -> None:
    """과목별 · 모의고사 회차별 기출 DB 탐색 및 파일 내보내기 UI를 렌더링한다."""
    st.markdown("---")
    st.subheader("3️⃣ 과목별 · 모의고사 회차별 기출 DB 탐색 & 파일 내보내기")
    st.caption(
        "💾 로컬 `./exam_db/과목별_기출DB/<과목>/<회차명>/` 폴더에 "
        "`기출정리.html`, `기출정리.md`, `문제목록.json`, 크롭 PNG, 원본 PDF가 자동 정리됩니다."
    )

    if db_records is None:
        db_records = ctx.storage.load_db_readonly()

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
                can_admin_db = can_admin_shared_db(ctx.current_user)
                can_retag_exam = can_admin_db
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
                            st.markdown(f"#### 📘 {exam_name}")
                            st.caption(
                                f"과목: **{subj_cat}** · 저장 문항: **{len(exam_recs)}문항** · 단원: {', '.join(units_preview[:4])}"
                                + (" 등" if len(units_preview) > 4 else "")
                            )

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
                                    from src.pdf_parser import CroppedQuestion
                                    from src.ui.tab1_tagging import _crop_single_exam_pdf, _tag_and_save_single_exam

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
                                            allow_edit=can_admin_db,
                                            allow_delete=can_admin_db,
                                        )
                                    st.markdown("---")
