from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Sequence

from src.curriculum_2022.catalog import (
    CURRICULUM_CATEGORIES,
    _canon_subject_key,
    _load_curriculum_2022_catalog_readonly,
    normalize_subject_2022,
)

if TYPE_CHECKING:
    from src.models import ConceptBookRecord, QuestionExtractionOutput

_EVERYDAY_TERM_STOPWORDS = frozenset(
    {
        "그림",
        "표",
        "자료",
        "보기",
        "다음",
        "문항",
        "문제",
        "정답",
        "해설",
        "과정",
        "결과",
        "값",
        "합",
        "차",
        "곱",
        "비",
        "수",
        "점",
        "선",
        "면",
        "ㄱ",
        "ㄴ",
        "ㄷ",
        "ㄹ",
        "개",
        "번",
        "명",
        "회",
        "가지",
        "원",
        "cm",
        "mm",
        "km",
        "g",
        "kg",
        "mg",
        "ml",
        "mL",
        "L",
        "s",
        "sec",
        "min",
        "h",
    }
)
_STD_CODE_ONLY_RE = re.compile(r"^\[?(?:10|12)[가-힣0-9ⅠⅡ]+(?:\d+)?(?:-\d{2}){1,2}\]?$")
def validate_standards_only(
    standard_code: str,
    standard_codes: Sequence[str],
    subject_category: str,
    catalog: dict[str, Any] | None = None,
) -> tuple[str, list[str], bool, list[str]]:
    """
    국어·영어 등 별도 스키마 교과용 성취기준 코드 정제 및 할루시네이션(가짜 코드) 검증 헬퍼.
    반환: (정제된_대표코드, 정제된_코드목록, 검토필요_여부, 검토_노트)
    """
    cat_obj = catalog or _load_curriculum_2022_catalog_readonly()
    cat_entry = cat_obj.get("categories", {}).get(subject_category, {})
    loaded_stds = {
        (s.get("code") or "").strip(): s
        for s in (cat_entry.get("standards") or [])
        if s.get("code")
    }

    std_code = re.sub(r"[\[\]\s]+", "", standard_code or "")
    cleaned_codes: list[str] = []
    seen_codes: set[str] = set()
    for rc in ([std_code] if std_code else []) + list(standard_codes or []):
        c = re.sub(r"[\[\]\s]+", "", rc or "")
        if c and c not in seen_codes:
            seen_codes.add(c)
            cleaned_codes.append(c)

    if not std_code and cleaned_codes:
        std_code = cleaned_codes[0]

    needs_review = False
    review_notes: list[str] = []
    if loaded_stds and cleaned_codes:
        fake_codes = [c for c in cleaned_codes if _STD_CODE_ONLY_RE.match(c) and c not in loaded_stds]
        if fake_codes:
            needs_review = True
            joined_fake = ", ".join(f"[{c}]" for c in fake_codes)
            review_notes.append(f"로드된 NCIC 공식 성취기준 목록에 없는 코드입니다(할루시네이션 의심): {joined_fake}")

    return std_code, cleaned_codes, needs_review, review_notes


def _infer_target_subject_from_exam_name(exam_name: str, cat_entry: dict[str, Any]) -> str:
    """시험 회차명(예: '2026년 3월 고3 물리학1')에서 2022 공식 과목명을 정밀 추론한다."""
    canon_exam = _canon_subject_key(exam_name)
    if not canon_exam:
        return ""
    candidates: list[tuple[str, str]] = []
    for subj in cat_entry.get("subjects") or []:
        official_name = (subj.get("name") or "").strip()
        if not official_name:
            continue
        candidates.append((_canon_subject_key(official_name), official_name))
        for alias in subj.get("aliases") or []:
            ck = _canon_subject_key(alias)
            if ck:
                candidates.append((ck, official_name))
    # 긴 과목/별칭 우선 매칭 ('미적분2'가 '미적분'보다 우선)
    candidates.sort(key=lambda x: len(x[0]), reverse=True)
    for key, official_name in candidates:
        if key and key in canon_exam:
            return official_name
    return ""


def build_tagging_grounding_context(
    subject_category: str,
    exam_name: str = "",
    concept_books: Sequence[ConceptBookRecord] | None = None,
    max_standards: int = 45,
    max_book_units: int = 28,
    catalog: dict[str, Any] | None = None,
) -> str:
    """
    Tab 1 태깅 프롬프트에 주입할 2022 개정 공식 기준표 + NCIC 성취기준 + 공용 개념서 DB 요약 컨텍스트를 조립한다.
    현재 시험지의 과목이 판별되면 해당 과목과 고1 공통과목만 동적 필터링해 프롬프트 토큰을 절감한다.
    """
    cat_obj = catalog or _load_curriculum_2022_catalog_readonly()
    cat_data = cat_obj.get("categories", {})
    entry = cat_data.get(subject_category)
    if not entry:
        return ""

    target_subject = _infer_target_subject_from_exam_name(exam_name, entry)
    common_subjects = {
        (s.get("name") or "").strip()
        for s in (entry.get("subjects") or [])
        if s.get("track") == "공통" and s.get("name")
    }
    allowed_scope_subjects: set[str] | None = (
        ({target_subject} | common_subjects) if target_subject else None
    )

    header = f"[2022 개정 교육과정 공식 기준표 — 대분류: {subject_category}"
    if target_subject:
        header += f" · 시험 과목 필터: {target_subject}"
    header += "]"
    blocks: list[str] = [header]

    # 1) 과목 편제표 (구분별 공식 과목명)
    by_track: dict[str, list[str]] = {}
    for subj in entry.get("subjects") or []:
        track = subj.get("track", "선택")
        by_track.setdefault(track, []).append(subj.get("name", ""))
    for track, names in by_track.items():
        if names:
            grade_hint = "고1 권장" if track == "공통" else "고2~3 선택"
            blocks.append(f"- {track}({grade_hint}): {', '.join(names)}")

    # 2) 2015→2022 개정 주요 변경 주의사항
    shifts = entry.get("shift_notes") or []
    if shifts:
        blocks.append("\n[2015→2022 개정 이동 주의사항 (할루시네이션 방지)]")
        for note in shifts[:5]:
            blocks.append(f"- {note}")

    # 3) 과목별 핵심 단원 개요 (대상 과목이 판별되면 해당 과목 + 고1 공통만 포함)
    core_units = entry.get("core_units") or {}
    if core_units:
        filtered_units = {
            s_name: u_list
            for s_name, u_list in core_units.items()
            if allowed_scope_subjects is None or s_name in allowed_scope_subjects
        }
        if filtered_units:
            blocks.append("\n[2022 관련 과목 단원 체계]")
            for s_name, u_list in filtered_units.items():
                blocks.append(f"- {s_name}: {', '.join(u_list)}")

    # 4) NCIC 공식 고시 PDF에서 파싱된 성취기준 (대상 과목 + 고1 공통만 동적 필터링)
    standards = entry.get("standards") or []
    if standards:
        primary_stds: list[dict[str, Any]] = []
        common_stds: list[dict[str, Any]] = []
        fallback_stds: list[dict[str, Any]] = []
        for st_item in standards:
            raw_st_subj = st_item.get("subject", "")
            norm_st_subj, _, _, _ = normalize_subject_2022(raw_st_subj, subject_category, cat_obj)
            eff_st_subj = norm_st_subj or raw_st_subj
            if target_subject and eff_st_subj == target_subject:
                primary_stds.append(st_item)
            elif eff_st_subj in common_subjects or st_item.get("min_grade") == 1:
                common_stds.append(st_item)
            elif allowed_scope_subjects is None:
                fallback_stds.append(st_item)

        chosen = (primary_stds + common_stds + fallback_stds)[:max_standards]
        if chosen:
            blocks.append(f"\n[NCIC 공식 고시 성취기준 발췌 ({len(chosen)}/{len(standards)}개 · 과목 필터링 적용)]")
            for st_item in chosen:
                blocks.append(
                    f"- [{st_item.get('code')}] ({st_item.get('subject')} · 고{st_item.get('min_grade')}) "
                    f"{st_item.get('statement', '')[:120]}"
                )

    # 5) 공용 개념서 DB(concept_db) 단원·허용 용어 연동 (대상 과목 + 고1 공통 교재만 선택)
    if concept_books:
        matching_books: list[ConceptBookRecord] = []
        cat_subject_tokens = {subj.get("name", "") for subj in (entry.get("subjects") or []) if subj.get("name")}
        for bk in concept_books:
            bk_blob = f"{bk.book_title} {bk.subject}"
            in_category = subject_category in bk_blob or any(tok in bk_blob for tok in cat_subject_tokens)
            if not in_category:
                continue
            if allowed_scope_subjects is not None:
                bk_subj_norm, bk_ok, _, _ = normalize_subject_2022(bk.subject, subject_category, cat_obj)
                if not bk_ok:
                    bk_subj_norm = _infer_target_subject_from_exam_name(bk_blob, entry)
                is_grade1_book = "1" in (bk.grade or "")
                if bk_subj_norm and bk_subj_norm not in allowed_scope_subjects:
                    continue
                if not bk_subj_norm and not is_grade1_book and target_subject not in bk_blob:
                    continue
            matching_books.append(bk)

        if matching_books:
            blocks.append("\n[등록된 공용 개념서 DB(concept_db) 단원 및 허용 용어]")
            unit_budget = max_book_units
            for bk in matching_books[:4]:
                if unit_budget <= 0:
                    break
                blocks.append(f"• {bk.book_title} ({bk.grade} · {bk.subject}):")
                for u in bk.units[: min(len(bk.units), unit_budget)]:
                    concepts_str = ", ".join(u.core_concepts[:5])
                    terms_str = ", ".join(u.allowed_terms_and_units[:8])
                    blocks.append(f"  - {u.unit_name}: 원리[{concepts_str}] / 허용용어[{terms_str}]")
                    unit_budget -= 1

    return "\n".join(blocks)


def _clean_term_list(terms: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in terms:
        t = re.sub(r"\s+", " ", (raw or "").strip())
        if not t or t in _EVERYDAY_TERM_STOPWORDS or t.lower() in _EVERYDAY_TERM_STOPWORDS:
            continue
        if len(t) == 1 and t.isdigit():
            continue
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def _collect_lower_grade_allowed_vocab(
    subject_category: str,
    min_grade: int,
    concept_books: Sequence[ConceptBookRecord] | None,
) -> set[str]:
    vocab: set[str] = set()
    if not concept_books:
        return vocab
    for bk in concept_books:
        m = re.search(r"[123]", bk.grade or "")
        bk_grade = int(m.group(0)) if m else 3
        if bk_grade <= min_grade:
            for u in bk.units:
                for term in u.allowed_terms_and_units:
                    clean_t = (term or "").strip()
                    if clean_t:
                        vocab.add(clean_t.lower())
    return vocab


def validate_and_normalize_extracted_tag(
    extracted: QuestionExtractionOutput,
    subject_category: str,
    concept_books: Sequence[ConceptBookRecord] | None = None,
    catalog: dict[str, Any] | None = None,
) -> QuestionExtractionOutput:
    """
    Gemini 태깅 출력을 2022 개정 공식 기준표 및 공용 개념서 DB와 대조해
    1) 2015→2022 과목명 자동 정규화
    2) 기본 용어(basic_terms) vs 상위 전문 용어(advanced_terms) 분리 및 일상어 필터링
    3) 과목-학년 모순 및 가짜 성취기준 코드 할루시네이션 검출(needs_review)
    을 수행한다.
    """
    cat_obj = catalog or _load_curriculum_2022_catalog_readonly()
    cat_entry = cat_obj.get("categories", {}).get(subject_category, {})
    loaded_stds = {
        (s.get("code") or "").strip(): s
        for s in (cat_entry.get("standards") or [])
        if s.get("code")
    }

    review_notes: list[str] = list(getattr(extracted, "review_notes", []) or [])
    needs_review = bool(getattr(extracted, "needs_review", False))

    norm_subj, is_official, subj_meta, was_alias = normalize_subject_2022(
        extracted.curriculum_subject,
        subject_category=subject_category,
        catalog=cat_obj,
    )
    if norm_subj:
        extracted.curriculum_subject = norm_subj
    if was_alias:
        review_notes.append(f"2022 공식 과목명으로 자동 정규화됨: {norm_subj}")
    if not is_official:
        needs_review = True
        review_notes.append(f"2022 '{subject_category}' 공식 과목 목록에서 확인되지 않은 과목명입니다: {extracted.curriculum_subject}")

    # 복수 성취기준 코드 정제·동기화 및 할루시네이션 검출
    std_code = re.sub(r"[\[\]\s]+", "", getattr(extracted, "standard_code", "") or "")
    raw_codes = list(getattr(extracted, "standard_codes", []) or [])
    cleaned_codes: list[str] = []
    seen_codes: set[str] = set()
    for rc in ([std_code] if std_code else []) + raw_codes:
        c = re.sub(r"[\[\]\s]+", "", rc or "")
        if c and c not in seen_codes:
            seen_codes.add(c)
            cleaned_codes.append(c)

    if not std_code and cleaned_codes:
        std_code = cleaned_codes[0]

    extracted.standard_code = std_code
    extracted.standard_codes = cleaned_codes

    if loaded_stds:
        fake_codes = [c for c in cleaned_codes if _STD_CODE_ONLY_RE.match(c) and c not in loaded_stds]
        if fake_codes:
            needs_review = True
            joined_fake = ", ".join(f"[{c}]" for c in fake_codes)
            review_notes.append(f"로드된 NCIC 공식 성취기준 목록에 없는 코드입니다(할루시네이션 의심): {joined_fake}")

    # 과목 편제(공통 vs 선택) ↔ 최저 학년 일관성 점검
    grade_evidence = (getattr(extracted, "grade_evidence", "") or "").strip()
    extracted.grade_evidence = grade_evidence
    if subj_meta:
        track = subj_meta.get("track", "")
        min_allowed = int(subj_meta.get("min_allowed_grade") or 1)
        if track == "공통" and extracted.min_grade > 1:
            needs_review = True
            review_notes.append(f"'{extracted.curriculum_subject}'는 고1 공통과목이나 min_grade={extracted.min_grade}로 판정되어 확인이 필요합니다.")
        elif min_allowed >= 2 and extracted.min_grade == 1:
            has_only_grade2_codes = bool(cleaned_codes) and all(c.startswith("12") for c in cleaned_codes)
            if has_only_grade2_codes:
                needs_review = True
                joined_c = ", ".join(f"[{c}]" for c in cleaned_codes)
                review_notes.append(
                    f"성취기준 코드({joined_c})는 고2~3 선택과목 범위인데 min_grade=1로 판정되어 모순이 있습니다."
                )
            elif not grade_evidence:
                needs_review = True
                review_notes.append(
                    f"선택과목('{extracted.curriculum_subject}') 문항을 고1(min_grade=1)로 판정했으나 고1 공통 원리 근거가 비어 있습니다."
                )

    if not extracted.core_principles:
        needs_review = True
        review_notes.append("핵심 풀이 원리(core_principles)가 비어 있습니다.")

    # 용어 정제: everyday 불용어 제거 + concept_db 저학년 허용 용어는 basic_terms로 교정
    adv_clean = _clean_term_list(getattr(extracted, "advanced_terms", []) or [])
    basic_clean = _clean_term_list(getattr(extracted, "basic_terms", []) or [])
    req_clean = _clean_term_list(extracted.required_terms or [])

    lower_vocab = _collect_lower_grade_allowed_vocab(subject_category, extracted.min_grade, concept_books)
    if lower_vocab and adv_clean:
        rebalanced_adv: list[str] = []
        for term in adv_clean:
            if term.lower() in lower_vocab:
                if term.lower() not in {b.lower() for b in basic_clean}:
                    basic_clean.append(term)
            else:
                rebalanced_adv.append(term)
        adv_clean = rebalanced_adv

    # required_terms는 기존 Tab 2 매칭 호환을 위해 상위·기본 학술 용어 전체 합집합으로 유지
    merged_req = _clean_term_list([*adv_clean, *basic_clean, *req_clean])
    if not adv_clean and not basic_clean and merged_req:
        adv_clean = list(merged_req)

    extracted.advanced_terms = adv_clean
    extracted.basic_terms = basic_clean
    extracted.required_terms = merged_req
    extracted.needs_review = needs_review
    extracted.review_notes = list(dict.fromkeys(review_notes))
    return extracted
