from __future__ import annotations

import hashlib
import os
import re
from typing import Any

import streamlit as st

from src.config import EXAM_TAGGING_MODEL
from src.gemini import (
    classify_mixed_exam_pdfs_with_ai,
    extract_exam_header_metadata,
    match_problem_and_answer_pdfs_with_ai,
)
from src.models import normalize_exam_title_year, normalize_exam_year
from src.pdf_parser import (
    extract_exam_metadata_fast,
    extract_first_page_for_metadata,
    extract_pdf_preview_for_pairing,
    is_likely_answer_filename,
    pair_problem_and_answer_files,
    render_pdf_page_png,
)
from src.ui.common import _ensure_token_quota, _usage_scope

def _pdf_display_basename(raw_name: str) -> str:
    """폴더 업로드 시 포함된 상대 경로를 제거해 순수 파일명만 반환한다."""
    return os.path.basename((raw_name or "").replace("\\", "/")) or (raw_name or "exam.pdf")


def _detect_single_exam_cover(
    prob_bytes: bytes,
    prob_name: str,
    api_key: str,
) -> tuple[dict[str, str], bool, bool]:
    """단일 문제지 PDF의 1페이지 표지 메타데이터를 1차 PyMuPDF + 2차 AI Vision으로 교차 판독한다."""
    parsed_local: dict[str, str] = {
        "year": "",
        "month": "",
        "grade": "",
        "subject": "",
        "exam_title": "",
    }
    has_p1_text = False
    ai_verified = False
    try:
        parsed_local, has_p1_text = extract_exam_metadata_fast(prob_bytes, filename=prob_name)
    except Exception:
        has_p1_text = False

    if api_key and _ensure_token_quota(show_error=False):
        try:
            p1_text, p1_png = extract_first_page_for_metadata(prob_bytes, dpi=140)
            if p1_png is None:
                p1_png = render_pdf_page_png(prob_bytes, 1, dpi=140)
            with _usage_scope():
                ai_meta = extract_exam_header_metadata(
                    api_key=api_key,
                    model_name=EXAM_TAGGING_MODEL,
                    first_page_text=p1_text,
                    first_page_png_bytes=p1_png,
                    filename=prob_name,
                )
            raw_ai_year = (ai_meta.year or "").strip().rstrip("년")
            merged_month = (ai_meta.month or parsed_local.get("month") or "").strip()
            if merged_month and not merged_month.endswith("월"):
                merged_month = f"{merged_month}월"
            merged_grade = (ai_meta.grade or parsed_local.get("grade") or "").strip()
            merged_subject = (ai_meta.subject or parsed_local.get("subject") or "").strip()
            merged_year = normalize_exam_year(
                raw_ai_year,
                text=p1_text,
                filename=prob_name,
                exam_title=ai_meta.exam_title or "",
                month=merged_month,
                grade=merged_grade,
            ) or parsed_local.get("year", "")
            if merged_year and merged_month and merged_grade and merged_subject:
                merged_title = f"{merged_year}년 {merged_month} {merged_grade} {merged_subject}"
            else:
                merged_title = normalize_exam_title_year(
                    (ai_meta.exam_title or parsed_local.get("exam_title") or "").strip(),
                    text=p1_text,
                    filename=prob_name,
                )
            parsed_local = {
                "year": merged_year,
                "month": merged_month,
                "grade": merged_grade,
                "subject": merged_subject,
                "exam_title": merged_title,
            }
            ai_verified = True
        except Exception:
            ai_verified = False

    return parsed_local, has_p1_text, ai_verified


def _compute_pairing_signature(prob_files: list[Any], ans_files: list[Any]) -> tuple[Any, Any]:
    return (
        tuple((f.name, int(getattr(f, "size", 0))) for f in prob_files),
        tuple((f.name, int(getattr(f, "size", 0))) for f in ans_files),
    )


def _compute_pairing_widget_hash(pair_sig: Any, version: int) -> str:
    return hashlib.md5(f"{pair_sig!r}:{version}".encode("utf-8")).hexdigest()[:10]


def _build_pairing_preview_items(uploaded_files: list[Any]) -> list[dict[str, Any]]:
    """업로드된 PDF 목록의 1페이지 텍스트/저해상도 썸네일을 세션 캐시로 추출한다."""
    cache: dict[tuple[str, int], dict[str, Any]] = st.session_state.setdefault("tab1_pdf_preview_cache", {})
    items: list[dict[str, Any]] = []
    for f in uploaded_files:
        sig = (f.name, int(getattr(f, "size", 0)))
        cached = cache.get(sig)
        if cached is None:
            p_text, p_png, page_cnt = extract_pdf_preview_for_pairing(f.getvalue())
            cached = {
                "name": _pdf_display_basename(f.name),
                "first_page_text": p_text,
                "first_page_png": p_png,
                "page_count": page_cnt,
            }
            cache[sig] = cached
        items.append(cached)
    return items


def _resolve_mixed_pdf_split_with_ai(
    raw_files: list[Any],
    api_key: str,
) -> tuple[list[int], list[int], str]:
    """단일 업로더에 문제지·해설지가 섞여 들어온 경우 Gemini 3.5 Flash-Lite로 자동 분리한다."""
    raw_sig = (tuple((f.name, int(getattr(f, "size", 0))) for f in raw_files), bool(api_key))
    cache: dict[Any, tuple[list[int], list[int], str]] = st.session_state.setdefault("tab1_ai_split_cache", {})
    if raw_sig in cache:
        return cache[raw_sig]

    if api_key and _ensure_token_quota(show_error=False):
        try:
            with st.spinner("🤖 Gemini 3.5 Flash-Lite가 업로드된 PDF 중 문제지·해설지를 자동 구분 중..."):
                preview_items = _build_pairing_preview_items(raw_files)
                with _usage_scope():
                    p_idxs, a_idxs = classify_mixed_exam_pdfs_with_ai(
                        api_key=api_key,
                        pdf_items=preview_items,
                        model_name=EXAM_TAGGING_MODEL,
                    )
            res = (p_idxs, a_idxs, "ai")
            cache[raw_sig] = res
            return res
        except Exception:
            pass

    fb_prob = [i for i, f in enumerate(raw_files) if not is_likely_answer_filename(f.name)]
    fb_ans = [i for i, f in enumerate(raw_files) if is_likely_answer_filename(f.name)]
    res = (fb_prob, fb_ans, "filename")
    cache[raw_sig] = res
    return res


def _resolve_auto_answer_matches(
    prob_files: list[Any],
    ans_files: list[Any],
    api_key: str,
) -> tuple[list[int | None], dict[int, str], str, str]:
    """
    업로드된 문제지·해설지 PDF 조합에 대해 Gemini 3.5 Flash-Lite 자동 매칭을 수행하고
    결과를 세션에 캐시한다. 실패 또는 API 키 미설정 시 로컬 파일명 매칭으로 폴백한다.
    """
    n_prob = len(prob_files)
    if n_prob == 0 or not ans_files:
        return [None] * n_prob, {}, "none", ""
    if n_prob == 1 and len(ans_files) == 1:
        return [0], {0: "단일 문제지·해설지 자동 연결"}, "single", ""

    pair_sig = _compute_pairing_signature(prob_files, ans_files)
    cache_key = (pair_sig, bool(api_key))
    cache: dict[Any, tuple[list[int | None], dict[int, str], str, str]] = st.session_state.setdefault(
        "tab1_ai_pair_cache", {}
    )
    if cache_key in cache:
        return cache[cache_key]

    err_msg = ""
    if api_key and _ensure_token_quota(show_error=False):
        try:
            with st.spinner("🤖 Gemini 3.5 Flash-Lite가 문제지·해설지 PDF 1페이지를 비교해 자동 매칭 중..."):
                prob_items = _build_pairing_preview_items(prob_files)
                ans_items = _build_pairing_preview_items(ans_files)
                with _usage_scope():
                    matches, reasons = match_problem_and_answer_pdfs_with_ai(
                        api_key=api_key,
                        problem_items=prob_items,
                        answer_items=ans_items,
                        model_name=EXAM_TAGGING_MODEL,
                    )
            res = (matches, reasons, "ai", "")
            cache[cache_key] = res
            return res
        except Exception as exc:
            err_msg = str(exc)
    else:
        err_msg = "서버 API 키 미설정"

    fb_matches = pair_problem_and_answer_files(
        [f.name for f in prob_files],
        [f.name for f in ans_files],
    )
    res = (fb_matches, {}, "fallback", err_msg)
    cache[cache_key] = res
    return res


def _invalidate_auto_pair_cache(pair_sig: Any, api_key: str) -> None:
    """현재 업로드 조합의 AI 매칭 캐시를 비우고 드롭다운 키 버전을 올려 즉시 재매칭한다."""
    cache_key = (pair_sig, bool(api_key))
    st.session_state.get("tab1_ai_pair_cache", {}).pop(cache_key, None)
    st.session_state["tab1_ai_pair_ver"] = int(st.session_state.get("tab1_ai_pair_ver", 0)) + 1
