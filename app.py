from __future__ import annotations

import concurrent.futures
import contextlib
import hashlib
import html as _html
import json
import os
import random
import re
import threading
import time
from datetime import datetime
from typing import Any, Callable

import streamlit as st

from src.auth import (
    FREE_EXAMS_PER_UNLOCK,
    FREE_MAX_BATCH_QUESTIONS,
    FREE_MONTHLY_TOKEN_LIMIT,
    PRO_MONTHLY_PRICE_KRW,
    PRO_MONTHLY_TOKEN_LIMIT,
    PRO_WEEKLY_PREMIUM_LIMIT,
    TIER_LABELS,
    ULTRA_MONTHLY_PRICE_KRW,
    ULTRA_MONTHLY_TOKEN_LIMIT,
    ULTRA_WEEKLY_PREMIUM_LIMIT,
    VALID_TIERS,
    authenticate_user,
    batch_update_user_tiers,
    can_access_shared_concept_db,
    can_use_premium_features,
    check_token_quota,
    clear_user_custom_api_key,
    consume_extraction_quota,
    consume_premium_model_usage,
    create_session_token,
    delete_users_by_usernames,
    format_krw,
    format_token_count,
    get_kst_month_str,
    get_max_batch_questions,
    get_monthly_token_limit,
    get_remaining_monthly_tokens,
    get_remaining_pro_weekly_quota,
    get_used_monthly_tokens,
    get_user_by_username,
    get_weekly_premium_limit,
    is_premium_model,
    is_server_api_key_user,
    mask_api_key,
    record_exam_upload_for_user,
    record_token_usage,
    register_user,
    reset_user_monthly_tokens,
    resolve_active_gemini_api_key,
    resolve_model_for_user,
    save_user_custom_api_key,
    verify_session_token,
)
from src.config import (
    CONCEPT_DB_BUILD_MODEL,
    CONCEPT_SCOPE_MODEL,
    DEFAULT_MODEL,
    EXAM_NO_ANSWER_SOLVER_MODEL,
    EXAM_TAGGING_MODEL,
    FLASH_LITE_CACHED_INPUT_USD_PER_1M,
    FLASH_LITE_INPUT_USD_PER_1M,
    FLASH_LITE_MODEL,
    FLASH_LITE_OUTPUT_USD_PER_1M,
    MODEL_LABELS,
    resolve_credentials,
    resolve_usd_to_krw_rate,
)
from src.curriculum_2022 import (
    build_tagging_grounding_context,
    get_catalog_summary_stats,
    get_official_subjects_for_category,
    merge_ncic_standards_to_catalog,
    parse_ncic_curriculum_pdf,
)
from src.ebsi_fetcher import attempt_ebsi_pdf_fetch
from src.exporters import build_exam_ai_jsonl, build_exam_html, build_exam_markdown
from src.gemini_service import (
    ForbiddenTermRemainingError,
    TokenUsageRecord,
    analyze_english_exam_pdf_batch_with_gemini,
    analyze_english_question_with_gemini,
    analyze_exam_pdf_batch_with_gemini,
    analyze_question_with_gemini,
    bind_usage_tracker,
    build_gemini_client,
    chat_followup_with_english,
    chat_followup_with_pdf,
    classify_mixed_exam_pdfs_with_ai,
    detect_concept_pages_with_ai,
    detect_scanned_question_boxes,
    extract_concept_book_cover_metadata,
    extract_concept_book_units,
    extract_concept_whitelist,
    extract_exam_header_metadata,
    generate_english_explanation,
    generate_english_transformed_question,
    generate_pdf_grounded_explanation,
    has_active_usage_tracker,
    match_problem_and_answer_pdfs_with_ai,
    match_questions_to_whitelist,
    prepare_pdf_part,
    rewrite_question_for_pdf,
    submit_with_usage_context,
)
from src.models import (
    ENGLISH_DEFAULT_EXCLUDED_Q_NUMS,
    ENGLISH_TRANSFORM_TYPES,
    ConceptBookRecord,
    ConceptUnit,
    ConceptWhitelist,
    QuestionMatchJudgment,
    QuestionRecord,
    RewrittenQuestion,
    build_whitelist_from_concept_units,
    infer_subject_category,
    normalize_english_group_id,
    normalize_exam_title_year,
    normalize_exam_year,
    resolve_question_concept_reference,
)
from src.pdf_generator import generate_custom_exam_pdf
from src.pdf_parser import (
    CroppedQuestion,
    crop_normalized_box,
    extract_answer_snippets_from_pdf,
    extract_concept_pages_by_ranges,
    extract_exam_metadata_fast,
    extract_first_page_for_metadata,
    extract_pdf_page_catalog,
    extract_pdf_preview_for_pairing,
    get_page_text_lengths,
    get_pdf_page_count,
    is_likely_answer_filename,
    pair_problem_and_answer_files,
    parse_exam_pdf,
    parse_page_ranges_spec,
    render_pdf_page_png,
)
from src.storage import StorageManager
from src.text_formatter import (
    format_chemical_scripts,
    format_explanation_text,
    has_visual_reference,
    render_english_inline_html,
    split_english_stem_blocks,
    split_stem_blocks,
)
from src.ui_styles import inject_responsive_css, render_chips

st.set_page_config(
    page_title="2022 개정 교육과정 맞춤형 기출 플랫폼",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="auto",
)

inject_responsive_css()

CIRCLED_MAP = {"1": "①", "2": "②", "3": "③", "4": "④", "5": "⑤"}
REVERSE_CIRCLED_MAP = {v: k for k, v in CIRCLED_MAP.items()}


def normalize_answer(ans: str) -> str:
    s = (ans or "").strip().replace(" ", "")
    if s in REVERSE_CIRCLED_MAP:
        return REVERSE_CIRCLED_MAP[s]
    if len(s) == 2 and s[0] in CIRCLED_MAP and s[1] == "번":
        return s[0]
    return s.lower()


def is_multiple_choice_question(q: QuestionRecord, rewritten: RewrittenQuestion | None) -> bool:
    if rewritten and len(rewritten.rewritten_choices) >= 2:
        return True
    return normalize_answer(q.answer) in {"1", "2", "3", "4", "5"}


def sanitize_id_part(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z가-힣_\-]", "_", text.strip())
    return re.sub(r"_+", "_", cleaned).strip("_") or "exam"


def _init_session_defaults() -> None:
    defaults: dict[str, Any] = {
        "parsed_crops": [],
        "parsed_exam_meta": {},
        "ebsi_problem_bytes": None,
        "ebsi_answer_bytes": None,
        "ebsi_problem_name": "",
        "ebsi_answer_name": "",
        "concept_whitelist": None,
        "concept_sliced_text": "",
        "concept_source_label": "",
        "concept_pdf_hash": "",
        "matched_items": [],
        "match_diagnostics": [],
        "generated_exam_pdf": None,
        "user_answers": {},
        "graded": False,
        "ai_explanations": {},
        "ai_chats": {},
        "drive_sync_key": "",
        "drive_sync_result": None,
        "logged_in_username": "",
        "tab1_last_loaded_sig": "",
        "tab1_meta_status": "",
        "meta_year": "",
        "meta_month": "",
        "meta_grade": "",
        "meta_subject": "",
        "meta_title": "",
        "tab1_subject_cat": "과학",
        "stage2_exam_name": "",
        "stage2_saved_q_nums": [],
        "stage2_failed_q_nums": [],
        "stage2_failed_errors": [],
        "stage2_sync_err": "",
        "trigger_retry_failed": False,
        "db_search_submitted": False,
        "db_search_filters": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session_defaults()

# ---------------------------------------------------------------------------
# 백그라운드 인증 및 저장소 초기화 (서버 캐시 적용으로 새로고침/리런 속도 최적화)
# ---------------------------------------------------------------------------
@st.cache_resource(ttl=1800, show_spinner=False)
def _get_cached_storage(
    local_root: str,
    drive_folder_id: str,
    auth_sig: str,
    _service_account_info: dict[str, Any] | None,
    _oauth_info: dict[str, Any] | None,
) -> StorageManager:
    return StorageManager(
        local_root=local_root,
        service_account_info=_service_account_info,
        drive_folder_id=drive_folder_id,
        oauth_info=_oauth_info,
    )


@st.cache_data(ttl=300, show_spinner=False)
def _run_startup_drive_sync(_storage: StorageManager, sync_fingerprint: str) -> dict[str, Any]:
    return _storage.sync_on_startup(sync_media=False)


@contextlib.contextmanager
def _usage_scope(
    storage_obj: StorageManager | None = None,
    username: str | None = None,
    used_server_key: bool | None = None,
):
    """현재 로그인 사용자의 실제 Gemini API 토큰·USD 비용을 스레드 안전하게 누적해 종료 시 1회 저장한다."""
    if has_active_usage_tracker():
        yield
        return

    st_mgr = storage_obj or globals().get("storage")
    uname = (
        username if username is not None else st.session_state.get("logged_in_username", "")
    ) or ""
    if not st_mgr or not uname:
        yield
        return

    cur_user = globals().get("current_user")
    if used_server_key is None:
        used_server_key = is_server_api_key_user(cur_user)

    acc_lock = threading.Lock()
    acc = {
        "prompt": 0,
        "cached": 0,
        "output": 0,
        "total": 0,
        "calls": 0,
        "cost_usd": 0.0,
    }

    def _on_usage(rec: TokenUsageRecord) -> None:
        with acc_lock:
            acc["prompt"] += rec.prompt_tokens
            acc["cached"] += rec.cached_tokens
            acc["output"] += rec.output_tokens
            acc["total"] += rec.total_tokens
            acc["calls"] += 1
            acc["cost_usd"] += rec.cost_usd

    with bind_usage_tracker(_on_usage):
        try:
            yield
        finally:
            if acc["total"] > 0 or acc["calls"] > 0 or acc["cost_usd"] > 0.0:
                try:
                    record_token_usage(
                        storage=st_mgr,
                        username=uname,
                        prompt_tokens=acc["prompt"],
                        cached_tokens=acc["cached"],
                        output_tokens=acc["output"],
                        total_tokens=acc["total"],
                        cost_usd=acc["cost_usd"],
                        api_calls=acc["calls"],
                        used_server_key=bool(used_server_key),
                        sync_drive=True,
                    )
                except Exception:
                    pass


def _ensure_token_quota(
    storage_obj: StorageManager | None = None,
    username: str | None = None,
    show_error: bool = True,
) -> bool:
    """API 호출 직전 월간 토큰 한도를 검사하고 초과 시 안내 메시지를 출력한 뒤 False를 반환한다."""
    st_mgr = storage_obj or globals().get("storage")
    uname = (
        username if username is not None else st.session_state.get("logged_in_username", "")
    ) or ""
    if not st_mgr or not uname:
        return True
    allowed, _rem, _limit, msg = check_token_quota(st_mgr, uname)
    if not allowed:
        if show_error and msg:
            st.error(msg)
        return False
    return True


@st.cache_data(ttl=300, show_spinner=False)
def _build_cached_exam_exports(
    _storage: StorageManager,
    exam_name: str,
    subj_cat: str,
    records_sig: str,
    _exam_recs: list[QuestionRecord],
) -> tuple[bytes, bytes, bytes, bytes]:
    """회차별 HTML/MD/JSONL/JSON 내보내기 바이트를 캐싱해 불필요한 크롭 이미지 반복 로드·인코딩을 방지한다."""
    html_str = build_exam_html(
        exam_name,
        subj_cat,
        _exam_recs,
        crop_loader=lambda fn, s=subj_cat, e=exam_name: _storage.get_crop_bytes(
            fn, subject_category=s, exam_name=e
        ),
    )
    md_str = build_exam_markdown(exam_name, subj_cat, _exam_recs)
    ai_jsonl_str = build_exam_ai_jsonl(_exam_recs)
    json_str = json.dumps(
        [r.model_dump() for r in _exam_recs],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        html_str.encode("utf-8"),
        md_str.encode("utf-8"),
        ai_jsonl_str.encode("utf-8"),
        json_str.encode("utf-8"),
    )


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


def _expected_max_q_for_subject(subject_hint: str) -> int:
    s = (subject_hint or "").strip()
    if any(k in s for k in ("국어", "영어")):
        return 45
    if any(k in s for k in ("수학", "미적분", "기하", "확률", "대수")):
        return 30
    if any(k in s for k in ("사회", "과학", "물리", "화학", "생명", "지구", "한국사", "윤리", "지리", "경제", "정치", "역사")):
        return 20
    return 0


def _compute_vision_recovery_pages(
    crops: list[CroppedQuestion],
    total_p: int,
    page_text_lens: list[int],
    has_text: bool,
    subject_hint: str = "",
) -> tuple[list[int], list[int]]:
    """
    텍스트가 부족한 스캔 페이지뿐 아니라, PyMuPDF 텍스트 파싱에서 번호가 누락된 문항(예: 도형·수식 위주 문항)이
    위치했을 가능성이 높은 페이지 구간을 추려 Gemini Vision 추가 탐지 대상으로 반환한다.
    """
    if total_p <= 0:
        return [], []
    if not crops or not has_text:
        return list(range(1, total_p + 1)), []

    found_by_q = {c.q_num: c.page_num for c in crops}
    cropped_pages = set(found_by_q.values())
    max_found = max(found_by_q.keys())
    expected_max = _expected_max_q_for_subject(subject_hint)
    range_end = max(max_found, expected_max) if total_p >= 3 else max_found
    missing_q_nums = [n for n in range(1, range_end + 1) if n not in found_by_q]

    pages_set = {
        idx + 1
        for idx, tlen in enumerate(page_text_lens)
        if tlen < 40 and (idx + 1) not in cropped_pages
    }
    if missing_q_nums:
        for p in range(1, total_p + 1):
            if p not in cropped_pages:
                pages_set.add(p)
        found_qs = sorted(found_by_q.keys())
        for m_q in missing_q_nums:
            lower_qs = [q for q in found_qs if q < m_q]
            higher_qs = [q for q in found_qs if q > m_q]
            prev_p = found_by_q[lower_qs[-1]] if lower_qs else 1
            next_p = found_by_q[higher_qs[0]] if higher_qs else total_p
            p_lo, p_hi = min(prev_p, next_p), max(prev_p, next_p)
            pages_set.update(range(p_lo, p_hi + 1))

    target_pages = sorted(p for p in pages_set if 1 <= p <= total_p)
    return target_pages, missing_q_nums


def _crop_single_exam_pdf(
    prob_bytes: bytes,
    use_vision_fallback: bool,
    api_key: str,
    subject_hint: str = "",
    status_cb: Callable[[str], None] | None = None,
    progress_cb: Callable[[float], None] | None = None,
) -> tuple[list[CroppedQuestion], int, bool]:
    """단일 문제지 PDF를 2단 편집 기준으로 크롭하고, 스캔·누락 의심 페이지는 순차 Vision 보완한다."""
    crops, has_text = parse_exam_pdf(prob_bytes, dpi=220)
    total_p = get_pdf_page_count(prob_bytes)

    if use_vision_fallback and api_key and _ensure_token_quota(show_error=False):
        page_text_lens = get_page_text_lengths(prob_bytes)
        target_vision_pages, missing_q_nums = _compute_vision_recovery_pages(
            crops,
            total_p,
            page_text_lens,
            has_text,
            subject_hint=subject_hint,
        )

        if target_vision_pages:
            if status_cb:
                model_disp = MODEL_LABELS.get(EXAM_TAGGING_MODEL, EXAM_TAGGING_MODEL)
                if missing_q_nums:
                    preview = ", ".join(f"{n}번" for n in missing_q_nums[:8])
                    suffix = "..." if len(missing_q_nums) > 8 else ""
                    status_cb(
                        f"누락 의심 문항 {len(missing_q_nums)}개({preview}{suffix}) 복구를 위해 "
                        f"후보 {len(target_vision_pages)}개 페이지를 {model_disp} Vision으로 추가 탐지합니다..."
                    )
                else:
                    status_cb(
                        f"텍스트가 부족한 {len(target_vision_pages)}개 페이지를 "
                        f"{model_disp} Vision으로 1페이지씩 순차 탐지합니다..."
                    )
            existing_by_q: dict[int, CroppedQuestion] = {c.q_num: c for c in crops}
            with _usage_scope():
                for step_i, p_num in enumerate(target_vision_pages, start=1):
                    page_png = render_pdf_page_png(prob_bytes, p_num, dpi=180)
                    boxes = detect_scanned_question_boxes(api_key, EXAM_TAGGING_MODEL, page_png)
                    boxes.sort(key=lambda b: (0 if b.xmin < 500 else 1, b.ymin, b.xmin))
                    for b in boxes:
                        if b.q_num in existing_by_q:
                            continue
                        png_b, pw, ph = crop_normalized_box(
                            prob_bytes,
                            p_num,
                            b.ymin,
                            b.xmin,
                            b.ymax,
                            b.xmax,
                            dpi=250,
                        )
                        existing_by_q[b.q_num] = CroppedQuestion(
                            q_num=b.q_num,
                            page_num=p_num,
                            png_bytes=png_b,
                            width=pw,
                            height=ph,
                            extracted_text="",
                        )
                    if progress_cb:
                        progress_cb(step_i / max(1, len(target_vision_pages)))
            crops = sorted(existing_by_q.values(), key=lambda x: x.q_num)

    return crops, total_p, has_text


def _tag_and_save_single_exam(
    storage: StorageManager,
    api_key: str,
    meta: dict[str, Any],
    parsed_crops: list[CroppedQuestion],
    target_q_nums: list[int],
    is_retry: bool = False,
    push_status: Callable[[str], None] | None = None,
    append_log: Callable[[str], None] | None = None,
    on_progress: Callable[[float], None] | None = None,
    on_warning: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """단일 시험지 PDF와 해설지를 일괄 태깅·크롭 저장·DB 반영·Drive 동기화한다."""
    with _usage_scope(storage_obj=storage):
        return _tag_and_save_single_exam_inner(
            storage=storage,
            api_key=api_key,
            meta=meta,
            parsed_crops=parsed_crops,
            target_q_nums=target_q_nums,
            is_retry=is_retry,
            push_status=push_status,
            append_log=append_log,
            on_progress=on_progress,
            on_warning=on_warning,
        )


def _tag_and_save_single_exam_inner(
    storage: StorageManager,
    api_key: str,
    meta: dict[str, Any],
    parsed_crops: list[CroppedQuestion],
    target_q_nums: list[int],
    is_retry: bool = False,
    push_status: Callable[[str], None] | None = None,
    append_log: Callable[[str], None] | None = None,
    on_progress: Callable[[float], None] | None = None,
    on_warning: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    safe_exam = sanitize_id_part(meta["exam_name"])
    tagging_model = EXAM_TAGGING_MODEL
    tagging_model_label = MODEL_LABELS.get(tagging_model, tagging_model)
    solver_model = EXAM_NO_ANSWER_SOLVER_MODEL
    solver_model_label = MODEL_LABELS.get(solver_model, solver_model)

    def _status(msg: str) -> None:
        if push_status:
            push_status(msg)

    def _log(msg: str) -> None:
        if append_log:
            append_log(msg)

    def _prog(val: float) -> None:
        if on_progress:
            on_progress(val)

    if not meta.get("ans_bytes"):
        raise ValueError(
            "해설지 PDF가 첨부되지 않아 태깅을 진행할 수 없습니다. 문제지와 해설지 PDF를 함께 업로드해 주세요."
        )

    pending_pdf_files: list[str] = []
    pending_crop_files: list[str] = []
    if not is_retry:
        if meta.get("prob_bytes"):
            prob_fn = f"{safe_exam}__문제지.pdf"
            storage.save_pdf_file(
                prob_fn,
                meta["prob_bytes"],
                subject_category=meta["subject_category"],
                exam_name=meta["exam_name"],
                sync_drive=False,
            )
            pending_pdf_files.append(prob_fn)
        if meta.get("ans_bytes"):
            ans_fn = f"{safe_exam}__해설지.pdf"
            storage.save_pdf_file(
                ans_fn,
                meta["ans_bytes"],
                subject_category=meta["subject_category"],
                exam_name=meta["exam_name"],
                sync_drive=False,
            )
            pending_pdf_files.append(ans_fn)

    answer_snippets: dict[int, str] = {}
    answer_pdf_part = None
    ans_prep_err = ""
    if meta.get("ans_bytes"):
        _status("📄 해설지 PDF 분석 중 (텍스트 스니펫 우선 추출)...")
        try:
            answer_snippets, ans_has_text = extract_answer_snippets_from_pdf(meta["ans_bytes"])
            missing_ans_q = [q for q in target_q_nums if q not in answer_snippets]
            if ans_has_text and answer_snippets:
                _log(f"📄 해설지 텍스트 스니펫 추출 완료 ({len(answer_snippets)}개 문항 매칭 · 경량 전송)")
            if not ans_has_text or not answer_snippets or missing_ans_q:
                tmp_client = build_gemini_client(api_key, timeout_seconds=30)
                answer_pdf_part = prepare_pdf_part(
                    tmp_client,
                    meta["ans_bytes"],
                    display_name=f"{safe_exam}_answer.pdf",
                )
                if ans_has_text and answer_snippets and missing_ans_q:
                    _log(
                        f"📄 해설지 스니펫 미탐지 문항({len(missing_ans_q)}개) 대비 원본 해설지 PDF 파트도 함께 준비"
                    )
                else:
                    _log("📄 스캔/통합 해설지 PDF 파트 준비 완료 (30초 타임아웃)")
        except Exception as exc:
            ans_prep_err = str(exc)
            warn_msg = f"해설지 첨부 준비 오류: {exc}"
            if on_warning:
                on_warning(warn_msg)
            _log(f"❌ {warn_msg}")

    if not answer_snippets and answer_pdf_part is None:
        detail = f" ({ans_prep_err})" if ans_prep_err else ""
        raise ValueError(
            f"해설지 PDF를 읽지 못해 태깅을 중단했습니다. 해설지 PDF 파일을 확인해 주세요.{detail}"
        )

    target_set = set(target_q_nums)
    target_crops = sorted(
        [c for c in parsed_crops if c.q_num in target_set],
        key=lambda c: c.q_num,
    )
    existing_records = storage.load_db()
    rec_map = {r.id: r for r in existing_records}

    _prog(0.05)
    newly_saved: list[QuestionRecord] = []
    failed_q_nums: list[int] = []
    errors: list[str] = []

    batch_t0 = time.perf_counter()
    extracted_by_q: dict[int, Any] = {}
    is_english_exam = meta["subject_category"] == "영어"

    if is_english_exam:
        if meta.get("prob_bytes"):
            _status(
                f"⚡ 영어 문제지 PDF({len(target_crops)}문항 · 듣기/독해/장문세트) 일괄 분석 중 (모델: {tagging_model_label})..."
            )
            try:
                batch_map = analyze_english_exam_pdf_batch_with_gemini(
                    api_key=api_key,
                    model_name=tagging_model,
                    prob_pdf_bytes=meta["prob_bytes"],
                    target_q_nums=[c.q_num for c in target_crops],
                    exam_name=meta["exam_name"],
                    answer_snippets=answer_snippets,
                    answer_pdf_part=answer_pdf_part,
                    status_callback=_status,
                )
                extracted_by_q.update(batch_map)
                _log(
                    f"⚡ 영어 PDF 일괄 추출 완료: {len(batch_map)}/{len(target_crops)}문항 "
                    f"({time.perf_counter() - batch_t0:.1f}초 · {tagging_model_label})"
                )
            except Exception as exc:
                _log(f"⚠️ 영어 PDF 일괄 추출 실패 → 개별 문항 보완 모드로 전환: {exc}")
        _prog(0.65)

        missing_crops = [c for c in target_crops if c.q_num not in extracted_by_q]
        if missing_crops:
            _status(f"🔁 누락된 영어 {len(missing_crops)}개 문항 보완 추출 중 ({tagging_model_label})...")

            def _analyze_eng_worker(cq_item: CroppedQuestion) -> tuple[CroppedQuestion, Any, float]:
                t_start = time.perf_counter()
                q_ans_hint = answer_snippets.get(cq_item.q_num, "")
                extracted_out = analyze_english_question_with_gemini(
                    api_key=api_key,
                    model_name=tagging_model,
                    crop_png_bytes=cq_item.png_bytes,
                    q_num=cq_item.q_num,
                    exam_name=meta["exam_name"],
                    answer_pdf_part=answer_pdf_part,
                    answer_text_hint=q_ans_hint,
                    question_text_hint=cq_item.extracted_text or "",
                    shared_passage_hint=(
                        (getattr(cq_item, "shared_passage_text", "") or "")
                        if normalize_english_group_id(cq_item.q_num)
                        else ""
                    ),
                )
                return cq_item, extracted_out, time.perf_counter() - t_start

            done_cnt = 0
            total_missing = len(missing_crops)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_cq = {
                    submit_with_usage_context(executor, _analyze_eng_worker, cq): cq
                    for cq in missing_crops
                }
                for fut in concurrent.futures.as_completed(future_to_cq):
                    cq = future_to_cq[fut]
                    done_cnt += 1
                    try:
                        _, extracted, elapsed_s = fut.result()
                        extracted_by_q[cq.q_num] = extracted
                        _log(f"✅ 영어 {cq.q_num}번 보완 추출 완료 (정답: {extracted.answer} · {elapsed_s:.1f}초)")
                    except Exception as exc:
                        err_msg = f"{cq.q_num}번 오류 (건너뜀): {exc}"
                        failed_q_nums.append(cq.q_num)
                        errors.append(err_msg)
                        _log(f"❌ {err_msg}")
                    _prog(0.65 + 0.20 * (done_cnt / max(1, total_missing)))

        # 41~42, 43~45 세트 문항 간 공통 지문 최종 동기화
        eng_group_passages: dict[str, str] = {}
        for cq in target_crops:
            ext = extracted_by_q.get(cq.q_num)
            gid = normalize_english_group_id(cq.q_num)
            if not gid:
                continue
            cand_p = (
                (getattr(ext, "passage_text", "") if ext else "")
                or getattr(cq, "shared_passage_text", "")
                or ""
            ).strip()
            if len(cand_p) > len(eng_group_passages.get(gid, "")):
                eng_group_passages[gid] = cand_p
    else:
        eng_group_passages = {}
        concept_books = storage.load_concept_books()
        grounding_ctx = build_tagging_grounding_context(
            subject_category=meta["subject_category"],
            exam_name=meta["exam_name"],
            concept_books=concept_books,
        )

        has_answer_sheet = bool(answer_snippets or answer_pdf_part is not None)
        if has_answer_sheet:
            if meta.get("prob_bytes"):
                _status(f"⚡ 전체 문제지 PDF({len(target_crops)}문항) 한 번에 태깅 중 (모델: {tagging_model_label})...")
                try:
                    batch_map = analyze_exam_pdf_batch_with_gemini(
                        api_key=api_key,
                        model_name=tagging_model,
                        prob_pdf_bytes=meta["prob_bytes"],
                        target_q_nums=[c.q_num for c in target_crops],
                        subject_category=meta["subject_category"],
                        exam_name=meta["exam_name"],
                        answer_snippets=answer_snippets,
                        answer_pdf_part=answer_pdf_part,
                        status_callback=_status,
                        concept_books=concept_books,
                        grounding_context=grounding_ctx,
                    )
                    extracted_by_q.update(batch_map)
                    _log(
                        f"⚡ 전체 PDF 일괄 태깅 완료: {len(batch_map)}/{len(target_crops)}문항 "
                        f"({time.perf_counter() - batch_t0:.1f}초 · {tagging_model_label})"
                    )
                except Exception as exc:
                    _log(f"⚠️ 전체 PDF 일괄 태깅 실패 → 개별 보완 모드로 전환: {exc}")
            _prog(0.65)
        else:
            crops_no_img = [c for c in target_crops if not c.png_bytes]
            if crops_no_img and meta.get("prob_bytes"):
                try:
                    batch_map = analyze_exam_pdf_batch_with_gemini(
                        api_key=api_key,
                        model_name=solver_model,
                        prob_pdf_bytes=meta["prob_bytes"],
                        target_q_nums=[c.q_num for c in crops_no_img],
                        subject_category=meta["subject_category"],
                        exam_name=meta["exam_name"],
                        answer_snippets=None,
                        answer_pdf_part=None,
                        status_callback=_status,
                        concept_books=concept_books,
                        grounding_context=grounding_ctx,
                    )
                    extracted_by_q.update(batch_map)
                except Exception as exc:
                    _log(f"⚠️ 이미지 없는 문항 배치 풀이 경고: {exc}")
            _log(
                f"🧠 해설지 미첨부 감지 → 정답 정확도를 위해 {solver_model_label} · HIGH 추론 · 문항별 고해상도 직접 풀이·검산 모드 실행 ({len(target_crops)}문항)"
            )
            _prog(0.10)

        missing_crops = [c for c in target_crops if c.q_num not in extracted_by_q]
        if missing_crops:
            if has_answer_sheet:
                _status(f"🔁 누락된 {len(missing_crops)}개 문항 보완 태깅 중 ({tagging_model_label})...")
            else:
                _status(f"🧠 {solver_model_label} · HIGH 추론 · 문항별 고해상도 직접 풀이·검산 중 ({len(missing_crops)}문항)...")

            def _analyze_worker(cq_item: CroppedQuestion) -> tuple[CroppedQuestion, Any, float, bool]:
                t_start = time.perf_counter()
                q_ans_hint = answer_snippets.get(cq_item.q_num, "")
                q_has_answer = bool((q_ans_hint and q_ans_hint.strip()) or answer_pdf_part is not None)
                worker_model = tagging_model if q_has_answer else solver_model
                extracted_out = analyze_question_with_gemini(
                    api_key=api_key,
                    model_name=worker_model,
                    crop_png_bytes=cq_item.png_bytes,
                    q_num=cq_item.q_num,
                    subject_category=meta["subject_category"],
                    exam_name=meta["exam_name"],
                    answer_pdf_part=answer_pdf_part,
                    answer_text_hint=q_ans_hint,
                    status_callback=None,
                    concept_books=concept_books,
                    grounding_context=grounding_ctx,
                    question_text_hint=cq_item.extracted_text or "",
                    solve_without_answer=not q_has_answer,
                )
                return cq_item, extracted_out, time.perf_counter() - t_start, q_has_answer

            done_cnt = 0
            total_missing = len(missing_crops)
            base_prog = 0.65 if has_answer_sheet else 0.10
            span_prog = 0.85 - base_prog
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_cq = {
                    submit_with_usage_context(executor, _analyze_worker, cq): cq
                    for cq in missing_crops
                }
                for fut in concurrent.futures.as_completed(future_to_cq):
                    cq = future_to_cq[fut]
                    done_cnt += 1
                    try:
                        _, extracted, elapsed_s, q_has_answer = fut.result()
                        extracted_by_q[cq.q_num] = extracted
                        mode_txt = "보완 태깅" if q_has_answer else "직접 풀이·태깅"
                        _log(f"✅ {cq.q_num}번 {mode_txt} 완료 (정답: {extracted.answer} · {elapsed_s:.1f}초)")
                    except Exception as exc:
                        err_msg = f"{cq.q_num}번 오류 (건너뜀): {exc}"
                        failed_q_nums.append(cq.q_num)
                        errors.append(err_msg)
                        _log(f"❌ {err_msg}")
                    _prog(base_prog + span_prog * (done_cnt / max(1, total_missing)))
                    if not has_answer_sheet:
                        _status(f"🧠 {solver_model_label} 직접 풀이·검산 진행 중 ({done_cnt}/{total_missing}문항)...")

    _prog(0.85)

    _status("💾 전체 문항 크롭 이미지 및 DB 한 번에 저장 중...")
    now_iso = datetime.now().isoformat(timespec="seconds")
    existing_by_qnum = {
        r.q_num: r for r in existing_records if r.exam_name == meta["exam_name"]
    }
    skip_crop_save = bool(meta.get("skip_crop_save", False))
    for cq in target_crops:
        extracted = extracted_by_q.get(cq.q_num)
        if extracted is None:
            continue
        old_rec = existing_by_qnum.get(cq.q_num)
        q_id = old_rec.id if old_rec is not None else f"{safe_exam}__Q{cq.q_num:02d}"
        img_filename = (
            old_rec.image_filename
            if (old_rec is not None and old_rec.image_filename)
            else f"{q_id}.png"
        )
        if cq.png_bytes and (old_rec is None or not skip_crop_save):
            storage.save_crop_png(
                img_filename,
                cq.png_bytes,
                subject_category=meta["subject_category"],
                exam_name=meta["exam_name"],
                sync_drive=False,
            )
            pending_crop_files.append(img_filename)

        eff_source_pdf = (
            meta.get("source_pdf")
            or (old_rec.source_pdf if (old_rec is not None and old_rec.source_pdf) else f"{safe_exam}__문제지.pdf")
        )
        eff_answer_pdf = (
            meta.get("answer_pdf")
            or (
                old_rec.answer_pdf
                if (old_rec is not None and old_rec.answer_pdf)
                else (f"{safe_exam}__해설지.pdf" if meta.get("ans_bytes") else "")
            )
        )

        if is_english_exam:
            eng_sec = getattr(extracted, "english_section", "") or ("듣기" if cq.q_num <= 17 else "독해")
            eng_type = (getattr(extracted, "english_question_type", "") or "").strip() or eng_sec
            p_group = normalize_english_group_id(cq.q_num)
            raw_p_text = (getattr(extracted, "passage_text", "") or "").strip()
            if p_group and len(raw_p_text) < len(eng_group_passages.get(p_group, "")) * 0.7:
                raw_p_text = eng_group_passages[p_group]
            p_text = "" if eng_sec == "듣기" else raw_p_text
            p_topic = (getattr(extracted, "passage_topic", "") or "").strip()
            k_vocab = [v.strip() for v in (getattr(extracted, "key_vocabulary", []) or []) if v and v.strip()]

            record = QuestionRecord(
                id=q_id,
                exam_name=meta["exam_name"],
                q_num=cq.q_num,
                subject_category="영어",
                curriculum_subject="영어",
                unit_name=eng_type,
                min_grade=1,
                core_principles=[p_topic] if p_topic else [eng_type],
                required_terms=k_vocab,
                advanced_terms=[],
                basic_terms=k_vocab,
                standard_code="",
                standard_codes=[],
                grade_evidence="",
                needs_review=False,
                review_notes=[],
                is_single_concept=True,
                answer=extracted.answer,
                official_explanation=extracted.official_explanation,
                question_text=re.sub(r"\s+", " ", (cq.extracted_text or "")).strip(),
                english_section=eng_sec,
                english_question_type=eng_type,
                passage_text=p_text,
                passage_group_id=p_group,
                passage_topic=p_topic,
                key_vocabulary=k_vocab,
                image_filename=img_filename,
                source_pdf=eff_source_pdf,
                answer_pdf=eff_answer_pdf,
                created_at=now_iso,
            )
        else:
            record = QuestionRecord(
                id=q_id,
                exam_name=meta["exam_name"],
                q_num=cq.q_num,
                subject_category=meta["subject_category"],
                curriculum_subject=extracted.curriculum_subject,
                unit_name=extracted.unit_name,
                min_grade=extracted.min_grade,
                core_principles=extracted.core_principles,
                required_terms=extracted.required_terms,
                advanced_terms=list(getattr(extracted, "advanced_terms", []) or []),
                basic_terms=list(getattr(extracted, "basic_terms", []) or []),
                standard_code=(getattr(extracted, "standard_code", "") or "").strip(),
                standard_codes=list(getattr(extracted, "standard_codes", []) or []),
                grade_evidence=(getattr(extracted, "grade_evidence", "") or "").strip(),
                needs_review=bool(getattr(extracted, "needs_review", False)),
                review_notes=list(getattr(extracted, "review_notes", []) or []),
                is_single_concept=extracted.is_single_concept,
                answer=extracted.answer,
                official_explanation=extracted.official_explanation,
                question_text=re.sub(r"\s+", " ", (cq.extracted_text or "")).strip(),
                image_filename=img_filename,
                source_pdf=eff_source_pdf,
                answer_pdf=eff_answer_pdf,
                created_at=now_iso,
            )
        rec_map[q_id] = record
        newly_saved.append(record)

    newly_saved.sort(key=lambda r: r.q_num)
    sorted_all_records = sorted(rec_map.values(), key=lambda r: (r.exam_name, r.q_num))
    pending_crop_files.sort()

    storage.save_db(
        sorted_all_records,
        sync_organized_drive=False,
        sync_db_drive=False,
        sync_local_organized=False,
    )

    _status(
        "☁️ 전체 문항 로컬 저장 및 Google Drive 일괄 동기화 중... "
        "('Google Drive 동기화 완료'가 뜰 때까지 창을 유지해 주세요)"
    )
    sync_err = storage.sync_exam_batch(
        subject_category=meta["subject_category"],
        exam_name=meta["exam_name"],
        records=sorted_all_records,
        crop_filenames=pending_crop_files,
        pdf_filenames=pending_pdf_files,
        include_organized_exports=True,
    )
    if newly_saved:
        _build_cached_exam_exports.clear()
    _prog(1.0)
    total_batch_s = time.perf_counter() - batch_t0
    _log(f"🏁 전체 PDF 태깅 및 일괄 저장·동기화 완료 ({len(newly_saved)}문항 · 총 {total_batch_s:.1f}초)")
    return {
        "newly_saved": newly_saved,
        "failed_q_nums": sorted(set(failed_q_nums)),
        "errors": errors,
        "sync_err": sync_err or "",
        "elapsed_s": total_batch_s,
    }


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


def _render_ncic_curriculum_manager(key_prefix: str = "ncic") -> None:
    """2022 개정 교육과정(NCIC 고시 별책 7·8·9 PDF) 기준표 현황 조회 및 성취기준 누적 등록 UI."""
    with st.expander("🧭 2022 개정 교육과정 공식 기준표(NCIC 고시 PDF) 관리 · 성취기준 보강", expanded=False):
        stats = get_catalog_summary_stats()
        st.caption(
            "기본적으로 2022 교육부 고시(제2022-33호) 공식 과목 편제표가 내장되어 있으며, "
            "국가교육과정정보센터(ncic.re.kr)의 [별책 7(사회) / 별책 8(수학) / 별책 9(과학)] PDF를 올리면 "
            "공식 성취기준 코드가 기준표에 누적되어 태깅 할루시네이션 검증이 한층 엄격해집니다."
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("수학 기준표", f"{stats['수학']['subjects']}과목", f"성취기준 {stats['수학']['standards']}개")
        m2.metric("사회 기준표", f"{stats['사회']['subjects']}과목", f"성취기준 {stats['사회']['standards']}개")
        m3.metric("과학 기준표", f"{stats['과학']['subjects']}과목", f"성취기준 {stats['과학']['standards']}개")

        nc1, nc2 = st.columns([2, 1])
        with nc1:
            ncic_pdf = st.file_uploader(
                "NCIC 2022 교육과정 고시 PDF 업로드 (별책 7/8/9)",
                type=["pdf"],
                key=f"{key_prefix}_ncic_pdf",
            )
        with nc2:
            ncic_cat_hint = st.selectbox(
                "기본 교과 대분류",
                options=["과학", "수학", "사회"],
                key=f"{key_prefix}_ncic_cat",
            )
        if ncic_pdf is not None and st.button("📥 NCIC 고시 PDF 성취기준 추출·반영", key=f"{key_prefix}_ncic_apply"):
            with st.spinner("NCIC 고시 원문 PDF에서 2022 성취기준 코드 추출 중..."):
                extracted = parse_ncic_curriculum_pdf(
                    ncic_pdf.getvalue(),
                    subject_category_hint=ncic_cat_hint,
                )
            if not extracted:
                st.warning("PDF에서 `[10...]` / `[12...]` 형식의 2022 성취기준 코드를 찾지 못했습니다. 텍스트 레이어가 있는 공식 NCIC 원문 PDF인지 확인해 주세요.")
            else:
                res = merge_ncic_standards_to_catalog(extracted)
                st.success(
                    f"✅ 성취기준 {len(extracted)}개 추출 완료! (신규 {res['added']}개 추가 · 기존 {res['updated']}개 갱신)"
                )
                st.rerun()


creds = resolve_credentials()
auth_payload = creds.oauth_credentials_info or creds.service_account_info
sa_fingerprint = (
    hashlib.sha256(json.dumps(auth_payload, sort_keys=True).encode()).hexdigest()[:12]
    if auth_payload
    else "none"
)
storage = _get_cached_storage(
    local_root="./exam_db",
    drive_folder_id=creds.drive_folder_id,
    auth_sig=sa_fingerprint,
    _service_account_info=creds.service_account_info,
    _oauth_info=creds.oauth_credentials_info,
)
sync_fingerprint = f"{creds.drive_folder_id}:{storage.drive_auth_mode}:{sa_fingerprint}"
if storage.drive_connected and st.session_state["drive_sync_key"] != sync_fingerprint:
    sync_out = _run_startup_drive_sync(storage, sync_fingerprint)
    st.session_state["drive_sync_result"] = sync_out
    if not sync_out.get("error"):
        st.session_state["drive_sync_key"] = sync_fingerprint

if not st.session_state["logged_in_username"]:
    raw_session_tok = st.query_params.get("session")
    if isinstance(raw_session_tok, list):
        raw_session_tok = raw_session_tok[-1] if raw_session_tok else ""
    if raw_session_tok:
        restored_username = verify_session_token(str(raw_session_tok), storage=storage)
        if restored_username:
            st.session_state["logged_in_username"] = restored_username
        else:
            st.query_params.pop("session", None)

current_user = (
    get_user_by_username(storage, st.session_state["logged_in_username"])
    if st.session_state["logged_in_username"]
    else None
)
if st.session_state["logged_in_username"] and current_user is None:
    st.session_state["logged_in_username"] = ""
    st.query_params.pop("session", None)
elif current_user is not None and not st.query_params.get("session"):
    st.query_params["session"] = create_session_token(current_user.username, storage=storage)

is_admin = bool(current_user is not None and current_user.tier == "ADMIN")
selected_model = DEFAULT_MODEL
active_gemini_key = ""
active_key_label = ""
upload_gemini_key = ""
upload_key_label = ""

# ---------------------------------------------------------------------------
# 사이드바: 비로그인 시 [로그인 / 회원가입] 탭만 표시, 로그인 시 프로필·개인키·설정 표시
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 👤 계정 & 멤버십")
    if current_user is None:
        auth_tab_login, auth_tab_signup = st.tabs(["🔑 로그인", "📝 회원가입"])
        with auth_tab_login:
            with st.form("sidebar_login_form"):
                login_id = st.text_input("아이디 (학번 또는 영문 ID)", placeholder="예: 2026101")
                login_pw = st.text_input("비밀번호", type="password")
                login_submit = st.form_submit_button("로그인", type="primary", width="stretch")
            if login_submit:
                ok, msg, logged_user = authenticate_user(storage, login_id, login_pw)
                if ok and logged_user is not None:
                    st.session_state["logged_in_username"] = logged_user.username
                    st.query_params["session"] = create_session_token(
                        logged_user.username, storage=storage
                    )
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

        with auth_tab_signup:
            with st.form("sidebar_signup_form"):
                reg_id = st.text_input("아이디 (영문·숫자 3~30자)", placeholder="예: student01")
                reg_nick = st.text_input("이름", placeholder="예: 김민준")
                reg_pw = st.text_input("비밀번호 (4자 이상)", type="password")
                reg_pw2 = st.text_input("비밀번호 확인", type="password")
                reg_submit = st.form_submit_button("회원가입", type="primary", width="stretch")
            if reg_submit:
                if reg_pw != reg_pw2:
                    st.error("비밀번호 확인이 일치하지 않습니다.")
                else:
                    ok, msg, new_user = register_user(storage, reg_id, reg_nick, reg_pw)
                    if ok and new_user is not None:
                        st.session_state["logged_in_username"] = new_user.username
                        st.query_params["session"] = create_session_token(
                            new_user.username, storage=storage
                        )
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
    else:
        tier_code = current_user.tier
        tier_kor = TIER_LABELS.get(tier_code, tier_code)
        active_gemini_key, active_key_label = resolve_active_gemini_api_key(
            current_user, creds.gemini_api_key
        )
        upload_gemini_key, upload_key_label = resolve_active_gemini_api_key(
            current_user, creds.gemini_api_key, force_server_key=True
        )
        monthly_limit = get_monthly_token_limit(current_user)
        monthly_used = get_used_monthly_tokens(current_user)
        monthly_rem = get_remaining_monthly_tokens(current_user)

        if tier_code == "FREE":
            status_line = (
                f"<div style='font-size:0.83rem; color:#334155; margin-top:0.45rem;'>"
                f"모델: <b>Gemini 3.5 Flash-Lite</b> (1회 최대 {FREE_MAX_BATCH_QUESTIONS}문항)<br/>"
                f"월간 토큰 잔여: <b>{monthly_rem:,} / {format_token_count(monthly_limit or 0)}</b> "
                f"<span style='color:#64748B;'>(사용 {monthly_used:,})</span>"
                f"</div>"
            )
        elif tier_code == "CUSTOM":
            masked = _html.escape(mask_api_key(current_user.custom_api_key))
            status_line = (
                f"<div style='font-size:0.83rem; color:#166534; margin-top:0.45rem;'>"
                f"등록된 개인 키: <code>{masked}</code>"
                f"<br/><b>Gemini 3.5 Flash-Lite · 개인 키 무제한</b>"
                f"<br/><span style='color:#64748B;'>이번 달 사용: {monthly_used:,} 토큰</span></div>"
            )
        elif tier_code == "PRO":
            status_line = (
                f"<div style='font-size:0.83rem; color:#1D4ED8; margin-top:0.45rem;'>"
                f"✨ 모델: <b>Gemini 3.5 Flash-Lite</b> · 배치 무제한<br/>"
                f"월간 토큰 잔여: <b>{monthly_rem:,} / {format_token_count(monthly_limit or 0)}</b> "
                f"<span style='color:#64748B;'>(사용 {monthly_used:,})</span>"
                f"</div>"
            )
        elif tier_code == "ULTRA":
            status_line = (
                "<div style='font-size:0.83rem; color:#C2410C; font-weight:600; margin-top:0.45rem;'>"
                "🔥 모델: <b>Gemini 3.5 Flash-Lite</b> · PRO 대비 4배<br/>"
                f"월간 토큰 잔여: <b>{monthly_rem:,} / {format_token_count(monthly_limit or 0)}</b> "
                f"<span style='color:#9A3412; font-weight:400;'>(사용 {monthly_used:,})</span></div>"
            )
        else:
            status_line = (
                "<div style='font-size:0.83rem; color:#1D4ED8; font-weight:600; margin-top:0.45rem;'>"
                "🛡️ 최고 관리자 · 전체 기능 및 토큰·비용 관리<br/>"
                f"<span style='color:#64748B; font-weight:400;'>이번 달 사용: {monthly_used:,} 토큰</span></div>"
            )

        st.markdown(
            f"""
            <div class="account-card">
              <div style="display:flex; justify-content:space-between; align-items:center; gap:6px;">
                <div>
                  <div style="font-size:1rem; font-weight:800; color:#0F172A;">{_html.escape(current_user.nickname)}</div>
                  <div style="font-size:0.78rem; color:#64748B;">@{_html.escape(current_user.username)}</div>
                </div>
                <span class="tier-badge tier-{tier_code}">{tier_kor}</span>
              </div>
              {status_line}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if monthly_limit is not None and monthly_rem is not None:
            rem_ratio = max(0.0, min(1.0, monthly_rem / max(1, monthly_limit)))
            st.progress(rem_ratio)

        # CUSTOM 개인 API 키 관리 UI
        if tier_code == "FREE":
            show_custom_form = st.toggle(
                "🔑 개인 API 키 등록하여 무제한 쓰기",
                key="sidebar_toggle_custom_key",
            )
            if show_custom_form:
                with st.form("sidebar_custom_key_form", clear_on_submit=True):
                    user_api_input = st.text_input(
                        "내 Gemini API 키 (AIza...)",
                        type="password",
                        placeholder="AIzaSy...",
                    )
                    save_key_btn = st.form_submit_button(
                        "💾 개인 키 저장 & 무제한 모드 전환",
                        type="primary",
                        width="stretch",
                    )
                if save_key_btn:
                    ok_k, msg_k, updated_u = save_user_custom_api_key(
                        storage, current_user.username, user_api_input
                    )
                    if ok_k:
                        st.success(msg_k)
                        st.rerun()
                    else:
                        st.error(msg_k)
        elif tier_code == "CUSTOM" or bool(current_user.custom_api_key):
            st.caption(f"저장된 개인 키: `{mask_api_key(current_user.custom_api_key)}`")
            if st.button(
                "🗑️ 개인 키 삭제 / 무료 회원으로 복귀",
                key="clear_custom_key_btn",
                width="stretch",
            ):
                ok_c, msg_c, _ = clear_user_custom_api_key(storage, current_user.username)
                if ok_c:
                    st.info(msg_c)
                    st.rerun()
                else:
                    st.error(msg_c)

        if st.button("🚪 로그아웃", key="logout_btn", width="stretch"):
            st.session_state["logged_in_username"] = ""
            st.session_state["matched_items"] = []
            st.session_state["generated_exam_pdf"] = None
            st.query_params.pop("session", None)
            st.rerun()

        st.markdown("---")
        # 전 등급 · 전 기능 Gemini 3.5 Flash-Lite 단일 모델 통일
        selected_model = FLASH_LITE_MODEL

        # 저장소 & DB 요약
        all_records = storage.load_db()
        all_concept_books = storage.load_concept_books()
        st.markdown("### 📊 기출 · 개념서 DB & 저장소")
        c1, c2, c3 = st.columns(3)
        c1.metric("기출 문항", f"{len(all_records)}개")
        unique_exams = sorted({r.exam_name for r in all_records})
        c2.metric("등록 회차", f"{len(unique_exams)}회")
        c3.metric("공용 개념서", f"{len(all_concept_books)}권")
        if storage.drive_connected:
            mode_tag = "OAuth" if storage.drive_auth_mode == "oauth" else "서비스 계정"
            st.caption(f"☁️ Google Drive 연동됨 ({storage.drive_folder_name} · {mode_tag})")
            sync_res = st.session_state.get("drive_sync_result") or {}
            if sync_res.get("error"):
                st.warning(f"Drive 동기화 오류: {sync_res['error'][:160]}")
            if st.button("🔄 Drive 최신화", key="sidebar_force_sync_btn", width="stretch"):
                _run_startup_drive_sync.clear()
                st.session_state["drive_sync_key"] = ""
                st.rerun()
        else:
            st.caption("💾 로컬 영구 저장소 (`./exam_db`, `./concept_db`)")
            if storage.drive_error:
                st.caption(f"⚠️ Drive 연결 오류: {storage.drive_error[:140]}")

# ---------------------------------------------------------------------------
# 메인 헤더 & 2개 탭
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-hero">
      <h1>📘 2022 개정 교육과정 맞춤형 기출 추출 & 상위 문제 개조 플랫폼</h1>
      <p>개념서·교과서 PDF에 실제로 나온 원리와 용어만 남기고, 원리는 같지만 용어가 어려운 상위 기출은 내 개념서 언어로 자동 개조합니다.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def _render_english_rewritten_card_html(rw: RewrittenQuestion) -> str:
    """영어 수능형 변형 문항의 밑줄(<u>)·빈칸·[주어진 문장]·(A)-(B)-(C) 순서·[요약문] 박스 HTML을 조립한다."""
    blocks = split_english_stem_blocks(rw.rewritten_stem)
    body_parts: list[str] = []
    for kind, content in blocks:
        if kind == "instruction":
            body_parts.append(f"<div class='eng-instruction'>{render_english_inline_html(str(content))}</div>")
        elif kind == "given_box":
            g_title = str(content.get("title") or "주어진 글") if isinstance(content, dict) else "주어진 글"
            g_body = str(content.get("body") or "") if isinstance(content, dict) else str(content)
            body_parts.append(
                f"<div class='eng-given-box'><div class='eng-given-title'>[{_html.escape(g_title)}]</div>"
                f"<div>{render_english_inline_html(g_body)}</div></div>"
            )
        elif kind == "order_chunks":
            order_rows = "".join(
                f"<div class='eng-order-item'><span class='eng-order-mk'>{_html.escape(mk)}</span>"
                f"<span>{render_english_inline_html(txt)}</span></div>"
                for mk, txt in content
            )
            body_parts.append(f"<div class='eng-order-wrap'>{order_rows}</div>")
        elif kind == "summary_box":
            body_parts.append(
                f"<div class='eng-summary-arrow'>↓</div>"
                f"<div class='eng-summary-box'>{render_english_inline_html(str(content))}</div>"
            )
        elif kind == "footnote":
            body_parts.append(f"<div class='eng-footnote'>{render_english_inline_html(str(content))}</div>")
        else:
            body_parts.append(f"<div class='eng-passage'>{render_english_inline_html(str(content))}</div>")

    choices_html = ""
    if rw.rewritten_choices:
        choice_divs = []
        for ci, c_text in enumerate(rw.rewritten_choices):
            sym = CIRCLED_MAP.get(str(ci + 1), f"{ci + 1})")
            cleaned = (c_text or "").strip()
            if cleaned == sym or not cleaned:
                choice_divs.append(f"<div class='rewritten-choice'><b>{sym}</b></div>")
            else:
                choice_divs.append(
                    f"<div class='rewritten-choice'><b>{sym}</b> {render_english_inline_html(cleaned)}</div>"
                )
        choices_html = f"<div class='rewritten-choices-wrap'>{''.join(choice_divs)}</div>"

    return f"<div class='rewritten-card'>{''.join(body_parts)}{choices_html}</div>"


def _render_rewritten_card_html(rw: RewrittenQuestion) -> str:
    """개조 문항의 <자료>·<보기> 분리 박스, 화학식 첨자(영어는 수능형 밑줄/박스), 선지 HTML을 모던 학습지 스타일로 조립한다."""
    if getattr(rw, "subject_domain", "") == "영어":
        return _render_english_rewritten_card_html(rw)
    blocks = split_stem_blocks(rw.rewritten_stem)
    body_parts: list[str] = []
    for kind, content in blocks:
        if kind == "data":
            if isinstance(content, dict):
                d_title = str(content.get("title") or "< 자 료 >")
                d_body = str(content.get("body") or "")
            else:
                d_title = "< 자 료 >"
                d_body = str(content)
            line_divs = "".join(
                "<div style='height:0.35rem;'></div>"
                if not ln.strip()
                else f"<div class='data-line'>{_html.escape(ln.strip())}</div>"
                for ln in d_body.split("\n")
            )
            body_parts.append(
                f"<div class='data-box'><div class='data-box-title'>{_html.escape(d_title)}</div><div>{line_divs}</div></div>"
            )
        elif kind == "bogi":
            bogi_rows = "".join(
                f"<div class='bogi-item'><span class='bogi-marker'>{_html.escape(mk)}</span>"
                f"<span>{_html.escape(txt).replace(chr(10), '<br/>')}</span></div>"
                for mk, txt in content
            )
            body_parts.append(
                f"<div class='bogi-box'><div class='bogi-title'>〈 보 기 〉</div>{bogi_rows}</div>"
            )
        else:
            stem_divs = "".join(
                "<div style='height:0.35rem;'></div>"
                if not ln.strip()
                else f"<div class='stem-line'>{_html.escape(ln.strip())}</div>"
                for ln in str(content).split("\n")
            )
            body_parts.append(f"<div class='rewritten-stem'>{stem_divs}</div>")

    choices_html = ""
    if rw.rewritten_choices:
        choice_divs = []
        for ci, c_text in enumerate(rw.rewritten_choices):
            sym = CIRCLED_MAP.get(str(ci + 1), f"{ci + 1})")
            formatted_choice = format_chemical_scripts(c_text.strip())
            c_html = _html.escape(formatted_choice).replace("\n", "<br/>")
            choice_divs.append(f"<div class='rewritten-choice'><b>{sym}</b> {c_html}</div>")
        choices_html = f"<div class='rewritten-choices-wrap'>{''.join(choice_divs)}</div>"

    return f"<div class='rewritten-card'>{''.join(body_parts)}{choices_html}</div>"


def _gate_regen_model(model_name: str) -> str:
    """재변환 직전 최신 사용자 한도를 확인해 상위 모델 한도 소진 시 3.5 Flash-Lite로 우회시킨다."""
    if not is_premium_model(model_name):
        return model_name
    fresh_user = get_user_by_username(storage, current_user.username) or current_user
    gated_model, gate_notice = resolve_model_for_user(fresh_user, model_name)
    if gate_notice:
        st.warning(gate_notice)
    return gated_model


def _charge_regen_usage(model_name: str) -> None:
    """재변환에 상위 모델이 실제 사용되었다면 1문항 분량을 주간 한도/해금 크레딧에서 차감한다."""
    if is_premium_model(model_name):
        consume_premium_model_usage(storage, current_user.username, model_name, count=1)


@st.fragment
def _render_mode_b_solve_fragment(
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
            ai_col1, ai_col2, _ = st.columns([1.15, 1.25, 1.4])
            with ai_col1:
                if st.button("✨ AI 자동 해설", key=f"ai_btn_{q.id}", width="stretch"):
                    if not active_gemini_key:
                        st.error(f"사용 가능한 Gemini API 키({active_key_label})가 설정되지 않았습니다.")
                    elif not _ensure_token_quota():
                        pass
                    elif is_eng_item:
                        with st.spinner("영어 구문·논리·오답 분석 해설 생성 중..."):
                            try:
                                with _usage_scope():
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
                                st.error(f"영어 AI 해설 생성 실패: {exc}")
                    elif whitelist is None:
                        st.error("개념서 화이트리스트가 필요합니다.")
                    else:
                        with st.spinner("개념서 범위 내 용어만 사용하여 맞춤 해설 작성 중..."):
                            try:
                                with _usage_scope():
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
                                st.error(f"AI 해설 생성 실패: {exc}")

            with ai_col2:
                if st.button("🔄 이 문제 다시 변환", key=f"re_rw_btn_{q.id}", width="stretch"):
                    if not active_gemini_key:
                        st.error(f"사용 가능한 Gemini API 키({active_key_label})가 설정되지 않았습니다.")
                    elif not _ensure_token_quota():
                        pass
                    elif is_eng_item:
                        eng_meta = item.get("english_meta") or {}
                        src_passage = (
                            eng_meta.get("source_passage")
                            or getattr(q, "passage_text", "")
                            or getattr(q, "question_text", "")
                        )
                        eng_rw_model = EXAM_TAGGING_MODEL
                        eng_precision = False
                        eng_rw_key, eng_rw_key_lbl = resolve_active_gemini_api_key(
                            current_user,
                            creds.gemini_api_key,
                            model_name=eng_rw_model,
                        )
                        eff_eng_key = eng_rw_key or active_gemini_key
                        with st.spinner(f"'{eng_target_type}' 유형으로 이 지문을 새롭게 다시 변형 중..."):
                            try:
                                with _usage_scope():
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
                                        storage=storage,
                                        use_cache=False,
                                        username=current_user.username if current_user is not None else "",
                                    )
                                st.session_state["matched_items"][seq_num - 1]["rewritten"] = new_rw
                                st.session_state["ai_explanations"].pop(q.id, None)
                                st.session_state["ai_chats"].pop(q.id, None)
                                st.session_state["generated_exam_pdf"] = None
                                st.session_state["generated_exam_pdf_sig"] = ""
                                st.rerun(scope="fragment")
                            except Exception as exc:
                                st.error(f"영어 문항 재변환 실패: {exc}")
                    elif whitelist is None:
                        st.error("개념서 화이트리스트가 필요합니다.")
                    else:
                        with st.spinner("이전 변환의 미흡 용어를 보완해 이 문항만 다시 변환 중..."):
                            try:
                                target_oos = list(
                                    getattr(j, "out_of_scope_terms", [])
                                    or getattr(q, "advanced_terms", [])
                                    or []
                                )
                                math_rw_model = EXAM_TAGGING_MODEL
                                with _usage_scope():
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
                                st.rerun(scope="fragment")
                            except ForbiddenTermRemainingError as fexc:
                                st.warning(f"⛔ 재변환 후에도 금지 용어가 남아 반영하지 않았습니다: {fexc}")
                            except Exception as exc:
                                st.error(f"문항 재변환 실패: {exc}")

            if q.id in st.session_state["ai_explanations"]:
                st.info("✨ **수능 영어 구문·논리·오답 AI 맞춤 해설**" if is_eng_item else "✨ **개념서 범위 기준 AI 맞춤 해설**")
                st.markdown(st.session_state["ai_explanations"][q.id])
                if concept_ref_str not in st.session_state["ai_explanations"][q.id]:
                    st.markdown(f"**{_html.escape(concept_ref_str)}**")

                # 문항별 추가 질문 채팅
                st.markdown("💬 **추가 질문 채팅 (1:1 질의응답)**")
                q_chat_history: list[dict[str, str]] = st.session_state["ai_chats"].get(q.id, [])
                for msg in q_chat_history:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

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
                    elif not _ensure_token_quota():
                        pass
                    else:
                        q_chat_history.append({"role": "user", "content": followup_q.strip()})
                        with st.spinner("답변 생성 중..."):
                            try:
                                with _usage_scope():
                                    if is_eng_item:
                                        reply = chat_followup_with_english(
                                            api_key=active_gemini_key,
                                            model_name=selected_model,
                                            question=q,
                                            rewritten=rw,
                                            chat_history=q_chat_history[:-1],
                                            user_query=followup_q.strip(),
                                        )
                                    else:
                                        reply = chat_followup_with_pdf(
                                            api_key=active_gemini_key,
                                            model_name=selected_model,
                                            question=q,
                                            crop_png_bytes=crop_b,
                                            whitelist=whitelist,
                                            chat_history=q_chat_history[:-1],
                                            user_question=followup_q.strip(),
                                            rewritten=rw,
                                            sliced_text=st.session_state.get("concept_sliced_text", ""),
                                        )
                                q_chat_history.append({"role": "assistant", "content": reply})
                                st.session_state["ai_chats"][q.id] = q_chat_history
                                st.rerun(scope="fragment")
                            except Exception as exc:
                                st.error(f"답변 생성 오류: {exc}")

    # 하단 채점하기 버튼
    grade_c1, grade_c2 = st.columns([2, 1])
    with grade_c1:
        if st.button("📝 채점하기", type="primary", width="stretch"):
            st.session_state["graded"] = True
            _record_english_seen_for_user(matched_items)
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
            _auto_save_current_exam(
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


def _render_admin_panel() -> None:
    st.subheader("🛡️ 관리자 회원 관리 (등급 설정 · 실제 API 토큰 & 원화 비용 모니터링)")
    st.caption(
        "가입한 회원 목록 및 실제 API 토큰·비용 내역은 `./exam_db/users.json`에 영구 저장되며 Google Drive 연동 시 자동 동기화됩니다."
    )

    col_top1, col_top2 = st.columns([2, 1])
    with col_top1:
        search_q = st.text_input(
            "회원 검색 (아이디 또는 이름)",
            placeholder="검색어를 입력하세요...",
            key="admin_user_search",
        ).strip().lower()
    with col_top2:
        st.markdown("<div style='height: 1.65rem;'></div>", unsafe_allow_html=True)
        if st.button("🔄 회원 목록 새로고침", width="stretch", key="admin_refresh_btn"):
            if storage.drive_connected:
                _run_startup_drive_sync.clear()
                storage.sync_on_startup(sync_media=False)
            st.rerun()

    users_list = storage.load_users()
    if not users_list:
        st.info("아직 가입한 회원이 없습니다.")
        return

    tier_counts = {t: sum(1 for u in users_list if u.tier == t) for t in VALID_TIERS}
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("전체 회원", f"{len(users_list)}명")
    m2.metric("FREE (0원)", f"{tier_counts['FREE']}명")
    m3.metric("CUSTOM (0원)", f"{tier_counts['CUSTOM']}명")
    m4.metric("PRO (9,900원/월)", f"{tier_counts['PRO']}명")
    m5.metric("ULTRA (19,900원/월)", f"{tier_counts['ULTRA']}명")
    m6.metric("ADMIN", f"{tier_counts['ADMIN']}명")

    # ------------------------------------------------------------------
    # 💰 실제 API 토큰 사용량 & 원화(₩) 환산 대시보드
    # ------------------------------------------------------------------
    current_month = get_kst_month_str()
    with st.container(border=True):
        rate_c1, rate_c2 = st.columns([2.4, 1.0])
        with rate_c1:
            st.markdown(f"#### 💰 API 토큰 사용량 & 원화(₩) 환산 ({current_month} KST 기준)")
            st.caption(
                f"적용 단가 (Gemini 3.5 Flash-Lite): "
                f"입력 `${FLASH_LITE_INPUT_USD_PER_1M:.2f}` · "
                f"캐시 `${FLASH_LITE_CACHED_INPUT_USD_PER_1M:.2f}` · "
                f"출력(+사고) `${FLASH_LITE_OUTPUT_USD_PER_1M:.2f}` / 1M 토큰"
            )
        with rate_c2:
            usd_to_krw = float(
                st.number_input(
                    "적용 환율 (1 USD → KRW)",
                    min_value=500.0,
                    max_value=5000.0,
                    value=float(resolve_usd_to_krw_rate()),
                    step=10.0,
                    format="%.0f",
                    key="admin_usd_to_krw_rate",
                )
            )

        def _u_month_attr(u_obj: Any, attr: str, default: Any = 0) -> Any:
            if getattr(u_obj, "last_used_month", "") != current_month:
                return default
            return getattr(u_obj, attr, default) or default

        month_prompt_sum = sum(int(_u_month_attr(u, "monthly_prompt_tokens", 0)) for u in users_list)
        month_cached_sum = sum(int(_u_month_attr(u, "monthly_cached_tokens", 0)) for u in users_list)
        month_output_sum = sum(int(_u_month_attr(u, "monthly_output_tokens", 0)) for u in users_list)
        month_total_tok_sum = sum(int(_u_month_attr(u, "monthly_total_tokens", 0)) for u in users_list)
        month_calls_sum = sum(int(_u_month_attr(u, "monthly_api_calls", 0)) for u in users_list)
        month_server_usd = sum(float(_u_month_attr(u, "monthly_server_cost_usd", 0.0)) for u in users_list)
        month_all_usd = sum(float(_u_month_attr(u, "monthly_cost_usd", 0.0)) for u in users_list)

        total_tok_sum = sum(int(getattr(u, "total_total_tokens", 0) or 0) for u in users_list)
        total_server_usd = sum(float(getattr(u, "total_server_cost_usd", 0.0) or 0.0) for u in users_list)
        total_all_usd = sum(float(getattr(u, "total_cost_usd", 0.0) or 0.0) for u in users_list)

        est_monthly_rev_krw = (
            tier_counts["PRO"] * PRO_MONTHLY_PRICE_KRW
            + tier_counts["ULTRA"] * ULTRA_MONTHLY_PRICE_KRW
        )
        month_server_krw = month_server_usd * usd_to_krw
        cost_ratio_str = (
            f"{(month_server_krw / est_monthly_rev_krw) * 100:.1f}%"
            if est_monthly_rev_krw > 0
            else "-"
        )

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "이번 달 서버 API 비용",
            format_krw(month_server_usd, usd_to_krw),
            help=f"서버 키 부담분 기준 (${month_server_usd:.4f} / 개인키 포함 전체 ${month_all_usd:.4f})",
        )
        k1.caption(f"USD `${month_server_usd:.4f}` (개인키 제외)")
        k2.metric(
            "이번 달 전체 사용 토큰",
            f"{month_total_tok_sum:,}",
        )
        k2.caption(
            f"입력 {month_prompt_sum + month_cached_sum:,} · 출력 {month_output_sum:,} · {month_calls_sum:,}회"
        )
        k3.metric(
            "누적 서버 API 비용",
            format_krw(total_server_usd, usd_to_krw),
            help=f"전체 누적 서버 키 부담분 (${total_server_usd:.4f} / 개인키 포함 전체 ${total_all_usd:.4f})",
        )
        k3.caption(f"USD `${total_server_usd:.4f}` · 누적 {total_tok_sum:,} 토큰")
        k4.metric(
            "예상 월 매출 대비 API 원가율",
            cost_ratio_str,
        )
        k4.caption(f"예상 월 매출 ₩{est_monthly_rev_krw:,}")

    filtered_users = [
        u
        for u in users_list
        if not search_q or search_q in u.username.lower() or search_q in u.nickname.lower()
    ]
    filtered_users.sort(key=lambda u: u.created_at or u.username, reverse=True)

    table_rows = []
    for u in filtered_users:
        u_m_tok = int(_u_month_attr(u, "monthly_total_tokens", 0))
        u_m_calls = int(_u_month_attr(u, "monthly_api_calls", 0))
        u_m_usd = float(_u_month_attr(u, "monthly_cost_usd", 0.0))
        u_t_tok = int(getattr(u, "total_total_tokens", 0) or 0)
        u_t_usd = float(getattr(u, "total_cost_usd", 0.0) or 0.0)
        u_lim = get_monthly_token_limit(u)
        if u_lim is not None and u_lim > 0:
            usage_pct = f"{min(999.9, (u_m_tok / u_lim) * 100):.1f}% ({format_token_count(u_lim)})"
        else:
            usage_pct = "무제한"
        table_rows.append(
            {
                "아이디": u.username,
                "이름": u.nickname,
                "현재 등급": u.tier,
                "이번 달 토큰": u_m_tok,
                "한도 사용률": usage_pct,
                "이번 달 비용(₩)": format_krw(u_m_usd, usd_to_krw),
                "누적 토큰": u_t_tok,
                "누적 비용(₩)": format_krw(u_t_usd, usd_to_krw),
                "월 호출수": u_m_calls,
                "기출 업로드": int(getattr(u, "uploaded_exam_count", 0) or 0),
                "개인 키": mask_api_key(u.custom_api_key) or "-",
            }
        )

    edited_rows = st.data_editor(
        table_rows,
        column_config={
            "아이디": st.column_config.TextColumn("아이디", disabled=True),
            "이름": st.column_config.TextColumn("이름", disabled=True),
            "현재 등급": st.column_config.SelectboxColumn(
                "현재 등급 (변경 가능)",
                options=list(VALID_TIERS),
                required=True,
            ),
            "이번 달 토큰": st.column_config.NumberColumn("이번 달 토큰", format="%d", disabled=True),
            "한도 사용률": st.column_config.TextColumn("한도 사용률", disabled=True),
            "이번 달 비용(₩)": st.column_config.TextColumn("이번 달 비용(₩)", disabled=True),
            "누적 토큰": st.column_config.NumberColumn("누적 토큰", format="%d", disabled=True),
            "누적 비용(₩)": st.column_config.TextColumn("누적 비용(₩)", disabled=True),
            "월 호출수": st.column_config.NumberColumn("월 호출수", format="%d", disabled=True),
            "기출 업로드": st.column_config.NumberColumn("기출 업로드", disabled=True),
            "개인 키": st.column_config.TextColumn("개인 키", disabled=True),
        },
        hide_index=True,
        width="stretch",
        key="admin_users_editor",
    )

    if st.button("💾 회원 등급 변경사항 저장", type="primary", key="admin_save_tiers_btn"):
        updates = {
            row["아이디"]: row["현재 등급"]
            for row in edited_rows
            if row.get("아이디") and row.get("현재 등급") in VALID_TIERS
        }
        changed_cnt, err = batch_update_user_tiers(storage, updates)
        if err:
            st.warning(f"{changed_cnt}명 변경됨 / 저장 경고: {err}")
        elif changed_cnt > 0:
            st.success(f"✅ {changed_cnt}명의 회원 등급을 즉시 변경하여 `users.json`에 저장했습니다!")
            st.rerun()
        else:
            st.info("변경된 등급이 없습니다.")

    with st.expander("🔄 관리자: 회원 이번 달 토큰 사용량 초기화", expanded=False):
        reset_options = [u.username for u in filtered_users]
        reset_label_map = {
            u.username: f"@{u.username} ({u.nickname} · {u.tier} · 이번 달 {int(_u_month_attr(u, 'monthly_total_tokens', 0)):,} 토큰)"
            for u in filtered_users
        }
        reset_targets = st.multiselect(
            "이번 달 토큰 사용량을 0으로 초기화할 회원 선택 (누적 토큰·비용 기록은 유지됨)",
            options=reset_options,
            format_func=lambda un: reset_label_map.get(un, un),
            key="admin_reset_token_select",
        )
        if st.button(
            "🔄 선택한 회원 월간 토큰 초기화",
            disabled=not reset_targets,
            key="admin_exec_reset_tokens",
        ):
            r_cnt, r_err = reset_user_monthly_tokens(storage, reset_targets)
            if r_err:
                st.warning(f"{r_cnt}명 초기화됨 / Drive 저장 경고: {r_err}")
            elif r_cnt > 0:
                st.success(f"✅ {r_cnt}명의 이번 달 토큰 사용량을 0으로 초기화했습니다.")
                st.rerun()

    with st.expander("🗑️ 관리자: 회원 계정 삭제", expanded=False):
        actor_id = current_user.username if current_user is not None else ""
        deletable_users = [u for u in filtered_users if u.username != actor_id]
        if not deletable_users:
            st.info("삭제 가능한 다른 회원 계정이 없습니다.")
        else:
            del_options = [u.username for u in deletable_users]
            user_label_map = {
                u.username: f"@{u.username} ({u.nickname} · {u.tier})"
                for u in deletable_users
            }
            del_targets = st.multiselect(
                "삭제할 회원 계정 선택 (본인 계정 및 마지막 관리자는 보호됨)",
                options=del_options,
                format_func=lambda un: user_label_map.get(un, un),
                key="admin_delete_user_select",
            )
            confirm_del_users = st.checkbox(
                "선택한 회원 계정을 영구 삭제합니다. (복구 불가)",
                key="admin_confirm_delete_users",
            )
            if st.button(
                "🗑️ 선택한 계정 영구 삭제",
                disabled=not (del_targets and confirm_del_users),
                key="admin_exec_delete_users",
            ):
                del_cnt, del_err, del_note = delete_users_by_usernames(
                    storage,
                    del_targets,
                    actor_username=actor_id,
                )
                if del_err:
                    st.warning(f"{del_cnt}개 계정 삭제됨 / Drive 저장 경고: {del_err}")
                elif del_cnt > 0:
                    st.success(f"🗑️ {del_cnt}개 회원 계정을 영구 삭제했습니다.")
                    if del_note:
                        st.info(del_note)
                    st.rerun()
                elif del_note:
                    st.warning(del_note)

    _render_english_pool_admin_section()


def _render_english_pool_admin_section() -> None:
    st.divider()
    st.subheader("🔤 영어 변형 문제 풀(Pool) 캐시 관리")
    st.caption(
        "파이썬 검증을 통과한 영어 변형 문항은 서버의 `./exam_db/english_question_pool.json`에 "
        "지문·유형·옵션 해시별로 최대 3개씩 저장됩니다."
    )

    pool = storage.load_english_pool()
    total_keys = len(pool)
    flat_entries: list[tuple[str, int, dict]] = []
    type_counts: dict[str, int] = {}
    for c_key, items in pool.items():
        for idx, item in enumerate(items):
            t_name = str(item.get("target_type") or "기타").strip() or "기타"
            type_counts[t_name] = type_counts.get(t_name, 0) + 1
            flat_entries.append((c_key, idx, item))

    total_items = len(flat_entries)
    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("캐시 키(지문×유형) 수", f"{total_keys}개")
    c_m2.metric("저장된 변형 문항 수", f"{total_items}문항")
    c_m3.metric("저장된 변형 유형 수", f"{len(type_counts)}종")

    if type_counts:
        summary_parts = [f"{k}({v})" for k, v in sorted(type_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        st.caption("유형별 분포: " + " · ".join(summary_parts))

    if total_items == 0:
        st.info("현재 저장된 영어 변형 캐시 문항이 없습니다.")
        return

    col_clr1, col_clr2 = st.columns([2, 1])
    with col_clr1:
        confirm_clear = st.checkbox(
            "영어 변형 캐시만 전체 비우기 확인 (기출 DB·회원 DB는 유지됨)",
            key="admin_confirm_clear_eng_pool",
        )
    with col_clr2:
        if st.button(
            "🧹 영어 변형 캐시 전체 비우기",
            disabled=not confirm_clear,
            width="stretch",
            key="admin_clear_eng_pool_btn",
        ):
            removed_cnt = storage.clear_english_pool()
            st.success(f"🧹 영어 변형 캐시 {removed_cnt}문항을 모두 비웠습니다.")
            st.rerun()

    with st.expander(f"📋 저장된 영어 변형 캐시 상세 조회 / 개별 삭제 ({total_items}문항)", expanded=False):
        f_col1, f_col2 = st.columns([1, 2])
        with f_col1:
            type_options = ["전체"] + sorted(type_counts.keys())
            sel_type = st.selectbox("유형 필터", options=type_options, key="admin_eng_pool_type_filter")
        with f_col2:
            pool_q = st.text_input(
                "지문/해설/키 검색",
                placeholder="키워드 또는 캐시 키 입력...",
                key="admin_eng_pool_search",
            ).strip().lower()

        filtered_entries = [
            (ck, idx, it)
            for ck, idx, it in flat_entries
            if (sel_type == "전체" or (it.get("target_type") or "기타").strip() == sel_type)
            and (
                not pool_q
                or pool_q in ck.lower()
                or pool_q in str(it.get("rewritten_stem") or "").lower()
                or pool_q in str(it.get("pdf_grounded_explanation") or "").lower()
            )
        ]

        st.caption(f"검색 결과: {len(filtered_entries)}문항 (최대 50건 표시)")
        for ck, idx, it in filtered_entries[:50]:
            t_label = str(it.get("target_type") or "기타")
            ans_label = str(it.get("answer") or "-")
            stem_preview = " ".join(str(it.get("rewritten_stem") or "").split())[:130]
            r_col1, r_col2 = st.columns([5, 1])
            with r_col1:
                st.markdown(f"**[{t_label}]** · 키 `{ck}` (#{idx + 1}) · 정답 `{ans_label}`")
                if stem_preview:
                    st.caption(f"{stem_preview}...")
            with r_col2:
                if st.button("🗑️ 삭제", key=f"del_eng_pool_{ck}_{idx}", width="stretch"):
                    removed = storage.delete_english_pool_entry(ck, idx)
                    if removed > 0:
                        st.success(f"캐시 항목(`{ck}` #{idx + 1})을 삭제했습니다.")
                    st.rerun()


def _record_english_seen_for_user(items: list[dict[str, Any]]) -> None:
    """현재 계정이 PDF 생성 또는 풀이 채점을 완료한 영어 변형 문항을 기록해 중복 출제를 막는다."""
    if current_user is None or not items:
        return
    pairs: list[tuple[str, str]] = []
    for it in items:
        rw_item: RewrittenQuestion | None = it.get("rewritten")
        eng_meta = it.get("english_meta") or {}
        if rw_item is None or not eng_meta:
            continue
        passage = str(eng_meta.get("source_passage") or "").strip()
        t_type = str(eng_meta.get("target_type") or rw_item.target_question_type or "").strip()
        if not passage or not t_type:
            continue
        c_key = storage.build_english_pool_key(
            source_passage=passage,
            target_type=t_type,
            difficulty=str(eng_meta.get("difficulty") or ""),
            blank_unit=str(eng_meta.get("blank_unit") or ""),
            vocab_style=str(eng_meta.get("vocab_style") or ""),
        )
        pairs.append((c_key, str(rw_item.rewritten_stem or "")))
    if pairs:
        storage.mark_user_seen_english_variants(current_user.username, pairs)


def _infer_exam_subject(items: list[dict[str, Any]]) -> str:
    subjects: list[str] = []
    for it in items:
        q_obj: QuestionRecord = it["question"]
        rw_obj: RewrittenQuestion | None = it.get("rewritten")
        if it.get("english_meta") or (rw_obj and getattr(rw_obj, "subject_domain", "") == "영어"):
            subj = "영어"
        else:
            subj = str(getattr(q_obj, "subject_category", "") or "").strip()
            if not subj:
                subj = infer_subject_category(getattr(q_obj, "curriculum_subject", "") or getattr(q_obj, "exam_name", ""))
        if subj and subj not in subjects:
            subjects.append(subj)
    if not subjects:
        return "기타"
    if len(subjects) == 1:
        return subjects[0]
    return "통합"


def _build_exam_sig(items: list[dict[str, Any]]) -> str:
    payload = [
        (
            it["question"].id,
            getattr(it.get("rewritten"), "rewritten_stem", ""),
            getattr(it.get("rewritten"), "rewritten_choices", []),
        )
        for it in items
    ]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def _serialize_matched_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for it in items:
        q_obj = it.get("question")
        j_obj = it.get("judgment")
        rw_obj = it.get("rewritten")
        serialized.append(
            {
                "question": q_obj.model_dump() if hasattr(q_obj, "model_dump") else q_obj,
                "judgment": j_obj.model_dump() if hasattr(j_obj, "model_dump") else None,
                "rewritten": rw_obj.model_dump() if hasattr(rw_obj, "model_dump") else None,
                "concept_reference": str(it.get("concept_reference") or ""),
                "english_meta": dict(it.get("english_meta") or {}) if it.get("english_meta") else None,
            }
        )
    return serialized


def _deserialize_matched_items(raw_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    restored: list[dict[str, Any]] = []
    for raw in raw_items:
        try:
            q_obj = QuestionRecord.model_validate(raw["question"])
        except Exception:
            continue
        j_obj = None
        if raw.get("judgment"):
            try:
                j_obj = QuestionMatchJudgment.model_validate(raw["judgment"])
            except Exception:
                j_obj = None
        rw_obj = None
        if raw.get("rewritten"):
            try:
                rw_obj = RewrittenQuestion.model_validate(raw["rewritten"])
            except Exception:
                rw_obj = None
        crop_b = storage.get_crop_bytes(q_obj.crop_path) if getattr(q_obj, "crop_path", "") else None
        restored.append(
            {
                "question": q_obj,
                "judgment": j_obj,
                "crop_png_bytes": crop_b,
                "rewritten": rw_obj,
                "concept_reference": str(raw.get("concept_reference") or ""),
                "english_meta": raw.get("english_meta"),
            }
        )
    return restored


def _auto_save_current_exam(
    items: list[dict[str, Any]],
    title: str = "",
    pdf_bytes: bytes | None = None,
    score_info: dict[str, Any] | None = None,
    sig: str = "",
) -> None:
    if current_user is None or not items:
        return
    exam_id = (sig or _build_exam_sig(items))[:16]
    subject = _infer_exam_subject(items)
    record: dict[str, Any] = {
        "exam_id": exam_id,
        "title": (title or f"{subject} 맞춤형 시험지 ({len(items)}문항)").strip(),
        "subject": subject,
        "question_count": len(items),
        "items": _serialize_matched_items(items),
    }
    if score_info is not None:
        record["last_score"] = score_info
    storage.save_user_exam(current_user.username, record, pdf_bytes=pdf_bytes)


def _render_my_exams_tab() -> None:
    st.subheader("📁 나의 시험지 보관함 (과목별 자동 저장)")
    st.caption(
        "맞춤형 시험지 PDF를 생성하거나 웹에서 직접 풀고 채점한 시험지는 계정별로 과목에 맞춰 자동 저장됩니다."
    )
    if current_user is None:
        st.info("로그인 후 이용할 수 있습니다.")
        return

    all_exams = storage.list_user_exams(current_user.username)
    if not all_exams:
        st.info("아직 저장된 시험지가 없습니다. **탭 2**에서 PDF를 만들거나 문제를 풀고 채점하면 이곳에 자동 저장됩니다.")
        return

    subj_counts: dict[str, int] = {}
    for ex in all_exams:
        s = str(ex.get("subject") or "기타")
        subj_counts[s] = subj_counts.get(s, 0) + 1

    subj_options = ["전체"] + sorted(subj_counts.keys())
    sel_subj = st.radio(
        "과목별 보기",
        options=subj_options,
        format_func=lambda s: f"{s} ({len(all_exams)}건)" if s == "전체" else f"{s} ({subj_counts.get(s, 0)}건)",
        horizontal=True,
        key="my_exams_subject_filter",
    )

    shown_exams = [e for e in all_exams if sel_subj == "전체" or str(e.get("subject") or "기타") == sel_subj]
    if not shown_exams:
        st.info(f"'{sel_subj}' 과목에 저장된 시험지가 없습니다.")
        return

    for ex in shown_exams:
        ex_id = str(ex.get("exam_id") or "")
        ex_title = str(ex.get("title") or "맞춤형 시험지")
        ex_subj = str(ex.get("subject") or "기타")
        ex_q_cnt = int(ex.get("question_count") or len(ex.get("items") or []))
        ex_updated = str(ex.get("updated_at") or ex.get("created_at") or "")
        last_score = ex.get("last_score")

        with st.container(border=True):
            top_c1, top_c2 = st.columns([4, 1.5])
            with top_c1:
                st.markdown(f"**[{ex_subj}] {ex_title}**")
                meta_parts = [f"총 {ex_q_cnt}문항", f"저장일시: {ex_updated}"]
                if isinstance(last_score, dict):
                    meta_parts.append(
                        f"최근 점수: {last_score.get('percent', 0)}점 ({last_score.get('correct', 0)}/{last_score.get('total', ex_q_cnt)})"
                    )
                st.caption(" · ".join(meta_parts))
            with top_c2:
                pdf_data = storage.get_user_exam_pdf_bytes(current_user.username, ex_id)
                if pdf_data:
                    st.download_button(
                        label=f"⬇️ PDF ({len(pdf_data) // 1024}KB)",
                        data=pdf_data,
                        file_name=f"{sanitize_id_part(ex_title)}.pdf",
                        mime="application/pdf",
                        key=f"my_exam_dl_{ex_id}",
                        width="stretch",
                    )
                else:
                    st.caption("📄 PDF 미생성 (풀이 전용)")

            b_c1, b_c2, b_c3 = st.columns([1.4, 1.2, 3.4])
            with b_c1:
                if st.button("✏️ 이 시험지 다시 풀기", key=f"my_exam_load_{ex_id}", width="stretch"):
                    restored = _deserialize_matched_items(ex.get("items") or [])
                    if restored:
                        st.session_state["matched_items"] = restored
                        st.session_state["user_answers"] = {}
                        st.session_state["graded"] = False
                        st.session_state["ai_explanations"] = {}
                        st.session_state["ai_chats"] = {}
                        st.session_state["generated_exam_pdf"] = pdf_data
                        st.session_state["generated_exam_pdf_sig"] = ""
                        st.success("시험지를 불러왔습니다! 상단 **[탭 2]**로 이동해 바로 풀거나 PDF를 생성하세요.")
                    else:
                        st.warning("시험지 문항 데이터를 복원하지 못했습니다.")
            with b_c2:
                if st.button("🗑️ 삭제", key=f"my_exam_del_{ex_id}", width="stretch"):
                    storage.delete_user_exam(current_user.username, ex_id)
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


def _build_recommended_exam_catalog() -> list[dict[str, str]]:
    """최근 3개년(2024~2026) 고1·고2·고3 주요 학평/모평/수능 + 과목별 추천 기출 카탈로그를 생성한다."""
    years = ["2026", "2025", "2024"]
    grade_specs: list[tuple[str, list[str], list[tuple[str, str]]]] = [
        (
            "고1",
            ["3월", "6월", "9월", "10월"],
            [("영어", "영어"), ("수학", "수학"), ("과학", "통합과학"), ("사회", "통합사회")],
        ),
        (
            "고2",
            ["3월", "6월", "9월", "10월"],
            [
                ("영어", "영어"),
                ("수학", "수학"),
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
                ("영어", "영어"),
                ("수학", "수학"),
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
                    catalog.append(
                        {
                            "year": y,
                            "month": m,
                            "grade": grade,
                            "subject_cat": subj_cat,
                            "subject_name": subj_name,
                            "exam_name": f"{y}년 {m} {grade} {subj_name}",
                        }
                    )
    return catalog


def _render_exam_coverage_board() -> None:
    """탭 1 상단에 이미 등록된 기출 목록과 추천 미등록 기출 목록을 나란히 표시한다."""
    db_records = storage.load_db()
    registered_map: dict[str, dict[str, Any]] = {}
    registered_ids: set[str] = set()
    for rec in db_records:
        en = re.sub(r"\s+", " ", (rec.exam_name or "").strip())
        if not en:
            continue
        norm_id = _normalize_exam_identity(en)
        if norm_id:
            registered_ids.add(norm_id)
        entry = registered_map.setdefault(
            en,
            {
                "회차명": en,
                "과목": rec.subject_category or infer_subject_category(rec.curriculum_subject or en),
                "문항 수": 0,
            },
        )
        entry["문항 수"] += 1

    catalog = _build_recommended_exam_catalog()
    missing_all = [
        item
        for item in catalog
        if _normalize_exam_identity(item["exam_name"]) not in registered_ids
    ]

    with st.expander(
        f"📋 기출 DB 등록 현황 & 추천 미등록 기출 보기 (등록 {len(registered_map)}회차 · 추천 미등록 {len(missing_all)}건)",
        expanded=True,
    ):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            f_subj = st.selectbox(
                "과목 필터",
                options=["전체", "영어", "수학", "과학", "사회"],
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


if current_user is None:
    st.info("👈 좌측 사이드바 상단의 **[로그인 / 회원가입]** 탭에서 로그인하시면 기출 수집·추출·문제 개조 기능을 이용하실 수 있습니다.")
    t_col1, t_col2, t_col3, t_col4, t_col5 = st.columns(5)
    with t_col1:
        with st.container(border=True):
            st.markdown("<span class='tier-badge tier-FREE'>FREE · 0원</span>", unsafe_allow_html=True)
            st.markdown("#### 무료 회원")
            st.write(f"- Gemini 3.5 Flash-Lite **월 {format_token_count(FREE_MONTHLY_TOKEN_LIMIT)} 토큰**")
            st.write(f"- 1회 생성 **최대 {FREE_MAX_BATCH_QUESTIONS}문항** 제한")
            st.write("- 매월 1일(KST) 한도 자동 초기화")
    with t_col2:
        with st.container(border=True):
            st.markdown("<span class='tier-badge tier-CUSTOM'>CUSTOM · 0원</span>", unsafe_allow_html=True)
            st.markdown("#### 개인 API 키")
            st.write("- **본인 Gemini API 키** 등록")
            st.write("- Gemini 3.5 Flash-Lite **토큰 무제한**")
            st.write("- 1회 배치 제한 없음 · 즉시 전환")
    with t_col3:
        with st.container(border=True):
            st.markdown("<span class='tier-badge tier-PRO'>PRO · 9,900원/월</span>", unsafe_allow_html=True)
            st.markdown("#### 스탠다드 유료")
            st.write(f"- Gemini 3.5 Flash-Lite **월 {format_token_count(PRO_MONTHLY_TOKEN_LIMIT)} 토큰**")
            st.write("- 1회 생성 문항 수 **제한 없음**")
            st.write("- **공용 개념서 DB** 구축·이용 & 재태깅")
    with t_col4:
        with st.container(border=True):
            st.markdown("<span class='tier-badge tier-ULTRA'>ULTRA · 19,900원/월</span>", unsafe_allow_html=True)
            st.markdown("#### 대용량 유료 (4배)")
            st.write("- PRO 모든 혜택 포함")
            st.write(f"- Gemini 3.5 Flash-Lite **월 {format_token_count(ULTRA_MONTHLY_TOKEN_LIMIT)} 토큰** (PRO 4배)")
            st.write("- **공용 개념서 DB** 구축·이용 & 재태깅")
    with t_col5:
        with st.container(border=True):
            st.markdown("<span class='tier-badge tier-ADMIN'>ADMIN</span>", unsafe_allow_html=True)
            st.markdown("#### 최고 관리자")
            st.write("- 모든 기능·토큰 **무제한** (관리자 전용)")
            st.write("- **공용 개념서 DB** 구축 & 관리")
            st.write("- **회원 등급 · API 원화 비용** 관리")
    st.stop()


def _render_concept_book_builder_section(key_prefix: str = "tab3") -> None:
    """관리자(ADMIN) 및 Pro(VIP) 등급용 개념서·교과서 PDF 단원별 공용 DB 구축 & 관리 UI."""
    st.subheader("📚 개념서·교과서 PDF 단원별 공용 DB 구축 (관리자 · Pro 전용)")
    st.caption(
        "지정한 개념 페이지 구간만 PyMuPDF로 추출해 목차·정답지·문제 풀이를 제외하고, "
        "사이드바에서 선택한 Gemini 모델로 단원별 핵심 개념과 허용 용어·단위를 구조화하여 "
        "로컬(`./concept_db`) 및 Google Drive 공용 폴더에 자동 동기화합니다."
    )

    if not can_access_shared_concept_db(current_user):
        st.warning(
            "🔒 **공용 개념서·교과서 DB 구축 및 공유 기능은 Pro(VIP) · ADMIN 등급 전용입니다.**\n\n"
            f"현재 로그인 등급: **{TIER_LABELS.get(current_user.tier, current_user.tier)} ({current_user.tier})** — "
            "관리자에게 Pro(VIP) 승급을 요청하시거나, [🎯 탭 2]의 **개별 개념서 PDF 직접 업로드** 기능을 이용해 주세요."
        )
        return

    with st.container(border=True):
        st.markdown("#### ➕ 새 개념서·교과서 PDF 업로드 및 개념 구간 추출")
        cb_pdf_file = st.file_uploader(
            "교재 / 개념서 PDF 파일 업로드 (올리는 즉시 표지 자동 인식)",
            type=["pdf"],
            key=f"{key_prefix}_pdf_uploader",
        )
        cb_pdf_bytes: bytes | None = cb_pdf_file.getvalue() if cb_pdf_file is not None else None
        cb_sig = f"{cb_pdf_file.name}:{cb_pdf_file.size}" if cb_pdf_file is not None else ""

        if not cb_sig:
            st.session_state[f"{key_prefix}_last_cb_sig"] = ""
            st.session_state[f"{key_prefix}_cb_meta_status"] = ""
        elif cb_sig != st.session_state.get(f"{key_prefix}_last_cb_sig"):
            auto_title = cb_pdf_file.name.rsplit(".", 1)[0].replace("_", " ").strip()
            auto_subj_grade = ""
            if cb_pdf_bytes and active_gemini_key and _ensure_token_quota(show_error=False):
                try:
                    with st.spinner("🔍 개념서 표지(교재명 · 과목 및 학년) 자동 인식 중..."):
                        p1_txt, p1_png = extract_first_page_for_metadata(cb_pdf_bytes, dpi=140)
                        if p1_png is None:
                            p1_png = render_pdf_page_png(cb_pdf_bytes, 1, dpi=140)
                        with _usage_scope():
                            cover_meta = extract_concept_book_cover_metadata(
                                api_key=active_gemini_key,
                                model_name=EXAM_TAGGING_MODEL,
                                first_page_text=p1_txt,
                                first_page_png_bytes=p1_png,
                                filename=cb_pdf_file.name,
                            )
                        if cover_meta.book_title:
                            auto_title = cover_meta.book_title
                        if cover_meta.subject_and_grade:
                            auto_subj_grade = cover_meta.subject_and_grade
                except Exception:
                    pass
            st.session_state[f"{key_prefix}_book_title"] = auto_title
            st.session_state[f"{key_prefix}_subject_grade"] = auto_subj_grade
            st.session_state[f"{key_prefix}_last_cb_sig"] = cb_sig
            st.session_state[f"{key_prefix}_cb_meta_status"] = (
                "parsed" if (auto_title and auto_subj_grade) else "partial"
            )

        if cb_sig:
            cb_meta_st = st.session_state.get(f"{key_prefix}_cb_meta_status", "")
            if cb_meta_st == "parsed":
                st.success(
                    f"🎯 표지 자동 인식 완료: **{st.session_state.get(f'{key_prefix}_book_title', '')}** "
                    f"({st.session_state.get(f'{key_prefix}_subject_grade', '')}) — 필요 시 아래에서 수정하세요"
                )
            elif cb_meta_st == "partial":
                st.info("📄 파일명/표지 일부를 채웠습니다. 빈칸이나 수정이 필요한 부분을 확인해 주세요.")

        cb_col1, cb_col2 = st.columns(2)
        with cb_col1:
            book_title_in = st.text_input(
                "교재명",
                placeholder="예: 완자 통합과학, 개념원리 공통수학1",
                key=f"{key_prefix}_book_title",
            )
        with cb_col2:
            subject_grade_in = st.text_input(
                "과목 및 학년",
                placeholder="예: 고1 통합과학",
                key=f"{key_prefix}_subject_grade",
            )

        extract_mode = st.radio(
            "개념 페이지 추출 방식",
            options=[
                "🤖 자동 추출 (AI가 개념 부분 자동 판별 · 단원 정리)",
                "✍️ 수동 추출 (추출할 개념 페이지 범위 직접 지정)",
            ],
            horizontal=True,
            key=f"{key_prefix}_extract_mode",
        )
        is_auto_mode = extract_mode.startswith("🤖")

        page_ranges_in = ""
        if not is_auto_mode:
            page_ranges_in = st.text_input(
                "추출할 개념 페이지 범위 입력 (쉼표 `,`와 하이픈 `-`으로 개념 설명 구간만 여러 개 지정 가능)",
                placeholder="예: 12-25, 34-48, 60-75",
                key=f"{key_prefix}_page_ranges",
            )
        else:
            st.info(
                "🤖 **AI 자동 추출 모드**: 페이지 범위를 직접 입력하지 않아도 "
                "Gemini 3.5 Flash-Lite가 먼저 목차를 분석한 뒤 본문과 대조해 **표지·목차·문제 풀이·정답지를 제외하고 "
                "순수 개념 설명 구간만 빠짐없이 판별**하여 단원별로 구조화합니다."
            )

        parsed_intervals: list[tuple[int, int]] = []
        if cb_pdf_bytes is not None:
            total_cb_pages = get_pdf_page_count(cb_pdf_bytes)
            st.caption(f"📄 업로드된 PDF: `{cb_pdf_file.name}` (전체 **{total_cb_pages}쪽**)")
            if not is_auto_mode and page_ranges_in.strip():
                try:
                    parsed_intervals = parse_page_ranges_spec(page_ranges_in, total_cb_pages)
                    total_sel_pages = sum(e - s + 1 for s, e in parsed_intervals)
                    interval_str = ", ".join(
                        f"p.{s}~p.{e}" if s != e else f"p.{s}" for s, e in parsed_intervals
                    )
                    st.info(
                        f"✅ 파싱된 개념 구간: **{interval_str}** (총 {len(parsed_intervals)}개 구간 · {total_sel_pages}쪽)"
                    )
                    with st.expander("👀 첫 구간 시작/끝 페이지 미리보기", expanded=False):
                        first_s, first_e = parsed_intervals[0][0], parsed_intervals[-1][1]
                        pv1, pv2 = st.columns(2)
                        with pv1:
                            st.image(
                                render_pdf_page_png(cb_pdf_bytes, first_s, dpi=130),
                                caption=f"첫 시작 페이지 (p.{first_s})",
                                width="stretch",
                            )
                        with pv2:
                            if first_e != first_s:
                                st.image(
                                    render_pdf_page_png(cb_pdf_bytes, first_e, dpi=130),
                                    caption=f"마지막 끝 페이지 (p.{first_e})",
                                    width="stretch",
                                )
                except ValueError as ve:
                    st.error(str(ve))

        def _finalize_and_save_concept_book(
            intervals_to_extract: list[tuple[int, int]],
            spec_label: str,
            status_box: Any,
        ) -> None:
            status_box.info(
                "2단계: PyMuPDF로 선택 구간에서 목차·정답지·문제 풀이를 제외하고 순수 개념 텍스트와 실제 페이지 번호를 매핑 중..."
            )
            sliced_bytes, mapped_text, stats = extract_concept_pages_by_ranges(
                cb_pdf_bytes, intervals_to_extract
            )
            skipped_info = stats.get("skipped_pages") or []
            if skipped_info:
                skip_summary = ", ".join(
                    f"p.{sp['page']}({sp['reason']})" for sp in skipped_info[:10]
                )
                st.caption(f"🧹 불필요 페이지 자동 제외 ({len(skipped_info)}쪽): {skip_summary}")

            db_build_model, db_model_notice = resolve_model_for_user(
                current_user, CONCEPT_DB_BUILD_MODEL
            )
            model_disp = MODEL_LABELS.get(db_build_model, db_build_model)
            status_box.info(
                f"3단계: {model_disp} 모델로 단원별 핵심 개념·허용 용어·페이지 범위를 구조화하고 공용 DB에 동기화 중..."
                + (f" ({db_model_notice})" if db_model_notice else "")
            )
            with _usage_scope():
                extracted_book = extract_concept_book_units(
                    api_key=active_gemini_key,
                    model_name=db_build_model,
                    book_title=book_title_in.strip(),
                    subject_and_grade=subject_grade_in.strip(),
                    page_ranges_display=stats["ranges_display"],
                    sliced_pdf_bytes=sliced_bytes,
                    mapped_concept_text=mapped_text,
                    original_pdf_bytes=cb_pdf_bytes,
                    kept_pages=stats.get("kept_pages"),
                    status_callback=lambda m: status_box.info(m),
                )
            grade_match = re.search(r"(고\s*[123])", subject_grade_in)
            grade_str = (
                grade_match.group(1).replace(" ", "")
                if grade_match
                else (extracted_book.grade or "고1")
            )
            book_id = f"{sanitize_id_part(book_title_in)}__{sanitize_id_part(subject_grade_in)}"
            record = ConceptBookRecord(
                id=book_id,
                book_title=book_title_in.strip(),
                subject=subject_grade_in.strip(),
                grade=grade_str,
                page_ranges_spec=spec_label,
                units=extracted_book.units,
                created_by=current_user.username,
                created_at=datetime.now().isoformat(timespec="seconds"),
            )
            save_err = storage.save_concept_book(record)
            status_box.empty()
            if save_err:
                st.warning(save_err)
            else:
                st.success(
                    f"🎉 공용 개념서 DB 저장 완료! (`{record.book_title}` · 총 {len(record.units)}개 단원 구조화 · "
                    + (
                        "로컬 ./concept_db 및 Google Drive 동기화 완료)"
                        if storage.drive_connected
                        else "로컬 ./concept_db 저장 완료)"
                    )
                )

        if is_auto_mode:
            auto_b1, auto_b2 = st.columns([1.6, 1])
            with auto_b1:
                auto_build_btn = st.button(
                    "🤖 AI 개념 구간 자동 판별 & 단원별 공용 DB 저장",
                    type="primary",
                    width="stretch",
                    key=f"{key_prefix}_auto_build_btn",
                )
            with auto_b2:
                auto_preview_btn = st.button(
                    "🔍 AI 개념 구간만 먼저 미리보기",
                    width="stretch",
                    key=f"{key_prefix}_auto_preview_btn",
                )

            preview_state_key = f"{key_prefix}_auto_detect_preview"
            if cb_pdf_bytes is not None:
                cur_auto_sig = hashlib.sha256(cb_pdf_bytes).hexdigest()[:12]
                prev_auto = st.session_state.get(preview_state_key)
                if prev_auto and prev_auto.get("pdf_sig") != cur_auto_sig:
                    st.session_state.pop(preview_state_key, None)

            if auto_preview_btn or auto_build_btn:
                if not cb_pdf_bytes:
                    st.error("교재/개념서 PDF 파일을 먼저 업로드해 주세요.")
                elif not active_gemini_key:
                    st.error(f"사용 가능한 Gemini API 키({active_key_label})가 설정되지 않았습니다.")
                elif not _ensure_token_quota():
                    pass
                elif auto_build_btn and not book_title_in.strip():
                    st.error("교재명을 입력해 주세요. (예: 완자 통합과학)")
                elif auto_build_btn and not subject_grade_in.strip():
                    st.error("과목 및 학년을 입력해 주세요. (예: 고1 통합과학)")
                else:
                    status_box = st.empty()
                    try:
                        with _usage_scope():
                            cur_auto_sig = hashlib.sha256(cb_pdf_bytes).hexdigest()[:12]
                            cached_prev = st.session_state.get(preview_state_key)
                            if (
                                auto_build_btn
                                and cached_prev
                                and cached_prev.get("pdf_sig") == cur_auto_sig
                                and cached_prev.get("intervals")
                            ):
                                detected_out = cached_prev["detected_out"]
                                detected_intervals = cached_prev["intervals"]
                            else:
                                status_box.info("1단계: PyMuPDF로 PDF 전체 페이지 구조 스캔 중...")
                                catalog, has_text_layer = extract_pdf_page_catalog(cb_pdf_bytes)
                                detect_model, _ = resolve_model_for_user(
                                    current_user, CONCEPT_DB_BUILD_MODEL
                                )
                                detected_out, detected_intervals = detect_concept_pages_with_ai(
                                    api_key=active_gemini_key,
                                    model_name=detect_model,
                                    pdf_bytes=cb_pdf_bytes,
                                    page_catalog=catalog,
                                    has_text_layer=has_text_layer,
                                    book_title=book_title_in.strip(),
                                    subject_and_grade=subject_grade_in.strip(),
                                    status_callback=lambda m: status_box.info(m),
                                )
                                st.session_state[preview_state_key] = {
                                    "pdf_sig": cur_auto_sig,
                                    "detected_out": detected_out,
                                    "intervals": detected_intervals,
                                }

                            failed_chunks_list = list(getattr(detected_out, "failed_chunks", []) or [])
                            if failed_chunks_list:
                                st.warning(
                                    "⚠️ **일부 PDF 페이지 구간 판별에 실패했습니다** (자동 재시도 후 누락 방지를 위해 표시):\n- "
                                    + "\n- ".join(failed_chunks_list)
                                    + "\n\n해당 구간이 필요하다면 **[✍️ 수동 추출]**에서 직접 범위를 추가해 보완해 주세요."
                                )

                            if not detected_intervals:
                                status_box.empty()
                                st.warning(
                                    "AI가 순수 개념 설명 구간을 찾지 못했습니다. [✍️ 수동 추출]로 전환해 페이지 범위를 직접 지정해 주세요."
                                )
                            elif auto_preview_btn:
                                status_box.empty()
                                st.success(
                                    f"🔍 AI 개념 구간 판별 완료! (총 {len(detected_intervals)}개 구간 · "
                                    f"{sum(e - s + 1 for s, e in detected_intervals)}쪽 감지)"
                                )
                            else:
                                spec_auto = ", ".join(
                                    f"{s}-{e}" if s != e else f"{s}" for s, e in detected_intervals
                                )
                                _finalize_and_save_concept_book(
                                    intervals_to_extract=detected_intervals,
                                    spec_label=f"{spec_auto} (AI 자동 추출)",
                                    status_box=status_box,
                                )
                    except Exception as exc:
                        status_box.empty()
                        st.error(f"AI 자동 추출 실패: {exc}")

            saved_preview = st.session_state.get(preview_state_key)
            if saved_preview and (
                saved_preview.get("intervals")
                or getattr(saved_preview.get("detected_out"), "failed_chunks", None)
            ):
                det_out = saved_preview["detected_out"]
                det_ints = saved_preview.get("intervals") or []
                det_failed = list(getattr(det_out, "failed_chunks", []) or [])
                det_spec = ", ".join(f"p.{s}~p.{e}" if s != e else f"p.{s}" for s, e in det_ints)
                exp_label = (
                    f"🤖 AI가 자동 판별한 개념 구간 ({det_spec} · {sum(e - s + 1 for s, e in det_ints)}쪽)"
                    if det_ints
                    else "⚠️ AI 개념 구간 자동 판별 결과 (일부 구간 실패)"
                )
                with st.expander(exp_label, expanded=True):
                    if det_failed:
                        st.warning(
                            "⚠️ **자동 판별에 실패한 청크 구간:** " + " / ".join(det_failed)
                        )
                    if getattr(det_out, "toc_units", None):
                        st.caption(
                            f"📑 **목차 분석 기준 단원 ({len(det_out.toc_units)}개):** "
                            + " · ".join(det_out.toc_units)
                        )
                    for r_item in det_out.detected_ranges:
                        st.write(
                            f"- **{r_item.unit_hint}**: `p.{r_item.start_page}~p.{r_item.end_page}`"
                            + (f" — {r_item.reason}" if r_item.reason else "")
                        )
                    if det_out.excluded_summary:
                        st.caption(f"🧹 AI 제외 판단 구간: {det_out.excluded_summary}")
        else:
            build_db_btn = st.button(
                "🧠 PyMuPDF 개념 구간 추출 & Gemini 단원별 공용 DB 저장",
                type="primary",
                width="stretch",
                key=f"{key_prefix}_build_btn",
            )

            if build_db_btn:
                if not cb_pdf_bytes:
                    st.error("교재/개념서 PDF 파일을 먼저 업로드해 주세요.")
                elif not book_title_in.strip():
                    st.error("교재명을 입력해 주세요. (예: 완자 통합과학)")
                elif not subject_grade_in.strip():
                    st.error("과목 및 학년을 입력해 주세요. (예: 고1 통합과학)")
                elif not page_ranges_in.strip():
                    st.error("추출할 개념 페이지 범위를 입력해 주세요. (예: 12-25, 34-48, 60-75)")
                elif not active_gemini_key:
                    st.error(f"사용 가능한 Gemini API 키({active_key_label})가 설정되지 않았습니다.")
                elif not _ensure_token_quota():
                    pass
                else:
                    try:
                        total_cb_pages = get_pdf_page_count(cb_pdf_bytes)
                        intervals = parse_page_ranges_spec(page_ranges_in, total_cb_pages)
                    except ValueError as ve:
                        st.error(str(ve))
                        intervals = []

                    if intervals:
                        status_box = st.empty()
                        try:
                            _finalize_and_save_concept_book(
                                intervals_to_extract=intervals,
                                spec_label=f"{page_ranges_in.strip()} (수동 추출)",
                                status_box=status_box,
                            )
                        except Exception as exc:
                            status_box.empty()
                            st.error(f"개념서 단원별 구조화 추출 실패: {exc}")

    # 등록된 공용 개념서 DB 목록 조회 및 관리
    st.markdown("---")
    st.markdown("#### 🗂️ 등록된 공용 개념서·교과서 DB 목록 (`./concept_db` & Google Drive)")
    saved_books = storage.load_concept_books()
    if not saved_books:
        st.info("아직 등록된 공용 개념서 DB가 없습니다. 위에서 개념서 PDF를 업로드해 첫 공용 DB를 구축해 보세요.")
    else:
        for b_idx, book in enumerate(saved_books):
            with st.container(border=True):
                b_col1, b_col2, b_col3 = st.columns([3, 1.2, 0.8])
                with b_col1:
                    st.markdown(f"##### 📗 {book.book_title} (`{book.subject}`)")
                    st.caption(
                        f"단원 수: **{len(book.units)}개** · 추출 페이지 구간: `{book.page_ranges_spec}` · "
                        f"등록자: @{book.created_by or 'admin'}"
                    )
                with b_col2:
                    schema_dict = {
                        "book_title": book.book_title,
                        "subject": book.subject,
                        "grade": book.grade,
                        "units": [u.model_dump() for u in book.units],
                    }
                    st.download_button(
                        "📦 단원별 JSON 다운로드",
                        data=json.dumps(schema_dict, ensure_ascii=False, indent=2).encode("utf-8"),
                        file_name=f"{sanitize_id_part(book.book_title)}_concept_db.json",
                        mime="application/json",
                        key=f"{key_prefix}_dl_cb_{b_idx}_{book.id}",
                        width="stretch",
                    )
                with b_col3:
                    if st.button(
                        "🗑️ 삭제",
                        key=f"{key_prefix}_del_cb_{b_idx}_{book.id}",
                        width="stretch",
                    ):
                        del_err = storage.delete_concept_book(book.id)
                        if del_err:
                            st.warning(del_err)
                        st.rerun()

                with st.expander(f"📋 `{book.book_title}` 단원별 개념 및 허용 용어 상세 보기 ({len(book.units)}개 단원)", expanded=False):
                    for u_idx, unit in enumerate(book.units, start=1):
                        st.markdown(f"**{u_idx}. {unit.unit_name}** (`{unit.page_range}`)")
                        uc1, uc2 = st.columns(2)
                        with uc1:
                            st.caption("핵심 원리 및 정의 요약 (`core_concepts`)")
                            render_chips(unit.core_concepts, variant="blue", limit=60)
                        with uc2:
                            st.caption("허용 용어·공식·단위 (`allowed_terms_and_units`)")
                            render_chips(unit.allowed_terms_and_units, variant="green", limit=80)
                        st.markdown("---")


if is_admin:
    tab1, tab2, tab3, my_exams_tab, admin_tab = st.tabs(
        [
            "📥 탭 1: 모의고사 수집 & 2022 원리·용어 분해",
            "🎯 탭 2: 개념서 범위 추출 & 상위 문제 개조",
            "📚 탭 3: 공용 개념서·교과서 DB 구축",
            "📁 나의 시험지",
            "🛡️ 관리자: 회원 등급 관리",
        ]
    )
else:
    tab1, tab2, tab3, my_exams_tab = st.tabs(
        [
            "📥 탭 1: 모의고사 수집 & 2022 원리·용어 분해",
            "🎯 탭 2: 개념서 범위 추출 & 상위 문제 개조",
            "📚 탭 3: 공용 개념서·교과서 DB 구축",
            "📁 나의 시험지",
        ]
    )
    admin_tab = None

# ===========================================================================
# TAB 1: 모의고사 수집 & 2022 개정 교육과정 원리·용어 분해 태깅
# ===========================================================================
with tab1:
    st.subheader("1️⃣ 기출 PDF 수집 및 고해상도 자동 크롭 (250 DPI)")
    _render_exam_coverage_board()

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
        prob_up_label = (
            "문제지 폴더 업로드 (폴더 내 모든 PDF 자동 수집)"
            if is_dir_upload
            else "문제지 PDF 업로드 (1개 또는 여러 개 선택 가능)"
        )
        ans_up_label = (
            "해설지 폴더 업로드 (필수 · AI(Gemini 3.5 Flash-Lite) 자동 1:1 매칭)"
            if is_dir_upload
            else "해설지 PDF 업로드 (필수 · 여러 개 선택 시 AI 자동 매칭)"
        )

        col_up1, col_up2 = st.columns(2)
        with col_up1:
            raw_prob_uploads = st.file_uploader(
                prob_up_label,
                type=["pdf"],
                accept_multiple_files=accept_mode,
                key="tab1_prob_pdf_dir" if is_dir_upload else "tab1_prob_pdf_files",
            )
        with col_up2:
            raw_ans_uploads = st.file_uploader(
                ans_up_label,
                type=["pdf"],
                accept_multiple_files=accept_mode,
                key="tab1_ans_pdf_dir" if is_dir_upload else "tab1_ans_pdf_files",
            )

        raw_prob_files = sorted(
            [f for f in (raw_prob_uploads or []) if f.name.lower().endswith(".pdf")],
            key=lambda f: f.name,
        )
        raw_ans_files = sorted(
            [f for f in (raw_ans_uploads or []) if f.name.lower().endswith(".pdf")],
            key=lambda f: f.name,
        )

        prob_files = list(raw_prob_files)
        ans_files = list(raw_ans_files)

        # 문제지 업로드 목록/폴더 안에 해설·정답 PDF가 함께 들어있고 해설지 업로더가 비어있을 때 AI(3.5 Flash-Lite) 자동 분리
        if not raw_ans_files and len(raw_prob_files) >= 2:
            split_prob_idxs, split_ans_idxs, split_source = _resolve_mixed_pdf_split_with_ai(
                raw_prob_files, upload_gemini_key
            )
            if split_ans_idxs and split_prob_idxs:
                split_prob = [raw_prob_files[i] for i in split_prob_idxs]
                split_ans = [raw_prob_files[i] for i in split_ans_idxs]
                split_badge = "🤖 Gemini 3.5 Flash-Lite가 " if split_source == "ai" else "📁 "
                auto_split_ans = st.checkbox(
                    f"{split_badge}문제지 업로드 목록에서 해설/정답 PDF {len(split_ans)}개를 해설지로 자동 분리 매칭 "
                    f"(문제지 {len(split_prob)}개 + 해설지 {len(split_ans)}개)",
                    value=True,
                    key="tab1_auto_split_ans",
                )
                if auto_split_ans:
                    prob_files = split_prob
                    ans_files = split_ans

        is_batch_queue_mode = len(prob_files) >= 2

        if is_batch_queue_mode:
            auto_matches, match_reasons, match_source, match_err = _resolve_auto_answer_matches(
                prob_files,
                ans_files,
                upload_gemini_key,
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
                            _invalidate_auto_pair_cache(pair_sig, upload_gemini_key)
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
                    current_user, creds.gemini_api_key, force_server_key=True
                )
                if not verified_key:
                    st.error(
                        f"서버 업로드·태깅용 API 키가 설정되지 않았습니다 ({verified_key_label}). "
                        "`.streamlit/secrets.toml`의 `GEMINI_API_KEY`를 확인해 주세요."
                    )
                elif not _ensure_token_quota():
                    pass
                else:
                    db_batch_snapshot = storage.load_db()
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
                        if not _ensure_token_quota():
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
                                storage=storage,
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
                                record_exam_upload_for_user(storage, current_user.username)
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
                        upload_gemini_key,
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
                            _invalidate_auto_pair_cache(s_pair_sig, upload_gemini_key)
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
                            upload_gemini_key,
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
                    options=["과학", "수학", "사회", "영어"],
                    key="tab1_subject_cat",
                )
            with col_meta2:
                exam_name_input = st.text_input(
                    "회차명 (자동 조합)",
                    key="meta_title",
                    placeholder="예: 2026년 9월 고1 통합사회",
                )

            db_snapshot = storage.load_db()
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
                            api_key=upload_gemini_key,
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
                current_user, creds.gemini_api_key, force_server_key=True
            )
            if not verified_key:
                st.error(
                    f"업로드·태깅용 API 키가 설정되지 않았습니다 ({verified_key_label}). "
                    "`.streamlit/secrets.toml`의 `GEMINI_API_KEY`를 확인해 주세요."
                )
                return
            if not _ensure_token_quota():
                return
            if not target_q_nums:
                st.warning("태깅할 문항을 최소 1개 이상 선택해 주세요.")
                return

            tagging_model_label = MODEL_LABELS.get(EXAM_TAGGING_MODEL, EXAM_TAGGING_MODEL)
            sync_notice = (
                st.warning(
                    "☁️ **\"Google Drive 동기화 완료\"** 문구가 뜰 때까지 브라우저 창을 닫지 말고 유지해 주세요. "
                    "(전체 PDF를 한 번에 태깅한 뒤 로컬 DB와 Google Drive에 일괄 저장합니다.)"
                )
                if storage.drive_connected
                else st.info(
                    "💾 전체 PDF를 한 번에 태깅한 뒤 서버 영구 디스크(`./exam_db`)에 일괄 저장합니다."
                )
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
                storage=storage,
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
                record_exam_upload_for_user(storage, current_user.username)

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
                    f"🎉 `{meta['exam_name']}` 총 {len(saved_q_nums)}개 문항 태깅 및 영구 저장 완료!"
                    + (" (Google Drive 동기화 완료)" if storage.drive_connected else " (로컬 ./exam_db 저장)")
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
                        for r in storage.load_db()
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
                            b = storage.get_crop_bytes(
                                rec.image_filename,
                                subject_category=rec.subject_category,
                                exam_name=rec.exam_name,
                            )
                            if b:
                                st.image(b, width="stretch")
                        with c_info:
                            _render_question_tag_details(
                                rec,
                                storage=storage,
                                key_prefix="stage2_saved",
                                allow_delete=False,
                            )

    # 저장된 DB 조회 및 과목·회차별 관리
    st.markdown("---")
    st.subheader("3️⃣ 과목별 · 모의고사 회차별 기출 DB 탐색 & 파일 내보내기")
    if storage.drive_connected:
        st.caption(
            f"☁️ Google Drive `{storage.drive_folder_name}/<과목>/<회차명>/` 폴더에 "
            "`기출정리.html`, `기출정리.md`, `문제목록.json`, 크롭 PNG, 원본 PDF가 자동 정리됩니다."
        )
    else:
        st.caption(
            "💾 로컬 `./exam_db/과목별_기출DB/<과목>/<회차명>/` 폴더에 "
            "`기출정리.html`, `기출정리.md`, `문제목록.json`, 크롭 PNG, 원본 PDF가 자동 정리됩니다."
        )

    db_records = storage.load_db()

    if is_admin:
        with st.expander("🧹 관리자: 누적 기출 DB 전체 초기화 (회원 계정은 유지)", expanded=False):
            st.warning("로컬(`./exam_db`) 및 Google Drive에 누적된 기출 문항·크롭 이미지·PDF 파일을 모두 삭제합니다. (`users.json` 회원 정보는 유지됩니다.)")
            confirm_wipe = st.checkbox("정말 모든 기출 DB 파일을 초기화합니다.", key="confirm_db_wipe")
            if st.button("🗑️ 기출 DB 전체 초기화 실행", disabled=not confirm_wipe, key="exec_db_wipe"):
                wipe_res = storage.reset_exam_database(delete_users=False)
                if wipe_res.get("drive_error"):
                    st.error(f"Drive 초기화 중 오류: {wipe_res['drive_error']}")
                else:
                    st.success(
                        f"초기화 완료! (로컬 항목 {wipe_res['removed_local']}개, Drive 항목 {wipe_res['trashed_drive']}개 정리)"
                    )
                st.rerun()

    def _extract_exam_search_attrs(exam_name: str, subj_cat: str) -> dict[str, str]:
        ym = re.search(r"(20\d{2})", exam_name or "")
        mm = re.search(r"(\d{1,2})\s*월", exam_name or "")
        gm = re.search(r"고\s*([123])", exam_name or "")
        return {
            "year": f"{ym.group(1)}년" if ym else "",
            "month": f"{int(mm.group(1))}월" if mm else "",
            "grade": f"고{gm.group(1)}" if gm else "",
            "subject": subj_cat or "",
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
        avail_subjects = ["수학", "사회", "과학", "영어"]

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
                    current_user is not None and current_user.tier in ("PRO", "ULTRA", "ADMIN")
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
                                    cnt = storage.delete_exam(exam_name)
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
                                    cnt = storage.delete_exam(exam_name)
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
                                            rn_cnt = storage.rename_exam(exam_name, new_en)
                                            _build_cached_exam_exports.clear()
                                            st.success(f"회차명을 '{new_en}'(으)로 변경했습니다 ({rn_cnt}문항 반영).")
                                            st.rerun()
                                        else:
                                            st.info("변경할 회차명이 기존과 같습니다.")

                        if retag_clicked and can_retag_exam:
                            corrected_en = normalize_exam_title_year(exam_name)
                            if corrected_en and corrected_en != exam_name:
                                storage.rename_exam(exam_name, corrected_en)
                                exam_name = corrected_en
                            subj_cat = infer_subject_category("", exam_title=exam_name, default=subj_cat)
                            verified_key, verified_key_label = resolve_active_gemini_api_key(
                                current_user, creds.gemini_api_key, force_server_key=True
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
                                retag_prob_bytes = storage.get_pdf_bytes(
                                    retag_prob_name,
                                    subject_category=subj_cat,
                                    exam_name=exam_name,
                                )
                                retag_ans_bytes = storage.get_pdf_bytes(
                                    retag_ans_name,
                                    subject_category=subj_cat,
                                    exam_name=exam_name,
                                )
                                if not retag_ans_bytes:
                                    st.error(
                                        f"🚫 **'{exam_name}'** 회차에 저장된 해설지 PDF가 없어 다시 태깅할 수 없습니다. "
                                        "기존 회차를 삭제한 뒤 문제지·해설지 PDF를 함께 다시 업로드해 주세요."
                                    )
                                elif not _ensure_token_quota():
                                    pass
                                else:
                                    saved_crops_by_q = {
                                        r.q_num: CroppedQuestion(
                                            q_num=r.q_num,
                                            page_num=1,
                                            png_bytes=(
                                                storage.get_crop_bytes(
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
                                            storage=storage,
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

                        # 회차별 정리 파일 & 원본 PDF 다운로드 바 (캐싱 적용)
                        records_sig = f"{len(exam_recs)}:{','.join(r.id for r in exam_recs)}"
                        html_bytes, md_bytes, ai_jsonl_bytes, json_bytes = _build_cached_exam_exports(
                            storage,
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
                                storage.get_pdf_bytes(prob_pdf_name, subject_category=subj_cat, exam_name=exam_name)
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
                                storage.get_pdf_bytes(ans_pdf_name, subject_category=subj_cat, exam_name=exam_name)
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

                        with st.expander(f"📋 '{exam_name}' 문항 상세 보기 ({len(exam_recs)}문항)", expanded=False):
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
                                    img_b = storage.get_crop_bytes(
                                        rec.image_filename,
                                        subject_category=rec.subject_category,
                                        exam_name=rec.exam_name,
                                    )
                                    if img_b:
                                        st.image(img_b, width="stretch")
                                with col_b:
                                    _render_question_tag_details(
                                        rec,
                                        storage=storage,
                                        key_prefix="db_search",
                                        allow_delete=True,
                                    )
                                st.markdown("---")


# ===========================================================================
# TAB 2: 개념서·교과서 PDF 분석 (수학·사회·과학) & 영어 지문 수능형 변형 엔진
# ===========================================================================
def _resolve_tab2_english_rewrite_model(key_suffix: str = "eng") -> tuple[str, bool]:
    """탭 2 영어 지문 변형 모델을 전 등급 동일하게 초경량 Gemini 3.5 Flash-Lite로 통일한다."""
    rw_model = EXAM_TAGGING_MODEL
    use_precision_mode = False
    st.session_state["active_rewrite_model"] = rw_model
    st.session_state["active_eng_precision_mode"] = False
    return rw_model, use_precision_mode


def _execute_english_transform_jobs(
    jobs: list[dict[str, Any]],
    rewrite_model: str,
    source_summary_label: str,
    use_precision_mode: bool = False,
) -> None:
    """(영어 지문, 목표 수능 유형) 작업 목록을 Gemini 3.5 Flash-Lite 단일 모델 및 등급별 월간 토큰 한도 하에 실행한다."""
    if not jobs:
        st.warning("변형할 영어 지문과 목표 수능 유형을 최소 1개 이상 선택해 주세요.")
        return

    if not _ensure_token_quota():
        return

    user_max_jobs = get_max_batch_questions(current_user)
    if user_max_jobs is not None and len(jobs) > user_max_jobs:
        st.warning(
            f"⚠️ 현재 등급({TIER_LABELS.get(current_user.tier, current_user.tier)})은 1회 실행당 최대 **{user_max_jobs}문항**까지만 생성됩니다. "
            f"(요청 {len(jobs)}건 중 앞의 {user_max_jobs}문항만 실행)"
        )
        jobs = jobs[:user_max_jobs]

    rewrite_model = FLASH_LITE_MODEL
    use_precision_mode = False

    rewrite_api_key, rewrite_key_label = resolve_active_gemini_api_key(
        current_user, creds.gemini_api_key, model_name=rewrite_model
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

    with _usage_scope():
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
                    storage=storage,
                    use_cache=True,
                    username=current_user.username if current_user is not None else "",
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

    consume_extraction_quota(storage, current_user.username, len(assembled))

    st.session_state["matched_items"] = assembled
    st.session_state["match_diagnostics"] = []
    st.session_state["concept_source_label"] = source_summary_label
    st.session_state["generated_exam_pdf"] = None
    st.session_state["user_answers"] = {}
    st.session_state["graded"] = False
    st.session_state["ai_explanations"] = {}
    st.session_state["ai_chats"] = {}

    rem_msg = ""
    fresh_u = get_user_by_username(storage, current_user.username)
    if fresh_u is not None:
        m_lim = get_monthly_token_limit(fresh_u)
        m_rem = get_remaining_monthly_tokens(fresh_u)
        if m_lim is not None and m_rem is not None:
            rem_msg = f" | 이번 달 잔여 토큰: {m_rem:,} / {format_token_count(m_lim)}"

    st.success(f"🎯 총 **{len(assembled)}문항**의 수능형 영어 변형 문제 출제가 완료되었습니다! 아래 모드 A(시험지 PDF) 또는 모드 B(웹 바로 풀기)에서 확인하세요.{rem_msg}")


def _render_tab2_english_transform_section() -> None:
    """탭 2 영어 과목 전용: 기출 영어 지문 검색/직접 입력 및 수능 11개 독해 유형 변형 UI."""
    st.subheader("1️⃣ 영어 지문 선택 & 수능 독해 유형 변형 엔진")
    rewrite_model, use_precision_mode = _resolve_tab2_english_rewrite_model(key_suffix="eng")
    user_max_jobs = get_max_batch_questions(current_user)
    max_limit_badge = f"1회 최대 **{user_max_jobs}문항**" if user_max_jobs is not None else "1회 문항 수 **제한 없음**"

    eng_tab_db, eng_tab_direct = st.tabs(
        [
            "📂 기출 DB 영어 지문에서 선택 (회차·원래 유형별 검색)",
            "✍️ 새 영어 지문 직접 붙여넣기",
        ]
    )

    with eng_tab_db:
        all_db = storage.load_db()
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
        else:
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
                    key="tab2_eng_filter_exam",
                )
            with f_c2:
                sel_eng_orig_type = st.selectbox(
                    "원래 문항 유형 필터",
                    options=["전체"] + avail_eng_types,
                    key="tab2_eng_filter_type",
                )
            with f_c3:
                eng_kw = st.text_input(
                    "주제·어휘·지문 키워드 검색",
                    placeholder="예: artificial intelligence, climate, 감정...",
                    key="tab2_eng_filter_kw",
                )

            opt_row1, opt_row2 = st.columns([1.3, 1.2])
            with opt_row1:
                include_excluded_q = st.checkbox(
                    "목적·심경·도표·내용일치·실용문·장문(18, 19, 25, 26, 27, 28, 43, 44, 45번) 지문도 목록에 포함",
                    value=False,
                    key="tab2_eng_include_excluded_q",
                )
            with opt_row2:
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
                    jobs=direct_jobs,
                    rewrite_model=rewrite_model,
                    source_summary_label=base_src,
                    use_precision_mode=use_precision_mode,
                )


def _render_tab2_concept_source_tabs() -> None:
    st.subheader("1️⃣ 기준 개념서 범위 설정 (공용 개념서 DB 선택 또는 PDF 직접 업로드)")

    source_tab_db, source_tab_pdf = st.tabs(
        [
            "📚 공용 개념서·교과서 DB에서 단원 선택 (Pro · ADMIN 전용)",
            "📄 개념서·교과서 PDF 1회성 직접 업로드",
        ]
    )

    with source_tab_db:
        has_concept_db_access = can_access_shared_concept_db(current_user)
        shared_books = storage.load_concept_books()

        if not has_concept_db_access:
            st.warning(
                "🔒 **공용 개념서·교과서 DB 불러오기는 Pro(VIP) · ADMIN 등급 전용 기능입니다.**\n\n"
                f"현재 등급(**{TIER_LABELS.get(current_user.tier, current_user.tier)}**)에서는 잠금 처리되어 있으며, "
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

                with st.form(key=f"tab2_unit_select_form_{chosen_book.id}"):
                    st.markdown("**✅ 출제 및 개조 기준으로 사용할 단원을 체크한 뒤 아래 적용 버튼을 누르세요:**")
                    select_all_units = st.checkbox(
                        "전체 단원 일괄 선택 (체크 후 적용 시 전체 단원 반영)",
                        value=False,
                        key=f"sel_all_units_{chosen_book.id}",
                    )
                    selected_units_list: list[ConceptUnit] = []
                    for u_i, unit_obj in enumerate(chosen_book.units):
                        checked = st.checkbox(
                            f"{unit_obj.unit_name} ({unit_obj.page_range}) — 핵심개념 {len(unit_obj.core_concepts)}개 · 허용용어 {len(unit_obj.allowed_terms_and_units)}개",
                            value=False,
                            key=f"unit_chk_{chosen_book.id}_{u_i}",
                        )
                        if checked:
                            selected_units_list.append(unit_obj)

                    apply_units_btn = st.form_submit_button(
                        "📚 선택한 단원으로 지식 화이트리스트 즉시 적용",
                        type="primary",
                        width="stretch",
                    )

                if apply_units_btn:
                    effective_units = (
                        list(chosen_book.units) if select_all_units else selected_units_list
                    )
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

            with st.expander("➕ 새 개념서·교과서 PDF를 공용 DB에 바로 추가하기", expanded=not bool(shared_books)):
                _render_concept_book_builder_section(key_prefix="tab2_inline")

    with source_tab_pdf:
        concept_file = st.file_uploader(
            "통합과학·통합사회·공통수학 등 개념서/교과서 PDF 업로드",
            type=["pdf"],
            key="tab2_concept_pdf",
        )

        concept_pdf_bytes: bytes | None = concept_file.getvalue() if concept_file is not None else None
        if concept_pdf_bytes is not None:
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

                if st.button("🔍 1단계: 업로드된 PDF 지식 화이트리스트 추출", type="primary"):
                    scope_api_key, scope_key_label = resolve_active_gemini_api_key(
                        current_user, creds.gemini_api_key, model_name=CONCEPT_SCOPE_MODEL
                    )
                    if not scope_api_key:
                        st.error(f"사용 가능한 Gemini API 키({scope_key_label})가 설정되지 않았습니다.")
                    elif not _ensure_token_quota():
                        pass
                    else:
                        with st.spinner(f"PDF ({range_disp})의 순수 개념 본문·도표·용어를 정독해 화이트리스트를 구축 중..."):
                            sliced_bytes, sliced_text, slice_stats = extract_concept_pages_by_ranges(
                                concept_pdf_bytes, parsed_direct_ranges
                            )
                            try:
                                with _usage_scope():
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
                                st.success("✅ PDF 지식 화이트리스트 추출이 완료되었습니다!")
                            except Exception as exc:
                                st.error(f"화이트리스트 추출 실패: {exc}")


with tab2:
    tab2_domain_mode = st.radio(
        "작업 과목 모드 선택",
        options=[
            "📚 수학 · 사회 · 과학 (개념서 범위 추출 & 상위 용어 개조)",
            "🔤 영어 (기출 지문 선택 & 수능 독해 유형 변형)",
        ],
        horizontal=True,
        key="tab2_domain_mode",
    )
    is_tab2_english_mode = tab2_domain_mode.startswith("🔤")
    if is_tab2_english_mode:
        _render_tab2_english_transform_section()
    else:
        _render_tab2_concept_source_tabs()

    # 추출된 화이트리스트 표시
    whitelist: ConceptWhitelist | None = st.session_state.get("concept_whitelist")
    if (not is_tab2_english_mode) and (whitelist is not None):
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

        is_premium = can_use_premium_features(current_user)
        match_model = EXAM_TAGGING_MODEL
        rewrite_model = EXAM_TAGGING_MODEL
        st.session_state["active_rewrite_model"] = rewrite_model
        tab2_max_batch = get_max_batch_questions(current_user)

        with st.container(border=True):
            with st.form("tab2_extract_options_form", border=False):
                opt_c1, opt_c2, opt_c3 = st.columns(3)
                with opt_c1:
                    filter_subject = st.selectbox("과목 필터", ["전체", "수학", "사회", "과학"], index=0)
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
                    "🚀 기출 매칭 & 개념서 맞춤 문제 추출·개조 실행",
                    type="primary",
                )

            if run_extract_btn:
                db_list = storage.load_db()
                match_api_key, match_key_label = resolve_active_gemini_api_key(
                    current_user, creds.gemini_api_key, model_name=match_model
                )
                rewrite_api_key, rewrite_key_label = resolve_active_gemini_api_key(
                    current_user, creds.gemini_api_key, model_name=rewrite_model
                )
                if not match_api_key:
                    st.error(f"사용 가능한 Gemini API 키({match_key_label})가 설정되지 않았습니다.")
                elif not _ensure_token_quota():
                    pass
                elif not db_list:
                    st.warning("저장된 기출 DB가 비어 있습니다. 먼저 [탭 1]에서 모의고사를 수집·태깅해 주세요.")
                else:
                    # 1차 메타데이터 필터링
                    candidates: list[QuestionRecord] = []
                    for r in db_list:
                        if filter_subject != "전체" and r.subject_category != filter_subject:
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
                        with _usage_scope():
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
                                assembled: list[dict[str, Any]] = []
                                prog = st.progress(0.0)
                                stat = st.empty()
                                sliced_text_ctx = st.session_state.get("concept_sliced_text", "")
                                fallback_source_ctx = st.session_state.get("concept_source_label", "")

                                prepared_items: list[tuple[QuestionRecord, Any, bytes | None, str]] = []
                                rewrite_indices: list[int] = []
                                for idx, (q, j) in enumerate(selected_pairs):
                                    crop_bytes = storage.get_crop_bytes(
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
                                    storage, current_user.username, len(assembled)
                                )
                                rewritten_cnt = sum(1 for it in assembled if it["rewritten"] is not None)
                                st.session_state["matched_items"] = assembled
                                st.session_state["generated_exam_pdf"] = None
                                st.session_state["user_answers"] = {}
                                st.session_state["graded"] = False
                                st.session_state["ai_explanations"] = {}
                                st.session_state["ai_chats"] = {}
                                st.session_state["_tab2_last_extract_summary"] = (
                                    len(assembled),
                                    sum(1 for it in assembled if it["rewritten"] is None),
                                    rewritten_cnt,
                                )

                        last_sum = st.session_state.pop("_tab2_last_extract_summary", None)
                        if last_sum is not None:
                            tot_c, orig_c, rw_c = last_sum
                            rem_msg = ""
                            fresh_u = get_user_by_username(storage, current_user.username)
                            if fresh_u is not None:
                                m_lim = get_monthly_token_limit(fresh_u)
                                m_rem = get_remaining_monthly_tokens(fresh_u)
                                if m_lim is not None and m_rem is not None:
                                    rem_msg = f" | 이번 달 잔여 토큰: {m_rem:,} / {format_token_count(m_lim)}"
                            st.success(
                                f"🎯 총 {tot_c}문항 추출 완료! "
                                f"(원문 그대로 {orig_c}문항 · 개념서 기준 개조 {rw_c}문항){rem_msg}"
                            )

    # -----------------------------------------------------------------------
    # 결과 활용 2가지 모드 (모드 A: PDF 다운로드 / 모드 B: 웹에서 풀기 + AI 해설)
    # -----------------------------------------------------------------------
    matched_items: list[dict[str, Any]] = st.session_state.get("matched_items") or []
    if (not is_tab2_english_mode) and st.session_state.get("match_diagnostics"):
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

    if matched_items:
        st.markdown("---")
        if len(matched_items) >= 2:
            shuf_c1, shuf_c2, shuf_c3 = st.columns([1.15, 1.05, 2.3])
            with shuf_c1:
                if st.button("🔀 문항 순서 섞기", key="tab2_shuffle_matched_items", width="stretch"):
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
            if not can_use_premium_features(current_user):
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
                    default_pdf_title = (
                        "수능·모의고사 영어 지문 맞춤형 변형 모의고사"
                        if is_all_eng_matched
                        else "2022 개정 교육과정 개념서 맞춤형 기출 모의고사"
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
                            if is_all_eng_matched:
                                st.caption("📌 영어 변형 시험지는 원본 이미지를 제외하고 기출 출처(회차·번호)만 표기됩니다.")
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

                    if build_pdf_btn:
                        with st.spinner("PDF 조판 및 뒷장 정답·해설 페이지 생성 중..."):
                            pdf_bytes = generate_custom_exam_pdf(
                                items=matched_items,
                                exam_title=pdf_title.strip() or "맞춤형 기출 모의고사",
                                source_label=st.session_state.get("concept_source_label", "업로드 개념서 PDF"),
                                whitelist=whitelist,
                                two_column=(layout_choice == "2단 모의고사 스타일"),
                                include_original_for_rewritten=include_orig_fig,
                            )
                            st.session_state["generated_exam_pdf"] = pdf_bytes
                            st.session_state["generated_exam_pdf_sig"] = items_sig
                            _record_english_seen_for_user(matched_items)
                            _auto_save_current_exam(
                                matched_items,
                                title=pdf_title.strip() or "맞춤형 기출 모의고사",
                                pdf_bytes=pdf_bytes,
                                sig=_build_exam_sig(matched_items),
                            )
                        st.success("✅ 시험지 PDF(뒷장 정답·개념서 맞춤 해설 포함)가 생성되고 '나의 시험지'에 자동 저장되었습니다!")

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
            _render_mode_b_solve_fragment(
                matched_items=matched_items,
                whitelist=whitelist,
                active_gemini_key=active_gemini_key,
                active_key_label=active_key_label,
                selected_model=selected_model,
                rewrite_model=st.session_state.get("active_rewrite_model") or selected_model,
            )

# ===========================================================================
# TAB 3: 관리자 / Pro 전용 공용 개념서·교과서 DB 구축 및 관리
# ===========================================================================
with tab3:
    _render_ncic_curriculum_manager(key_prefix="tab3")
    _render_concept_book_builder_section(key_prefix="tab3")

with my_exams_tab:
    _render_my_exams_tab()

if admin_tab is not None:
    with admin_tab:
        _render_admin_panel()
