from __future__ import annotations

import concurrent.futures
import os
import re
import time
from datetime import datetime
from typing import Any, Callable

import streamlit as st

from src.config import EXAM_NO_ANSWER_SOLVER_MODEL, EXAM_TAGGING_MODEL, MODEL_LABELS
from src.curriculum_2022 import build_tagging_grounding_context
from src.gemini import (
    analyze_english_exam_pdf_batch_with_gemini,
    analyze_english_question_with_gemini,
    analyze_exam_pdf_batch_with_gemini,
    analyze_korean_exam_pdf_batch_with_gemini,
    analyze_korean_question_with_gemini,
    analyze_question_with_gemini,
    build_gemini_client,
    detect_scanned_question_boxes,
    prepare_pdf_part,
    submit_with_usage_context,
)
from src.models import (
    QuestionRecord,
    infer_subject_category,
    normalize_english_group_id,
    normalize_exam_title_year,
    normalize_exam_year,
    normalize_korean_group_id,
)
from src.pdf_parser import (
    CroppedQuestion,
    crop_normalized_box,
    extract_answer_snippets_from_pdf,
    get_page_text_lengths,
    get_pdf_page_count,
    parse_exam_pdf,
    render_pdf_page_png,
)
from src.storage import StorageManager
from src.ui.common import (
    _build_cached_exam_exports,
    _ensure_token_quota,
    _usage_scope,
    sanitize_id_part,
)
from src.ui.tab1_pairing import (  # noqa: F401
    _build_pairing_preview_items,
    _compute_pairing_signature,
    _compute_pairing_widget_hash,
    _detect_single_exam_cover,
    _invalidate_auto_pair_cache,
    _pdf_display_basename,
    _resolve_auto_answer_matches,
    _resolve_mixed_pdf_split_with_ai,
)

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
    is_korean_hint = "국어" in (subject_hint or "")
    crops, has_text = parse_exam_pdf(
        prob_bytes,
        dpi=220,
        split_passage_and_question=is_korean_hint,
    )
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
            )
            pending_pdf_files.append(prob_fn)
        if meta.get("ans_bytes"):
            ans_fn = f"{safe_exam}__해설지.pdf"
            storage.save_pdf_file(
                ans_fn,
                meta["ans_bytes"],
                subject_category=meta["subject_category"],
                exam_name=meta["exam_name"],
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
    is_korean_exam = meta["subject_category"] == "국어"

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
                    question_texts={c.q_num: (c.extracted_text or "") for c in target_crops},
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
    elif is_korean_exam:
        eng_group_passages = {}
        if meta.get("prob_bytes"):
            _status(
                f"📖 국어 문제지 PDF({len(target_crops)}문항 · 독서/문학 공통 지문 세트) 일괄 분석 중 (모델: {tagging_model_label})..."
            )
            try:
                batch_map = analyze_korean_exam_pdf_batch_with_gemini(
                    api_key=api_key,
                    model_name=tagging_model,
                    prob_pdf_bytes=meta["prob_bytes"],
                    target_q_nums=[c.q_num for c in target_crops],
                    exam_name=meta["exam_name"],
                    answer_snippets=answer_snippets,
                    answer_pdf_part=answer_pdf_part,
                    status_callback=_status,
                    parsed_crops=target_crops,
                )
                extracted_by_q.update(batch_map)
                _log(
                    f"⚡ 국어 PDF 일괄 추출 완료: {len(batch_map)}/{len(target_crops)}문항 "
                    f"({time.perf_counter() - batch_t0:.1f}초 · {tagging_model_label})"
                )
            except Exception as exc:
                _log(f"⚠️ 국어 PDF 일괄 추출 실패 → 개별 문항 보완 모드로 전환: {exc}")
        _prog(0.65)

        missing_crops = [c for c in target_crops if c.q_num not in extracted_by_q]
        if missing_crops:
            _status(f"🔁 누락된 국어 {len(missing_crops)}개 문항 보완 추출 중 ({tagging_model_label})...")

            def _analyze_kor_worker(cq_item: CroppedQuestion) -> tuple[CroppedQuestion, Any, float]:
                t_start = time.perf_counter()
                q_ans_hint = answer_snippets.get(cq_item.q_num, "")
                extracted_out = analyze_korean_question_with_gemini(
                    api_key=api_key,
                    model_name=tagging_model,
                    crop_png_bytes=cq_item.png_bytes,
                    q_num=cq_item.q_num,
                    exam_name=meta["exam_name"],
                    answer_pdf_part=answer_pdf_part,
                    answer_text_hint=q_ans_hint,
                    question_text_hint=cq_item.extracted_text or "",
                    shared_passage_hint=(getattr(cq_item, "shared_passage_text", "") or ""),
                )
                return cq_item, extracted_out, time.perf_counter() - t_start

            done_cnt = 0
            total_missing = len(missing_crops)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_cq = {
                    submit_with_usage_context(executor, _analyze_kor_worker, cq): cq
                    for cq in missing_crops
                }
                for fut in concurrent.futures.as_completed(future_to_cq):
                    cq = future_to_cq[fut]
                    done_cnt += 1
                    try:
                        _, extracted, elapsed_s = fut.result()
                        extracted_by_q[cq.q_num] = extracted
                        _log(f"✅ 국어 {cq.q_num}번 보완 추출 완료 ({extracted.korean_section} · 정답: {extracted.answer} · {elapsed_s:.1f}초)")
                    except Exception as exc:
                        err_msg = f"{cq.q_num}번 오류 (건너뜀): {exc}"
                        failed_q_nums.append(cq.q_num)
                        errors.append(err_msg)
                        _log(f"❌ {err_msg}")
                    _prog(0.65 + 0.20 * (done_cnt / max(1, total_missing)))

        # 전 영역(독서 · 문학 · 화법과 작문 · 언어와 매체) 공통 지문 세트 동기화
        kor_group_passages: dict[str, str] = {}
        for q_n, ext in list(extracted_by_q.items()):
            gid = normalize_korean_group_id(getattr(ext, "passage_group_id", ""), q_n)
            p_txt = (getattr(ext, "passage_text", "") or "").strip()
            if gid and len(p_txt) > len(kor_group_passages.get(gid, "")):
                kor_group_passages[gid] = p_txt
    else:
        eng_group_passages = {}
        kor_group_passages = {}
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
                standard_code=(getattr(extracted, "standard_code", "") or ""),
                standard_codes=list(getattr(extracted, "standard_codes", []) or []),
                grade_evidence="",
                needs_review=bool(getattr(extracted, "needs_review", False)),
                review_notes=list(getattr(extracted, "review_notes", []) or []),
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
        elif is_korean_exam:
            kor_sec = getattr(extracted, "korean_section", "") or "독서"
            kor_sub = (getattr(extracted, "korean_subgenre", "") or "").strip()
            kor_qtype = (getattr(extracted, "question_type", "") or "").strip() or kor_sec
            curr_subj = f"{kor_sec}({kor_sub})" if kor_sub else kor_sec
            p_group = normalize_korean_group_id(
                getattr(extracted, "passage_group_id", "") or getattr(cq, "passage_group_id", ""),
                cq.q_num,
            )
            raw_p_text = (getattr(extracted, "passage_text", "") or "").strip()
            if p_group and len(raw_p_text) < len(kor_group_passages.get(p_group, "")) * 0.7:
                raw_p_text = kor_group_passages[p_group]
            if not raw_p_text:
                raw_p_text = (getattr(cq, "shared_passage_text", "") or getattr(cq, "extracted_text", "") or "").strip()
            p_topic = (getattr(extracted, "passage_topic", "") or "").strip()
            k_vocab = [v.strip() for v in (getattr(extracted, "key_vocabulary", []) or []) if v and v.strip()]

            p_img_filename = old_rec.passage_image_filename if (old_rec is not None and getattr(old_rec, "passage_image_filename", "")) else ""
            cq_passage_png = getattr(cq, "passage_png_bytes", b"") or b""
            if cq_passage_png and p_group:
                safe_grp = re.sub(r"[^0-9_~]+", "", p_group).replace("~", "_")
                p_img_filename = f"{safe_exam}__P{safe_grp}.png"
                if p_img_filename not in pending_crop_files:
                    storage.save_crop_png(
                        p_img_filename,
                        cq_passage_png,
                        subject_category=meta["subject_category"],
                        exam_name=meta["exam_name"],
                    )
                    pending_crop_files.append(p_img_filename)

            record = QuestionRecord(
                id=q_id,
                exam_name=meta["exam_name"],
                q_num=cq.q_num,
                subject_category="국어",
                curriculum_subject=curr_subj,
                unit_name=kor_qtype,
                min_grade=1,
                core_principles=[p_topic] if p_topic else [kor_qtype],
                required_terms=k_vocab,
                advanced_terms=[],
                basic_terms=k_vocab,
                standard_code=(getattr(extracted, "standard_code", "") or ""),
                standard_codes=list(getattr(extracted, "standard_codes", []) or []),
                grade_evidence="",
                needs_review=bool(getattr(extracted, "needs_review", False)),
                review_notes=list(getattr(extracted, "review_notes", []) or []),
                is_single_concept=True,
                answer=extracted.answer,
                official_explanation=extracted.official_explanation,
                question_text=re.sub(r"\s+", " ", (cq.extracted_text or "")).strip(),
                english_section="",
                english_question_type=kor_qtype,
                passage_text=raw_p_text,
                passage_group_id=p_group,
                passage_topic=p_topic,
                key_vocabulary=k_vocab,
                image_filename=img_filename,
                passage_image_filename=p_img_filename,
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

    _status("💾 전체 문항 및 과목·회차별 정리 파일 로컬 영구 저장 중...")
    sync_err = storage.save_db(
        sorted_all_records,
        target_exams={meta["exam_name"]},
        sync_local_organized=True,
    )
    if newly_saved:
        _build_cached_exam_exports.clear()
    _prog(1.0)
    total_batch_s = time.perf_counter() - batch_t0
    _log(f"🏁 전체 PDF 태깅 및 로컬 DB 일괄 저장 완료 ({len(newly_saved)}문항 · 총 {total_batch_s:.1f}초)")
    return {
        "newly_saved": newly_saved,
        "failed_q_nums": sorted(set(failed_q_nums)),
        "errors": errors,
        "sync_err": sync_err or "",
        "elapsed_s": total_batch_s,
    }

