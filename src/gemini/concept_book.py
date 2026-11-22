from __future__ import annotations

import concurrent.futures
import json
import random
import re
import time
from typing import Any, Callable

from google import genai
from google.genai import types

from src.config import DEFAULT_MODEL
from src.gemini.core import (
    _build_client,
    _generate_content_with_fallback,
    _is_fatal_config_or_auth_error,
    _is_rate_limit_error,
    _is_server_overload_error,
    _parse_structured,
    optimize_image_for_vision,
    prepare_pdf_part,
    submit_with_usage_context,
)
from src.models import (
    AutoConceptPageDetectionOutput,
    AutoConceptRangeItem,
    BatchMatchResult,
    ConceptBookCoverMetadata,
    ConceptBookExtractionOutput,
    ConceptBookRecord,
    ConceptUnit,
    ConceptWhitelist,
    QuestionMatchJudgment,
    QuestionRecord,
    TocExtractionOutput,
)
from src.pdf_parser import extract_specific_pages_slice


def _extract_concept_toc_with_ai(
    client: genai.Client,
    model_name: str,
    pdf_bytes: bytes,
    page_catalog: list[dict[str, Any]],
    has_text_layer: bool,
    book_title: str = "",
    subject_and_grade: str = "",
    status_callback: Callable[[str], None] | None = None,
) -> TocExtractionOutput:
    """
    개념서·교과서 PDF 앞부분(p.1~15) 및 목차 페이지를 1차로 분석해
    전체 대단원/중단원 목록(toc_units)과 표지·구성안내·목차가 끝나는 마지막 실제 PDF 페이지(last_toc_page)를 추출한다.
    """
    total_pages = len(page_catalog)
    front_pages = set(range(1, min(total_pages, 15) + 1))
    for entry in page_catalog:
        if "목차" in (entry.get("noise_hint") or "") or "차례" in (entry.get("snippet") or "")[:60]:
            front_pages.add(entry["page"])
    target_pages = sorted(front_pages)[:18]
    if not target_pages:
        return TocExtractionOutput()

    if status_callback is not None:
        status_callback(
            f"📑 1단계: 표지·목차(차례) 경계 및 단원 분석 중 ({target_pages[0]}~{target_pages[-1]}쪽 · 모델: {model_name})..."
        )

    toc_system = (
        "당신은 한국 교과서·개념서(완자, 개념원리, 수능특강 등)의 앞부분(표지·차례)을 정밀 추출하는 AI입니다.\n"
        "제공된 PDF 앞부분에서 목차/차례 부분을 찾아 **모든 대단원과 중단원**을 순서대로 나열하고, "
        "표지·교재 구성 안내·차례(목차)가 모두 끝나고 1단원 개념 본문이 시작되기 직전의 마지막 비-개념 PDF 쪽수(`last_toc_page`)를 판별하세요.\n"
        "- last_toc_page: 반드시 책 하단 인쇄 쪽수가 아닌 `[p.X]` 또는 빨간색 `[ACTUAL_PDF_PAGE: X]` 라벨 기준의 실제 PDF 페이지 번호 정수\n"
        "- unit_name: 목차에 적힌 단원명 (예: 'Ⅰ. 물질의 규칙성과 결합 - 1. 지구와 생명체를 이루는 원소의 생성')\n"
        "- book_page_hint: 목차에 적힌 책 쪽수 (있으면 적고, 없으면 빈 문자열)"
    )

    try:
        if has_text_layer:
            lines = []
            by_page = {e["page"]: e for e in page_catalog}
            for p_num in target_pages:
                e = by_page.get(p_num, {})
                snip = (e.get("snippet") or "").replace("\n", " | ")[:450]
                lines.append(f"[ACTUAL_PDF_PAGE: {p_num}] {snip}")
            prompt = (
                f"- 교재명: {book_title or '(미입력)'} | 과목/학년: {subject_and_grade or '(미입력)'}\n\n"
                "아래 앞부분 페이지 발췌에서 표지/구성안내/목차가 끝나는 `last_toc_page`와 전체 단원 목록(`toc_units`)을 JSON으로 반환하세요.\n\n"
                + "\n".join(lines)
            )
            contents: Any = prompt
        else:
            slice_b, _ = extract_specific_pages_slice(
                pdf_bytes,
                target_pages[:15],
                stamp_actual_page=True,
            )
            pdf_part = prepare_pdf_part(client, slice_b, display_name="concept_toc_front.pdf")
            prompt = (
                f"- 교재명: {book_title or '(미입력)'} | 과목/학년: {subject_and_grade or '(미입력)'}\n"
                "각 페이지 상단에 표시된 `[ACTUAL_PDF_PAGE: 번호]`를 기준으로 표지·교재구성·목차가 끝나는 "
                "`last_toc_page`와 전체 대단원/중단원 목록(`toc_units`)을 JSON으로 반환하세요."
            )
            contents = [pdf_part, prompt]

        resp = _generate_content_with_fallback(
            client=client,
            model_name=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=toc_system,
                response_mime_type="application/json",
                response_schema=TocExtractionOutput,
                temperature=0.0,
            ),
            status_callback=status_callback,
        )
        parsed = _parse_structured(resp, TocExtractionOutput)
        parsed.toc_units = [u for u in parsed.toc_units if u.unit_name.strip()]
        return parsed
    except Exception:
        return TocExtractionOutput()


def _is_same_unit_hint(hint_a: str, hint_b: str) -> bool:
    """두 구간이 같은 단원(대단원/중단원)에 속하는지 판별해 2쪽 이내 짧은 간격 병합 여부를 결정한다."""
    a = re.sub(r"\s+", " ", (hint_a or "").strip())
    b = re.sub(r"\s+", " ", (hint_b or "").strip())
    if not a or not b:
        return True
    if a == b or a in b or b in a:
        return True

    # 대단원 기호/번호 접두어 비교 (예: 'Ⅰ.', 'Ⅱ.', '1.', '2.')
    maj_a = re.match(r"^([ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]|\d{1,2})\s*[.\-)\s]", a)
    maj_b = re.match(r"^([ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]|\d{1,2})\s*[.\-)\s]", b)
    if maj_a and maj_b:
        if maj_a.group(1) != maj_b.group(1):
            return False
        # 같은 대단원 번호이고 앞부분 주제가 비슷하면 동일 단원
        return True

    # 하이픈/대시 앞의 상위 단원명 비교
    pref_a = re.split(r"\s*[-–—:]\s*", a)[0].strip()
    pref_b = re.split(r"\s*[-–—:]\s*", b)[0].strip()
    if pref_a and pref_b and (pref_a == pref_b or pref_a in pref_b or pref_b in pref_a):
        return True

    toks_a = {t for t in re.split(r"[\s·,.\-\(\)]+", a) if len(t) >= 2}
    toks_b = {t for t in re.split(r"[\s·,.\-\(\)]+", b) if len(t) >= 2}
    return bool(toks_a & toks_b)


def _coalesce_concept_ranges(
    items: list[AutoConceptRangeItem],
    max_same_unit_gap: int = 2,
) -> list[AutoConceptRangeItem]:
    """정렬된 개념 구간 리스트에서 겹치거나, 같은 단원 내 2쪽 이내로 떨어진 짧은 간격을 하나로 병합한다."""
    if not items:
        return []
    ordered = sorted(items, key=lambda x: (x.start_page, x.end_page))
    merged: list[AutoConceptRangeItem] = [ordered[0].model_copy()]
    for cur in ordered[1:]:
        prev = merged[-1]
        gap = cur.start_page - prev.end_page - 1
        same_unit = _is_same_unit_hint(prev.unit_hint, cur.unit_hint)
        if gap <= 0 or (same_unit and gap <= max_same_unit_gap):
            prev.end_page = max(prev.end_page, cur.end_page)
            if cur.unit_hint and cur.unit_hint not in prev.unit_hint and len(prev.unit_hint) < 40:
                pass
            if gap > 0:
                prev.reason = (prev.reason or "개념 설명") + f" (단원 내 {gap}쪽 간격 병합)"
        else:
            merged.append(cur.model_copy())
    return merged


def _run_chunk_with_backoff(
    fn: Callable[[], AutoConceptPageDetectionOutput],
    chunk_label: str,
    max_retries: int = 2,
    status_callback: Callable[[str], None] | None = None,
) -> AutoConceptPageDetectionOutput:
    """
    429, 타임아웃, 503, JSON 파싱 실패 등 일시 오류 발생 시 지수 백오프(2s→4s)로 최소 2회 재시도한다.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            if _is_fatal_config_or_auth_error(msg) or attempt >= max_retries:
                raise
            delay = min(10.0, 2.0 * (2**attempt) + random.uniform(0.15, 0.75))
            if status_callback is not None:
                status_callback(
                    f"⚠️ {chunk_label} 일시 오류({type(exc).__name__}) → {delay:.1f}초 후 재시도 ({attempt + 1}/{max_retries})..."
                )
            time.sleep(delay)
    raise RuntimeError(str(last_exc))


def detect_concept_pages_with_ai(
    api_key: str,
    model_name: str,
    pdf_bytes: bytes,
    page_catalog: list[dict[str, Any]],
    has_text_layer: bool,
    book_title: str = "",
    subject_and_grade: str = "",
    status_callback: Callable[[str], None] | None = None,
) -> tuple[AutoConceptPageDetectionOutput, list[tuple[int, int]]]:
    """
    1) 먼저 앞부분(p.1~15)에서 목차 및 표지·목차 끝 페이지(last_toc_page)를 추출하고,
    2) 목차 단원과 본문 페이지를 `[ACTUAL_PDF_PAGE]` 기준으로 대조해
       표지·목차·대단원 종합문제·정답지를 제외한 개념 구간(탐구/개념확인퀴즈 포함)을 판별하며,
    3) 실패 청크를 누락 없이 failed_chunks에 기록하고 같은 단원 2쪽 이내 간격을 병합한다.
    """
    total_pages = len(page_catalog)
    if total_pages == 0:
        raise ValueError("PDF에 페이지가 없습니다.")

    client = _build_client(api_key, timeout_seconds=90)

    # 1단계: 앞부분 표지/목차 분석
    toc_out = _extract_concept_toc_with_ai(
        client=client,
        model_name=model_name,
        pdf_bytes=pdf_bytes,
        page_catalog=page_catalog,
        has_text_layer=has_text_layer,
        book_title=book_title,
        subject_and_grade=subject_and_grade,
        status_callback=status_callback,
    )
    toc_items = toc_out.toc_units
    toc_unit_names = [t.unit_name.strip() for t in toc_items if t.unit_name.strip()]

    # 앞부분(p.1~15) 표지·차례 마지막 페이지 컷오프 계산
    heuristic_toc_end = 0
    for entry in page_catalog[:15]:
        snip_head = (entry.get("snippet") or "")[:120]
        if entry.get("noise_hint") in ("목차/차례", "표지/안내") or any(
            k in snip_head for k in ("차례", "목차", "이 책의 구성", "구성과 특징")
        ):
            heuristic_toc_end = max(heuristic_toc_end, int(entry["page"]))

    front_cutoff = max(int(toc_out.last_toc_page or 0), heuristic_toc_end)
    if front_cutoff == 0 and total_pages > 8:
        front_cutoff = 1  # 최소 1쪽 표지는 항상 제외
    front_cutoff = min(front_cutoff, min(total_pages - 1, 20))

    toc_guide_block = ""
    if toc_items:
        toc_lines = [
            f"  {idx}. {t.unit_name.strip()}"
            + (f" (목차 쪽수: {t.book_page_hint.strip()})" if t.book_page_hint.strip() else "")
            for idx, t in enumerate(toc_items, start=1)
        ]
        toc_guide_block = (
            "[1단계 목차 분석 결과 — 본문 대조 기준표]\n"
            + "\n".join(toc_lines)
            + f"\n※ 표지·교재구성·목차는 1~{front_cutoff}쪽까지로 판단되었으므로 개념 구간은 최소 {front_cutoff + 1}쪽 이후부터 시작하세요.\n\n"
        )
        if status_callback is not None:
            status_callback(
                f"📋 목차 {len(toc_items)}개 단원 · 앞부분 제외 경계(p.1~p.{front_cutoff}) 확인 → 2단계 본문 구간 판별 중..."
            )

    system_inst = (
        "당신은 대한민국 교과서 및 개념서(완자, 개념원리, 수능특강 등) PDF 구조 판별 AI입니다.\n"
        "1단계에서 추출한 [목차 단원 기준표]와 각 페이지를 대조하여 개념 설명 및 관련 학습 구간을 빠짐없이 찾으세요.\n\n"
        "**[매우 중요 — 페이지 번호 기준]**\n"
        "- 책 하단에 인쇄된 쪽수나 첨부 파일의 순서가 아니라, 각 페이지 상단 여백에 적힌 **`[ACTUAL_PDF_PAGE: 번호]`** 를 그대로 `start_page`, `end_page`로 반환하세요.\n\n"
        "**[제외 대상 (반드시 제외)]**\n"
        "- 표지, 머리말, 교재 구성 안내, 차례(목차)\n"
        "- 대단원 종합 문제, 실전 문제, 내신 만점 테스트, 수능 실전 문제 전용 페이지 구간\n"
        "- 정답 및 해설지, 빠른 정답 찾기, 채점 기준표 페이지\n\n"
        "**[포함 대상 (파편화 방지를 위해 반드시 개념 구간에 포함)]**\n"
        "- 대단원/중단원 도입, 핵심 개념 정리, 개념 설명 본문\n"
        "- 개념 설명에 이어지거나 개념 사이에 포함된 **탐구 활동, 자료 분석, 실험 원리, 용어 정리, 개념 확인 퀴즈(개념 체크 문제)**\n\n"
        "**[출력 규칙]**\n"
        "- detected_ranges: 같은 단원의 연속 개념 구간마다 unit_hint(가급적 목차 단원명과 통일), start_page, end_page(`[ACTUAL_PDF_PAGE]` 정수), reason을 적으세요.\n"
        "- excluded_summary: 제외한 표지/목차/대단원 종합문제/정답 구간을 간단히 요약하세요."
    )

    all_items: list[AutoConceptRangeItem] = []
    excluded_notes: list[str] = []
    failed_chunks: list[str] = []

    def _normalize_chunk_items(
        raw_ranges: list[AutoConceptRangeItem],
        chunk_pages: list[int],
    ) -> list[AutoConceptRangeItem]:
        c_min, c_max = chunk_pages[0], chunk_pages[-1]
        c_len = len(chunk_pages)
        normalized: list[AutoConceptRangeItem] = []
        for it in raw_ranges:
            s_p = int(it.start_page)
            e_p = int(it.end_page)
            if s_p > e_p:
                s_p, e_p = e_p, s_p
            # AI가 ACTUAL_PDF_PAGE 대신 1..len(chunk) 로컬 번호를 반환한 경우 자동 오프셋 복원
            if c_min > c_len and 1 <= s_p <= c_len and 1 <= e_p <= c_len:
                s_p = chunk_pages[s_p - 1]
                e_p = chunk_pages[e_p - 1]
            s_p = max(c_min, min(c_max, s_p))
            e_p = max(c_min, min(c_max, e_p))
            if s_p <= e_p:
                normalized.append(
                    AutoConceptRangeItem(
                        unit_hint=it.unit_hint.strip() or f"개념 구간 p.{s_p}~p.{e_p}",
                        start_page=s_p,
                        end_page=e_p,
                        reason=it.reason.strip(),
                    )
                )
        return normalized

    if has_text_layer:
        batch_size = 80
        batches = [page_catalog[i : i + batch_size] for i in range(0, total_pages, batch_size)]
        if status_callback is not None:
            status_callback(
                f"⚡ AI 목차 대조 기반 개념 구간 병렬 판별 중 ({len(batches)}개 구간 · 전체 {total_pages}쪽)..."
            )

        def _detect_text_batch(batch: list[dict[str, Any]]) -> tuple[list[int], AutoConceptPageDetectionOutput]:
            b_pages = [int(e["page"]) for e in batch]
            b_label = f"p.{b_pages[0]}~p.{b_pages[-1]}"

            def _call() -> AutoConceptPageDetectionOutput:
                local_client = _build_client(api_key, timeout_seconds=90)
                lines = []
                for entry in batch:
                    hint_tag = f" [로컬감지:{entry['noise_hint']}]" if entry["noise_hint"] else ""
                    snip = entry["snippet"].replace("\n", " | ")[:340]
                    lines.append(
                        f"[ACTUAL_PDF_PAGE: {entry['page']}{hint_tag}] "
                        f"{snip if snip else '(텍스트 없음/이미지 페이지)'}"
                    )
                prompt = (
                    f"- 교재명: {book_title or '(미입력)'} | 과목/학년: {subject_and_grade or '(미입력)'}\n"
                    f"- 분석 대상 구간: [ACTUAL_PDF_PAGE: {b_pages[0]}] ~ [ACTUAL_PDF_PAGE: {b_pages[-1]}] (전체 {total_pages}쪽)\n\n"
                    f"{toc_guide_block}"
                    "아래 페이지별 `[ACTUAL_PDF_PAGE: 번호]`를 기준으로 개념 구간(`detected_ranges`)을 JSON으로 반환하세요.\n\n"
                    + "\n".join(lines)
                )
                resp = _generate_content_with_fallback(
                    client=local_client,
                    model_name=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_inst,
                        response_mime_type="application/json",
                        response_schema=AutoConceptPageDetectionOutput,
                        temperature=0.0,
                    ),
                    fast_mode=True,
                    status_callback=None,
                )
                return _parse_structured(resp, AutoConceptPageDetectionOutput)

            out_parsed = _run_chunk_with_backoff(_call, b_label, max_retries=2, status_callback=status_callback)
            return b_pages, out_parsed

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(2, max(1, len(batches)))) as ex:
            fut_to_pages = {
                submit_with_usage_context(ex, _detect_text_batch, b): [int(e["page"]) for e in b]
                for b in batches
            }
            for fut, b_pages in fut_to_pages.items():
                try:
                    _, parsed = fut.result()
                    all_items.extend(_normalize_chunk_items(parsed.detected_ranges, b_pages))
                    if parsed.excluded_summary.strip():
                        excluded_notes.append(parsed.excluded_summary.strip())
                except Exception as exc:
                    err_label = f"p.{b_pages[0]}~p.{b_pages[-1]} ({exc})"
                    failed_chunks.append(err_label)
                    if status_callback is not None:
                        status_callback(f"❌ 구간 판별 실패: {err_label}")
    else:
        # 스캔(이미지) PDF: 20쪽씩 자른 슬라이스 각 페이지 상단에 [ACTUAL_PDF_PAGE: X] 각인 후 max_workers=2로 처리
        batch_size = 20
        page_batches = [
            [int(entry["page"]) for entry in page_catalog[i : i + batch_size]]
            for i in range(0, total_pages, batch_size)
        ]
        if status_callback is not None:
            status_callback(
                f"🖼️ 스캔 PDF AI 병렬 시각 판별 중 ({len(page_batches)}개 구간 · 동시 워커 최대 2개 · 전체 {total_pages}쪽)..."
            )

        def _detect_scan_batch(batch_pages: list[int]) -> AutoConceptPageDetectionOutput:
            b_label = f"p.{batch_pages[0]}~p.{batch_pages[-1]}"
            slice_b, _ = extract_specific_pages_slice(
                pdf_bytes,
                batch_pages,
                stamp_actual_page=True,
            )

            def _call() -> AutoConceptPageDetectionOutput:
                local_client = _build_client(api_key, timeout_seconds=120)
                pdf_part = prepare_pdf_part(
                    local_client,
                    slice_b,
                    display_name=f"scan_batch_{batch_pages[0]}_{batch_pages[-1]}.pdf",
                )
                prompt = (
                    f"- 교재명: {book_title or '(미입력)'} | 과목/학년: {subject_and_grade or '(미입력)'}\n"
                    f"- 이 슬라이스 PDF는 원본의 **`[ACTUAL_PDF_PAGE: {batch_pages[0]}]` ~ `[ACTUAL_PDF_PAGE: {batch_pages[-1]}]`** 입니다.\n\n"
                    f"{toc_guide_block}"
                    "각 페이지 상단 여백에 각인된 빨간색 라벨 **`[ACTUAL_PDF_PAGE: 번호]`** 를 보고, "
                    "반드시 그 `ACTUAL_PDF_PAGE` 번호 기준으로 `start_page`와 `end_page`를 반환하세요."
                )
                resp = _generate_content_with_fallback(
                    client=local_client,
                    model_name=model_name,
                    contents=[pdf_part, prompt],
                    config=types.GenerateContentConfig(
                        system_instruction=system_inst,
                        response_mime_type="application/json",
                        response_schema=AutoConceptPageDetectionOutput,
                        temperature=0.0,
                    ),
                    fast_mode=True,
                    status_callback=None,
                )
                return _parse_structured(resp, AutoConceptPageDetectionOutput)

            return _run_chunk_with_backoff(_call, b_label, max_retries=2, status_callback=status_callback)

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(2, max(1, len(page_batches)))) as ex:
            fut_to_pages = {
                submit_with_usage_context(ex, _detect_scan_batch, bp): bp for bp in page_batches
            }
            for fut, bp in fut_to_pages.items():
                try:
                    parsed = fut.result()
                    all_items.extend(_normalize_chunk_items(parsed.detected_ranges, bp))
                    if parsed.excluded_summary.strip():
                        excluded_notes.append(parsed.excluded_summary.strip())
                except Exception as exc:
                    err_label = f"p.{bp[0]}~p.{bp[-1]} ({exc})"
                    failed_chunks.append(err_label)
                    if status_callback is not None:
                        status_callback(f"❌ 스캔 구간 판별 실패: {err_label}")

    # 앞부분(표지·교재구성·목차) 컷오프 적용 (p.1 ~ front_cutoff 자동 제외)
    trimmed_items: list[AutoConceptRangeItem] = []
    for it in all_items:
        s_p = max(1, min(total_pages, int(it.start_page)))
        e_p = max(1, min(total_pages, int(it.end_page)))
        if s_p > e_p:
            s_p, e_p = e_p, s_p
        if e_p <= front_cutoff:
            continue
        if s_p <= front_cutoff:
            s_p = front_cutoff + 1
        if s_p <= e_p:
            trimmed_items.append(
                AutoConceptRangeItem(
                    unit_hint=it.unit_hint,
                    start_page=s_p,
                    end_page=e_p,
                    reason=it.reason,
                )
            )

    if front_cutoff >= 1:
        excluded_notes.insert(0, f"p.1~p.{front_cutoff} 표지/목차 자동 제외")

    # 같은 단원 내 2쪽 이내 짧은 간격 병합 + 최종 연속 구간 생성
    coalesced_items = _coalesce_concept_ranges(trimmed_items, max_same_unit_gap=2)

    raw_intervals = sorted((it.start_page, it.end_page) for it in coalesced_items)
    merged_intervals: list[tuple[int, int]] = []
    for s_p, e_p in raw_intervals:
        if not merged_intervals or s_p > merged_intervals[-1][1] + 1:
            merged_intervals.append((s_p, e_p))
        else:
            merged_intervals[-1] = (merged_intervals[-1][0], max(merged_intervals[-1][1], e_p))

    merged_result = AutoConceptPageDetectionOutput(
        toc_units=toc_unit_names,
        detected_ranges=coalesced_items,
        excluded_summary=" / ".join(dict.fromkeys(excluded_notes)),
        failed_chunks=failed_chunks,
    )
    return merged_result, merged_intervals


def _merge_concept_units(units: list[ConceptUnit]) -> list[ConceptUnit]:
    """청크 분할 추출 시 동일 단원명이 나뉜 경우 핵심 개념·허용 용어·페이지 범위를 병합한다."""
    merged: list[ConceptUnit] = []
    by_name: dict[str, ConceptUnit] = {}
    for u in units:
        key = u.unit_name.strip()
        if not key:
            continue
        if key not in by_name:
            clone = ConceptUnit(
                unit_name=key,
                page_range=u.page_range.strip(),
                core_concepts=list(dict.fromkeys(c.strip() for c in u.core_concepts if c.strip())),
                allowed_terms_and_units=list(
                    dict.fromkeys(t.strip() for t in u.allowed_terms_and_units if t.strip())
                ),
            )
            by_name[key] = clone
            merged.append(clone)
        else:
            existing = by_name[key]
            if u.page_range.strip() and u.page_range.strip() not in existing.page_range:
                existing.page_range = f"{existing.page_range}, {u.page_range.strip()}".strip(", ")
            for c in u.core_concepts:
                cs = c.strip()
                if cs and cs not in existing.core_concepts:
                    existing.core_concepts.append(cs)
            for t in u.allowed_terms_and_units:
                ts = t.strip()
                if ts and ts not in existing.allowed_terms_and_units:
                    existing.allowed_terms_and_units.append(ts)
    return merged


def _has_rich_extracted_text(text: str, page_count: int = 1) -> bool:
    """추출된 텍스트에 페이지당 충분한 본문 텍스트가 있는지 판별한다 (스캔 PDF 여부 구분)."""
    if not text:
        return False
    stripped = re.sub(r"={2,}.*?={2,}|-{2,}.*?-{2,}|\(도표/이미지 중심 개념 페이지\)", "", text)
    compact_len = len(re.sub(r"\s+", "", stripped))
    return compact_len >= max(80, max(1, page_count) * 60)


def extract_concept_book_units(
    api_key: str,
    model_name: str,
    book_title: str,
    subject_and_grade: str,
    page_ranges_display: str,
    sliced_pdf_bytes: bytes,
    mapped_concept_text: str,
    original_pdf_bytes: bytes | None = None,
    kept_pages: list[int] | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> ConceptBookExtractionOutput:
    """
    지정된 페이지 범위에서 추출된 개념 텍스트(및 필요 시 슬라이스 PDF)를
    Gemini 모델(model_name)로 병렬 분석하여 단원별(대단원/중단원) 개념 + 페이지 번호 구조화 DB를 생성한다.
    텍스트가 풍부한 PDF는 텍스트 전용 경량 프롬프트 + 50쪽 청크 병렬 처리로 속도를 극대화한다.
    """
    system_inst = (
        "당신은 2022 개정 교육과정 개념서·교과서 구조화 전문가입니다.\n"
        "제공된 개념서 페이지 번호 매핑 텍스트(`=== [실제 페이지: p.X] ===`) 또는 PDF를 분석하여 "
        "단원(대단원/중단원)별로 정확히 정리하세요.\n"
        "엄격 준수 규칙:\n"
        "1. 불필요한 목차, 정답지, 문제 풀이 내용은 제외하고 순수 개념 설명만 구조화하세요.\n"
        "2. 각 단원(`units`)의 `unit_name`은 대단원/중단원명을 명확히 적으세요 (예: 'Ⅰ. 물질과 규칙성 - 1. 원소의 생성', 'Ⅲ. 변화와 다양성').\n"
        "3. `page_range`는 반드시 텍스트(`=== [실제 페이지: p.X] ===`) 또는 페이지 상단 여백의 빨간색 `[ACTUAL_PDF_PAGE: X]` 라벨에 적힌 실제 페이지 번호를 기준으로 `'p.12 ~ p.25'` 형태로 정확히 표기하세요.\n"
        "4. `core_concepts`에는 해당 단원 페이지에 실제로 설명된 핵심 원리, 법칙, 정의, 공식을 구체적으로 나열하세요.\n"
        "5. `allowed_terms_and_units`에는 해당 단원 페이지에 실제로 등장하여 학생이 사용할 수 있는 모든 교과 용어, 기호, 공식, 단위(예: mL, 온도, 이온 수 등)를 빠짐없이 나열하세요.\n"
        "6. 제공된 페이지 범위에 없는 상위 과목 개념이나 다른 단원 용어를 절대 임의로 추가하지 마세요."
    )

    def _run_chunk(
        chunk_pdf: bytes | None,
        chunk_text: str,
        chunk_label: str,
        page_cnt: int,
    ) -> ConceptBookExtractionOutput:
        local_client = _build_client(api_key, timeout_seconds=90)
        text_is_rich = _has_rich_extracted_text(chunk_text, page_cnt)
        excerpt = chunk_text[:65000] if chunk_text else "(이미지 중심 개념서 PDF)"
        prompt = (
            f"- 교재명(book_title): {book_title}\n"
            f"- 과목 및 학년(subject / grade): {subject_and_grade}\n"
            f"- 추출 대상 개념 페이지 범위: {chunk_label}\n\n"
            "아래 페이지별 추출 개념 텍스트를 정독한 뒤, "
            "단원별로 구분하여 JSON 스키마(book_title, subject, grade, units)에 맞게 반환하세요.\n\n"
            f"[페이지별 개념 텍스트]\n{excerpt}"
        )
        if text_is_rich or chunk_pdf is None:
            contents: Any = prompt
        else:
            pdf_part = prepare_pdf_part(local_client, chunk_pdf, display_name="concept_book_slice.pdf")
            contents = [pdf_part, prompt]

        resp = _generate_content_with_fallback(
            client=local_client,
            model_name=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_inst,
                response_mime_type="application/json",
                response_schema=ConceptBookExtractionOutput,
                temperature=0.0,
            ),
            fast_mode=True,
            status_callback=None,
        )
        return _parse_structured(resp, ConceptBookExtractionOutput)

    total_kept = len(kept_pages) if kept_pages else 1
    is_text_rich_book = _has_rich_extracted_text(mapped_concept_text, total_kept)
    chunk_size = 50 if is_text_rich_book else 25

    if original_pdf_bytes is not None and kept_pages and len(kept_pages) > chunk_size:
        chunk_specs: list[tuple[bytes | None, str, str, int]] = []
        total_chunks = (len(kept_pages) + chunk_size - 1) // chunk_size
        for c_idx, offset in enumerate(range(0, len(kept_pages), chunk_size), start=1):
            sub_pages = kept_pages[offset : offset + chunk_size]
            sub_label = f"p.{sub_pages[0]}~p.{sub_pages[-1]} ({c_idx}/{total_chunks} 구간)"
            sub_pdf, sub_txt = extract_specific_pages_slice(original_pdf_bytes, sub_pages)
            chunk_specs.append((None if is_text_rich_book else sub_pdf, sub_txt, sub_label, len(sub_pages)))

        if status_callback is not None:
            status_callback(
                f"⚡ 단원별 개념 구조화 병렬 분석 중 ({len(chunk_specs)}개 구간 동시 처리 · "
                f"{'텍스트 고속 모드' if is_text_rich_book else 'Vision 모드'})..."
            )

        collected_units: list[ConceptUnit] = []
        detected_grade = ""
        worker_limit = 4 if is_text_rich_book else 2
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(worker_limit, len(chunk_specs))) as ex:
            futs = [submit_with_usage_context(ex, _run_chunk, *sp) for sp in chunk_specs]
            results = [f.result() for f in futs]

        for sub_out in results:
            collected_units.extend(sub_out.units)
            if not detected_grade and sub_out.grade:
                detected_grade = sub_out.grade

        merged_units = _merge_concept_units(collected_units)
        return ConceptBookExtractionOutput(
            book_title=book_title.strip(),
            subject=subject_and_grade.strip(),
            grade=detected_grade or "고1",
            units=merged_units,
        )

    parsed = _run_chunk(
        None if is_text_rich_book else sliced_pdf_bytes,
        mapped_concept_text,
        page_ranges_display,
        total_kept,
    )
    if book_title.strip():
        parsed.book_title = book_title.strip()
    if subject_and_grade.strip():
        parsed.subject = subject_and_grade.strip()
    return parsed


def extract_concept_whitelist(
    api_key: str,
    model_name: str,
    sliced_pdf_bytes: bytes,
    sliced_text: str,
    start_page: int,
    end_page: int,
    book_title: str = "",
    range_display: str = "",
) -> ConceptWhitelist:
    """업로드된 개념서·교과서 PDF 지정 페이지 범위에서 허용 개념과 허용 용어 화이트리스트를 추출한다."""
    client = _build_client(api_key, timeout_seconds=60)
    page_span = max(1, end_page - start_page + 1)
    text_is_rich = _has_rich_extracted_text(sliced_text, page_span)

    system_inst = (
        "당신은 엄격한 교육과정 범위 검증관입니다. "
        "제공된 개념서/교과서 페이지에 **실제로 설명되어 있는 내용만** 추출해야 합니다. "
        "뒤 단원이나 상위 과목에 나오는 개념·용어를 절대 추측하거나 추가하지 마세요.\n"
        "1. pdf_concepts: 이 페이지 범위에서 실제로 설명·정의·예시된 개념, 원리, 법칙, 계산 방법을 빠짐없이 나열하세요.\n"
        "2. pdf_vocabulary: 이 페이지 범위 본문, 그림, 표, 공식, 용어 사전 박스에 실제로 등장하는 "
        "모든 교과 용어, 기호, 단위를 나열하세요.\n"
        "3. units: 이 페이지 범위에 포함된 단원명(unit_name), 페이지 범위(page_range, 예: 'p.12 ~ p.25'), "
        "핵심 개념(core_concepts), 허용 용어·단위(allowed_terms_and_units)를 단원별로 함께 정리하세요."
    )

    effective_range = range_display.strip() or f"p.{start_page} ~ p.{end_page}"
    excerpt = sliced_text[:30000] if sliced_text else "(이미지 중심 PDF)"
    prompt = (
        f"교재명: {book_title or '업로드 개념서 PDF'}\n"
        f"대상 PDF 페이지 범위: {effective_range}\n"
        "제공된 개념서 페이지 내용을 정독한 뒤, "
        "이 페이지 범위 안에 실제로 존재하는 개념·원리(pdf_concepts), 허용 용어(pdf_vocabulary), 그리고 단원별 정보(units)를 추출하세요.\n\n"
        f"[추출 텍스트]\n{excerpt}"
    )

    if text_is_rich:
        contents: Any = prompt
    else:
        pdf_part = prepare_pdf_part(client, sliced_pdf_bytes, display_name="concept_slice.pdf")
        contents = [pdf_part, prompt]

    resp = _generate_content_with_fallback(
        client=client,
        model_name=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=ConceptWhitelist,
            temperature=0.0,
        ),
        fast_mode=True,
    )
    wl = _parse_structured(resp, ConceptWhitelist)
    if book_title.strip():
        wl.book_title = book_title.strip()
    if not wl.units:
        wl.units = [
            ConceptUnit(
                unit_name=wl.subject_guess or "지정 개념 범위",
                page_range=effective_range,
                core_concepts=list(wl.pdf_concepts),
                allowed_terms_and_units=list(wl.pdf_vocabulary),
            )
        ]
    return wl


def match_questions_to_whitelist(
    api_key: str,
    model_name: str,
    candidates: list[QuestionRecord],
    whitelist: ConceptWhitelist,
) -> list[QuestionMatchJudgment]:
    """후보 기출 문항들의 단원과 core_principles가 사용자가 선택한 개념서 단원·pdf_concepts 범위만으로 해결 가능한지 엄격히 판정한다."""
    if not candidates:
        return []

    client = _build_client(api_key)
    selected_unit_names = [u.unit_name for u in whitelist.units if u.unit_name]
    system_inst = (
        "당신은 기출 문항과 개념서 PDF 범위의 정합성을 판정하는 매우 엄격한 교육과정 심사관입니다.\n"
        "억지 매칭·억지 개조를 차단하기 위해 아래 규칙을 반드시 지키세요:\n"
        "1. is_solvable_by_pdf_only (엄격 일치 필수):\n"
        "   - 후보 문항의 **단원(unit_name)**과 **본질 원리(core_principles)**, 그리고 **실제 풀이(official_explanation)**가 "
        "사용자가 선택한 개념서 단원 목록(`whitelist.units`)의 핵심 주제 및 `pdf_concepts`와 **직접 일치**하고, "
        "해당 범위(+ 초중등 기초 연산·독해)만으로 **100% 완전히 해결 가능할 때만 True**로 판정하세요.\n"
        "   - **반드시 False로 제외해야 하는 경우:**\n"
        "     (a) 선택한 단원과 무관한 다른 단원 문항 (예: '산, 염기와 중화 반응' 단원을 선택했는데 '동적 평형', '가역 반응', '물 분자의 이온화' 단원 문항이 섞인 경우)\n"
        "     (b) 개념서 범위를 벗어나는 상위 실험 기법·계산 원리 (예: 고1 통합과학 범위를 벗어난 '중화 적정 실험', '피펫 분취 및 희석 계산', '이상기체 상태방정식' 등)를 말장난식 용어 치환으로 우회해야만 풀리는 문항\n"
        "     (c) 개념서에 없는 상위 원리·공식이 하나라도 필수인 문항\n"
        "2. out_of_scope_terms: 문항의 required_terms 중 pdf_vocabulary에 없는 상위/범위 외 전문 용어·단위(`M`, `몰농도`, `mol`, `몰수`, `분자량`, `화학식량`, `엔탈피` 등)를 빠짐없이 적으세요.\n"
        "3. needs_rewriting: is_solvable_by_pdf_only가 True이고 out_of_scope_terms가 1개 이상이면 True, 용어까지 모두 개념서 범위 안이면 False입니다.\n"
        "4. matched_unit_name 및 matched_page_range: is_solvable_by_pdf_only가 True인 경우, "
        "제공된 [개념서 단원별 상세 목록] 중 이 문항의 핵심 원리가 수록된 가장 정확한 단원명(`unit_name`)과 페이지 범위(`page_range`, 예: 'p.184~189')를 그대로 적으세요."
    )

    units_block = ""
    if whitelist.units:
        unit_lines = [
            f"- 단원명: {u.unit_name} | 페이지: {u.page_range} | 핵심 개념: {u.core_concepts[:15]} | 허용 용어: {u.allowed_terms_and_units[:20]}"
            for u in whitelist.units
        ]
        units_block = "[개념서 단원별 상세 목록 — 이 단원 외 타 단원 문항은 모두 제외]\n" + "\n".join(unit_lines) + "\n\n"

    all_judgments: list[QuestionMatchJudgment] = []
    batch_size = 25

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        cand_lines = [
            json.dumps(
                {
                    "id": q.id,
                    "subj": f"{q.subject_category}/{q.curriculum_subject}",
                    "unit": q.unit_name,
                    "gr": q.min_grade,
                    "prin": q.core_principles,
                    "terms": q.required_terms,
                    "exp": q.official_explanation,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for q in batch
        ]

        prompt = (
            f"[참고 교재명] {whitelist.book_title or '업로드 개념서 PDF'}\n"
            f"[선택된 단원명 목록] {selected_unit_names or [whitelist.subject_guess]}\n"
            f"[개념서 범위 요약]\n{whitelist.pdf_summary}\n\n"
            f"{units_block}"
            f"[허용 개념·원리 (pdf_concepts)]\n{whitelist.pdf_concepts}\n\n"
            f"[허용 용어 목록 (pdf_vocabulary)]\n{whitelist.pdf_vocabulary}\n\n"
            "[평가할 기출 문항 목록 (JSONL)]\n" + "\n".join(cand_lines)
        )

        resp = _generate_content_with_fallback(
            client=client,
            model_name=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_inst,
                response_mime_type="application/json",
                response_schema=BatchMatchResult,
                temperature=0.0,
            ),
        )
        batch_res = _parse_structured(resp, BatchMatchResult)
        all_judgments.extend(batch_res.judgments)

    return all_judgments


def extract_concept_book_cover_metadata(
    api_key: str,
    model_name: str,
    first_page_text: str,
    first_page_png_bytes: bytes | None = None,
    filename: str = "",
) -> ConceptBookCoverMetadata:
    """
    개념서·교과서 PDF 1페이지(표지) 이미지와 텍스트·파일명을 분석해
    교재명(book_title)과 과목 및 학년(subject_and_grade)을 자동 추출한다.
    """
    client = _build_client(api_key, timeout_seconds=20)
    system_inst = (
        "당신은 한국 고등학교 교과서·개념서(완자, 개념원리, 마플, 수능특강, 자이스토리 등) 표지 판독기입니다.\n"
        "첨부된 표지(1페이지) 이미지, 텍스트, 파일명을 종합하여 다음 2개 필드를 정확히 추출하세요:\n"
        "1) book_title: 교재명 또는 출판사+교과서명 (예: '완자 통합과학', '개념원리 공통수학1', '비상 통합사회1')\n"
        "2) subject_and_grade: 학년과 과목명 (예: '고1 통합과학', '고1 공통수학1', '고2 물리학Ⅰ', '고3 화학Ⅱ')"
    )
    text_excerpt = (first_page_text or "").strip()[:2500]
    fn_hint = f"[업로드 파일명]\n{filename.strip()}\n\n" if filename and filename.strip() else ""
    prompt = (
        f"{fn_hint}"
        "[표지/1페이지 텍스트 발췌]\n"
        f"{text_excerpt if text_excerpt else '(텍스트가 거의 없는 스캔 표지입니다. 첨부된 표지 이미지를 확인하세요.)'}\n\n"
        "위 개념서·교과서의 교재명(book_title)과 과목 및 학년(subject_and_grade)을 JSON으로 반환하세요."
    )
    contents: list[Any] = []
    if first_page_png_bytes:
        opt_png = optimize_image_for_vision(first_page_png_bytes, max_width=1200)
        contents.append(types.Part.from_bytes(data=opt_png, mime_type="image/png"))
    contents.append(prompt)

    resp = _generate_content_with_fallback(
        client=client,
        model_name=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=ConceptBookCoverMetadata,
            temperature=0.0,
        ),
        fast_mode=True,
    )
    meta = _parse_structured(resp, ConceptBookCoverMetadata)
    return ConceptBookCoverMetadata(
        book_title=meta.book_title.strip(),
        subject_and_grade=meta.subject_and_grade.strip(),
    )
