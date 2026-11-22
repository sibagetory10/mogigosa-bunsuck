from __future__ import annotations

import re

import pymupdf

def extract_pdf_slice(pdf_bytes: bytes, start_page: int, end_page: int) -> tuple[bytes, str, int]:
    """지정한 페이지 범위(1-based inclusive)의 슬라이스 PDF 바이트와 추출 텍스트를 반환한다."""
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        total = doc.page_count
        s_idx = max(0, min(start_page - 1, total - 1))
        e_idx = max(s_idx, min(end_page - 1, total - 1))

        sliced_doc = pymupdf.open()
        sliced_doc.insert_pdf(doc, from_page=s_idx, to_page=e_idx)
        sliced_bytes = sliced_doc.tobytes(deflate=True)
        sliced_doc.close()

        text_parts: list[str] = []
        for idx in range(s_idx, e_idx + 1):
            page_txt = doc[idx].get_text("text").strip()
            if page_txt:
                text_parts.append(f"--- [PDF {idx + 1}쪽] ---\n{page_txt}")

        return sliced_bytes, "\n\n".join(text_parts), (e_idx - s_idx + 1)


_PAGE_RANGE_TOKEN_RE = re.compile(r"^\s*(\d+)\s*(?:[\-~～–—]\s*(\d+)\s*)?$")
_TOC_HEADER_RE = re.compile(r"^(?:목\s*차|차\s*례|contents|이\s*책의\s*차례)\b", re.IGNORECASE)
_TOC_DOT_LINE_RE = re.compile(r"(?:\.{3,}|…{2,}|·{4,})\s*\d+\s*$")
_ANSWER_HEADER_RE = re.compile(
    r"(?:정답과\s*해설|정답\s*및\s*해설|정답\s*및\s*풀이|빠른\s*정답\s*찾기|빠른\s*정답|채점\s*기준표)",
    re.IGNORECASE,
)
_EXERCISE_HEADER_RE = re.compile(
    r"^(?:실전\s*문제|연습\s*문제|단원\s*마무리\s*문제|내신\s*만점\s*문제|수능\s*실전\s*문제|수능\s*맛보기|서술형\s*문제|기출\s*문제\s*맛보기|대표\s*기출\s*문제)\b"
)
_INLINE_NOISE_LINE_RE = re.compile(
    r"^(?:\[?\s*정답\s*\]?\s*[:：]?|\s*정답\s*[①②③④⑤1-5]\b|\[?\s*해설\s*\]?\s*[:：]|\[?\s*바로\s*알기\s*\]?|\[?\s*오답\s*피하기\s*\]?|채점\s*기준\s*[:：])"
)
_PROB_NUMBERED_LINE_RE = re.compile(r"^(?:0[1-9]|[1-9]\d?)\s*[.)]\s+\S+")


def parse_page_ranges_spec(range_spec: str, max_page: int) -> list[tuple[int, int]]:
    """
    '12-25, 34-48, 60-75' 형태의 다중 페이지 범위 문자열을 파싱하여
    정렬·병합된 1-based (start_page, end_page) 튜플 리스트로 반환한다.
    """
    raw = (range_spec or "").strip()
    if not raw:
        raise ValueError("추출할 개념 페이지 범위를 입력해 주세요. (예: 12-25, 34-48)")

    tokens = [t.strip() for t in re.split(r"[,，;/\n]+", raw) if t.strip()]
    if not tokens:
        raise ValueError("유효한 페이지 범위 형식이 아닙니다. (예: 12-25, 34-48)")

    intervals: list[tuple[int, int]] = []
    for tok in tokens:
        cleaned_tok = re.sub(r"[pP쪽페이지\s]+", "", tok)
        m = _PAGE_RANGE_TOKEN_RE.match(cleaned_tok)
        if not m:
            raise ValueError(
                f"페이지 범위 토큰 '{tok}'을(를) 해석할 수 없습니다. '12-25, 34-48' 형식으로 입력해 주세요."
            )
        s_val = int(m.group(1))
        e_val = int(m.group(2)) if m.group(2) is not None else s_val
        if s_val > e_val:
            s_val, e_val = e_val, s_val
        if s_val < 1 or e_val < 1:
            raise ValueError("페이지 번호는 1 이상이어야 합니다.")
        if s_val > max_page:
            raise ValueError(
                f"입력한 시작 페이지({s_val}쪽)가 PDF 전체 페이지 수({max_page}쪽)를 초과합니다."
            )
        e_clamped = min(max_page, e_val)
        intervals.append((s_val, e_clamped))

    intervals.sort(key=lambda x: (x[0], x[1]))
    merged: list[tuple[int, int]] = []
    for cur_s, cur_e in intervals:
        if not merged or cur_s > merged[-1][1] + 1:
            merged.append((cur_s, cur_e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], cur_e))
    return merged


def _clean_and_classify_concept_page(raw_text: str) -> tuple[str, str | None]:
    """
    단일 페이지 텍스트에서 목차·정답지·문제 풀이 노이즈를 판별 및 제거하고
    (정제된 개념 텍스트, 페이지 전체 제외 사유 또는 None)을 반환한다.
    """
    lines = [ln.strip() for ln in (raw_text or "").splitlines() if ln.strip()]
    if not lines:
        return "", None

    head_blob = "\n".join(lines[:6])

    # 1) 목차(TOC) 페이지 판별
    toc_dot_count = sum(1 for ln in lines if _TOC_DOT_LINE_RE.search(ln))
    if (_TOC_HEADER_RE.search(head_blob) and toc_dot_count >= 2) or toc_dot_count >= 6:
        return "", "목차(차례)"

    # 2) 정답 및 해설 전용 페이지 판별
    ans_marker_count = sum(
        1
        for ln in lines
        if _INLINE_NOISE_LINE_RE.match(ln) or re.search(r"[①②③④⑤]\s*(?:정답|해설|바로알기)", ln)
    )
    if _ANSWER_HEADER_RE.search(head_blob) and ans_marker_count >= 2:
        return "", "정답·해설지"
    if ans_marker_count >= 6:
        return "", "정답·해설지"

    # 3) 순수 문제 풀이(연습문제/실전문제) 페이지 판별
    prob_start_count = sum(1 for ln in lines if _PROB_NUMBERED_LINE_RE.match(ln))
    circled_choice_count = sum(len(re.findall(r"[①②③④⑤]", ln)) for ln in lines)
    if _EXERCISE_HEADER_RE.search(head_blob) and (prob_start_count >= 2 or circled_choice_count >= 5):
        return "", "문제 풀이(연습/실전문제)"
    if prob_start_count >= 4 and circled_choice_count >= 10:
        return "", "문제 풀이(문항 페이지)"

    # 4) 개념 페이지 내부의 인라인 문제/정답/목차 라인 제거
    cleaned_lines: list[str] = []
    skip_exercise_tail = False
    for ln in lines:
        if _TOC_DOT_LINE_RE.search(ln):
            continue
        if _INLINE_NOISE_LINE_RE.match(ln):
            continue
        if _EXERCISE_HEADER_RE.match(ln):
            # 페이지 하단에 붙은 확인문제/실전문제 섹션 시작 시 이하 생략
            skip_exercise_tail = True
            continue
        if skip_exercise_tail:
            continue
        cleaned_lines.append(ln)

    return "\n".join(cleaned_lines).strip(), None


def extract_concept_pages_by_ranges(
    pdf_bytes: bytes,
    page_ranges: list[tuple[int, int]],
) -> tuple[bytes, str, dict[str, Any]]:
    """
    지정된 다중 페이지 범위(예: [(12, 25), (34, 48)])만 PyMuPDF로 추출하고,
    불필요한 목차·정답지·문제 풀이 페이지를 제외한 순수 개념 텍스트와 실제 페이지 번호를 매핑한다.
    """
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        total_pages = doc.page_count
        requested_pages: list[int] = []
        for s_p, e_p in page_ranges:
            for p_num in range(max(1, s_p), min(total_pages, e_p) + 1):
                if p_num not in requested_pages:
                    requested_pages.append(p_num)

        kept_pages: list[int] = []
        skipped_pages: list[dict[str, Any]] = []
        mapped_blocks: list[str] = []

        for p_num in requested_pages:
            raw_txt = (doc[p_num - 1].get_text("text") or "").strip()
            cleaned_txt, skip_reason = _clean_and_classify_concept_page(raw_txt)
            if skip_reason is not None:
                skipped_pages.append({"page": p_num, "reason": skip_reason})
                continue
            kept_pages.append(p_num)
            if cleaned_txt:
                mapped_blocks.append(f"=== [실제 페이지: p.{p_num}] ===\n{cleaned_txt}")
            else:
                mapped_blocks.append(f"=== [실제 페이지: p.{p_num} (도표/이미지 중심 개념 페이지)] ===")

        # 만약 모든 요청 페이지가 필터링되었다면(예: 스캔본이거나 특수 포맷), 원본 지정 페이지를 그대로 보존
        if not kept_pages and requested_pages:
            kept_pages = list(requested_pages)
            skipped_pages.clear()
            for p_num in kept_pages:
                raw_txt = (doc[p_num - 1].get_text("text") or "").strip()
                mapped_blocks.append(f"=== [실제 페이지: p.{p_num}] ===\n{raw_txt}")

        sliced_doc = pymupdf.open()
        for p_num in kept_pages:
            sliced_doc.insert_pdf(doc, from_page=p_num - 1, to_page=p_num - 1)
            _stamp_actual_pdf_page_badge(sliced_doc[-1], p_num)
        sliced_bytes = sliced_doc.tobytes(deflate=True)
        sliced_doc.close()

        stats = {
            "total_pdf_pages": total_pages,
            "requested_pages": requested_pages,
            "kept_pages": kept_pages,
            "skipped_pages": skipped_pages,
            "ranges_display": ", ".join(
                f"p.{s}~p.{e}" if s != e else f"p.{s}" for s, e in page_ranges
            ),
        }
        return sliced_bytes, "\n\n".join(mapped_blocks), stats


def _stamp_actual_pdf_page_badge(page: pymupdf.Page, actual_page_num: int) -> None:
    """스캔 PDF 슬라이스 페이지 상단 여백에 실제 원본 PDF 페이지 라벨을 직접 각인한다."""
    rect = page.rect
    w = float(rect.width or 595.0)
    h = float(rect.height or 842.0)
    badge_w = min(max(185.0, w * 0.34), max(120.0, w - 20.0))
    badge_h = min(22.0, max(16.0, h * 0.032))
    box = pymupdf.Rect(
        rect.x0 + 10.0,
        rect.y0 + 6.0,
        min(rect.x1 - 10.0, rect.x0 + 10.0 + badge_w),
        min(rect.y1 - 6.0, rect.y0 + 6.0 + badge_h),
    )
    try:
        page.clean_contents()
    except Exception:
        pass
    page.draw_rect(
        box,
        color=(0.75, 0.05, 0.05),
        fill=(1.0, 0.98, 0.78),
        width=1.0,
        overlay=True,
    )
    font_size = max(9.5, min(12.0, badge_h * 0.56))
    page.insert_text(
        pymupdf.Point(box.x0 + 6.0, box.y0 + box.height * 0.72),
        f"[ACTUAL_PDF_PAGE: {actual_page_num}]",
        fontsize=font_size,
        fontname="helv",
        color=(0.7, 0.0, 0.0),
        overlay=True,
    )


def extract_pdf_page_catalog(
    pdf_bytes: bytes,
    snippet_chars: int = 500,
) -> tuple[list[dict[str, Any]], bool]:
    """
    AI 자동 개념 페이지 판별을 위해 PDF 전체 각 페이지(1..N)의 요약 카탈로그를 생성한다.
    텍스트가 충분한 PDF인지(has_text_layer) 여부도 함께 반환한다.
    """
    catalog: list[dict[str, Any]] = []
    text_pages_count = 0

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        total_pages = doc.page_count
        for idx in range(total_pages):
            raw_txt = (doc[idx].get_text("text") or "").strip()
            compact_len = len(re.sub(r"\s+", "", raw_txt))
            if compact_len >= 25:
                text_pages_count += 1
            lines = [ln.strip() for ln in raw_txt.splitlines() if ln.strip()]
            head_snippet = "\n".join(lines)[:snippet_chars]
            _, noise_hint = _clean_and_classify_concept_page(raw_txt)
            catalog.append(
                {
                    "page": idx + 1,
                    "char_count": compact_len,
                    "noise_hint": noise_hint or "",
                    "snippet": head_snippet,
                }
            )

    has_text_layer = total_pages > 0 and (text_pages_count / total_pages) >= 0.25
    return catalog, has_text_layer


def extract_specific_pages_slice(
    pdf_bytes: bytes,
    page_numbers: list[int],
    stamp_actual_page: bool = True,
) -> tuple[bytes, str]:
    """
    지정된 1-based 페이지 번호 리스트만 순서를 유지해 별도 PDF 바이트와 페이지 매핑 텍스트로 추출한다.
    stamp_actual_page=True이면 각 페이지 상단 여백에 `[ACTUAL_PDF_PAGE: X]` 라벨을 각인해
    스캔 PDF Vision 판별 시 슬라이스 내부 쪽수 혼동을 원천 차단한다.
    """
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        total = doc.page_count
        valid_pages = [p for p in page_numbers if 1 <= p <= total]
        sliced_doc = pymupdf.open()
        blocks: list[str] = []
        for p_num in valid_pages:
            sliced_doc.insert_pdf(doc, from_page=p_num - 1, to_page=p_num - 1)
            if stamp_actual_page:
                _stamp_actual_pdf_page_badge(sliced_doc[-1], p_num)
            raw_txt = (doc[p_num - 1].get_text("text") or "").strip()
            cleaned_txt, _ = _clean_and_classify_concept_page(raw_txt)
            use_txt = cleaned_txt or raw_txt
            if use_txt:
                blocks.append(f"=== [실제 페이지: p.{p_num}] ===\n{use_txt}")
            else:
                blocks.append(f"=== [실제 페이지: p.{p_num} (도표/이미지 중심 개념 페이지)] ===")
        sliced_bytes = sliced_doc.tobytes(deflate=True)
        sliced_doc.close()
        return sliced_bytes, "\n\n".join(blocks)
