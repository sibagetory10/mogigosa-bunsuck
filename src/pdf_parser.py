from __future__ import annotations

import io
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Sequence

import pymupdf
from PIL import Image

from src.models import normalize_exam_year

QUESTION_START_RE = re.compile(r"^(\d{1,2})\s*\.(?!\d)\s*(.*)$")
SHARED_PASSAGE_RE = re.compile(r"^[\[\【]\s*(\d{1,2})\s*[~\-～–—∼〜]\s*(\d{1,2})\s*(?:번)?\s*[\]\】]")


@dataclass
class CroppedQuestion:
    q_num: int
    page_num: int  # 1-indexed
    png_bytes: bytes
    width: int
    height: int
    extracted_text: str
    passage_group_id: str = ""
    shared_passage_text: str = ""
    passage_png_bytes: bytes = b""
    passage_width: int = 0
    passage_height: int = 0


def get_pdf_page_count(pdf_bytes: bytes) -> int:
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        return doc.page_count


def get_page_text_lengths(pdf_bytes: bytes) -> list[int]:
    """각 페이지(0-based 순서)의 공백 제외 텍스트 길이를 반환한다."""
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        return [
            len(re.sub(r"\s+", "", doc[i].get_text("text") or ""))
            for i in range(doc.page_count)
        ]


def render_pdf_page_png(pdf_bytes: bytes, page_num: int, dpi: int = 160) -> bytes:
    """1-based page_num에 해당하는 PDF 페이지를 PNG 바이트로 렌더링한다."""
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        idx = max(0, min(page_num - 1, doc.page_count - 1))
        page = doc[idx]
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        return pix.tobytes("png")


def extract_first_page_for_metadata(
    pdf_bytes: bytes,
    dpi: int = 160,
    min_text_chars: int = 40,
) -> tuple[str, bytes | None]:
    """PDF 1페이지 텍스트를 추출하고, 스캔 PDF(텍스트 부족)인 경우 1페이지 PNG를 함께 반환한다."""
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        if doc.page_count == 0:
            return "", None
        page = doc[0]
        raw_text = (page.get_text("text") or "").strip()
        compact_len = len(re.sub(r"\s+", "", raw_text))
        if compact_len < min_text_chars:
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            return raw_text, pix.tobytes("png")
        return raw_text, None


def extract_pdf_preview_for_pairing(
    pdf_bytes: bytes,
    dpi: int = 95,
    min_text_chars: int = 40,
    max_text_chars: int = 550,
) -> tuple[str, bytes | None, int]:
    """
    문제지·해설지 자동 매칭 및 역할 분류용으로 PDF 앞부분 텍스트와
    (스캔 PDF인 경우) 저해상도 1페이지 PNG, 총 페이지 수를 추출한다.
    """
    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            page_count = int(doc.page_count)
            if page_count == 0:
                return "", None, 0

            p1_text = (doc[0].get_text("text") or "").strip()
            combined_text = p1_text
            if len(re.sub(r"\s+", "", p1_text)) < 80 and page_count >= 2:
                p2_text = (doc[1].get_text("text") or "").strip()
                if p2_text:
                    combined_text = f"{p1_text}\n[2페이지]\n{p2_text}".strip()

            compact_len = len(re.sub(r"\s+", "", combined_text))
            png_bytes: bytes | None = None
            if compact_len < min_text_chars:
                pix = doc[0].get_pixmap(dpi=dpi, alpha=False)
                png_bytes = pix.tobytes("png")

            return combined_text[:max_text_chars], png_bytes, page_count
    except Exception:
        return "", None, 0


_LOCAL_SUBJECT_CANDIDATES: tuple[str, ...] = (
    "통합사회",
    "통합과학",
    "생활과 윤리",
    "윤리와 사상",
    "한국지리",
    "세계지리",
    "동아시아사",
    "세계사",
    "정치와 법",
    "사회·문화",
    "사회문화",
    "물리학Ⅱ",
    "물리학Ⅰ",
    "물리학2",
    "물리학1",
    "물리학",
    "화학Ⅱ",
    "화학Ⅰ",
    "화학2",
    "화학1",
    "생명과학Ⅱ",
    "생명과학Ⅰ",
    "생명과학2",
    "생명과학1",
    "생명과학",
    "지구과학Ⅱ",
    "지구과학Ⅰ",
    "지구과학2",
    "지구과학1",
    "지구과학",
    "공통수학1",
    "공통수학2",
    "확률과 통계",
    "미적분",
    "기하",
    "한국사",
    "경제",
    "화학",
    "수학",
    "국어",
    "영어",
)


def parse_exam_header_locally(first_page_text: str, filename: str = "") -> dict[str, str] | None:
    """
    1페이지 상단 텍스트(및 보조 파일명)에서 시행 년도·월·학년·과목·표준 회차명을
    정규표현식으로 0.05초 내에 즉시 추출하는 로컬 파서.
    """
    header_text = (first_page_text or "").strip()[:2000]
    fname_clean = re.sub(r"\.pdf$", "", filename or "", flags=re.IGNORECASE)
    combined = f"{header_text}\n{fname_clean}".strip()
    if not combined:
        return None

    year = ""
    ym = re.search(r"(20\d{2})\s*(?:학년도|년)", combined) or re.search(
        r"(?:^|[\s_\-.(])(20\d{2})(?:$|[\s_\-.)])", combined
    )
    if ym:
        year = ym.group(1)

    month = ""
    mm = re.search(r"(?:^|[\s_\-.(])(1[0-2]|[1-9])\s*월", combined)
    if mm:
        month = f"{int(mm.group(1))}월"
    elif "대학수학능력시험" in combined or "수능" in combined:
        month = "11월"

    grade = ""
    gm = re.search(r"(고\s*[123])", combined)
    if gm:
        grade = gm.group(1).replace(" ", "")
    else:
        gm2 = re.search(r"제?\s*([123])\s*학년", combined)
        if gm2:
            grade = f"고{gm2.group(1)}"
        elif "대학수학능력시험" in combined or "모의평가" in combined or "수능" in combined:
            grade = "고3"

    subject = ""
    # 시험지 괄호 표기 예: 과학탐구 영역(화학Ⅰ) 우선 매칭
    paren_m = re.search(r"(?:영역|탐구)\s*\(\s*([^)]+)\s*\)", header_text)
    if paren_m:
        cand = paren_m.group(1).strip()
        if cand:
            subject = cand
    if not subject:
        for s_cand in _LOCAL_SUBJECT_CANDIDATES:
            if s_cand in combined:
                subject = s_cand
                break

    year = normalize_exam_year(
        year,
        text=header_text,
        filename=fname_clean,
        month=month,
        grade=grade,
    )

    found_count = sum(bool(v) for v in (year, month, grade, subject))
    if found_count == 0:
        return None

    tokens = [
        f"{year}년" if year else "",
        month,
        grade,
        subject,
    ]
    exam_title = " ".join(t for t in tokens if t).strip()
    return {
        "year": year,
        "month": month,
        "grade": grade,
        "subject": subject,
        "exam_title": exam_title,
    }


def extract_exam_metadata_fast(
    pdf_bytes: bytes,
    filename: str = "",
    min_text_chars: int = 15,
) -> tuple[dict[str, str], bool]:
    """
    fitz(PyMuPDF)로 1페이지 텍스트만 추출해 정규표현식(re)으로 즉시 메타데이터를 파싱한다.
    텍스트가 거의 없는 스캔 PDF인 경우 API 호출 없이 빈칸 dict와 False를 반환한다.
    """
    empty_meta = {
        "year": "",
        "month": "",
        "grade": "",
        "subject": "",
        "exam_title": "",
    }
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        if doc.page_count == 0:
            return empty_meta, False
        raw_text = (doc[0].get_text("text") or "").strip()

    compact_len = len(re.sub(r"\s+", "", raw_text))
    if compact_len < min_text_chars:
        return empty_meta, False

    parsed = parse_exam_header_locally(raw_text, filename=filename)
    if parsed is None:
        return empty_meta, True
    return parsed, True


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


def _detect_columns(page: pymupdf.Page) -> list[tuple[float, float]]:
    """
    모의고사 PDF의 가로 중앙(page_width / 2)을 기준으로 왼쪽 단 -> 오른쪽 단 영역을 반환한다.
    명백한 1단 통짜 본문(구분선 없고 중앙을 가로지르는 긴 문단 다수)만 예외적으로 1단 처리한다.
    """
    w = page.rect.width
    h = page.rect.height
    mid_x = w / 2.0

    has_center_divider = False
    for d in page.get_drawings():
        rect = d.get("rect")
        if not rect:
            continue
        rw = abs(rect.x1 - rect.x0)
        rh = abs(rect.y1 - rect.y0)
        cx = (rect.x0 + rect.x1) / 2.0
        if (0.44 * w <= cx <= 0.56 * w) and rw <= 5.0 and rh >= 0.20 * h:
            has_center_divider = True
            break

    if has_center_divider:
        return [(0.0, mid_x), (mid_x, w)]

    # 중앙을 가로지르는 순수 1단 문단이 다수이고 오른쪽 단독 컬럼이 전혀 없는 경우에만 1단으로 처리
    spanning_lines = 0
    right_only_lines = 0
    page_dict = page.get_text("dict")
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            lx0, ly0, lx1, ly1 = line["bbox"]
            if ly1 < 0.14 * h or ly0 > 0.92 * h:
                continue
            txt = "".join(sp.get("text", "") for sp in line.get("spans", [])).strip()
            if len(txt) < 8:
                continue
            if lx0 < 0.30 * w and lx1 > 0.70 * w:
                spanning_lines += 1
            elif lx0 >= mid_x - 4.0:
                right_only_lines += 1

    if spanning_lines >= 4 and right_only_lines == 0:
        return [(0.0, w)]
    return [(0.0, mid_x), (mid_x, w)]


_PDF_HYPHEN_CHARS_RE = re.compile(r"[\u00ad\u2010\u2011\u2012\u2212\uf02d]")
_PDF_ENDASH_ALNUM_RE = re.compile(r"(?<=[A-Za-z0-9])\u2013(?=[A-Za-z0-9])")


def _normalize_pdf_hyphens(text: str) -> str:
    """PDF 추출 텍스트의 소프트 하이픈·PUA 하이픈·유니코드 대시를 ASCII '-'로 정규화한다."""
    if not text:
        return ""
    out = _PDF_HYPHEN_CHARS_RE.sub("-", text)
    out = _PDF_ENDASH_ALNUM_RE.sub("-", out)
    return out


def _has_vector_hyphen_between(
    x_gap_start: float,
    x_gap_end: float,
    line_y0: float,
    line_y1: float,
    drawings: list[dict] | None,
) -> bool:
    """두 span 사이 간격 안에 벡터 선분으로 그려진 짧은 수평 하이픈이 있는지 판별한다."""
    if not drawings:
        return False
    line_h = max(4.0, line_y1 - line_y0)
    y_lo = line_y0 + 0.15 * line_h
    y_hi = line_y1 - 0.15 * line_h
    x_lo = x_gap_start - 1.5
    x_hi = x_gap_end + 1.5
    for d in drawings:
        rect = d.get("rect")
        if not rect:
            continue
        rw = float(rect.x1 - rect.x0)
        rh = float(rect.y1 - rect.y0)
        if 1.5 <= rw <= 11.0 and rh <= 2.2:
            rcx = (rect.x0 + rect.x1) / 2.0
            rcy = (rect.y0 + rect.y1) / 2.0
            if x_lo <= rcx <= x_hi and y_lo <= rcy <= y_hi:
                return True
    return False


def _extract_lines_in_column(
    page_dict: dict,
    col_x0: float,
    col_x1: float,
    page_h: float,
    page_w: float,
    page_drawings: list[dict] | None = None,
) -> list[dict]:
    """
    블록 경계가 2단을 가로지르더라도 span 단위로 칼럼 소속을 분리해
    해당 단 내 텍스트 라인만 y좌표 -> x좌표 순으로 정렬해 반환한다.
    """
    is_two_col = (col_x1 - col_x0) < (0.85 * page_w)
    lines: list[dict] = []

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            lx0, ly0, lx1, ly1 = line["bbox"]
            if ly1 < 0.035 * page_h or ly0 > 0.945 * page_h:
                continue

            col_spans: list[dict] = []
            for sp in line.get("spans", []):
                stext = sp.get("text", "")
                if not stext:
                    continue
                sx0, sy0, sx1, sy1 = sp.get("bbox", (lx0, ly0, lx1, ly1))
                scx = (sx0 + sx1) / 2.0
                if is_two_col:
                    if col_x0 == 0.0:
                        if not (scx < col_x1):
                            continue
                    else:
                        if not (col_x0 <= scx <= col_x1):
                            continue
                col_spans.append(sp)

            if not col_spans:
                continue

            parts: list[str] = []
            for idx_sp, sp in enumerate(col_spans):
                st = _normalize_pdf_hyphens(sp.get("text", ""))
                if idx_sp > 0 and parts:
                    prev_sp = col_spans[idx_sp - 1]
                    prev_txt = parts[-1]
                    prev_bbox = prev_sp.get("bbox", (lx0, ly0, lx1, ly1))
                    curr_bbox = sp.get("bbox", (lx0, ly0, lx1, ly1))
                    gap = float(curr_bbox[0]) - float(prev_bbox[2])
                    if (
                        0.5 <= gap <= 14.0
                        and prev_txt
                        and st
                        and prev_txt[-1].isalnum()
                        and st[0].isalnum()
                        and _has_vector_hyphen_between(
                            float(prev_bbox[2]),
                            float(curr_bbox[0]),
                            float(ly0),
                            float(ly1),
                            page_drawings,
                        )
                    ):
                        parts.append("-")
                parts.append(st)

            text = "".join(parts).strip()
            if not text:
                continue

            sx0 = min(sp.get("bbox", (lx0, ly0, lx1, ly1))[0] for sp in col_spans)
            sy0 = min(sp.get("bbox", (lx0, ly0, lx1, ly1))[1] for sp in col_spans)
            sx1 = max(sp.get("bbox", (lx0, ly0, lx1, ly1))[2] for sp in col_spans)
            sy1 = max(sp.get("bbox", (lx0, ly0, lx1, ly1))[3] for sp in col_spans)
            max_font_size = max((sp.get("size", 0.0) for sp in col_spans), default=0.0)
            is_bold = any(bool(sp.get("flags", 0) & 16) for sp in col_spans)
            lines.append(
                {
                    "x0": sx0,
                    "y0": sy0,
                    "x1": sx1,
                    "y1": sy1,
                    "text": text,
                    "font_size": max_font_size,
                    "is_bold": is_bold,
                }
            )

    lines.sort(key=lambda item: (round(item["y0"], 1), item["x0"]))
    return lines


def _is_footer_or_header_noise(text: str, y0: float, page_h: float) -> bool:
    # 본문 중앙에서 우연히 단어가 겹치는 오탐을 막기 위해 상·하단 여백 구간 위주로 필터링
    t = (text or "").strip()
    in_top_margin = y0 < 0.095 * page_h
    in_bottom_margin = y0 > 0.895 * page_h
    in_margin = in_top_margin or in_bottom_margin

    always_noise = (
        "이 문제지에 관한 저작권",
        "한국교육과정평가원",
        "답안지의 해당란에",
    )
    if any(p in t for p in always_noise):
        return True
    if re.match(r"^(?:국어|수학|영어|한국사|사회탐구|과학탐구|직업탐구|제2외국어)\s*영역$", t):
        return True
    # 단독 1~2자리 쪽번호('7', '[10]' 등)는 약간 더 넓은 상·하단 여백까지 필터링 (마침표 붙은 문항번호 '39.'는 제외)
    if (y0 < 0.115 * page_h or y0 > 0.885 * page_h) and re.match(r"^\[?\s*\d{1,2}\s*\]?$", t):
        return True
    if not in_margin:
        return False

    # 상·하단 여백의 단독 쪽수('[10]', '12', '6') 또는 '12 국어', '국어 11', '영역 7' 등
    if re.match(r"^\[?\s*\d{1,2}\s*\]?$", t):
        return True
    if re.match(r"^(?:\d{1,2}\s+)?(?:국어|수학|영어|한국사|사회탐구|과학탐구|직업탐구|제2외국어)(?:\s+\d{1,2})?$", t):
        return True
    if re.match(r"^(?:영역\s*\d{1,2}|\d{1,2}\s*영역)$", t):
        return True

    margin_noise = (
        "학년도 대학수학능력시험",
        "전국연합학력평가",
        "모의평가 문제지",
        "확인 사항",
        "제 1 교시",
        "제 2 교시",
        "제 3 교시",
        "제 4 교시",
        "영역",
    )
    return any(p in t for p in margin_noise)


def _collect_content_boxes_in_column(
    page: pymupdf.Page,
    page_dict: dict,
    col_x0: float,
    col_x1: float,
    page_h: float,
) -> list[tuple[float, float, float, float]]:
    """정밀 크롭 경계 계산을 위해 컬럼 내 텍스트/이미지/도형 박스를 수집한다 (상·하단 헤더/푸터 제외)."""
    boxes: list[tuple[float, float, float, float]] = []

    for block in page_dict.get("blocks", []):
        btype = block.get("type")
        bx0, by0, bx1, by1 = block["bbox"]
        if by1 < 0.055 * page_h or by0 > 0.905 * page_h:
            continue

        if btype == 0:
            for ln in block.get("lines", []):
                lx0, ly0, lx1, ly1 = ln["bbox"]
                if ly1 < 0.055 * page_h or ly0 > 0.905 * page_h:
                    continue
                ln_text = "".join(sp.get("text", "") for sp in ln.get("spans", [])).strip()
                if not ln_text or _is_footer_or_header_noise(ln_text, ly0, page_h):
                    continue
                for sp in ln.get("spans", []):
                    if not sp.get("text", "").strip():
                        continue
                    sx0, sy0, sx1, sy1 = sp.get("bbox", (lx0, ly0, lx1, ly1))
                    scx = (sx0 + sx1) / 2.0
                    if col_x0 <= scx <= col_x1:
                        boxes.append((max(col_x0, sx0), sy0, min(col_x1, sx1), sy1))
        else:
            bcx = (bx0 + bx1) / 2.0
            overlap_x = min(col_x1, bx1) - max(col_x0, bx0)
            if (col_x0 <= bcx <= col_x1) or overlap_x > 16.0:
                boxes.append((max(col_x0, bx0), by0, min(col_x1, bx1), by1))

    for d in page.get_drawings():
        rect = d.get("rect")
        if not rect:
            continue
        rw = rect.x1 - rect.x0
        rh = rect.y1 - rect.y0
        if rh > 0.72 * page_h and rw < 6.0:
            continue
        if rw > 0.85 * page.rect.width:
            continue
        # 상·하단 헤더/푸터 가로 구분선 및 페이지 번호 박스 제외
        if rect.y1 < 0.095 * page_h and rh < 6.0 and rw > 0.25 * (col_x1 - col_x0):
            continue
        dcx = (rect.x0 + rect.x1) / 2.0
        if not (col_x0 <= dcx <= col_x1):
            continue
        if rect.y1 < 0.055 * page_h or rect.y0 > 0.90 * page_h:
            continue
        boxes.append((max(col_x0, rect.x0), rect.y0, min(col_x1, rect.x1), rect.y1))

    return boxes


def _stitch_vertical_pngs(top_png: bytes, bottom_png: bytes, gap_px: int = 12) -> tuple[bytes, int, int]:
    """이전 단(컬럼)의 세트 지문 PNG와 문항 PNG를 위아래로 깔끔하게 이어 붙인다."""
    with Image.open(io.BytesIO(top_png)) as im_top, Image.open(io.BytesIO(bottom_png)) as im_bot:
        w = max(im_top.width, im_bot.width)
        h = im_top.height + gap_px + im_bot.height
        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        canvas.paste(im_top.convert("RGB"), (0, 0))
        canvas.paste(im_bot.convert("RGB"), (0, im_top.height + gap_px))
        buf = io.BytesIO()
        canvas.save(buf, format="PNG")
        return buf.getvalue(), w, h


def parse_exam_pdf(
    pdf_bytes: bytes,
    dpi: int = 250,
    split_passage_and_question: bool = False,
) -> tuple[list[CroppedQuestion], bool]:
    """
    2단 편집 한국어 모의고사/수능 문제지 PDF를 PyMuPDF로 분석해
    가로 중앙(page_width / 2) 기준 왼쪽 단(위->아래) 후 오른쪽 단(위->아래) 순서로
    1번~45번 전체 문항 및 세트 지문([N~M])을 고해상도(>=220 DPI) PNG로 크롭한다.
    split_passage_and_question=True(국어 등)일 때는 공통 지문 이미지와 개별 문제 이미지를 분리해 반환한다.
    """
    effective_dpi = max(220, dpi)
    results: list[CroppedQuestion] = []
    seen_q_nums: set[int] = set()
    total_text_chars = 0

    # 이전 단(또는 이전 페이지) 끝에서 미처 첫 문항을 만나지 못한 세트 지문 [N~M] 이월용 및 세트 문항 공통 텍스트 유지용
    carried_passage: dict | None = None
    active_passage_group: dict | None = None
    # 이전 단 맨 아래 문항이 다음 단 상단(새 문항/새 지문 시작 전)까지 이어지는 경우(선지 등) 꼬리 병합용
    prev_col_last_q_idx: int | None = None

    def _crop_region_png(
        page_obj: pymupdf.Page,
        c_boxes: list[tuple[float, float, float, float]],
        c_x0: float,
        c_x1: float,
        p_h: float,
        y_lo: float,
        y_hi: float,
    ) -> tuple[bytes, int, int] | None:
        r_boxes = [b for b in c_boxes if b[3] > y_lo - 4.0 and b[1] < y_hi]
        if not r_boxes:
            return None
        r_clip = pymupdf.Rect(
            max(c_x0 + 2.0, min(b[0] for b in r_boxes) - 10.0),
            max(0.055 * p_h, min(y_lo, min(b[1] for b in r_boxes)) - 6.0),
            min(c_x1 - 2.0, max(b[2] for b in r_boxes) + 10.0),
            min(y_hi, max(b[3] for b in r_boxes) + 8.0),
        )
        if r_clip.y1 <= r_clip.y0 + 6.0:
            return None
        r_pix = page_obj.get_pixmap(dpi=effective_dpi, clip=r_clip, alpha=False)
        return r_pix.tobytes("png"), r_pix.width, r_pix.height

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_idx in range(doc.page_count):
            page = doc[page_idx]
            page_h = page.rect.height
            page_w = page.rect.width
            page_dict = page.get_text("dict")
            page_drawings = page.get_drawings()
            total_text_chars += len(re.sub(r"\s+", "", page.get_text("text") or ""))

            columns = _detect_columns(page)

            for col_x0, col_x1 in columns:
                col_w = col_x1 - col_x0
                lines = _extract_lines_in_column(
                    page_dict,
                    col_x0,
                    col_x1,
                    page_h,
                    page_w,
                    page_drawings=page_drawings,
                )
                if not lines:
                    continue

                # 상단 여백 헤더('16 국어', '12 국어' 등)가 있으면 그 아래부터 본문 시작점으로 제한
                top_header_y1 = 0.055 * page_h
                for ln in lines:
                    if ln["y0"] < 0.105 * page_h and _is_footer_or_header_noise(ln["text"], ln["y0"], page_h):
                        top_header_y1 = max(top_header_y1, ln["y1"] + 3.0)

                # 컬럼 본문의 좌측 시작 기준 (과도한 좌측 여백 노이즈 방지를 위해 중앙값/대다수 기준 보조)
                candidate_xs = sorted(
                    ln["x0"]
                    for ln in lines
                    if not _is_footer_or_header_noise(ln["text"], ln["y0"], page_h)
                    and QUESTION_START_RE.match(ln["text"])
                )
                ref_q_x0 = candidate_xs[0] if candidate_xs else (col_x0 + 24.0)
                max_allowed_x0 = max(col_x0 + 0.34 * col_w, ref_q_x0 + 48.0)

                markers: list[dict] = []
                first_passage_y0_in_col: float | None = None
                pending_passage_y0: float | None = None
                pending_passage_start_q: int | None = None
                pending_passage_end_q: int | None = None
                same_col_passage_crops: list[tuple[str, float, float]] = []

                for ln in lines:
                    txt = ln["text"]
                    if _is_footer_or_header_noise(txt, ln["y0"], page_h):
                        continue

                    # 공통/세트 지문 헤더 [1~3], [16~17], [41~42], [43~45] 등 감지
                    pm = SHARED_PASSAGE_RE.match(txt)
                    if pm and ln["x0"] <= col_x0 + 0.45 * col_w:
                        p_start = int(pm.group(1))
                        p_end = int(pm.group(2))
                        if 1 <= p_start <= p_end <= 45:
                            if first_passage_y0_in_col is None:
                                first_passage_y0_in_col = ln["y0"]
                            pending_passage_y0 = ln["y0"]
                            pending_passage_start_q = p_start
                            pending_passage_end_q = p_end
                        continue

                    qm = QUESTION_START_RE.match(txt)
                    if not qm:
                        continue
                    q_num = int(qm.group(1))
                    if not (1 <= q_num <= 45):
                        continue

                    if ln["x0"] > max_allowed_x0:
                        continue

                    # 좌->우, 위->아래 순차 탐색 중 중복 번호 및 역순 오탐(<보기> 내부 숫자 등) 방지
                    if q_num in seen_q_nums:
                        continue
                    if seen_q_nums and q_num < max(seen_q_nums):
                        continue

                    start_y = ln["y0"]
                    if (
                        pending_passage_y0 is not None
                        and pending_passage_start_q is not None
                        and pending_passage_end_q is not None
                        and q_num <= pending_passage_end_q
                    ):
                        same_col_shared_txt = "\n".join(
                            l_item["text"]
                            for l_item in lines
                            if pending_passage_y0 - 2.0 <= l_item["y0"] < ln["y0"] - 1.0
                            and not _is_footer_or_header_noise(l_item["text"], l_item["y0"], page_h)
                        ).strip()
                        grp_id = f"{pending_passage_start_q}~{pending_passage_end_q}"
                        active_passage_group = {
                            "start_q": pending_passage_start_q,
                            "end_q": pending_passage_end_q,
                            "group_id": grp_id,
                            "text": same_col_shared_txt,
                            "png_bytes": b"",
                            "width": 0,
                            "height": 0,
                        }
                        if split_passage_and_question:
                            # 지문과 문제를 분리할 때는 지문 영역을 별도 기록하고 문항은 번호 시작점(ln["y0"])부터 크롭
                            same_col_passage_crops.append((grp_id, pending_passage_y0, ln["y0"] - 3.0))
                        else:
                            start_y = min(start_y, pending_passage_y0)
                        pending_passage_y0 = None
                        pending_passage_start_q = None
                        pending_passage_end_q = None

                    markers.append(
                        {
                            "q_num": q_num,
                            "y_start": start_y,
                            "anchor_y": ln["y0"],
                            "passage_group": (
                                dict(active_passage_group)
                                if (
                                    active_passage_group is not None
                                    and active_passage_group["start_q"] <= q_num <= active_passage_group["end_q"]
                                )
                                else None
                            ),
                        }
                    )
                    seen_q_nums.add(q_num)

                content_boxes = [
                    b
                    for b in _collect_content_boxes_in_column(page, page_dict, col_x0, col_x1, page_h)
                    if b[3] > top_header_y1
                ]

                # 같은 단에 있던 지문 영역을 별도 PNG로 크롭(split_passage_and_question=True 전용)
                if split_passage_and_question and same_col_passage_crops:
                    for s_gid, py0, py1 in same_col_passage_crops:
                        cropped_p = _crop_region_png(page, content_boxes, col_x0, col_x1, page_h, py0, py1)
                        if cropped_p:
                            p_b, p_w, p_h_px = cropped_p
                            if active_passage_group and active_passage_group.get("group_id") == s_gid:
                                active_passage_group["png_bytes"] = p_b
                                active_passage_group["width"] = p_w
                                active_passage_group["height"] = p_h_px
                            for m_item in markers:
                                mg = m_item.get("passage_group")
                                if mg and mg.get("group_id") == s_gid:
                                    mg["png_bytes"] = p_b
                                    mg["width"] = p_w
                                    mg["height"] = p_h_px

                # 직전 단에서 이월된 세트 지문(예: [43~45] (A))이 현재 단 상단에서 (B)(C)(D) 등으로 이어진 뒤 첫 문항(43.)이 나오는 경우
                if (
                    markers
                    and carried_passage is not None
                    and carried_passage["start_q"] <= markers[0]["q_num"] <= carried_passage["end_q"]
                ):
                    first_m = markers[0]
                    cont_lines = [
                        ln
                        for ln in lines
                        if top_header_y1 <= ln["y0"] < first_m["y_start"] - 1.0
                        and not _is_footer_or_header_noise(ln["text"], ln["y0"], page_h)
                    ]
                    if cont_lines:
                        cont_txt = "\n".join(ln["text"] for ln in cont_lines).strip()
                        if cont_txt:
                            carried_passage["text"] = f"{carried_passage.get('text', '')}\n{cont_txt}".strip()
                            if (
                                active_passage_group is not None
                                and active_passage_group["start_q"] == carried_passage["start_q"]
                            ):
                                active_passage_group["text"] = carried_passage["text"]
                        if split_passage_and_question:
                            # 이어지는 지문 이미지를 carried_passage 지문 이미지 아래에 이어 붙임 (문제 이미지에는 안 붙임)
                            cont_crop = _crop_region_png(
                                page,
                                content_boxes,
                                col_x0,
                                col_x1,
                                page_h,
                                min(ln["y0"] for ln in cont_lines),
                                first_m["y_start"] - 3.0,
                            )
                            if cont_crop:
                                st_p, st_pw, st_ph = _stitch_vertical_pngs(
                                    carried_passage["png_bytes"], cont_crop[0]
                                )
                                carried_passage["png_bytes"] = st_p
                                carried_passage["width"] = st_pw
                                carried_passage["height"] = st_ph
                                if active_passage_group and active_passage_group["start_q"] == carried_passage["start_q"]:
                                    active_passage_group["png_bytes"] = st_p
                                    active_passage_group["width"] = st_pw
                                    active_passage_group["height"] = st_ph
                        else:
                            first_m["y_start"] = min(first_m["y_start"], min(ln["y0"] for ln in cont_lines))
                    if (
                        active_passage_group is not None
                        and active_passage_group["start_q"] <= first_m["q_num"] <= active_passage_group["end_q"]
                    ):
                        for m_item in markers:
                            if active_passage_group["start_q"] <= m_item["q_num"] <= active_passage_group["end_q"]:
                                m_item["passage_group"] = dict(active_passage_group)
                    prev_col_last_q_idx = None
                elif (
                    not markers
                    and carried_passage is not None
                    and pending_passage_y0 is None
                ):
                    # 한 단 전체가 이어지는 지문으로만 채워져 있고 문항이 하나도 없는 경우 (긴 국어 지문 대응)
                    mid_top_y = first_passage_y0_in_col if first_passage_y0_in_col is not None else 0.902 * page_h
                    mid_lines = [
                        ln
                        for ln in lines
                        if top_header_y1 <= ln["y0"] < mid_top_y - 1.0
                        and not _is_footer_or_header_noise(ln["text"], ln["y0"], page_h)
                    ]
                    if mid_lines:
                        mid_txt = "\n".join(ln["text"] for ln in mid_lines).strip()
                        if mid_txt:
                            carried_passage["text"] = f"{carried_passage.get('text', '')}\n{mid_txt}".strip()
                        mid_crop = _crop_region_png(
                            page, content_boxes, col_x0, col_x1, page_h, min(ln["y0"] for ln in mid_lines), mid_top_y
                        )
                        if mid_crop:
                            st_p, st_pw, st_ph = _stitch_vertical_pngs(
                                carried_passage["png_bytes"], mid_crop[0]
                            )
                            carried_passage["png_bytes"] = st_p
                            carried_passage["width"] = st_pw
                            carried_passage["height"] = st_ph
                            if active_passage_group and active_passage_group["start_q"] == carried_passage["start_q"]:
                                active_passage_group["text"] = carried_passage["text"]
                                active_passage_group["png_bytes"] = st_p
                                active_passage_group["width"] = st_pw
                                active_passage_group["height"] = st_ph
                    prev_col_last_q_idx = None
                elif prev_col_last_q_idx is not None:
                    # 직전 단의 마지막 문항(예: 17번)이 현재 단 상단(첫 문항 또는 첫 세트 헤더 이전)까지 이어지는 경우 꼬리(①~⑤ 선지 등) 병합
                    # 단, 새 지문('다음 글을 읽고 물음에 답하시오', '[N~M]')이 시작되는 경우는 절대 병합하지 않음
                    boundary_candidates: list[float] = []
                    if markers:
                        boundary_candidates.append(markers[0]["y_start"])
                    if first_passage_y0_in_col is not None:
                        boundary_candidates.append(first_passage_y0_in_col)
                    top_boundary_y = min(boundary_candidates) if boundary_candidates else (0.905 * page_h)

                    tail_lines = [
                        ln
                        for ln in lines
                        if top_header_y1 <= ln["y0"] < top_boundary_y - 1.0
                        and not _is_footer_or_header_noise(ln["text"], ln["y0"], page_h)
                    ]
                    tail_head_blob = " ".join(ln["text"] for ln in tail_lines[:3])
                    is_new_passage_start = (
                        "다음 글을 읽고 물음에 답하시오" in tail_head_blob
                        or any(SHARED_PASSAGE_RE.match(ln["text"]) for ln in tail_lines[:3])
                    )
                    if tail_lines and not is_new_passage_start and len(tail_lines) <= 18:
                        tail_top = min(ln["y0"] for ln in tail_lines) - 4.0
                        tail_boxes = [
                            b
                            for b in content_boxes
                            if b[3] > tail_top and b[1] < top_boundary_y - 4.0
                        ]
                        if tail_boxes:
                            t_clip = pymupdf.Rect(
                                max(col_x0 + 2.0, min(b[0] for b in tail_boxes) - 10.0),
                                max(top_header_y1, min(tail_top, min(b[1] for b in tail_boxes)) - 6.0),
                                min(col_x1 - 2.0, max(b[2] for b in tail_boxes) + 10.0),
                                min(top_boundary_y - 3.0, max(b[3] for b in tail_boxes) + 8.0),
                            )
                            if 8.0 < (t_clip.y1 - t_clip.y0) <= 0.45 * page_h:
                                t_pix = page.get_pixmap(dpi=effective_dpi, clip=t_clip, alpha=False)
                                prev_cq = results[prev_col_last_q_idx]
                                stitched_png, st_w, st_h = _stitch_vertical_pngs(
                                    prev_cq.png_bytes, t_pix.tobytes("png")
                                )
                                prev_cq.png_bytes = stitched_png
                                prev_cq.width = st_w
                                prev_cq.height = st_h
                                tail_txt = "\n".join(ln["text"] for ln in tail_lines).strip()
                                if tail_txt:
                                    prev_cq.extracted_text = f"{prev_cq.extracted_text}\n{tail_txt}".strip()
                    prev_col_last_q_idx = None

                for i, marker in enumerate(markers):
                    is_last_in_col = i == len(markers) - 1
                    trailing_passage_cuts_col = (
                        is_last_in_col
                        and pending_passage_y0 is not None
                        and pending_passage_y0 > marker["y_start"]
                    )
                    y_top_limit = marker["y_start"] - 4.0
                    if not is_last_in_col:
                        y_bottom_limit = markers[i + 1]["y_start"] - 4.0
                    elif trailing_passage_cuts_col:
                        y_bottom_limit = pending_passage_y0 - 6.0
                    else:
                        # 각 단 맨 아래 문항은 하단 푸터('12 국어', '영역 7', 쪽번호) 위쪽까지만 확보
                        y_bottom_limit = 0.905 * page_h

                    matched_boxes = [
                        b
                        for b in content_boxes
                        if b[3] > y_top_limit and b[1] < y_bottom_limit and b[1] >= y_top_limit - 8.0
                    ]

                    if matched_boxes:
                        min_x = min(b[0] for b in matched_boxes)
                        min_y = min(marker["y_start"], min(b[1] for b in matched_boxes))
                        max_x = max(b[2] for b in matched_boxes)
                        max_y = max(b[3] for b in matched_boxes)
                    else:
                        min_x = max(col_x0, col_x0 + 18.0)
                        min_y = marker["y_start"]
                        max_x = col_x1 - 8.0
                        max_y = y_bottom_limit

                    if max_y - min_y < 40.0:
                        max_y = min(y_bottom_limit, min_y + 120.0)

                    pad_x = 12.0
                    pad_y = 8.0
                    clip = pymupdf.Rect(
                        max(col_x0 + 2.0, min_x - pad_x),
                        max(top_header_y1, min_y - pad_y),
                        min(col_x1 - 2.0, max_x + pad_x),
                        min(
                            0.908 * page_h if (is_last_in_col and not trailing_passage_cuts_col) else y_bottom_limit,
                            max_y + pad_y,
                        ),
                    )
                    if clip.y1 <= clip.y0:
                        clip.y1 = min(0.908 * page_h, clip.y0 + 120.0)

                    if clip.width < 0.30 * page_w:
                        clip.x0 = max(col_x0 + 4.0, col_x0 + 12.0)
                        clip.x1 = min(col_x1 - 4.0, max(clip.x1, col_x1 - 16.0))

                    pix = page.get_pixmap(dpi=effective_dpi, clip=clip, alpha=False)
                    png_out = pix.tobytes("png")
                    out_w, out_h = pix.width, pix.height
                    q_text = "\n".join(
                        ln["text"]
                        for ln in lines
                        if ln["y1"] >= clip.y0 and ln["y0"] <= clip.y1
                        and not _is_footer_or_header_noise(ln["text"], ln["y0"], page_h)
                    )

                    q_passage_png = b""
                    q_passage_w = 0
                    q_passage_h = 0

                    # 직전 단에서 이월된 세트 지문이 현재 문항 번호 구간(start_q <= q_num <= end_q)에 속할 때
                    if (
                        carried_passage is not None
                        and carried_passage["start_q"] <= marker["q_num"] <= carried_passage["end_q"]
                    ):
                        if split_passage_and_question:
                            q_passage_png = carried_passage.get("png_bytes", b"")
                            with Image.open(io.BytesIO(q_passage_png)) as im_p:
                                q_passage_w, q_passage_h = im_p.size
                        else:
                            png_out, out_w, out_h = _stitch_vertical_pngs(
                                carried_passage["png_bytes"],
                                png_out,
                            )
                        if carried_passage.get("text"):
                            q_text = f"{carried_passage['text']}\n\n{q_text}".strip()
                        carried_passage = None
                    elif carried_passage is not None and marker["q_num"] > carried_passage["end_q"]:
                        carried_passage = None

                    q_group_id = ""
                    q_shared_text = ""
                    m_group = marker.get("passage_group")
                    if m_group is None and active_passage_group is not None:
                        if active_passage_group["start_q"] <= marker["q_num"] <= active_passage_group["end_q"]:
                            m_group = active_passage_group
                        elif marker["q_num"] > active_passage_group["end_q"]:
                            active_passage_group = None
                    if m_group is not None:
                        q_group_id = str(m_group.get("group_id") or "")
                        q_shared_text = str(m_group.get("text") or "").strip()
                        if q_shared_text and q_shared_text not in q_text:
                            q_text = f"{q_shared_text}\n\n{q_text}".strip()
                        if split_passage_and_question and not q_passage_png and m_group.get("png_bytes"):
                            q_passage_png = m_group["png_bytes"]
                            q_passage_w = int(m_group.get("width") or 0)
                            q_passage_h = int(m_group.get("height") or 0)

                    results.append(
                        CroppedQuestion(
                            q_num=marker["q_num"],
                            page_num=page_idx + 1,
                            png_bytes=png_out,
                            width=out_w,
                            height=out_h,
                            extracted_text=q_text,
                            passage_group_id=q_group_id,
                            shared_passage_text=q_shared_text,
                            passage_png_bytes=q_passage_png,
                            passage_width=q_passage_w,
                            passage_height=q_passage_h,
                        )
                    )
                    if is_last_in_col and not trailing_passage_cuts_col:
                        # 이미 ⑤(또는 5)) 선지까지 완결된 문항은 다음 단으로 꼬리를 이어 붙이지 않음
                        q_has_choice_5 = ("⑤" in q_text) or bool(re.search(r"(?:^|\n)\s*5\)", q_text))
                        prev_col_last_q_idx = None if q_has_choice_5 else (len(results) - 1)
                    elif is_last_in_col:
                        prev_col_last_q_idx = None

                # 현재 단의 모든 문항을 먼저 처리한 뒤, 단 하단에 미처 첫 문항을 만나지 못한 [N~M] 지문이 남아 있으면 다음 단으로 이월한다.
                if (
                    pending_passage_y0 is not None
                    and pending_passage_start_q is not None
                    and pending_passage_end_q is not None
                ):
                    p_bottom = 0.895 * page_h
                    p_boxes = [
                        b
                        for b in content_boxes
                        if b[3] > pending_passage_y0 - 4.0 and b[1] < p_bottom
                    ]
                    if p_boxes:
                        p_clip = pymupdf.Rect(
                            max(col_x0 + 2.0, min(b[0] for b in p_boxes) - 10.0),
                            max(0.055 * page_h, min(pending_passage_y0, min(b[1] for b in p_boxes)) - 6.0),
                            min(col_x1 - 2.0, max(b[2] for b in p_boxes) + 10.0),
                            min(0.902 * page_h, max(b[3] for b in p_boxes) + 8.0),
                        )
                        p_pix = page.get_pixmap(dpi=effective_dpi, clip=p_clip, alpha=False)
                        p_txt = "\n".join(
                            ln["text"]
                            for ln in lines
                            if ln["y1"] >= p_clip.y0
                            and ln["y0"] <= p_clip.y1
                            and not _is_footer_or_header_noise(ln["text"], ln["y0"], page_h)
                        )
                        grp_str = f"{pending_passage_start_q}~{pending_passage_end_q}"
                        carried_passage = {
                            "start_q": pending_passage_start_q,
                            "end_q": pending_passage_end_q,
                            "group_id": grp_str,
                            "png_bytes": p_pix.tobytes("png"),
                            "text": p_txt,
                        }
                        active_passage_group = {
                            "start_q": pending_passage_start_q,
                            "end_q": pending_passage_end_q,
                            "group_id": grp_str,
                            "text": p_txt,
                        }
                        prev_col_last_q_idx = None

    results.sort(key=lambda r: r.q_num)
    has_text_layer = total_text_chars >= 60
    return results, has_text_layer


def crop_normalized_box(
    pdf_bytes: bytes,
    page_num: int,
    ymin: int,
    xmin: int,
    ymax: int,
    xmax: int,
    dpi: int = 250,
) -> tuple[bytes, int, int]:
    """0~1000 정규화 좌표를 이용해 스캔 PDF 페이지를 고해상도 크롭한다."""
    effective_dpi = max(220, dpi)
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc[max(0, min(page_num - 1, doc.page_count - 1))]
        w, h = page.rect.width, page.rect.height
        clip = pymupdf.Rect(
            max(0.0, (xmin / 1000.0) * w - 6.0),
            max(0.0, (ymin / 1000.0) * h - 6.0),
            min(w, (xmax / 1000.0) * w + 6.0),
            min(h, (ymax / 1000.0) * h + 6.0),
        )
        pix = page.get_pixmap(dpi=effective_dpi, clip=clip, alpha=False)
        return pix.tobytes("png"), pix.width, pix.height


_ANS_RANGE_MARKER_RE = re.compile(
    r"^(?:\[?\s*(\d{1,2})\s*[~\-～–—]\s*(\d{1,2})\s*\]?\s*[.)번]?|\[\s*(\d{1,2})\s*[~\-～–—]\s*(\d{1,2})\s*\])\s*(.*)$"
)
_ANS_LINE_MARKER_RE = re.compile(
    r"^(?:[•▶▷*]\s*)?(?:\[?\s*(\d{1,2})\s*\]?\s*[.)번]|\[\s*(\d{1,2})\s*\])\s*(.*)$"
)


def _trim_answer_snippet_preserving_explanation(text: str, max_chars: int) -> str:
    """장문(41~42, 43~45) 해설에서 앞부분 긴 [해석] 때문에 뒷부분 [해설]/정답이 잘리지 않도록 보존한다."""
    if len(text) <= max_chars:
        return text
    exp_idx = -1
    for marker in ("[해설]", "해설:", "[풀이]", "[출제의도]", "41.", "42.", "43.", "44.", "45."):
        pos = text.find(marker, 40)
        if pos != -1 and (exp_idx == -1 or pos < exp_idx):
            exp_idx = pos
    if exp_idx != -1 and len(text) - exp_idx < max_chars - 60:
        tail = text[exp_idx:]
        head_budget = max(60, max_chars - len(tail) - 5)
        return f"{text[:head_budget].rstrip()}\n...\n{tail}"
    half = max_chars // 2
    return f"{text[:half].rstrip()}\n...\n{text[-half:].lstrip()}"


def extract_answer_snippets_from_pdf(pdf_bytes: bytes, max_chars_per_q: int = 1200) -> tuple[dict[int, str], bool]:
    """
    텍스트 레이어가 있는 해설지 PDF에서 문항 번호별 해설 텍스트 스니펫을 빠르게 추출한다.
    [41~42], [43~45] 등 세트 해설 헤더도 해당 번호 전체에 매핑하고 [해설] 본문이 잘리지 않게 보존한다.
    """
    full_lines: list[str] = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i in range(doc.page_count):
            txt = (doc[i].get_text("text") or "").strip()
            if txt:
                full_lines.extend(ln.strip() for ln in txt.splitlines() if ln.strip())

    joined_len = sum(len(x) for x in full_lines)
    if joined_len < 20:
        return {}, False

    snippets: dict[int, list[str]] = {}
    cur_qs: list[int] = []
    active_range_qs: list[int] = []
    for ln in full_lines:
        rm = _ANS_RANGE_MARKER_RE.match(ln)
        if rm:
            s_str = rm.group(1) or rm.group(3)
            e_str = rm.group(2) or rm.group(4)
            if s_str and e_str:
                q_s, q_e = int(s_str), int(e_str)
                if 1 <= q_s <= q_e <= 45 and (q_e - q_s) <= 4:
                    cur_qs = list(range(q_s, q_e + 1))
                    active_range_qs = list(cur_qs)
                    for qn in cur_qs:
                        snippets.setdefault(qn, []).append(ln)
                    continue
        m = _ANS_LINE_MARKER_RE.match(ln)
        if m:
            q_str = m.group(1) or m.group(2)
            if q_str:
                q_num = int(q_str)
                if 1 <= q_num <= 45:
                    cur_qs = [q_num]
                    if q_num not in active_range_qs:
                        active_range_qs = []
                    snippets.setdefault(q_num, []).append(ln)
                    continue
        for qn in cur_qs:
            snippets.setdefault(qn, []).append(ln)

    out: dict[int, str] = {}
    for q_num, parts in snippets.items():
        merged = "\n".join(parts).strip()
        if merged:
            out[q_num] = _trim_answer_snippet_preserving_explanation(merged, max_chars_per_q)
    return out, True


_FILENAME_ROLE_SUFFIX_RE = re.compile(
    r"(?:정답\s*[및과]\s*해설|문제\s*[및과]\s*해설|문제\s*[및과]\s*정답|"
    r"정답해설|문제지|해설지|정답지|시험지|해설|정답|풀이|문제)"
    r"(?=$|[\s_.\-\(\)\[\]])"
)
_FILENAME_ROLE_TOKENS = frozenset(
    {
        "문제",
        "문제지",
        "시험지",
        "해설",
        "해설지",
        "정답",
        "정답지",
        "정답해설",
        "정답및해설",
        "정답과해설",
        "풀이",
        "오답",
        "problem",
        "problems",
        "prob",
        "question",
        "questions",
        "answer",
        "answers",
        "ans",
        "solution",
        "solutions",
        "sol",
        "key",
        "copy",
        "사본",
    }
)
_ANS_ONLY_HINT_RE = re.compile(r"(?:해설지?|정답지?|정답\s*[및과]\s*해설|풀이|\banswers?\b|\bsolutions?\b|\bans\b|\bsol\b)", re.I)
_PROB_ONLY_HINT_RE = re.compile(r"(?:문제지?|시험지|\bproblems?\b|\bprobs?\b|\bquestions?\b)", re.I)
_YEAR_TOKEN_RE = re.compile(r"^(?:19|20)\d{2}$")
_MONTH_TOKEN_RE = re.compile(r"^(?:1[0-2]|[1-9])월$|^수능$")
_GRADE_TOKEN_RE = re.compile(r"^고[123]$")


def is_likely_answer_filename(filename: str) -> bool:
    """파일명만 보고 해설/정답지 PDF인지 판별한다."""
    base = unicodedata.normalize("NFC", os.path.basename((filename or "").replace("\\", "/")))
    stem = re.sub(r"\.pdf$", "", base, flags=re.I)
    return bool(_ANS_ONLY_HINT_RE.search(stem)) and not bool(_PROB_ONLY_HINT_RE.search(stem))


def _tokenize_exam_filename(filename: str) -> list[str]:
    base = unicodedata.normalize("NFC", os.path.basename((filename or "").replace("\\", "/")))
    stem = re.sub(r"\.pdf$", "", base, flags=re.I).lower()
    # 로마자 표기(Ⅰ/Ⅱ/I/II)를 아라비아 숫자로 통일해 물리학I <-> 물리학1 매칭 지원
    stem = stem.replace("ⅱ", "2").replace("ⅰ", "1").replace("Ⅱ", "2").replace("Ⅰ", "1")
    stem = re.sub(r"([가-힣])ii(?=$|[\s_.\-\(\)\[\]])", r"\g<1>2", stem)
    stem = re.sub(r"([가-힣])i(?=$|[\s_.\-\(\)\[\]])", r"\g<1>1", stem)
    stem = _FILENAME_ROLE_SUFFIX_RE.sub(" ", stem)
    stem = re.sub(r"(?:^|(?<=[\s_.\-\(\)\[\]]))20(\d{2})(?=년)", lambda m: f"20{m.group(1)}", stem)
    raw_tokens = re.split(r"[\s_.\-\(\)\[\]/,+]+", stem)
    tokens: list[str] = []
    for tok in raw_tokens:
        cleaned = tok.strip()
        if not cleaned or cleaned in _FILENAME_ROLE_TOKENS:
            continue
        if cleaned.endswith("년") and _YEAR_TOKEN_RE.match(cleaned[:-1]):
            cleaned = cleaned[:-1]
        tokens.append(cleaned)
    return tokens


def normalize_exam_filename_stem(filename: str) -> str:
    """문제지/해설지 역할 키워드를 제거한 정규화 파일명 식별 문자열을 반환한다."""
    return " ".join(_tokenize_exam_filename(filename))


def _score_prob_ans_pair(p_tokens: list[str], a_tokens: list[str]) -> int:
    if not p_tokens or not a_tokens:
        return -1
    if p_tokens == a_tokens:
        return 1000

    p_years = {t for t in p_tokens if _YEAR_TOKEN_RE.match(t)}
    a_years = {t for t in a_tokens if _YEAR_TOKEN_RE.match(t)}
    if p_years and a_years and p_years.isdisjoint(a_years):
        return -1

    p_months = {t for t in p_tokens if _MONTH_TOKEN_RE.match(t)}
    a_months = {t for t in a_tokens if _MONTH_TOKEN_RE.match(t)}
    if p_months and a_months and p_months.isdisjoint(a_months):
        return -1

    p_grades = {t for t in p_tokens if _GRADE_TOKEN_RE.match(t)}
    a_grades = {t for t in a_tokens if _GRADE_TOKEN_RE.match(t)}
    if p_grades and a_grades and p_grades.isdisjoint(a_grades):
        return -1

    p_set = set(p_tokens)
    a_set = set(a_tokens)
    p_core = {t for t in p_set if t not in p_years and t not in p_months and t not in p_grades}
    a_core = {t for t in a_set if t not in a_years and t not in a_months and t not in a_grades}

    shared_core = p_core & a_core
    # 둘 다 과목 등 핵심 토큰을 갖고 있는데 겹치는 토큰이 전혀 없으면 서로 다른 시험지/과목
    if p_core and a_core and not shared_core:
        return -1

    shared_all = p_set & a_set
    if not shared_all:
        return -1

    score = 0
    score += len(p_years & a_years) * 30
    score += len(p_months & a_months) * 25
    score += len(p_grades & a_grades) * 15
    score += len(shared_core) * 40
    score -= len(p_core ^ a_core) * 5

    # 연도·월만 같고 핵심 토큰이 비어 있는 단일 과목 세트만 최소 기준 통과
    if not shared_core and not (p_years & a_years and p_months & a_months):
        return -1
    return score if score >= 25 else -1


def pair_problem_and_answer_files(
    problem_names: Sequence[str],
    answer_names: Sequence[str],
) -> list[int | None]:
    """문제지 파일명 목록에 가장 잘 대응하는 해설지 인덱스를 1:1 그리디 매칭한다."""
    n_prob = len(problem_names)
    n_ans = len(answer_names)
    matches: list[int | None] = [None] * n_prob
    if n_prob == 0 or n_ans == 0:
        return matches

    p_toks = [_tokenize_exam_filename(name) for name in problem_names]
    a_toks = [_tokenize_exam_filename(name) for name in answer_names]

    candidates: list[tuple[int, int, int]] = []
    for p_idx in range(n_prob):
        for a_idx in range(n_ans):
            sc = _score_prob_ans_pair(p_toks[p_idx], a_toks[a_idx])
            if sc > 0:
                candidates.append((sc, -p_idx, -a_idx))

    candidates.sort(reverse=True)
    used_p: set[int] = set()
    used_a: set[int] = set()
    for sc, neg_p, neg_a in candidates:
        p_idx = -neg_p
        a_idx = -neg_a
        if p_idx in used_p or a_idx in used_a:
            continue
        matches[p_idx] = a_idx
        used_p.add(p_idx)
        used_a.add(a_idx)

    # 단일 문제지 + 단일 해설지 업로드 시 파일명이 달라도 기본 매칭 제공
    if n_prob == 1 and n_ans == 1 and matches[0] is None:
        matches[0] = 0
    return matches
