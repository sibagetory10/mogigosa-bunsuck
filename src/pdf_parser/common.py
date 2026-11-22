from __future__ import annotations

import io
import re
from dataclasses import dataclass

import pymupdf
from PIL import Image

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
