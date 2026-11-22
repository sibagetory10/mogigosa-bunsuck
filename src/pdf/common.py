from __future__ import annotations

import io
import os
from typing import Any

import pymupdf

_CIRCLED_NUMS = ("①", "②", "③", "④", "⑤")

def _pick_korean_font_paths() -> tuple[str | None, str | None, str | None]:
    """시스템(Linux/Windows/macOS)에서 명조(시 본문) · 고딕(주석) · 고딕 볼드 폰트 경로를 찾는다 (11,172자 완성형 지원 폰트 우선)."""
    serif_candidates = [
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSerifCJK-Regular.ttc",
        "C:/Windows/Fonts/batang.ttc",
        "C:/Windows/Fonts/H2MJRE.TTF",
        "/System/Library/Fonts/Supplemental/AppleMyungjo.ttf",
    ]
    sans_candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/NanumBarunGothic.ttf",
        "C:/Windows/Fonts/NanumGothic.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    ]
    bold_candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "C:/Windows/Fonts/malgunbd.ttf",
        "C:/Windows/Fonts/NanumBarunGothicBold.ttf",
        "C:/Windows/Fonts/NanumGothicBold.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    ]
    serif_p = next((p for p in serif_candidates if os.path.exists(p)), None)
    sans_p = next((p for p in sans_candidates if os.path.exists(p)), None)
    bold_p = next((p for p in bold_candidates if os.path.exists(p)), None) or sans_p
    return serif_p, sans_p, bold_p


def _build_korean_story_font_bundle() -> tuple[Any | None, str]:
    """
    PyMuPDF Story에서 희귀 한글 음절까지 누락 없이 렌더링되도록
    시스템 한글 폰트 디렉터리를 Archive로 묶고 @font-face CSS를 생성한다.
    """
    _, sans_p, bold_p = _pick_korean_font_paths()
    font_dirs: list[str] = []
    face_rules: list[str] = []

    if sans_p and os.path.isfile(sans_p):
        d = os.path.dirname(sans_p)
        if d and d not in font_dirs:
            font_dirs.append(d)
        fname = os.path.basename(sans_p)
        face_rules.append(
            f"@font-face {{ font-family: 'MogiKorSans'; src: url('{fname}'); font-weight: normal; font-style: normal; }}"
        )
    if bold_p and os.path.isfile(bold_p):
        d = os.path.dirname(bold_p)
        if d and d not in font_dirs:
            font_dirs.append(d)
        fname = os.path.basename(bold_p)
        face_rules.append(
            f"@font-face {{ font-family: 'MogiKorSans'; src: url('{fname}'); font-weight: bold; font-style: normal; }}"
        )

    archive = None
    if font_dirs and hasattr(pymupdf, "Archive"):
        try:
            archive = pymupdf.Archive(font_dirs[0])
            for extra_d in font_dirs[1:]:
                archive.add(extra_d)
        except Exception:
            archive = None
            face_rules = []

    return archive, "\n".join(face_rules)
def _render_story_pdf(full_html: str, css_text: str) -> bytes:
    """HTML + CSS를 PyMuPDF Story로 A4 PDF 바이트로 렌더링한다."""
    font_archive, font_face_css = _build_korean_story_font_bundle()
    eff_css = f"{font_face_css}\n{css_text}" if font_face_css else css_text
    story_kwargs: dict[str, Any] = {"html": full_html, "user_css": eff_css}
    if font_archive is not None:
        story_kwargs["archive"] = font_archive
    story = pymupdf.Story(**story_kwargs)
    writer_buf = io.BytesIO()
    mediabox = pymupdf.paper_rect("a4")
    where = mediabox + (36, 40, -36, -42)
    writer = pymupdf.DocumentWriter(writer_buf)
    more = True
    while more:
        dev = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(dev, None)
        writer.end_page()
    writer.close()
    return writer_buf.getvalue()
def merge_korean_analysis_pdfs(pdf_bytes_list: list[bytes]) -> bytes:
    """지문별로 저장된 국어 분석지 PDF 바이트들을 순서대로 이어 붙이고 전체 통합 쪽번호를 덧입힌다."""
    valid_pdfs = [b for b in pdf_bytes_list if b]
    if not valid_pdfs:
        raise ValueError("이어 붙일 국어 분석지 PDF 데이터가 없습니다.")

    merged = pymupdf.open()
    try:
        for raw_pdf in valid_pdfs:
            with pymupdf.open("pdf", raw_pdf) as src_doc:
                merged.insert_pdf(src_doc)

        total_pages = merged.page_count
        if total_pages == 0:
            raise ValueError("병합된 국어 분석지 PDF에 페이지가 없습니다.")

        for page_idx in range(total_pages):
            page = merged[page_idx]
            rect = pymupdf.Rect(0, page.rect.height - 28, page.rect.width, page.rect.height - 10)
            page.insert_textbox(
                rect,
                f"- {page_idx + 1} / {total_pages} -",
                fontsize=8.5,
                fontname="helv",
                color=(0.39, 0.45, 0.55),
                align=pymupdf.TEXT_ALIGN_CENTER,
            )

        return merged.tobytes(garbage=3, deflate=True)
    finally:
        merged.close()
