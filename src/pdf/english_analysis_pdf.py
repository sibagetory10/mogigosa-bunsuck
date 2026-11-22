from __future__ import annotations

import base64
import html
import io
import re
from datetime import datetime
from typing import Any

import pymupdf
from PIL import Image

from src.diagram_renderer import render_korean_diagram_png
from src.pdf.common import _build_korean_story_font_bundle
from src.pdf.korean_analysis_pdf import _format_structure_flow_html
from src.pdf_parser.english_source_extractor import split_english_sentences_clean

# KS X 1001 완성형 한글 폰트에서 모두 안전하게 지원되는 ①~⑮ 원문자
_EXT_CIRCLED_NUMS: tuple[str, ...] = (
    "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩",
    "⑪", "⑫", "⑬", "⑭", "⑮",
)


def _format_sentence_badge(num: int) -> str:
    if 1 <= num <= len(_EXT_CIRCLED_NUMS):
        return _EXT_CIRCLED_NUMS[num - 1]
    return f"({num})"


def _build_english_landscape_css(scale: float = 1.0, font_face_css: str = "") -> str:
    """A4 가로(Landscape) 1페이지 1지문 영어 지문 분석지용 스케일 가변 CSS를 생성한다."""
    s = max(0.58, min(1.05, float(scale)))
    return f"""
    {font_face_css}
    body {{
        font-family: "MogiKorSans", "Noto Sans CJK KR", "NanumBarunGothic", "NanumGothic", "Malgun Gothic", "Apple SD Gothic Neo", sans-serif;
        font-size: {8.6 * s:.2f}pt;
        line-height: 1.42;
        color: #0F172A;
        margin: 0;
        padding: 0;
    }}
    .eng-l-hdr {{
        border: 1.2pt solid #1E293B;
        border-radius: 4pt;
        background-color: #F8FAFC;
        padding: {5.5 * s:.2f}pt {9 * s:.2f}pt;
    }}
    .eng-l-hdr-top {{
        font-size: {7.6 * s:.2f}pt;
        font-weight: bold;
        color: #1D4ED8;
    }}
    .eng-l-hdr-title {{
        font-size: {12.2 * s:.2f}pt;
        font-weight: bold;
        color: #0F172A;
        margin-top: {1.5 * s:.2f}pt;
    }}
    .eng-l-sec {{
        font-size: {9.2 * s:.2f}pt;
        font-weight: bold;
        color: #1E3A8A;
        border-left: 3pt solid #2563EB;
        padding-left: {5 * s:.2f}pt;
        margin-top: {2 * s:.2f}pt;
        margin-bottom: {4 * s:.2f}pt;
    }}
    .eng-l-theme {{
        background-color: #EFF6FF;
        border: 0.8pt solid #93C5FD;
        border-radius: 3.5pt;
        padding: {5.5 * s:.2f}pt {7.5 * s:.2f}pt;
        margin-bottom: {4.5 * s:.2f}pt;
        font-size: {8.3 * s:.2f}pt;
        line-height: 1.42;
    }}
    .eng-l-flow {{
        background-color: #F0FDF4;
        border: 0.8pt solid #86EFAC;
        border-radius: 3.5pt;
        padding: {5.5 * s:.2f}pt {7.5 * s:.2f}pt;
        margin-bottom: {4.5 * s:.2f}pt;
        font-size: {8.1 * s:.2f}pt;
        line-height: 1.42;
    }}
    .eng-l-footnote {{
        background-color: #F8FAFC;
        border: 0.6pt solid #CBD5E1;
        border-radius: 3pt;
        padding: {3.2 * s:.2f}pt {6.5 * s:.2f}pt;
        margin-bottom: {4 * s:.2f}pt;
        font-size: {7.3 * s:.2f}pt;
        line-height: 1.35;
        color: #475569;
    }}
    .eng-l-diag-wrap {{
        margin-top: {2 * s:.2f}pt;
        text-align: center;
    }}
    .eng-l-diag-img {{
        max-width: 100%;
        height: auto;
    }}
    .eng-sent-card {{
        border: 0.7pt solid #CBD5E1;
        border-radius: 3.5pt;
        background-color: #FFFFFF;
        padding: {4.5 * s:.2f}pt {7 * s:.2f}pt;
        margin-bottom: {4 * s:.2f}pt;
        break-inside: avoid;
        page-break-inside: avoid;
    }}
    .eng-sent-num {{
        font-weight: bold;
        color: #1E3A8A;
        font-size: {8.5 * s:.2f}pt;
        margin-right: 2.5pt;
    }}
    .eng-sent-eng {{
        font-size: {8.3 * s:.2f}pt;
        font-weight: bold;
        color: #0F172A;
        line-height: 1.4;
        margin-bottom: {1.8 * s:.2f}pt;
    }}
    .eng-sent-kor {{
        font-size: {7.9 * s:.2f}pt;
        color: #334155;
        line-height: 1.38;
        margin-bottom: 0;
    }}
    """


def _choose_balanced_split_index(weights: list[float]) -> int:
    """우측 2열 서브컬럼의 누적 텍스트 높이가 균등하도록 분할 인덱스를 결정한다."""
    n = len(weights)
    if n <= 1:
        return 1
    total = sum(weights)
    half = total / 2.0
    acc = 0.0
    best_idx = 1
    best_diff = float("inf")
    for idx in range(1, n):
        acc += weights[idx - 1]
        diff = abs(acc - half)
        if diff < best_diff:
            best_diff = diff
            best_idx = idx
    return best_idx


def _render_single_english_landscape_page(
    item: dict[str, Any],
    doc_title: str,
    today_str: str,
    include_diagrams: bool,
) -> bytes:
    """영어 지문 1개를 A4 가로(842x595pt) 정확히 1페이지 안에 좌·우 분할 레이아웃으로 렌더링한다."""
    an = item["analysis"]
    q_label = str(item.get("q_label") or item.get("exam_name") or "영어 독해").strip()
    orig_type = str(item.get("orig_type") or "영어 독해").strip()
    src_passage = str(item.get("passage_text") or "").strip()

    font_archive, font_face_css = _build_korean_story_font_bundle()

    def _make_story(html_str: str, css_str: str) -> Any:
        kw: dict[str, Any] = {"html": html_str, "user_css": css_str}
        if font_archive is not None:
            kw["archive"] = font_archive
        return pymupdf.Story(**kw)

    mediabox = pymupdf.paper_rect("a4-l")  # 842 x 595 pt
    page_w = mediabox.width
    page_h = mediabox.height
    margin_x = 24.0
    margin_top = 18.0
    margin_bottom = 26.0
    col_gap = 12.0

    # 시각 도식 PNG 사전 생성
    diag_b64 = ""
    diag_aspect = 1.0
    v_diag = getattr(an, "visual_diagram", None)
    if include_diagrams and v_diag is not None and (
        getattr(v_diag, "nodes", None)
        or getattr(v_diag, "graph_spec", None)
        or getattr(v_diag, "emotion_points", None)
    ):
        try:
            diag_png = render_korean_diagram_png(v_diag, compact_vertical=True)
            with Image.open(io.BytesIO(diag_png)) as _dim:
                dw_px, dh_px = _dim.size
            diag_aspect = float(max(1, dw_px)) / float(max(1, dh_px))
            diag_b64 = base64.b64encode(diag_png).decode("ascii")
        except Exception:
            diag_b64 = ""

    content_w = page_w - 2 * margin_x
    left_ratio = 0.43 if diag_b64 else 0.29
    left_w = content_w * left_ratio
    right_w = content_w - left_w - col_gap

    header_html = f"""
    <html><body>
      <div class="eng-l-hdr">
        <div class="eng-l-hdr-top">{html.escape(doc_title)} · {html.escape(orig_type)} &nbsp;|&nbsp; 출처: {html.escape(q_label)} &nbsp;|&nbsp; 생성일: {today_str}</div>
        <div class="eng-l-hdr-title">{html.escape(an.title)}</div>
      </div>
    </body></html>
    """

    flow_html_inner = _format_structure_flow_html(an.structure_flow)
    fn_list = [str(f).strip() for f in (getattr(an, "footnotes", None) or []) if str(f).strip()]
    fn_html = ""
    if fn_list:
        fn_joined = " &nbsp;·&nbsp; ".join(html.escape(f) for f in fn_list)
        fn_html = f"<div class='eng-l-footnote'><b>* 각주:</b> {fn_joined}</div>"

    left_top_html = f"""
    <html><body>
      <div class="eng-l-sec">1. 핵심 주제 및 논리 전개 흐름</div>
      <div class="eng-l-theme">
        <div><b>핵심 주제</b></div>
        <div>{html.escape(an.core_theme)}</div>
      </div>
      <div class="eng-l-flow">
        <div><b>논리 전개 흐름</b></div>
        <div>{flow_html_inner}</div>
      </div>
      {fn_html}
    </body></html>
    """

    sent_sec_num = "3" if diag_b64 else "2"
    raw_sent_list = list(getattr(an, "sentence_analyses", None) or [])
    sent_pairs: list[tuple[int, str, str]] = []
    if raw_sent_list:
        for s_idx, sa in enumerate(raw_sent_list):
            s_num = int(getattr(sa, "sentence_num", s_idx + 1) or (s_idx + 1))
            orig_eng = (
                getattr(sa, "original_english", "")
                or getattr(sa, "chunked_english", "")
                or ""
            ).strip()
            kor_txt = (getattr(sa, "korean_translation", "") or "").strip()
            if orig_eng or kor_txt:
                sent_pairs.append((s_num, orig_eng, kor_txt))
    elif src_passage:
        for s_idx, s_txt in enumerate(split_english_sentences_clean(src_passage), start=1):
            sent_pairs.append((s_idx, s_txt, ""))

    sent_cards_html: list[str] = []
    card_weights: list[float] = []
    expected_badges: list[str] = []
    last_sent_check_words: list[str] = []

    for p_idx, (s_num, orig_eng, kor_txt) in enumerate(sent_pairs):
        num_badge = _format_sentence_badge(s_num)
        expected_badges.append(num_badge)
        if p_idx == len(sent_pairs) - 1:
            last_words = re.findall(r"[A-Za-z]{3,}", orig_eng)
            if last_words:
                last_sent_check_words = [last_words[0].lower(), last_words[-1].lower()]

        kor_div = f"<div class='eng-sent-kor'>{html.escape(kor_txt)}</div>" if kor_txt else ""
        card_inner = (
            "<div class='eng-sent-card'>"
            f"<div class='eng-sent-eng'><span class='eng-sent-num'>{num_badge}</span>{html.escape(orig_eng)}</div>"
            f"{kor_div}"
            "</div>"
        )
        sent_cards_html.append(card_inner)
        card_weights.append(len(orig_eng) * 1.15 + len(kor_txt) * 1.30 + 45.0)

    right_html = f"""
    <html><body>
      <div class="eng-l-sec">{sent_sec_num}. 문장별 정밀 구문 분석</div>
      {''.join(sent_cards_html)}
    </body></html>
    """
    mid_idx = _choose_balanced_split_index(card_weights)
    right_html_col1 = f"""
    <html><body>
      <div class="eng-l-sec">{sent_sec_num}. 문장별 정밀 구문 분석</div>
      {''.join(sent_cards_html[:mid_idx])}
    </body></html>
    """
    right_html_col2 = f"""
    <html><body>
      <div class="eng-l-sec" style="visibility:hidden;">{sent_sec_num}. 문장별 정밀 구문 분석</div>
      {''.join(sent_cards_html[mid_idx:])}
    </body></html>
    """

    def _try_render_1page(scale: float, two_col_right: bool = False, canvas_h: float = page_h) -> bytes | None:
        css_txt = _build_english_landscape_css(scale, font_face_css=font_face_css)
        hdr_story = _make_story(header_html, css_txt)
        l_top_story = _make_story(left_top_html, css_txt)

        buf = io.BytesIO()
        c_rect = pymupdf.Rect(0, 0, page_w, canvas_h)
        writer = pymupdf.DocumentWriter(buf)
        dev = writer.begin_page(c_rect)

        hdr_rect = pymupdf.Rect(margin_x, margin_top, page_w - margin_x, canvas_h - margin_bottom)
        more_h, filled_h = hdr_story.place(hdr_rect)
        if more_h:
            writer.end_page()
            writer.close()
            return None
        hdr_story.draw(dev, None)

        body_top = float(filled_h[3]) + 7.0
        body_bottom = canvas_h - margin_bottom

        l_top_rect = pymupdf.Rect(margin_x, body_top, margin_x + left_w, body_bottom)
        more_lt, filled_lt = l_top_story.place(l_top_rect)
        if more_lt or float(filled_lt[3]) > body_bottom + 1.0:
            writer.end_page()
            writer.close()
            return None
        l_top_story.draw(dev, None)

        if diag_b64:
            diag_top = float(filled_lt[3]) + 5.0
            avail_h = body_bottom - diag_top
            if avail_h < 95.0:
                writer.end_page()
                writer.close()
                return None
            max_img_h = max(60.0, avail_h - 20.0)
            fit_w = min(left_w - 4.0, max_img_h * diag_aspect)
            diag_img_w = max(140, int(fit_w))
            diag_only_html = (
                "<html><body>"
                "<div class='eng-l-sec' style='margin-top:0;'>2. 한눈에 보는 논리 · 구조 도식</div>"
                f"<div class='eng-l-diag-wrap'><img class='eng-l-diag-img' width='{diag_img_w}' src='data:image/png;base64,{diag_b64}'/></div>"
                "</body></html>"
            )
            diag_story = _make_story(diag_only_html, css_txt)
            diag_rect = pymupdf.Rect(margin_x, diag_top, margin_x + left_w, body_bottom)
            more_d, _ = diag_story.place(diag_rect)
            if more_d:
                diag_img_w = max(120, int(fit_w * 0.92))
                diag_only_html = (
                    "<html><body>"
                    "<div class='eng-l-sec' style='margin-top:0;'>2. 한눈에 보는 논리 · 구조 도식</div>"
                    f"<div class='eng-l-diag-wrap'><img class='eng-l-diag-img' width='{diag_img_w}' src='data:image/png;base64,{diag_b64}'/></div>"
                    "</body></html>"
                )
                diag_story = _make_story(diag_only_html, css_txt)
                more_d, _ = diag_story.place(diag_rect)
                if more_d:
                    writer.end_page()
                    writer.close()
                    return None
            diag_story.draw(dev, None)

        r_start_x = margin_x + left_w + col_gap
        if not two_col_right:
            r_story = _make_story(right_html, css_txt)
            r_rect = pymupdf.Rect(r_start_x, body_top, page_w - margin_x, body_bottom)
            more_r, filled_r = r_story.place(r_rect)
            if more_r or float(filled_r[3]) > body_bottom + 1.0:
                writer.end_page()
                writer.close()
                return None
            r_story.draw(dev, None)
        else:
            sub_gap = 8.0
            sub_w = (right_w - sub_gap) / 2.0
            r_rect1 = pymupdf.Rect(r_start_x, body_top, r_start_x + sub_w, body_bottom)
            r_rect2 = pymupdf.Rect(r_start_x + sub_w + sub_gap, body_top, page_w - margin_x, body_bottom)
            r_story1 = _make_story(right_html_col1, css_txt)
            more_r1, filled_r1 = r_story1.place(r_rect1)
            if more_r1 or float(filled_r1[3]) > body_bottom + 1.0:
                writer.end_page()
                writer.close()
                return None
            r_story1.draw(dev, None)
            if sent_cards_html[mid_idx:]:
                r_story2 = _make_story(right_html_col2, css_txt)
                more_r2, filled_r2 = r_story2.place(r_rect2)
                if more_r2 or float(filled_r2[3]) > body_bottom + 1.0:
                    writer.end_page()
                    writer.close()
                    return None
                r_story2.draw(dev, None)

        writer.end_page()
        writer.close()
        pdf_out = buf.getvalue()
        if expected_badges:
            with pymupdf.open("pdf", pdf_out) as chk_doc:
                rendered_txt = chk_doc[0].get_text("text") if chk_doc.page_count else ""
            if any(b not in rendered_txt for b in expected_badges):
                return None
            if last_sent_check_words:
                low_rendered = rendered_txt.lower()
                if any(w not in low_rendered for w in last_sent_check_words):
                    return None
        return pdf_out

    # 1) 우측 1컬럼으로 들어가는 최대 스케일 탐색
    for sc in (1.0, 0.92, 0.85, 0.78, 0.72, 0.66):
        out_b = _try_render_1page(sc, two_col_right=False, canvas_h=page_h)
        if out_b is not None:
            return out_b

    # 2) 문장 수가 많은 경우 우측 영역을 2열 서브컬럼으로 자동 분할
    for sc in (0.85, 0.78, 0.72, 0.66, 0.60):
        out_b = _try_render_1page(sc, two_col_right=True, canvas_h=page_h)
        if out_b is not None:
            return out_b

    # 3) 초장문 지문 최종 폴백: 세로로 넉넉한 가상 캔버스에 그린 뒤 A4 가로 1페이지에 비율 유지 축소
    for virt_h in (760.0, 960.0, 1280.0, 1600.0):
        tall_b = _try_render_1page(0.75 if virt_h < 1600.0 else 0.62, two_col_right=True, canvas_h=virt_h)
        if tall_b is not None:
            src_doc = pymupdf.open("pdf", tall_b)
            dst_doc = pymupdf.open()
            try:
                dst_page = dst_doc.new_page(width=page_w, height=page_h)
                target_rect = pymupdf.Rect(0, 0, page_w, page_h - 14.0)
                dst_page.show_pdf_page(target_rect, src_doc, 0, keep_proportion=True)
                return dst_doc.tobytes(garbage=3, deflate=True)
            finally:
                src_doc.close()
                dst_doc.close()

    return _try_render_1page(0.60, two_col_right=True, canvas_h=1600.0) or b""


def generate_english_passage_analysis_pdf(
    analyses: list[dict[str, Any]],
    doc_title: str = "수능·모의고사 영어 지문 분석지",
    include_diagrams: bool = True,
) -> bytes:
    """
    영어 지문 분석 결과(EnglishPassageAnalysisOutput + 출처 메타) 리스트를
    A4 가로(Landscape) 1페이지당 1지문 레이아웃 PDF로 렌더링한다.
    구성: [1] 핵심 주제 & 논리 전개 흐름, [2] 이해용 시각 도식, [3] 문장별 정밀 구문 분석 (원문·해설지 해석).
    """
    if not analyses:
        raise ValueError("생성할 영어 지문 분석 데이터가 없습니다.")

    today_str = datetime.now().strftime("%Y.%m.%d")
    page_pdfs: list[bytes] = []
    for item in analyses:
        p_bytes = _render_single_english_landscape_page(
            item=item,
            doc_title=doc_title,
            today_str=today_str,
            include_diagrams=include_diagrams,
        )
        if p_bytes:
            page_pdfs.append(p_bytes)

    if not page_pdfs:
        raise ValueError("영어 지문 분석지 PDF 페이지를 생성하지 못했습니다.")

    merged = pymupdf.open()
    try:
        for raw_p in page_pdfs:
            with pymupdf.open("pdf", raw_p) as src_doc:
                merged.insert_pdf(src_doc)
        return merged.tobytes(garbage=3, deflate=True)
    finally:
        merged.close()
