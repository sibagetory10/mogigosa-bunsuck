from __future__ import annotations

import base64
import html
import io
import math
import os
import re
from datetime import datetime
from typing import Any

import pymupdf
from PIL import Image

from src.diagram_renderer import render_korean_diagram_png
from src.models import ConceptWhitelist, QuestionRecord, RewrittenQuestion
from src.text_formatter import (
    format_chemical_scripts,
    format_explanation_text,
    render_english_inline_html,
    split_english_stem_blocks,
    split_stem_blocks,
    strip_english_embedded_choices,
    unicode_scripts_to_html_tags,
)

CIRCLED_NUMS = ["①", "②", "③", "④", "⑤"]

EXAM_CSS = """
body {
    font-family: "NanumGothic", "Malgun Gothic", sans-serif;
    font-size: 9.4pt;
    line-height: 1.52;
    color: #0F172A;
    margin: 0;
    padding: 0;
}
sub {
    font-size: 72%;
    vertical-align: sub;
}
sup {
    font-size: 72%;
    vertical-align: super;
}
.exam-banner {
    border: 1.2pt solid #1E293B;
    border-radius: 4pt;
    padding: 7pt 10pt;
    background-color: #F8FAFC;
}
.exam-banner-sub {
    font-size: 8.2pt;
    font-weight: bold;
    color: #1D4ED8;
    letter-spacing: 0.04em;
}
.exam-banner-title {
    font-size: 14.5pt;
    font-weight: bold;
    color: #0F172A;
    margin-top: 2pt;
}
.student-box {
    width: 100%;
    border-collapse: collapse;
    margin-top: 5pt;
    font-size: 8.4pt;
    background-color: #FFFFFF;
}
.student-box td {
    border: 0.7pt solid #475569;
    padding: 3pt 6pt;
    color: #1E293B;
}
.double-rule-top {
    border-top: 2pt solid #0F172A;
    margin-top: 6pt;
}
.double-rule-bot {
    border-top: 0.7pt solid #334155;
    margin-top: 2pt;
}
.q-wrap {
    margin-bottom: 6pt;
}
.q-header {
    margin-bottom: 4pt;
    font-size: 9.2pt;
    color: #0F172A;
}
.q-num {
    font-weight: bold;
    font-size: 11.2pt;
    color: #1E3A8A;
}
.badge {
    background-color: #DBEAFE;
    color: #1E40AF;
    font-size: 7.4pt;
    padding: 1pt 4.5pt;
    border-radius: 3pt;
    font-weight: bold;
}
.badge-orig {
    background-color: #DCFCE7;
    color: #166534;
    font-size: 7.4pt;
    padding: 1pt 4.5pt;
    border-radius: 3pt;
    font-weight: bold;
}
.meta {
    color: #475569;
    font-size: 7.8pt;
}
.stem-p {
    margin-top: 2pt;
    margin-bottom: 4pt;
    word-break: keep-all;
}
.stem-line {
    margin-bottom: 2.5pt;
}
.stem-line:last-child { margin-bottom: 0; }
.data-line {
    margin-bottom: 2.2pt;
}
.data-line:last-child { margin-bottom: 0; }
.pdf-data-box {
    border: 0.8pt solid #94A3B8;
    border-left: 2.6pt solid #334155;
    background-color: #F8FAFC;
    padding: 5pt 7pt;
    margin: 5pt 0;
    border-radius: 3pt;
    font-size: 8.9pt;
    line-height: 1.48;
}
.pdf-data-title {
    font-size: 7.8pt;
    font-weight: bold;
    color: #334155;
    margin-bottom: 2pt;
}
.pdf-bogi-box {
    border: 1pt solid #1E293B;
    padding: 5pt 7pt;
    margin: 5pt 0;
    background-color: #FFFFFF;
}
.pdf-bogi-title {
    text-align: center;
    font-weight: bold;
    font-size: 8.6pt;
    letter-spacing: 0.25em;
    color: #0F172A;
    border-bottom: 0.6pt solid #CBD5E1;
    padding-bottom: 2.5pt;
    margin-bottom: 4pt;
}
.bogi-tbl {
    width: 100%;
    border-collapse: collapse;
    font-size: 9pt;
    line-height: 1.48;
}
.bogi-tbl td {
    vertical-align: top;
    padding: 1.2pt 0;
}
.bogi-mk {
    width: 16pt;
    font-weight: bold;
    color: #1E3A8A;
}
.choices {
    margin-top: 4pt;
    margin-left: 4pt;
}
.choice-item {
    margin-bottom: 2.5pt;
    word-break: keep-all;
}
.choice-grid {
    width: 100%;
    border-collapse: collapse;
    margin-top: 4pt;
    font-size: 9.1pt;
}
.choice-grid td {
    padding: 2pt 4pt 2pt 0;
    vertical-align: top;
    word-break: keep-all;
}
.ref-note {
    margin-top: 4pt;
    color: #1E40AF;
    font-weight: bold;
    font-size: 7.6pt;
}
.vis-note {
    margin-top: 2pt;
    padding: 3pt 5pt;
    background-color: #FFFBEB;
    border-left: 2pt solid #D97706;
    color: #92400E;
    font-size: 7.6pt;
}
.ans-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 4pt;
    margin-bottom: 10pt;
    font-size: 8.5pt;
}
.ans-table th, .ans-table td {
    border: 0.65pt solid #94A3B8;
    padding: 3.5pt 5pt;
    text-align: center;
}
.ans-table th {
    background-color: #EFF6FF;
    color: #1E3A8A;
    font-weight: bold;
}
.exp-card {
    border: 0.7pt solid #CBD5E1;
    background-color: #F8FAFC;
    padding: 6pt 7pt;
    margin-bottom: 6pt;
    border-radius: 4pt;
    font-size: 8.7pt;
    line-height: 1.48;
}
.exp-title {
    font-weight: bold;
    font-size: 9.2pt;
    color: #0F172A;
    margin-bottom: 3pt;
    border-bottom: 0.5pt solid #E2E8F0;
    padding-bottom: 2pt;
}
.exp-ans {
    color: #DC2626;
    font-weight: bold;
}
.exp-sub {
    font-size: 8pt;
    color: #1D4ED8;
    margin-bottom: 3pt;
}
.eng-u {
    text-decoration: underline;
    font-weight: bold;
}
.eng-blank {
    text-decoration: underline;
    font-weight: bold;
}
.eng-inst {
    font-weight: bold;
    margin-top: 2pt;
    margin-bottom: 4pt;
    word-break: keep-all;
}
.eng-given-box {
    border: 1pt solid #1E293B;
    background-color: #F8FAFC;
    padding: 5pt 7pt;
    margin: 4pt 0 5pt 0;
    font-size: 9.1pt;
    line-height: 1.5;
}
.eng-given-title {
    font-size: 7.6pt;
    font-weight: bold;
    color: #1E3A8A;
    margin-bottom: 2pt;
}
.eng-passage {
    margin-top: 2pt;
    margin-bottom: 4pt;
    font-size: 9.2pt;
    line-height: 1.52;
}
.eng-order-tbl {
    width: 100%;
    border-collapse: collapse;
    margin: 4pt 0;
    font-size: 9.1pt;
    line-height: 1.5;
}
.eng-order-tbl td {
    vertical-align: top;
    padding: 2.5pt 0;
}
.eng-order-mk {
    width: 20pt;
    font-weight: bold;
    color: #1E3A8A;
}
.eng-summary-arrow {
    text-align: center;
    font-weight: bold;
    font-size: 9.5pt;
    color: #334155;
    margin: 2pt 0;
}
.eng-summary-box {
    border: 1pt solid #1E3A8A;
    background-color: #F8FAFC;
    padding: 5pt 7pt;
    margin: 2pt 0 5pt 0;
    font-size: 9.1pt;
    line-height: 1.5;
}
.eng-footnote {
    font-size: 7.8pt;
    color: #475569;
    margin-top: 2pt;
    margin-bottom: 3pt;
}
"""


def _esc_pdf(text: str) -> str:
    formatted = format_chemical_scripts(text or "")
    escaped = html.escape(formatted).replace("\n", "<br/>")
    return unicode_scripts_to_html_tags(escaped)


def _format_answer_badge(ans: str) -> str:
    cleaned = (ans or "").strip()
    mapping = {"1": "①", "2": "②", "3": "③", "4": "④", "5": "⑤"}
    return mapping.get(cleaned, cleaned)


def _extract_short_page_ref(concept_ref: str) -> str:
    if not concept_ref:
        return "-"
    m = re.search(r"\(([^()]+)\)\s*$", concept_ref.strip())
    if m:
        return m.group(1).strip()
    cleaned = concept_ref.replace("📖 참고 개념:", "").replace("📚 출처:", "").strip()
    return cleaned[:26] if cleaned else "-"


def _build_english_stem_blocks_html(stem_text: str) -> str:
    blocks = split_english_stem_blocks(strip_english_embedded_choices(stem_text))
    parts: list[str] = []
    for kind, content in blocks:
        if kind == "instruction":
            parts.append(f"<div class='eng-inst'>{render_english_inline_html(str(content))}</div>")
        elif kind == "given_box":
            g_title = str(content.get("title") or "주어진 글") if isinstance(content, dict) else "주어진 글"
            g_body = str(content.get("body") or "") if isinstance(content, dict) else str(content)
            parts.append(
                f"""
                <div class="eng-given-box">
                  <div class="eng-given-title">[{html.escape(g_title)}]</div>
                  <div>{render_english_inline_html(g_body)}</div>
                </div>
                """
            )
        elif kind == "order_chunks":
            rows = "".join(
                f"<tr><td class='eng-order-mk'>{html.escape(mk)}</td><td>{render_english_inline_html(txt)}</td></tr>"
                for mk, txt in content
            )
            parts.append(f"<table class='eng-order-tbl'>{rows}</table>")
        elif kind == "summary_box":
            parts.append(
                f"""
                <div class="eng-summary-arrow">↓</div>
                <div class="eng-summary-box">{render_english_inline_html(str(content))}</div>
                """
            )
        elif kind == "footnote":
            parts.append(f"<div class='eng-footnote'>{render_english_inline_html(str(content))}</div>")
        else:
            parts.append(f"<div class='eng-passage'>{render_english_inline_html(str(content))}</div>")
    return "".join(parts)


def _build_stem_blocks_html(stem_text: str, is_english: bool = False) -> str:
    if is_english:
        return _build_english_stem_blocks_html(stem_text)
    blocks = split_stem_blocks(stem_text)
    parts: list[str] = []
    for kind, content in blocks:
        if kind == "data":
            if isinstance(content, dict):
                d_title = str(content.get("title") or "< 자 료 >")
                d_body = str(content.get("body") or "")
            else:
                d_title = "< 자 료 >"
                d_body = str(content)
            body_lines = "".join(
                "<div style='height:3pt;'></div>"
                if not ln.strip()
                else f"<div class='data-line'>{_esc_pdf(ln.strip())}</div>"
                for ln in d_body.split("\n")
            )
            parts.append(
                f"""
                <div class="pdf-data-box">
                  <div class="pdf-data-title">{html.escape(d_title)}</div>
                  <div>{body_lines}</div>
                </div>
                """
            )
        elif kind == "bogi":
            rows = "".join(
                f"<tr><td class='bogi-mk'>{html.escape(mk)}</td><td>{_esc_pdf(txt)}</td></tr>"
                for mk, txt in content
            )
            parts.append(
                f"""
                <div class="pdf-bogi-box">
                  <div class="pdf-bogi-title">〈 보 기 〉</div>
                  <table class="bogi-tbl">{rows}</table>
                </div>
                """
            )
        else:
            stem_lines = "".join(
                "<div style='height:3pt;'></div>"
                if not ln.strip()
                else f"<div class='stem-line'>{_esc_pdf(ln.strip())}</div>"
                for ln in str(content).split("\n")
            )
            parts.append(f"<div class='stem-p'>{stem_lines}</div>")
    return "".join(parts)


def _build_choices_html(choices: list[str], is_english: bool = False) -> str:
    if not choices:
        return ""
    capped = choices[:5]
    if is_english:
        formatted_items = [c.strip() for c in capped]
    else:
        formatted_items = [format_chemical_scripts(c.strip()) for c in capped]
    max_len = max((len(c) for c in formatted_items), default=0)
    total_len = sum(len(c) + 4 for c in formatted_items)

    def _fmt_choice_body(i: int, txt: str) -> str:
        sym = CIRCLED_NUMS[i] if i < len(CIRCLED_NUMS) else f"{i + 1})"
        cleaned = txt.strip()
        if cleaned == sym or not cleaned:
            return f"<b>{sym}</b>"
        rendered = render_english_inline_html(cleaned) if is_english else _esc_pdf(cleaned)
        return f"<b>{sym}</b> {rendered}"

    def _cell(i: int, txt: str) -> str:
        return f"<td>{_fmt_choice_body(i, txt)}</td>"

    # 짧은 선지는 평가원 모의고사처럼 1줄 또는 2줄(3+2) 가로 배열
    if len(capped) <= 5 and total_len <= 28 and all("\n" not in c for c in formatted_items):
        cells = "".join(_cell(i, c) for i, c in enumerate(formatted_items))
        return f"<table class='choice-grid'><tr>{cells}</tr></table>"
    if len(capped) == 5 and max_len <= 18 and all("\n" not in c for c in formatted_items):
        row1 = "".join(_cell(i, formatted_items[i]) for i in range(3))
        row2 = "".join(_cell(i, formatted_items[i]) for i in range(3, 5)) + "<td></td>"
        return f"<table class='choice-grid'><tr>{row1}</tr><tr>{row2}</tr></table>"

    rows = []
    for i, c in enumerate(formatted_items):
        rows.append(f"<div class='choice-item'>{_fmt_choice_body(i, c)}</div>")
    return f"<div class='choices'>{''.join(rows)}</div>"


def _measure_html_height(html_str: str, width: float, max_h: float = 760.0) -> float:
    """스크래치 페이지에서 HTML 박스의 비축소 실제 높이를 측정한다."""
    scratch_doc = pymupdf.open()
    try:
        sp = scratch_doc.new_page(width=max(600.0, width + 40.0), height=max_h + 40.0)
        spare_h, scale = sp.insert_htmlbox(
            pymupdf.Rect(0, 0, width, max_h),
            html_str,
            css=EXAM_CSS,
        )
        if scale <= 0:
            return max_h
        used = max_h - max(0.0, spare_h)
        return max(16.0, used + 4.0)
    finally:
        scratch_doc.close()


def _draw_page_footer(page: pymupdf.Page, page_num: int, two_column: bool, top_line_y: float = 38.0) -> None:
    w, h = page.rect.width, page.rect.height
    if two_column:
        page.draw_line(
            pymupdf.Point(w / 2.0, top_line_y),
            pymupdf.Point(w / 2.0, h - 38.0),
            color=(0.76, 0.80, 0.86),
            width=0.65,
        )
    page.insert_htmlbox(
        pymupdf.Rect(30, h - 30, w - 30, h - 10),
        f"<div style='text-align:center; font-size:8.5pt; color:#64748B;'>- {page_num} -</div>",
        css=EXAM_CSS,
    )


def _draw_first_page_banner(
    page: pymupdf.Page,
    exam_title: str,
    source_label: str,
    total_count: int,
    is_english_exam: bool = False,
) -> float:
    w = page.rect.width
    today = datetime.now().strftime("%Y.%m.%d")
    sub_banner = (
        "수능 · 평가원 · 교육청 영어 기출 지문 맞춤형 수능형 변형 모의고사"
        if is_english_exam
        else "2022 개정 교육과정 맞춤형 기출 · 개념서 범위 엄격 준수"
    )
    src_prefix = "지문 출처:" if is_english_exam else "기준 개념서:"
    banner_html = f"""
    <div>
      <div class="exam-banner">
        <div class="exam-banner-sub">{sub_banner}</div>
        <div class="exam-banner-title">{html.escape(exam_title)}</div>
        <table class="student-box">
          <tr>
            <td style="width:44%;"><b>{src_prefix}</b> {html.escape(source_label)}</td>
            <td style="width:20%;"><b>총 {total_count}문항</b> ({today})</td>
            <td style="width:36%;"><b>학년:</b> ____ &nbsp;<b>반:</b> ____ &nbsp;<b>번호:</b> ____ &nbsp;<b>이름:</b> _________</td>
          </tr>
        </table>
      </div>
      <div class="double-rule-top"></div>
      <div class="double-rule-bot"></div>
    </div>
    """
    banner_h = _measure_html_height(banner_html, w - 60.0, max_h=170.0)
    page.insert_htmlbox(pymupdf.Rect(30, 22, w - 60.0 + 30.0, 22 + banner_h + 4.0), banner_html, css=EXAM_CSS)
    return 22.0 + banner_h + 10.0


def _is_english_item_dict(item: dict[str, Any]) -> bool:
    q = item.get("question")
    rw = item.get("rewritten")
    return (
        getattr(q, "subject_category", "") == "영어"
        or getattr(rw, "subject_domain", "") == "영어"
        or bool(item.get("english_meta"))
    )


def _format_english_source_text(q: QuestionRecord) -> str:
    if "직접 입력" in (q.exam_name or ""):
        return f"출처: {q.exam_name}"
    return f"출처: {q.exam_name} {q.q_num}번"


def generate_custom_exam_pdf(
    items: list[dict[str, Any]],
    exam_title: str,
    source_label: str,
    whitelist: ConceptWhitelist | None = None,
    two_column: bool = True,
    include_original_for_rewritten: bool = True,
) -> bytes:
    """
    추출 및 개조된 문항들을 평가원·교육청 모의고사 스타일 PDF로 조판하고,
    뒷장에 [빠른 정답 한눈에 보기] 표와 2단 맞춤 해설 페이지를 부착한다.
    """
    doc = pymupdf.open()
    page_w, page_h = 595.0, 842.0  # A4
    margin_x = 28.0
    margin_bottom = 42.0
    col_gap = 22.0
    is_english_exam_set = bool(items) and all(_is_english_item_dict(it) for it in items)

    def get_col_rects(top_y: float, is_two_col: bool) -> list[tuple[float, float, float]]:
        if is_two_col:
            col_w = (page_w - 2 * margin_x - col_gap) / 2.0
            return [
                (margin_x, margin_x + col_w, top_y),
                (margin_x + col_w + col_gap, page_w - margin_x, top_y),
            ]
        return [(margin_x, page_w - margin_x, top_y)]

    # 1) 문제지 페이지 생성
    page = doc.new_page(width=page_w, height=page_h)
    first_top = _draw_first_page_banner(
        page,
        exam_title,
        source_label,
        len(items),
        is_english_exam=is_english_exam_set,
    )
    first_divider_y = first_top - 2.0
    cur_divider_y = first_divider_y
    col_rects = get_col_rects(first_top, two_column)
    col_idx = 0
    cur_y = col_rects[0][2]

    def advance_column() -> None:
        nonlocal page, col_rects, col_idx, cur_y, cur_divider_y
        if col_idx + 1 < len(col_rects):
            col_idx += 1
            cur_y = col_rects[col_idx][2]
        else:
            _draw_page_footer(page, doc.page_count, two_column, top_line_y=cur_divider_y)
            page = doc.new_page(width=page_w, height=page_h)
            cur_divider_y = 34.0
            col_rects = get_col_rects(36.0, two_column)
            col_idx = 0
            cur_y = col_rects[0][2]

    for idx, item in enumerate(items, start=1):
        q: QuestionRecord = item["question"]
        crop_bytes: bytes | None = item.get("crop_png_bytes")
        rewritten: RewrittenQuestion | None = item.get("rewritten")
        is_eng_item = _is_english_item_dict(item)
        concept_ref: str = str(item.get("concept_reference") or "").strip()
        ref_html = (
            f"<div class='ref-note'>{html.escape(concept_ref)}</div>"
            if concept_ref
            else ""
        )

        col_x0, col_x1, _ = col_rects[col_idx]
        col_w = col_x1 - col_x0
        max_col_h = (page_h - margin_bottom) - col_rects[col_idx][2]

        if rewritten is not None:
            stem_body_html = _build_stem_blocks_html(rewritten.rewritten_stem, is_english=is_eng_item)
            choices_html = _build_choices_html(rewritten.rewritten_choices, is_english=is_eng_item)
            if is_eng_item:
                t_type = (getattr(rewritten, "target_question_type", "") or "수능형 변형").strip()
                badge_lbl = f"수능형 변형 · {html.escape(t_type)}"
                meta_lbl = html.escape(_format_english_source_text(q))
            else:
                badge_lbl = "PDF 맞춤 개조"
                meta_lbl = f"{html.escape(q.curriculum_subject)} · 원본 {html.escape(q.exam_name)} {q.q_num}번"

            card_html = f"""
            <div class="q-wrap">
              <div class="q-header">
                <span class="q-num">{idx}.</span>
                <span class="badge">{badge_lbl}</span>
                <span class="meta">{meta_lbl}</span>
              </div>
              {stem_body_html}
              {choices_html}
              {ref_html}
            </div>
            """
            text_h = _measure_html_height(card_html, col_w, max_h=735.0)

            img_draw_h = 0.0
            img_draw_w = col_w
            # 영어 변형 문항은 원본 이미지(기존 밑줄·정답 노출)를 절대 수록하지 않음
            if include_original_for_rewritten and crop_bytes and not is_eng_item:
                with Image.open(io.BytesIO(crop_bytes)) as im:
                    iw, ih = im.size
                ratio = ih / max(1, iw)
                img_draw_w = min(col_w, col_w * 0.94)
                img_draw_h = min(img_draw_w * ratio, 175.0)
                if img_draw_w * ratio > 175.0:
                    img_draw_w = 175.0 / max(0.01, ratio)

            vis_term_note = (getattr(rewritten, "visual_term_note", "") or "").strip()
            vis_note_html = (
                f"<div class='vis-note'><b>[그림 용어 안내]</b> {_esc_pdf(vis_term_note)}</div>"
                if vis_term_note
                else ""
            )
            vis_note_h = _measure_html_height(vis_note_html, col_w, max_h=60.0) if vis_note_html else 0.0
            note_h = 15.0 if img_draw_h > 0 else 0.0

            # 한 문항의 지문·선지가 칼럼 경계에서 잘리지 않도록 통째로 공간 확인
            total_needed = text_h + note_h + img_draw_h + vis_note_h + 12.0
            if cur_y + total_needed > page_h - margin_bottom and cur_y > col_rects[col_idx][2] + 5.0:
                advance_column()
                col_x0, col_x1, _ = col_rects[col_idx]
                col_w = col_x1 - col_x0

            page.insert_htmlbox(
                pymupdf.Rect(col_x0, cur_y, col_x1, min(page_h - margin_bottom, cur_y + text_h + 4.0)),
                card_html,
                css=EXAM_CSS,
            )
            cur_y += text_h + 2.0

            if img_draw_h > 0 and crop_bytes:
                if cur_y + note_h + img_draw_h + vis_note_h > page_h - margin_bottom and cur_y > col_rects[col_idx][2] + 5.0:
                    advance_column()
                    col_x0, col_x1, _ = col_rects[col_idx]
                page.insert_htmlbox(
                    pymupdf.Rect(col_x0, cur_y, col_x1, cur_y + 15.0),
                    "<div class='meta'>[참고용 원본 문항 도표/이미지]</div>",
                    css=EXAM_CSS,
                )
                cur_y += 13.0
                page.insert_image(
                    pymupdf.Rect(col_x0, cur_y, col_x0 + img_draw_w, cur_y + img_draw_h),
                    stream=crop_bytes,
                    keep_proportion=True,
                )
                cur_y += img_draw_h + 3.0
                if vis_note_html:
                    page.insert_htmlbox(
                        pymupdf.Rect(col_x0, cur_y, col_x1, min(page_h - margin_bottom, cur_y + vis_note_h + 3.0)),
                        vis_note_html,
                        css=EXAM_CSS,
                    )
                    cur_y += vis_note_h + 4.0

            cur_y += 12.0

        else:
            header_html = f"""
            <div class="q-header">
              <span class="q-num">{idx}.</span>
              <span class="badge-orig">원문 출제</span>
              <span class="meta">{html.escape(q.curriculum_subject)} · {html.escape(q.unit_name)} ({html.escape(q.exam_name)} {q.q_num}번)</span>
              {ref_html}
            </div>
            """
            header_h = _measure_html_height(header_html, col_w, max_h=80.0)

            img_draw_w = col_w
            img_draw_h = 40.0
            if crop_bytes:
                with Image.open(io.BytesIO(crop_bytes)) as im:
                    iw, ih = im.size
                ratio = ih / max(1, iw)
                img_draw_h = img_draw_w * ratio
                max_allowed_img_h = max_col_h - header_h - 14.0
                if img_draw_h > max_allowed_img_h:
                    scale_f = max_allowed_img_h / img_draw_h
                    img_draw_h = max_allowed_img_h
                    img_draw_w = img_draw_w * scale_f

            total_needed = header_h + img_draw_h + 16.0
            if cur_y + total_needed > page_h - margin_bottom and cur_y > col_rects[col_idx][2] + 5.0:
                advance_column()
                col_x0, col_x1, _ = col_rects[col_idx]
                col_w = col_x1 - col_x0

            page.insert_htmlbox(
                pymupdf.Rect(col_x0, cur_y, col_x1, cur_y + header_h + 2.0),
                header_html,
                css=EXAM_CSS,
            )
            cur_y += header_h + 2.0
            if crop_bytes:
                page.insert_image(
                    pymupdf.Rect(col_x0, cur_y, col_x0 + img_draw_w, cur_y + img_draw_h),
                    stream=crop_bytes,
                    keep_proportion=True,
                )
                cur_y += img_draw_h + 14.0
            else:
                cur_y += 20.0

    _draw_page_footer(page, doc.page_count, two_column, top_line_y=cur_divider_y)

    # 2) 뒷장: [빠른 정답 한눈에 보기] 표 + 2단 맞춤 해설
    ans_page = doc.new_page(width=page_w, height=page_h)
    ans_sub_title = (
        "정답 및 수능 영어 구문·논리·오답 분석 해설"
        if is_english_exam_set
        else "정답 및 개념서 용어 기준 맞춤 해설"
    )
    ans_src_prefix = "지문 출처" if is_english_exam_set else "기준 개념서"
    ref_col_header = "출처 / 유형" if is_english_exam_set else "참고 개념서 페이지"
    ans_header = f"""
    <div class="exam-banner" style="padding:6pt 9pt;">
      <div class="exam-banner-sub">{ans_sub_title}</div>
      <div class="exam-banner-title" style="font-size:12.5pt;">{html.escape(exam_title)}</div>
      <div class="meta" style="margin-top:2pt;">{ans_src_prefix}: {html.escape(source_label)} &nbsp;|&nbsp; 총 {len(items)}문항</div>
    </div>
    """
    ah = _measure_html_height(ans_header, page_w - 2 * margin_x, max_h=120.0)
    ans_page.insert_htmlbox(pymupdf.Rect(margin_x, 24, page_w - margin_x, 24 + ah + 4.0), ans_header, css=EXAM_CSS)
    ay = 24.0 + ah + 8.0

    # [빠른 정답 한눈에 보기] 요약 표 (문항 번호 · 정답 · 참고 개념서 페이지)
    quick_title_html = "<div style='font-size:9.8pt; font-weight:bold; color:#1E3A8A; margin-bottom:3pt;'>[빠른 정답 한눈에 보기]</div>"
    qth = _measure_html_height(quick_title_html, page_w - 2 * margin_x, max_h=30.0)
    ans_page.insert_htmlbox(pymupdf.Rect(margin_x, ay, page_w - margin_x, ay + qth), quick_title_html, css=EXAM_CSS)
    ay += qth + 2.0

    half = (len(items) + 1) // 2 if len(items) > 6 else len(items)
    summary_rows: list[str] = []
    if len(items) > 6:
        for r_i in range(half):
            left_it = items[r_i]
            l_ans = _format_answer_badge(left_it["rewritten"].preserved_answer if left_it.get("rewritten") else left_it["question"].answer)
            l_pg = _extract_short_page_ref(str(left_it.get("concept_reference") or ""))
            right_cells = "<td></td><td></td><td></td>"
            if r_i + half < len(items):
                right_it = items[r_i + half]
                r_ans = _format_answer_badge(right_it["rewritten"].preserved_answer if right_it.get("rewritten") else right_it["question"].answer)
                r_pg = _extract_short_page_ref(str(right_it.get("concept_reference") or ""))
                right_cells = (
                    f"<td><b>{r_i + half + 1}번</b></td>"
                    f"<td style='color:#DC2626; font-weight:bold;'>{html.escape(r_ans)}</td>"
                    f"<td>{html.escape(r_pg)}</td>"
                )
            summary_rows.append(
                f"<tr><td><b>{r_i + 1}번</b></td><td style='color:#DC2626; font-weight:bold;'>{html.escape(l_ans)}</td><td>{html.escape(l_pg)}</td>{right_cells}</tr>"
            )
        ans_table_html = (
            "<table class='ans-table'>"
            f"<tr><th>문항 번호</th><th>정답</th><th>{ref_col_header}</th>"
            f"<th>문항 번호</th><th>정답</th><th>{ref_col_header}</th></tr>"
            + "".join(summary_rows)
            + "</table>"
        )
    else:
        for r_i, it in enumerate(items, start=1):
            a_disp = _format_answer_badge(it["rewritten"].preserved_answer if it.get("rewritten") else it["question"].answer)
            pg_disp = _extract_short_page_ref(str(it.get("concept_reference") or ""))
            summary_rows.append(
                f"<tr><td><b>{r_i}번</b></td><td style='color:#DC2626; font-weight:bold;'>{html.escape(a_disp)}</td><td>{html.escape(pg_disp)}</td></tr>"
            )
        ans_table_html = (
            "<table class='ans-table'>"
            f"<tr><th>문항 번호</th><th>정답</th><th>{ref_col_header}</th></tr>"
            + "".join(summary_rows)
            + "</table>"
        )

    th = _measure_html_height(ans_table_html, page_w - 2 * margin_x, max_h=460.0)
    ans_page.insert_htmlbox(pymupdf.Rect(margin_x, ay, page_w - margin_x, ay + th + 4.0), ans_table_html, css=EXAM_CSS)
    ay += th + 8.0

    # 아래부터 2단 구성으로 맞춤 해설 배치
    ans_div_top = ay
    ans_col_rects = get_col_rects(ay + 4.0, True)
    ans_col_idx = 0
    ans_y = ans_col_rects[0][2]

    def advance_ans_col() -> None:
        nonlocal ans_page, ans_col_rects, ans_col_idx, ans_y, ans_div_top
        if ans_col_idx + 1 < len(ans_col_rects):
            ans_col_idx += 1
            ans_y = ans_col_rects[ans_col_idx][2]
        else:
            _draw_page_footer(ans_page, doc.page_count, True, top_line_y=ans_div_top)
            ans_page = doc.new_page(width=page_w, height=page_h)
            ans_div_top = 34.0
            ans_col_rects = get_col_rects(36.0, True)
            ans_col_idx = 0
            ans_y = ans_col_rects[0][2]

    for idx, item in enumerate(items, start=1):
        q = item["question"]
        rewritten = item.get("rewritten")
        is_eng_item = _is_english_item_dict(item)
        concept_ref = str(item.get("concept_reference") or "").strip()
        exp_ref_html = (
            f"<div class='ref-note'>{html.escape(concept_ref)}</div>"
            if concept_ref
            else ""
        )

        if rewritten is not None:
            ans_disp = _format_answer_badge(rewritten.preserved_answer)
            swap_info = ""
            if rewritten.replaced_terms_map:
                swap_label = "핵심 어휘·구문:" if is_eng_item else "용어 치환:"
                swap_info = (
                    f"<div class='exp-sub'><b>{swap_label}</b> "
                    + (
                        render_english_inline_html(" / ".join(rewritten.replaced_terms_map))
                        if is_eng_item
                        else _esc_pdf(" / ".join(rewritten.replaced_terms_map))
                    )
                    + "</div>"
                )
            exp_txt = format_explanation_text(rewritten.pdf_grounded_explanation, for_markdown=False)
            exp_body = render_english_inline_html(exp_txt) if is_eng_item else _esc_pdf(exp_txt)
            badge = (getattr(rewritten, "target_question_type", "") or "수능형 영어 변형") if is_eng_item else "PDF 맞춤 개조"
        else:
            ans_disp = _format_answer_badge(q.answer)
            swap_info = (
                "<div class='exp-sub'><b>핵심 원리:</b> "
                + _esc_pdf(", ".join(q.core_principles))
                + "</div>"
            )
            exp_body = _esc_pdf(
                format_explanation_text(q.official_explanation, for_markdown=False)
            )
            badge = "원본 기출 해설"

        src_meta_str = _format_english_source_text(q) if is_eng_item else f"{q.exam_name} {q.q_num}번"
        exp_html = f"""
        <div class="exp-card">
          <div class="exp-title">
            {idx}번 &nbsp; 정답: <span class="exp-ans">{html.escape(ans_disp)}</span>
            <span class="meta">({html.escape(badge)} · {html.escape(src_meta_str)})</span>
          </div>
          {swap_info}
          <div>{exp_body}</div>
          {exp_ref_html}
        </div>
        """
        ac_x0, ac_x1, _ = ans_col_rects[ans_col_idx]
        ac_w = ac_x1 - ac_x0
        eh = _measure_html_height(exp_html, ac_w, max_h=720.0)
        if ans_y + eh > page_h - margin_bottom and ans_y > ans_col_rects[ans_col_idx][2] + 5.0:
            advance_ans_col()
            ac_x0, ac_x1, _ = ans_col_rects[ans_col_idx]
            ac_w = ac_x1 - ac_x0

        ans_page.insert_htmlbox(
            pymupdf.Rect(ac_x0, ans_y, ac_x1, min(page_h - margin_bottom, ans_y + eh + 4.0)),
            exp_html,
            css=EXAM_CSS,
        )
        ans_y += eh + 6.0

    _draw_page_footer(ans_page, doc.page_count, True, top_line_y=ans_div_top)

    try:
        doc.subset_fonts()
    except Exception:
        pass
    out_bytes = doc.tobytes(deflate=True, garbage=3)
    doc.close()
    return out_bytes



from src.korean_pdf_generator import (  # noqa: E402, F401
    _KOREAN_ANALYSIS_CSS,
    _build_korean_passage_html_block,
    _clean_and_reflow_korean_passage,
    _format_structure_flow_html,
    _highlight_poem_line_html,
    _is_korean_poetry_item,
    _join_korean_wrapped_line,
    _measure_rendered_line_chars,
    _pick_korean_font_paths,
    _render_poetry_work_png,
    _render_poetry_work_png_pages,
    _render_story_pdf,
    _strip_korean_passage_markers,
    _wrap_text_by_width,
    generate_korean_passage_analysis_pdf,
    generate_korean_passage_only_pdf,
    merge_korean_analysis_pdfs,
)
