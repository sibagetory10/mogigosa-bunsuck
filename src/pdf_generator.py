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


def _join_korean_wrapped_line(prev_line: str, next_line: str) -> str:
    """2단 PDF 줄바꿈으로 끊긴 두 줄을 이어 붙일 때 단어·조사 중간에 불필요한 공백이 들어가지 않도록 보정한다."""
    if not prev_line:
        return next_line
    if not next_line:
        return prev_line

    # 앞 줄 끝이 하이픈/공백이면 그대로 연결
    if prev_line.endswith((" ", "\t")):
        return prev_line + next_line

    # 다음 줄이 한국어 문장부호나 대표 조사·어미로 시작하면 붙여쓰기
    if re.match(
        r"^(?:[.,?!:;·\)\]\”\’]|(?:이다|입니다|있다|없다|한다|된다|하며|하여|하고|하면|해서|하는|되는|된|할|될|은|는|이|가|을|를|의|에|에서|에게|로|으로|와|과|도|만|까지|부터|처럼|보다|라고|이라고|라는|이라는|라|이라|며|으며|고|서|니|면|지만|으나|나|거나)\b)",
        next_line,
    ):
        return prev_line + next_line

    # 앞 줄 끝 어절(마지막 공백 이후)이 1~2글자이고 다음 줄 첫 어절도 1~2글자 한글이면 한 단어가 줄 끝에서 쪼개졌을 가능성이 높음
    last_tok = prev_line.split()[-1] if prev_line.split() else ""
    first_tok = next_line.split()[0] if next_line.split() else ""
    if (
        1 <= len(last_tok) <= 2
        and 1 <= len(first_tok) <= 2
        and re.fullmatch(r"[가-힣]+", last_tok)
        and re.fullmatch(r"[가-힣]+", first_tok)
        and len(prev_line) >= 18
    ):
        return prev_line + next_line

    return f"{prev_line} {next_line}"


_KOR_PASSAGE_MARKER_RE = re.compile(
    r"[\u3260-\u327B\u24B6-\u24E9]"
    r"|[\[\(]\s*[A-Za-zㄱ-ㅎ]\s*[\]\)]"
    r"|(?:^|(?<=\s))[ㄱ-ㅎA-Za-z]\.(?=\s)"
)


def _strip_korean_passage_markers(line: str) -> str:
    """국어 지문 원문에서 문제용 기호(㉠~㉭, ⓐ~ⓩ, [A]/[B], (a)/(b), ㄱ./a. 등)를 제거한다."""
    cleaned = _KOR_PASSAGE_MARKER_RE.sub("", line)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def _clean_and_reflow_korean_passage(passage_text: str, is_poetry_or_literature: bool = False) -> list[str]:
    """
    국어 지문 원문에서 시험지 쪽수('4', '12')·푸터('국어 영역')·문제용 기호(㉠~㉭, ⓐ~ⓩ, [A]/[B] 등)를 제거하고,
    2단 폭 때문에 문장 중간에서 끊긴 줄바꿈을 자연스러운 문단 단위로 재조립해 반환한다.
    """
    raw_lines = (passage_text or "").splitlines()
    cleaned: list[tuple[str, bool]] = []
    for raw_ln in raw_lines:
        is_indented = bool(re.match(r"^(?:[ \t\u3000]{2,}|\t|\u3000)", raw_ln))
        s = raw_ln.strip()
        if not s:
            cleaned.append(("", False))
            continue
        if re.match(r"^\[?\s*\d{1,2}\s*\]?$", s):
            continue
        if re.match(r"^(?:국어|수학|영어|한국사)\s*영역(?:\s*\d{1,2})?$", s):
            continue
        if re.match(r"^(?:\d{1,2}\s+국어|국어\s+\d{1,2})$", s):
            continue
        s = _strip_korean_passage_markers(s)
        if not s:
            continue
        cleaned.append((s, is_indented))

    if is_poetry_or_literature:
        # 시·문학은 연/행 구분을 보존하되 연속 빈 줄만 정리
        out_lit: list[str] = []
        prev_blank = False
        for ln, _ in cleaned:
            if not ln:
                if not prev_blank and out_lit:
                    out_lit.append("")
                prev_blank = True
            else:
                out_lit.append(ln)
                prev_blank = False
        while out_lit and not out_lit[-1]:
            out_lit.pop()
        return out_lit

    # 독서(산문) 지문: 빈 줄, 들여쓰기 시작, 헤더, 문장 종결 후 짧은 줄 기준으로 문단 나눔
    prose_lens = sorted(len(ln) for ln, _ in cleaned if len(ln) >= 15)
    full_len = prose_lens[int(len(prose_lens) * 0.8)] if prose_lens else 30
    short_end_thresh = max(24, int(full_len * 0.84))

    paragraphs: list[str] = []
    cur_para = ""
    prev_raw_ln = ""
    header_re = re.compile(r"^(?:\[\d{1,2}\s*[~\-～]\s*\d{1,2}\]|\([가나다라]\)|<[가-힣\s]+>|-\s*.*-\s*$)")
    sentence_end_re = re.compile(r"[.?!…\"”’\)\]]\s*$")

    for ln, is_indented in cleaned:
        if not ln:
            if cur_para:
                paragraphs.append(cur_para)
                cur_para = ""
                prev_raw_ln = ""
            continue
        if header_re.match(ln):
            if cur_para:
                paragraphs.append(cur_para)
                cur_para = ""
                prev_raw_ln = ""
            paragraphs.append(ln)
            continue
        if cur_para:
            if sentence_end_re.search(prev_raw_ln) and (is_indented or len(prev_raw_ln) <= short_end_thresh):
                paragraphs.append(cur_para)
                cur_para = ln
            else:
                cur_para = _join_korean_wrapped_line(cur_para, ln)
            prev_raw_ln = ln
        else:
            cur_para = ln
            prev_raw_ln = ln

    if cur_para:
        paragraphs.append(cur_para)
    return [p for p in paragraphs if p.strip()]


def _format_structure_flow_html(flow_text: str) -> str:
    """전개 흐름 문자열을 단계별 줄바꿈('도입 : ~~~' 형태) HTML로 변환한다."""
    raw = (flow_text or "").strip()
    if not raw:
        return ""
    parts = [p.strip() for p in re.split(r"\s*(?:→|->|\r?\n)+\s*", raw) if p.strip()]
    stage_paren_re = re.compile(
        r"^(도입|전개|심화(?:/전환)?|전환|결론|발단|위기|절정|결말|서론|본론)\s*\((.+)\)$"
    )
    stage_colon_re = re.compile(
        r"^(도입|전개|심화(?:/전환)?|전환|결론|발단|위기|절정|결말|서론|본론)\s*[:：]\s*(.+)$"
    )
    formatted: list[str] = []
    for part in parts:
        m_paren = stage_paren_re.match(part)
        if m_paren:
            formatted.append(f"<b>{html.escape(m_paren.group(1))}</b> : {html.escape(m_paren.group(2).strip())}")
            continue
        m_colon = stage_colon_re.match(part)
        if m_colon:
            formatted.append(f"<b>{html.escape(m_colon.group(1))}</b> : {html.escape(m_colon.group(2).strip())}")
            continue
        formatted.append(html.escape(part))
    return "<br/>".join(formatted)


_KOREAN_ANALYSIS_CSS = """
    body {
        font-family: "NanumGothic", "Malgun Gothic", sans-serif;
        font-size: 9.4pt;
        line-height: 1.54;
        color: #0F172A;
    }
    .hdr-banner {
        border: 1.4pt solid #1E293B;
        border-radius: 5pt;
        background-color: #F8FAFC;
        padding: 9pt 12pt;
        margin-bottom: 10pt;
        break-inside: avoid;
        page-break-inside: avoid;
    }
    .hdr-sub {
        font-size: 8.2pt;
        font-weight: bold;
        color: #1D4ED8;
    }
    .hdr-title {
        font-size: 14.5pt;
        font-weight: bold;
        color: #0F172A;
        margin-top: 2pt;
    }
    .hdr-meta {
        font-size: 8.2pt;
        color: #475569;
        margin-top: 3pt;
    }
    .sec-title {
        font-size: 10.4pt;
        font-weight: bold;
        color: #1E3A8A;
        border-left: 3.5pt solid #2563EB;
        padding-left: 6pt;
        margin-top: 10pt;
        margin-bottom: 5pt;
        break-after: avoid;
        page-break-after: avoid;
    }
    .sec-keep-tbl {
        width: 100%;
        border-collapse: collapse;
        border: none;
        margin: 0;
        padding: 0;
        break-inside: avoid;
        page-break-inside: avoid;
    }
    .sec-keep-tr {
        break-inside: avoid;
        page-break-inside: avoid;
    }
    .sec-keep-td {
        padding: 0;
        margin: 0;
        border: none;
        vertical-align: top;
    }
    .theme-box {
        background-color: #EFF6FF;
        border: 0.9pt solid #93C5FD;
        border-radius: 4pt;
        padding: 7pt 10pt;
        margin-bottom: 6pt;
        break-inside: avoid;
        page-break-inside: avoid;
    }
    .flow-box {
        background-color: #F0FDF4;
        border: 0.9pt solid #86EFAC;
        border-radius: 4pt;
        padding: 7pt 10pt;
        margin-bottom: 8pt;
        break-inside: avoid;
        page-break-inside: avoid;
    }
    .flow-lines {
        margin-top: 2pt;
        line-height: 1.58;
    }
    .para-card {
        border-bottom: 0.6pt solid #E2E8F0;
        padding: 4pt 0;
        break-inside: avoid;
        page-break-inside: avoid;
    }
    .para-lbl {
        font-weight: bold;
        color: #1E40AF;
    }
    .para-kp {
        color: #0F766E;
        font-size: 8.7pt;
        margin-top: 1.5pt;
    }
    .vocab-list {
        border: 0.8pt solid #CBD5E1;
        border-radius: 4pt;
        background-color: #FFFFFF;
        padding: 4pt 8pt;
        margin-bottom: 8pt;
    }
    .vocab-row {
        border-bottom: 0.5pt solid #E2E8F0;
        padding: 4pt 0;
        font-size: 9.0pt;
        break-inside: avoid;
        page-break-inside: avoid;
    }
    .vocab-row-last {
        padding: 4pt 0;
        font-size: 9.0pt;
        break-inside: avoid;
        page-break-inside: avoid;
    }
    .vocab-badge {
        display: inline;
        background-color: #FEF3C7;
        color: #92400E;
        font-weight: bold;
        padding: 1pt 5pt;
        border-radius: 3pt;
    }
    .checkpoint-box {
        background-color: #FFFBEB;
        border: 0.9pt solid #FCD34D;
        border-radius: 4pt;
        padding: 7pt 10pt;
        margin-bottom: 8pt;
    }
    .note-li {
        margin-bottom: 3pt;
    }
    .passage-box {
        background-color: #FAFAFA;
        border: 0.8pt solid #CBD5E1;
        border-radius: 4pt;
        padding: 8pt 11pt;
        font-size: 9.0pt;
        line-height: 1.62;
    }
    .passage-p {
        margin: 0 0 7.5pt 0;
        text-indent: 9pt;
        text-align: justify;
    }
    .passage-lit-line {
        margin: 0 0 2.5pt 0;
    }
    .passage-hdr {
        margin: 4pt 0 3pt 0;
        font-weight: bold;
        color: #1E293B;
    }
    .page-break {
        page-break-before: always;
    }
    .diagram-wrap {
        margin-top: 4pt;
        margin-bottom: 8pt;
        text-align: center;
    }
    .diagram-img {
        width: 505pt;
        max-width: 100%;
        height: auto;
    }
    .poem-ul {
        font-weight: bold;
        color: #1E3A8A;
        text-decoration: underline;
    }
    .poem-badge {
        font-size: 7.5pt;
        font-weight: bold;
        color: #1D4ED8;
    }
    .poem-notes-box {
        background-color: #EFF6FF;
        border: 0.8pt solid #BFDBFE;
        border-radius: 4pt;
        padding: 7pt 10pt;
        margin-top: 6pt;
        margin-bottom: 8pt;
    }
    .poem-notes-hdr {
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 4pt;
        font-size: 9.2pt;
    }
    .poem-note-row {
        margin-bottom: 3.5pt;
        font-size: 8.9pt;
        line-height: 1.5;
        break-inside: avoid;
        page-break-inside: avoid;
    }
    """


_CIRCLED_NUMS = ("①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩")


def _is_korean_poetry_item(
    section_label: str,
    subgenre: str = "",
    passage_text: str = "",
    an: Any = None,
) -> bool:
    """현대시·고전시가 등 시가 문학 갈래인지 판별한다."""
    if an is not None and (
        getattr(an, "poetry_works", None) or getattr(an, "poetry_underline_notes", None)
    ):
        return True
    sub = (subgenre or "").strip()
    sec = (section_label or "").strip()
    if any(k in sub for k in ("소설", "극", "시나리오", "희곡", "수필")):
        return False
    if any(k in sub for k in ("현대시", "고전시가", "시가", "시조", "가사", "향가", "한시")) or sub == "시":
        return True
    if "문학" in sec and not sub and passage_text:
        lines = [ln.strip() for ln in passage_text.splitlines() if ln.strip()]
        if len(lines) >= 6:
            lens = sorted(len(ln) for ln in lines)
            if lens[len(lens) // 2] <= 22:
                return True
    return False


def _pick_korean_font_paths() -> tuple[str | None, str | None, str | None]:
    """시스템(Linux/Windows/macOS)에서 명조(시 본문) · 고딕(주석) · 고딕 볼드 폰트 경로를 찾는다."""
    serif_candidates = [
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSerifCJK-Regular.ttc",
        "C:/Windows/Fonts/batang.ttc",
        "C:/Windows/Fonts/H2MJRE.TTF",
        "/System/Library/Fonts/Supplemental/AppleMyungjo.ttf",
    ]
    sans_candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/NanumGothic.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    ]
    bold_candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "C:/Windows/Fonts/malgunbd.ttf",
        "C:/Windows/Fonts/NanumGothicBold.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    ]
    serif_p = next((p for p in serif_candidates if os.path.exists(p)), None)
    sans_p = next((p for p in sans_candidates if os.path.exists(p)), None)
    bold_p = next((p for p in bold_candidates if os.path.exists(p)), None) or sans_p
    return serif_p, sans_p, bold_p


def _measure_rendered_line_chars(
    text: str,
    fontfile: str | None,
    fontsize: float,
) -> tuple[list[tuple[float, float]], float]:
    """PyMuPDF 실제 렌더링 좌표(rawdict)로 각 글자의 (x0, x1) 오프셋과 전체 폭을 정확히 측정한다."""
    if not text:
        return [], 0.0
    tmp_doc = pymupdf.open()
    try:
        p = tmp_doc.new_page(width=2000, height=120)
        if fontfile:
            p.insert_font(fontname="mfont", fontfile=fontfile)
            fn = "mfont"
        else:
            fn = "korea"
        p.insert_text((0.0, 60.0), text, fontname=fn, fontsize=fontsize, color=(0, 0, 0))
        rd = p.get_text("rawdict")
        chars_bbox: list[tuple[float, float]] = []
        for blk in rd.get("blocks", []):
            for ln in blk.get("lines", []):
                for sp in ln.get("spans", []):
                    for ch in sp.get("chars", []):
                        bb = ch.get("bbox")
                        if bb:
                            chars_bbox.append((float(bb[0]), float(bb[2])))
        if len(chars_bbox) == len(text):
            return chars_bbox, chars_bbox[-1][1]
        # fallback: 공백 등 rawdict에서 누락된 문자가 있을 때 선형 보간
        f_obj = pymupdf.Font(fontfile=fontfile) if fontfile else pymupdf.Font("korea")
        space_w = fontsize if not fontfile else f_obj.text_length(" ", fontsize=fontsize)
        out: list[tuple[float, float]] = []
        cx = 0.0
        for ch in text:
            cw = space_w if ch == " " else f_obj.text_length(ch, fontsize=fontsize)
            out.append((cx, cx + cw))
            cx += cw
        return out, cx
    finally:
        tmp_doc.close()


def _wrap_text_by_width(text: str, font_obj: pymupdf.Font, fontsize: float, max_width: float) -> list[str]:
    """주어진 폭(pt)에 맞게 한글/영문 문자열을 줄바꿈한다."""
    s = (text or "").strip()
    if not s:
        return []
    lines: list[str] = []
    cur = ""
    for ch in s:
        if ch == "\n":
            if cur:
                lines.append(cur)
                cur = ""
            continue
        test = cur + ch
        if cur and font_obj.text_length(test, fontsize=fontsize) > max_width:
            lines.append(cur.rstrip())
            cur = ch.lstrip()
        else:
            cur = test
    if cur:
        lines.append(cur.rstrip())
    return lines


def _render_poetry_work_png(work: Any) -> bytes:
    """
    KoreanPoetryWorkAnnotation을 입력받아 명조체 시 원문 + 컬러 고딕 행간 주석 +
    밑줄·물결선·네모·동그라미·세모 + 좌·우 여백 브라켓/수미상관 호를 고해상도 PNG로 렌더링한다.
    """
    serif_path, sans_path, bold_path = _pick_korean_font_paths()
    serif_font = pymupdf.Font(fontfile=serif_path) if serif_path else pymupdf.Font("korea")
    sans_font = pymupdf.Font(fontfile=sans_path) if sans_path else pymupdf.Font("korea")
    bold_font = pymupdf.Font(fontfile=bold_path) if bold_path else sans_font

    color_map = {
        "red": (0.84, 0.16, 0.16),
        "blue": (0.11, 0.35, 0.78),
        "dark": (0.20, 0.25, 0.33),
    }
    poem_color = (0.08, 0.10, 0.15)
    muted_color = (0.38, 0.44, 0.54)

    raw_lines = [str(ln or "") for ln in (getattr(work, "lines", None) or [])]
    if not raw_lines:
        raise ValueError("시 원문 행 데이터가 비어 있습니다.")

    marks = list(getattr(work, "marks", None) or [])
    brackets = list(getattr(work, "brackets", None) or [])

    W = 520.0
    has_left_bracket = any((getattr(b, "side", "") or "") == "left" for b in brackets)
    X_LEFT_LABEL = 6.0
    X_LEFT_ARC = 64.0
    X_POEM = 72.0 if has_left_bracket else 22.0
    POEM_FS = 10.6
    NOTE_FS = 7.4
    MARGIN_FS = 7.5

    # 실제 렌더링 좌표로 전체 시 행 폭을 측정하고, 우측 여백(>=105pt)을 침범하면 폰트 크기 자동 축소
    measured_lines = [_measure_rendered_line_chars(ltxt, serif_path, POEM_FS) for ltxt in raw_lines]
    max_poem_w = max((lw for _, lw in measured_lines), default=120.0)
    max_allowed_poem_w = W - X_POEM - 115.0
    if max_poem_w > max_allowed_poem_w and max_poem_w > 0:
        POEM_FS = max(8.4, round(POEM_FS * (max_allowed_poem_w / max_poem_w), 2))
        measured_lines = [_measure_rendered_line_chars(ltxt, serif_path, POEM_FS) for ltxt in raw_lines]
        max_poem_w = max((lw for _, lw in measured_lines), default=120.0)

    X_RIGHT_MARGIN = max(350.0, X_POEM + max_poem_w + 18.0)
    if X_RIGHT_MARGIN > W - 88.0:
        X_RIGHT_MARGIN = W - 88.0
    RIGHT_TEXT_MAX_W = max(72.0, W - X_RIGHT_MARGIN - 8.0)

    # 각 mark를 실제 행 번호(1-based)에 정확히 매칭
    line_marks: dict[int, list[tuple[Any, int, int]]] = {}
    for mk in marks:
        q = (getattr(mk, "quote", "") or "").strip()
        if not q:
            continue
        li = int(getattr(mk, "line_index", 1) or 1)
        pos = -1
        target_li = li
        if 1 <= li <= len(raw_lines):
            pos = raw_lines[li - 1].find(q)
        if pos == -1:
            for cand_i, cand_txt in enumerate(raw_lines, start=1):
                p = cand_txt.find(q)
                if p != -1:
                    pos = p
                    target_li = cand_i
                    break
        if pos != -1:
            line_marks.setdefault(target_li, []).append((mk, pos, pos + len(q)))

    # 1차 패스: 각 행의 Y 좌표 및 행간 주석 배치(겹침 회피) 계산
    work_label = (getattr(work, "work_label", "") or "").strip()
    title_author = (getattr(work, "title_author", "") or "").strip()
    header_str = f"{work_label} {title_author}".strip() if work_label else title_author

    cur_y = 14.0
    if header_str:
        cur_y += 22.0

    line_layouts: list[dict[str, Any]] = []
    for li, ltxt in enumerate(raw_lines, start=1):
        if not ltxt.strip():
            line_layouts.append(
                {
                    "line_index": li,
                    "text": "",
                    "is_blank": True,
                    "y_base": cur_y + 5.0,
                    "y_mid": cur_y + 5.0,
                    "end_x": X_POEM,
                    "above_items": [],
                    "below_items": [],
                }
            )
            cur_y += 12.0
            continue

        chars_bbox, line_w = measured_lines[li - 1]
        m_list = sorted(line_marks.get(li, []), key=lambda t: t[1])
        above_items: list[dict[str, Any]] = []
        below_items: list[dict[str, Any]] = []

        for mk, p0, p1 in m_list:
            if chars_bbox and 0 <= p0 < p1 <= len(chars_bbox):
                x0 = X_POEM + chars_bbox[p0][0]
                x1 = X_POEM + chars_bbox[p1 - 1][1]
            else:
                x0 = X_POEM + serif_font.text_length(ltxt[:p0], fontsize=POEM_FS)
                x1 = x0 + serif_font.text_length(ltxt[p0:p1], fontsize=POEM_FS)
            mid_x = (x0 + x1) * 0.5
            note_txt = (getattr(mk, "note", "") or "").strip()
            note_w = sans_font.text_length(note_txt, fontsize=NOTE_FS)
            # 주석 텍스트 X 범위 [nx, nx + note_w]
            nx = max(X_POEM - 8.0, min(mid_x - note_w * 0.35, X_RIGHT_MARGIN - 8.0 - note_w))
            pos_pref = getattr(mk, "position", "below") or "below"

            item = {
                "mk": mk,
                "x0": x0,
                "x1": x1,
                "mid_x": mid_x,
                "nx": nx,
                "nw": note_w,
                "note": note_txt,
                "tier": 0,
            }

            def _overlaps(target_list: list[dict[str, Any]], cand_nx: float, cand_nw: float) -> bool:
                for ex in target_list:
                    if not (cand_nx + cand_nw + 6.0 < ex["nx"] or cand_nx > ex["nx"] + ex["nw"] + 6.0):
                        return True
                return False

            if pos_pref == "above":
                if not _overlaps(above_items, nx, note_w):
                    above_items.append(item)
                elif not _overlaps(below_items, nx, note_w):
                    below_items.append(item)
                else:
                    item["tier"] = 1
                    above_items.append(item)
            else:
                if not _overlaps(below_items, nx, note_w):
                    below_items.append(item)
                elif not _overlaps(above_items, nx, note_w):
                    above_items.append(item)
                else:
                    item["tier"] = 1
                    below_items.append(item)

        above_tiers = max((it["tier"] + 1 for it in above_items), default=0)
        below_tiers = max((it["tier"] + 1 for it in below_items), default=0)
        above_pad = 4.0 + above_tiers * 14.0
        below_pad = 7.0 + below_tiers * 15.5

        y_base = cur_y + above_pad + POEM_FS
        line_layouts.append(
            {
                "line_index": li,
                "text": ltxt,
                "is_blank": False,
                "y_base": y_base,
                "y_mid": y_base - POEM_FS * 0.35,
                "end_x": X_POEM + line_w,
                "above_items": above_items,
                "below_items": below_items,
            }
        )
        cur_y = y_base + below_pad

    H = max(120.0, cur_y + 16.0)

    # PyMuPDF 임시 문서 생성 및 폰트 등록
    doc = pymupdf.open()
    page = doc.new_page(width=W, height=H)
    if serif_path:
        page.insert_font(fontname="kserif", fontfile=serif_path)
        serif_fn = "kserif"
    else:
        serif_fn = "korea"
    if sans_path:
        page.insert_font(fontname="ksans", fontfile=sans_path)
        sans_fn = "ksans"
    else:
        sans_fn = "korea"
    if bold_path:
        page.insert_font(fontname="kbold", fontfile=bold_path)
        bold_fn = "kbold"
    else:
        bold_fn = sans_fn

    # 배경 카드 테두리
    page.draw_rect(pymupdf.Rect(1.5, 1.5, W - 1.5, H - 1.5), color=(0.82, 0.86, 0.92), fill=(0.99, 0.995, 1.0), width=0.8)

    # 작품 제목 헤더
    if header_str:
        page.insert_text((14.0, 24.0), header_str, fontname=bold_fn, fontsize=11.0, color=(0.08, 0.14, 0.28))
        page.draw_line((14.0, 29.0), (W - 14.0, 29.0), color=(0.85, 0.88, 0.93), width=0.7)

    # 본문 시 행 + 마크 + 행간 주석 렌더링
    for lay in line_layouts:
        if lay["is_blank"]:
            continue
        y_base = lay["y_base"]
        page.insert_text((X_POEM, y_base), lay["text"], fontname=serif_fn, fontsize=POEM_FS, color=poem_color)

        for pos_kind, items in (("above", lay["above_items"]), ("below", lay["below_items"])):
            for it in items:
                mk = it["mk"]
                col = color_map.get(getattr(mk, "color", "red"), color_map["red"])
                mtype = getattr(mk, "mark_type", "underline") or "underline"
                x0, x1, mid_x = it["x0"], it["x1"], it["mid_x"]
                y_top = y_base - POEM_FS * 0.85
                y_bot = y_base + 2.2

                if mtype == "underline":
                    page.draw_line((x0, y_bot), (x1, y_bot), color=col, width=1.1)
                    anchor_y = y_bot if pos_kind == "below" else y_top
                elif mtype == "wavy":
                    pts = []
                    wx = x0
                    step = 3.0
                    idx_w = 0
                    while wx <= x1:
                        wy = y_bot + (1.4 if idx_w % 2 == 0 else -1.0)
                        pts.append((wx, wy))
                        wx += step
                        idx_w += 1
                    if pts and pts[-1][0] < x1:
                        pts.append((x1, y_bot))
                    if len(pts) >= 2:
                        page.draw_polyline(pts, color=col, width=1.0)
                    anchor_y = y_bot + 1.5 if pos_kind == "below" else y_top
                elif mtype == "box":
                    page.draw_rect(pymupdf.Rect(x0 - 1.5, y_top - 1.0, x1 + 1.5, y_bot + 0.8), color=col, width=1.0)
                    anchor_y = y_bot + 0.8 if pos_kind == "below" else y_top - 1.0
                elif mtype == "circle":
                    page.draw_oval(pymupdf.Rect(x0 - 2.2, y_top - 1.5, x1 + 2.2, y_bot + 1.2), color=col, width=1.0)
                    anchor_y = y_bot + 1.2 if pos_kind == "below" else y_top - 1.5
                elif mtype == "triangle":
                    q_len = len((getattr(mk, "quote", "") or "").strip())
                    if q_len <= 2:
                        tri_pts = [(mid_x, y_top - 4.0), (x0 - 3.5, y_bot + 1.2), (x1 + 3.5, y_bot + 1.2)]
                        page.draw_polyline(tri_pts + [tri_pts[0]], color=col, width=1.05)
                        anchor_y = y_bot + 1.2 if pos_kind == "below" else y_top - 4.0
                    else:
                        page.draw_line((x0, y_bot), (x1, y_bot), color=col, width=1.1)
                        tri_cx = mid_x
                        tri_top = y_top - 4.2
                        tri_bot = y_top - 0.5
                        tri_pts = [(tri_cx, tri_top), (tri_cx - 2.6, tri_bot), (tri_cx + 2.6, tri_bot)]
                        page.draw_polyline(tri_pts + [tri_pts[0]], color=col, width=0.95)
                        anchor_y = y_bot if pos_kind == "below" else tri_top - 1.5
                else:
                    page.draw_line((x0, y_bot), (x1, y_bot), color=col, width=1.0)
                    anchor_y = y_bot

                if it["note"]:
                    tier = it["tier"]
                    if pos_kind == "below":
                        note_y = y_base + 14.5 + tier * 14.5
                        line_end_y = note_y - NOTE_FS - 1.5
                        page.draw_line((mid_x, anchor_y + 0.5), (mid_x, max(anchor_y + 2.0, line_end_y)), color=col, width=0.75)
                    else:
                        note_y = min(y_top - 4.5, anchor_y - 2.5) - tier * 14.0
                        line_end_y = note_y + 1.5
                        page.draw_line((mid_x, anchor_y - 0.5), (mid_x, min(anchor_y - 1.5, line_end_y)), color=col, width=0.75)
                    page.insert_text((it["nx"], note_y), it["note"], fontname=sans_fn, fontsize=NOTE_FS, color=col)

    # 좌·우 여백 브라켓 / 수미상관 호 / 시상전환 구분선 렌더링
    n_lays = len(line_layouts)
    for br in brackets:
        s_idx = max(1, min(int(getattr(br, "start_line", 1) or 1), n_lays))
        e_idx = max(s_idx, min(int(getattr(br, "end_line", s_idx) or s_idx), n_lays))
        y_s = line_layouts[s_idx - 1]["y_mid"] - 3.0
        y_e = line_layouts[e_idx - 1]["y_mid"] + 3.0
        y_mid = (y_s + y_e) * 0.5
        col = color_map.get(getattr(br, "color", "blue"), color_map["blue"])
        style = getattr(br, "style", "brace") or "brace"
        side = getattr(br, "side", "right") or "right"
        note_txt = (getattr(br, "note", "") or "").strip()

        if style == "divider":
            y_div = (line_layouts[s_idx - 1]["y_base"] + line_layouts[e_idx - 1]["y_mid"]) * 0.5
            page.draw_line((X_POEM - 8.0, y_div), (W - 14.0, y_div), color=col, width=0.9, dashes="[3 3] 0")
            if note_txt:
                wrapped = _wrap_text_by_width(note_txt, sans_font, MARGIN_FS, RIGHT_TEXT_MAX_W)
                for wi, wl in enumerate(wrapped[:3]):
                    page.insert_text((X_RIGHT_MARGIN, y_div - 3.0 + wi * 10.0), wl, fontname=bold_fn, fontsize=MARGIN_FS, color=col)
        elif style == "arc":
            ax = X_LEFT_ARC if side == "left" else X_RIGHT_MARGIN - 8.0
            ctrl_x = ax - 22.0 if side == "left" else ax + 22.0
            dy = y_e - y_s
            page.draw_bezier(
                (ax, y_s),
                (ctrl_x, y_s + dy * 0.25),
                (ctrl_x, y_s + dy * 0.75),
                (ax, y_e),
                color=col,
                width=1.1,
            )
            if note_txt:
                if side == "left":
                    wrapped = _wrap_text_by_width(note_txt, sans_font, 7.2, 42.0)
                    start_ty = y_mid - (len(wrapped) - 1) * 4.5
                    for wi, wl in enumerate(wrapped[:4]):
                        page.insert_text((X_LEFT_LABEL, start_ty + wi * 9.5), wl, fontname=bold_fn, fontsize=7.2, color=col)
                else:
                    wrapped = _wrap_text_by_width(note_txt, sans_font, MARGIN_FS, RIGHT_TEXT_MAX_W)
                    start_ty = y_mid - (len(wrapped) - 1) * 4.5
                    for wi, wl in enumerate(wrapped[:4]):
                        page.insert_text((X_RIGHT_MARGIN, start_ty + wi * 10.0), wl, fontname=bold_fn, fontsize=MARGIN_FS, color=col)
        else:
            # brace (중괄호 } / {) — 시 본문 끝(max_end_x)보다 항상 우측에 배치
            max_end_x = max((line_layouts[k]["end_x"] for k in range(s_idx - 1, e_idx)), default=X_POEM + 100.0)
            bx = min(W - 82.0, max_end_x + 8.0) if side == "right" else X_LEFT_ARC
            tip = bx + 5.0 if side == "right" else bx - 5.0
            pts = [(bx, y_s), (bx + 2.5 if side == "right" else bx - 2.5, y_s + 2.0), (bx + 2.5 if side == "right" else bx - 2.5, y_mid - 2.0), (tip, y_mid), (bx + 2.5 if side == "right" else bx - 2.5, y_mid + 2.0), (bx + 2.5 if side == "right" else bx - 2.5, y_e - 2.0), (bx, y_e)]
            page.draw_polyline(pts, color=col, width=1.0)
            if note_txt:
                tx = tip + 5.0 if side == "right" else X_LEFT_LABEL
                tw = (W - tx - 8.0) if side == "right" else 44.0
                wrapped = _wrap_text_by_width(note_txt, sans_font, MARGIN_FS, max(38.0, tw))
                start_ty = y_mid - (len(wrapped) - 1) * 4.5 + 2.0
                for wi, wl in enumerate(wrapped[:4]):
                    page.insert_text((tx, start_ty + wi * 10.0), wl, fontname=sans_fn, fontsize=MARGIN_FS, color=col)

    try:
        pix = page.get_pixmap(dpi=240, alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()


def _render_poetry_work_png_pages(work: Any, max_height_pt: float = 560.0) -> list[bytes]:
    """시 원문 이미지가 너무 길어 페이지에 축소되지 않도록 연 단위로 분할해 렌더링한다."""
    from types import SimpleNamespace

    png0 = _render_poetry_work_png(work)
    with Image.open(io.BytesIO(png0)) as im:
        w_px, h_px = im.size
    h_pt = h_px * (520.0 / max(1, w_px))
    if h_pt <= max_height_pt:
        return [png0]

    raw_lines = [str(ln or "") for ln in (getattr(work, "lines", None) or [])]
    n_lines = len(raw_lines)
    if n_lines < 8:
        return [png0]

    n_chunks = max(2, math.ceil(h_pt / max_height_pt))
    blank_indices = [i for i, ln in enumerate(raw_lines) if not ln.strip()]

    split_points: list[int] = [0]
    for k in range(1, n_chunks):
        ideal = int(round(n_lines * k / n_chunks))
        # 가장 가까운 연 구분(빈 줄) 찾기
        best_bp = None
        best_dist = 9999
        for bi in blank_indices:
            if bi > split_points[-1] + 3 and bi < n_lines - 3:
                d = abs(bi - ideal)
                if d < best_dist:
                    best_dist = d
                    best_bp = bi
        if best_bp is not None and best_dist <= max(6, n_lines // (n_chunks * 2) + 2):
            split_points.append(best_bp + 1)
        elif ideal > split_points[-1] + 3 and ideal < n_lines - 2:
            split_points.append(ideal)
    split_points.append(n_lines)

    marks = list(getattr(work, "marks", None) or [])
    brackets = list(getattr(work, "brackets", None) or [])
    work_label = (getattr(work, "work_label", "") or "").strip()
    title_author = (getattr(work, "title_author", "") or "").strip()

    out_pngs: list[bytes] = []
    total_parts = len(split_points) - 1
    for part_idx in range(total_parts):
        s0 = split_points[part_idx]
        e0 = split_points[part_idx + 1]
        sub_lines = raw_lines[s0:e0]
        while sub_lines and not sub_lines[0].strip():
            sub_lines = sub_lines[1:]
            s0 += 1
        while sub_lines and not sub_lines[-1].strip():
            sub_lines = sub_lines[:-1]
            e0 -= 1
        if not sub_lines:
            continue

        # 1-based range [s0+1, e0]
        sub_marks = []
        for mk in marks:
            li = int(getattr(mk, "line_index", 0) or 0)
            if s0 + 1 <= li <= e0:
                sub_marks.append(
                    SimpleNamespace(
                        line_index=li - s0,
                        quote=getattr(mk, "quote", ""),
                        mark_type=getattr(mk, "mark_type", "underline"),
                        color=getattr(mk, "color", "red"),
                        position=getattr(mk, "position", "below"),
                        note=getattr(mk, "note", ""),
                    )
                )

        sub_brackets = []
        for br in brackets:
            bs = int(getattr(br, "start_line", 1) or 1)
            be = int(getattr(br, "end_line", bs) or bs)
            ov_s = max(bs, s0 + 1)
            ov_e = min(be, e0)
            if ov_s <= ov_e:
                sub_brackets.append(
                    SimpleNamespace(
                        start_line=ov_s - s0,
                        end_line=ov_e - s0,
                        side=getattr(br, "side", "right"),
                        style=getattr(br, "style", "brace"),
                        color=getattr(br, "color", "blue"),
                        note=getattr(br, "note", ""),
                    )
                )

        sub_title = f"{title_author} ({part_idx + 1}/{total_parts})" if total_parts > 1 else title_author
        sub_work = SimpleNamespace(
            work_label=work_label if part_idx == 0 else "",
            title_author=sub_title,
            lines=sub_lines,
            marks=sub_marks,
            brackets=sub_brackets,
        )
        out_pngs.append(_render_poetry_work_png(sub_work))

    return out_pngs or [png0]


def _highlight_poem_line_html(
    line_text: str,
    notes_with_idx: list[tuple[int, str]],
    matched_indices: set[int],
) -> str:
    """원문 시 행에서 아직 매칭되지 않은 시구(quote)를 찾아 밑줄 + 원문자 번호 HTML로 변환한다."""
    segments: list[tuple[bool, str, int]] = [(False, line_text, -1)]
    for note_idx, quote in notes_with_idx:
        if note_idx in matched_indices or not quote:
            continue
        new_segments: list[tuple[bool, str, int]] = []
        found_here = False
        for is_match, seg_txt, seg_idx in segments:
            if is_match or found_here:
                new_segments.append((is_match, seg_txt, seg_idx))
                continue
            pos = seg_txt.find(quote)
            if pos != -1:
                if pos > 0:
                    new_segments.append((False, seg_txt[:pos], -1))
                new_segments.append((True, seg_txt[pos : pos + len(quote)], note_idx))
                if pos + len(quote) < len(seg_txt):
                    new_segments.append((False, seg_txt[pos + len(quote) :], -1))
                matched_indices.add(note_idx)
                found_here = True
            else:
                new_segments.append((False, seg_txt, -1))
        segments = new_segments

    out_parts: list[str] = []
    for is_match, seg_txt, seg_idx in segments:
        if is_match:
            badge = _CIRCLED_NUMS[seg_idx] if seg_idx < len(_CIRCLED_NUMS) else f"({seg_idx + 1})"
            out_parts.append(
                f"<span class='poem-ul'><u>{html.escape(seg_txt)}</u></span>"
                f"<span class='poem-badge'> {badge}</span>"
            )
        else:
            out_parts.append(html.escape(seg_txt))
    return "".join(out_parts)


def _build_korean_passage_html_block(
    passage_text: str,
    is_lit: bool,
    sec_title: str = "6. 원문 지문 전체",
    underline_notes: list[Any] | None = None,
) -> str:
    """기호 제거 및 문단 재조립된 국어 원문 지문 HTML 블록(+ 시 밑줄 해설)을 생성한다."""
    reflowed_paragraphs = _clean_and_reflow_korean_passage(passage_text, is_poetry_or_literature=is_lit)
    clean_notes = [
        (i, (getattr(n, "quote", "") or "").strip(), (getattr(n, "explanation", "") or "").strip())
        for i, n in enumerate(underline_notes or [])
        if (getattr(n, "quote", "") or "").strip()
    ]
    notes_with_idx = [(i, q) for i, q, _ in clean_notes]
    matched_indices: set[int] = set()

    p_blocks: list[str] = []
    p_cls = "passage-lit-line" if is_lit else "passage-p"
    for rp in reflowed_paragraphs:
        if not rp:
            p_blocks.append("<div style='height:5pt;'></div>")
        elif re.match(r"^(?:\[\d{1,2}\s*[~\-～]\s*\d{1,2}\]|\([가나다라]\)|<[가-힣\s]+>)", rp):
            p_blocks.append(f"<div class='passage-hdr'>{html.escape(rp)}</div>")
        else:
            inner_html = (
                _highlight_poem_line_html(rp, notes_with_idx, matched_indices)
                if notes_with_idx
                else html.escape(rp)
            )
            p_blocks.append(f"<p class='{p_cls}'>{inner_html}</p>")
    if not p_blocks:
        return ""
    title_html = f"<div class='sec-title'>{html.escape(sec_title)}</div>" if sec_title else ""

    notes_html = ""
    if clean_notes:
        note_rows: list[str] = []
        for i, q, exp in clean_notes:
            badge = _CIRCLED_NUMS[i] if i < len(_CIRCLED_NUMS) else f"({i + 1})"
            note_rows.append(
                f"<div class='poem-note-row'><span class='poem-badge'>{badge}</span> "
                f"<span class='poem-ul'><u>{html.escape(q)}</u></span> : {html.escape(exp)}</div>"
            )
        notes_html = (
            "<div class='poem-notes-box'>"
            "<div class='poem-notes-hdr'>[밑줄 시구 정밀 해설]</div>"
            f"{''.join(note_rows)}"
            "</div>"
        )

    return f"{title_html}<div class='passage-box'>{''.join(p_blocks)}</div>{notes_html}"


def _render_story_pdf(full_html: str, css_text: str) -> bytes:
    """HTML + CSS를 PyMuPDF Story로 A4 PDF 바이트로 렌더링한다."""
    story = pymupdf.Story(html=full_html, user_css=css_text)
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


def generate_korean_passage_analysis_pdf(
    analyses: list[dict[str, Any]],
    doc_title: str = "수능·모의고사 국어 정밀 지문 분석지",
    include_diagrams: bool = True,
    include_passage: bool = True,
) -> bytes:
    """
    국어 지문 분석 결과(KoreanPassageAnalysisOutput + 원문 지문 + 출처 메타) 리스트를
    수능·학원형 정밀 지문 분석지 PDF로 렌더링한다 (PyMuPDF Story 기반 자동 페이지 넘김).
    """
    if not analyses:
        raise ValueError("생성할 국어 지문 분석 데이터가 없습니다.")

    analysis_css = _KOREAN_ANALYSIS_CSS

    sections_html: list[str] = []
    today_str = datetime.now().strftime("%Y.%m.%d")

    for idx, item in enumerate(analyses):
        an = item["analysis"]
        exam_name = item.get("exam_name", "")
        group_id = item.get("group_id", "")
        section_label = item.get("section_label", "독서")
        subgenre = item.get("subgenre", "")
        passage_text = (item.get("passage_text", "") or "").strip()
        is_lit = "문학" in section_label
        is_poetry = _is_korean_poetry_item(section_label, subgenre, passage_text, an)

        pb_cls = " class='page-break'" if idx > 0 else ""
        banner_html = f"""
        <div{pb_cls}>
          <div class="hdr-banner">
            <div class="hdr-sub">{html.escape(doc_title)} · {html.escape(section_label)}</div>
            <div class="hdr-title">{html.escape(an.title)}</div>
            <div class="hdr-meta">출처: {html.escape(exam_name)} [{html.escape(group_id)}번] &nbsp;|&nbsp; 생성일: {today_str}</div>
          </div>
        """

        def _keep_block(inner_html: str) -> str:
            return (
                "<table class='sec-keep-tbl'><tr class='sec-keep-tr'>"
                f"<td class='sec-keep-td'>{inner_html}</td>"
                "</tr></table>"
            )

        flow_html_inner = _format_structure_flow_html(an.structure_flow)
        theme_html = _keep_block(f"""
          <div class="sec-title">1. 핵심 주제 및 논리 · 시상 전개</div>
          <div class="theme-box">
            <div><b>핵심 주제</b></div>
            <div>{html.escape(an.core_theme)}</div>
          </div>
          <div class="flow-box">
            <div><b>전개 흐름:</b></div>
            <div class="flow-lines">{flow_html_inner}</div>
          </div>
        """)

        para_rows = []
        for ps in (an.paragraph_summaries or []):
            kp_html = f"<div class='para-kp'>▸ 포인트: {html.escape(ps.key_point)}</div>" if ps.key_point else ""
            para_rows.append(
                f"<div class='para-card'><span class='para-lbl'>[{html.escape(ps.label)}]</span> "
                f"{html.escape(ps.summary)}{kp_html}</div>"
            )
        para_html = (
            _keep_block(f"<div class='sec-title'>2. 문단 · 연별 정밀 요약</div>{''.join(para_rows)}")
            if para_rows
            else ""
        )

        vocab_items = [
            mv
            for mv in (an.marker_and_vocab_notes or [])
            if (mv.term or "").strip()
            and not re.match(r"^[㉠㉡㉢㉣㉤ⓐⓑⓒⓓⓔㄱ-ㅎa-eA-E]\b|^[㉠㉡㉢㉣㉤ⓐⓑⓒⓓⓔ]|^\[[A-E]\]", (mv.term or "").strip())
        ]
        vocab_rows = []
        for v_i, mv in enumerate(vocab_items):
            row_cls = "vocab-row-last" if v_i == len(vocab_items) - 1 else "vocab-row"
            vocab_rows.append(
                f"<div class='{row_cls}'><span class='vocab-badge'>{html.escape(mv.term)}</span> &nbsp; "
                f"{html.escape(mv.meaning)}</div>"
            )
        vocab_html = (
            _keep_block(
                "<div class='sec-title'>3. 지문 핵심 개념 정의 정리</div>"
                f"<div class='vocab-list'>{''.join(vocab_rows)}</div>"
            )
            if vocab_rows
            else ""
        )

        area_items = "".join(
            f"<div class='note-li'>• {html.escape(n)}</div>"
            for n in (an.area_specific_notes or [])
            if n.strip()
        )
        area_html = (
            _keep_block(f"<div class='sec-title'>4. 영역별 핵심 정리 ({html.escape(section_label)})</div>{area_items}")
            if area_items
            else ""
        )

        cp_items = "".join(
            f"<div class='note-li'>★ {html.escape(c)}</div>"
            for c in (an.exam_checkpoints or [])
            if c.strip()
        )
        cp_html = (
            _keep_block(
                f"<div class='sec-title'>5. 출제 예상 포인트 및 오답 주의사항</div><div class='checkpoint-box'>{cp_items}</div>"
            )
            if cp_items
            else ""
        )

        poetry_works = getattr(an, "poetry_works", None) or []
        if poetry_works:
            work_imgs_html: list[str] = []
            for pw in poetry_works:
                try:
                    for pw_png in _render_poetry_work_png_pages(pw, max_height_pt=560.0):
                        pw_b64 = base64.b64encode(pw_png).decode("ascii")
                        work_imgs_html.append(
                            _keep_block(
                                f"<div class='diagram-wrap' style='margin-top:6pt; margin-bottom:8pt;'>"
                                f"<img class='diagram-img' width='515' src='data:image/png;base64,{pw_b64}'/>"
                                "</div>"
                            )
                        )
                except Exception:
                    continue

            if work_imgs_html:
                compact_theme = _keep_block(
                    "<div class='theme-box' style='margin-bottom:6pt;'>"
                    "<div><b>핵심 주제</b></div>"
                    f"<div>{html.escape(an.core_theme)}</div>"
                    "</div>"
                )
                compact_cp = (
                    _keep_block(
                        f"<div class='sec-title'>출제 예상 포인트 및 오답 주의사항</div><div class='checkpoint-box'>{cp_items}</div>"
                    )
                    if cp_items
                    else ""
                )
                sections_html.append(
                    f"{banner_html}{compact_theme}{''.join(work_imgs_html)}{compact_cp}</div>"
                )
                continue

        poetry_notes = getattr(an, "poetry_underline_notes", None) or []
        should_include_passage = include_passage or is_poetry
        passage_sec_title = "6. 원문 시 전문 및 밑줄 정밀 해설" if is_poetry else "6. 원문 지문 전체"
        passage_html = (
            _build_korean_passage_html_block(
                passage_text,
                is_lit,
                sec_title=passage_sec_title,
                underline_notes=poetry_notes if is_poetry else None,
            )
            if should_include_passage
            else ""
        )

        diagram_html = ""
        v_diag = getattr(an, "visual_diagram", None)
        if include_diagrams and v_diag is not None and (
            getattr(v_diag, "nodes", None)
            or getattr(v_diag, "graph_spec", None)
            or getattr(v_diag, "emotion_points", None)
        ):
            try:
                diag_png = render_korean_diagram_png(v_diag)
                diag_b64 = base64.b64encode(diag_png).decode("ascii")
                diagram_html = _keep_block(
                    "<div class='sec-title'>한눈에 보는 개념 · 원리 도식</div>"
                    f"<div class='diagram-wrap'><img class='diagram-img' width='505' src='data:image/png;base64,{diag_b64}'/></div>"
                )
            except Exception:
                diagram_html = ""

        sections_html.append(
            f"{banner_html}{theme_html}{diagram_html}{para_html}{vocab_html}{area_html}{cp_html}{passage_html}</div>"
        )

    full_html = f"<html><body>{''.join(sections_html)}</body></html>"
    return _render_story_pdf(full_html, analysis_css)


def generate_korean_passage_only_pdf(item: dict[str, Any]) -> bytes:
    """국어 지문 1개 세트의 '원문 지문 전체'만 독립된 PDF로 렌더링한다 (원문 ON/OFF 병합용)."""
    exam_name = str(item.get("exam_name") or "")
    group_id = str(item.get("group_id") or "")
    section_label = str(item.get("section_label") or "독서")
    passage_text = (item.get("passage_text") or "").strip()
    an = item.get("analysis")
    title = getattr(an, "title", "") if an is not None else ""
    is_lit = "문학" in section_label

    passage_block = _build_korean_passage_html_block(passage_text, is_lit, sec_title="")
    if not passage_block:
        raise ValueError("렌더링할 원문 지문 텍스트가 비어 있습니다.")

    banner = f"""
    <div class="hdr-banner" style="padding:7pt 11pt; margin-bottom:8pt;">
      <div class="hdr-sub">원문 지문 전체 · {html.escape(section_label)}</div>
      <div class="hdr-title" style="font-size:12.5pt;">{html.escape(title or f"{exam_name} [{group_id}번]")}</div>
      <div class="hdr-meta">출처: {html.escape(exam_name)} [{html.escape(group_id)}번]</div>
    </div>
    """
    full_html = f"<html><body>{banner}{passage_block}</body></html>"
    return _render_story_pdf(full_html, _KOREAN_ANALYSIS_CSS)


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

