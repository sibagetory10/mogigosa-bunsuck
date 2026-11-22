from __future__ import annotations

import html
import io
import re
from datetime import datetime

import pymupdf

from src.models import UserVocabItem

_VOCAB_PDF_CSS = """
body {
    font-family: "NanumGothic", "Malgun Gothic", "DejaVu Sans", sans-serif;
    font-size: 9pt;
    line-height: 1.45;
    color: #0F172A;
}
.v-banner {
    border: 1.3pt solid #1E293B;
    border-radius: 5pt;
    background-color: #F8FAFC;
    padding: 8pt 11pt;
    margin-bottom: 10pt;
}
.v-sub {
    font-size: 8pt;
    font-weight: bold;
    color: #1D4ED8;
}
.v-title {
    font-size: 14pt;
    font-weight: bold;
    color: #0F172A;
    margin-top: 2pt;
}
.v-meta-tbl {
    width: 100%;
    border-collapse: collapse;
    margin-top: 5pt;
    font-size: 8.2pt;
    background-color: #FFFFFF;
}
.v-meta-tbl td {
    border: 0.6pt solid #64748B;
    padding: 2.5pt 6pt;
    color: #334155;
}
.study-tbl {
    width: 100%;
    border-collapse: separate;
    border-spacing: 5pt 5pt;
}
.study-cell {
    width: 50%;
    vertical-align: top;
    border: 0.75pt solid #CBD5E1;
    border-radius: 4pt;
    background-color: #FFFFFF;
    padding: 5.5pt 7pt;
}
.study-empty {
    width: 50%;
}
.w-num {
    font-size: 8.2pt;
    font-weight: bold;
    color: #2563EB;
}
.w-head {
    font-size: 10.8pt;
    font-weight: bold;
    color: #0F172A;
}
.w-ipa {
    font-size: 8pt;
    color: #64748B;
}
.w-pos {
    font-size: 7.4pt;
    font-weight: bold;
    color: #1E40AF;
    background-color: #DBEAFE;
    padding: 0.5pt 4pt;
    border-radius: 3pt;
}
.w-lvl {
    font-size: 7.2pt;
    font-weight: bold;
    color: #166534;
    background-color: #DCFCE7;
    padding: 0.5pt 4pt;
    border-radius: 3pt;
}
.w-mean {
    font-size: 9pt;
    font-weight: bold;
    color: #1E293B;
    margin-top: 2.5pt;
}
.w-syn {
    font-size: 7.7pt;
    color: #475569;
    margin-top: 2pt;
}
.w-ex {
    font-size: 7.6pt;
    color: #334155;
    margin-top: 2.5pt;
    padding-top: 2pt;
    border-top: 0.5pt dashed #E2E8F0;
}
.test-tbl {
    width: 100%;
    border-collapse: collapse;
    margin-top: 4pt;
    font-size: 8.8pt;
}
.test-tbl th {
    background-color: #1E3A8A;
    color: #FFFFFF;
    font-weight: bold;
    text-align: left;
    padding: 4.5pt 6pt;
    border: 0.6pt solid #1E3A8A;
    font-size: 8.4pt;
}
.test-tbl td {
    padding: 6.5pt 6pt;
    border: 0.6pt solid #CBD5E1;
    vertical-align: middle;
}
.test-alt td {
    background-color: #F8FAFC;
}
.blank-line {
    color: #64748B;
    font-size: 8.5pt;
}
.ans-break {
    page-break-before: always;
}
.ans-hdr {
    border-bottom: 1.5pt solid #1E3A8A;
    padding-bottom: 4pt;
    margin-bottom: 8pt;
    font-size: 12pt;
    font-weight: bold;
    color: #1E3A8A;
}
.ans-tbl {
    width: 100%;
    border-collapse: collapse;
    font-size: 8.2pt;
}
.ans-tbl th {
    background-color: #EFF6FF;
    color: #1E3A8A;
    border: 0.6pt solid #93C5FD;
    padding: 3.5pt 5pt;
    text-align: left;
}
.ans-tbl td {
    border: 0.6pt solid #CBD5E1;
    padding: 3.5pt 5pt;
    vertical-align: top;
}
"""


def _build_spelling_hint(headword: str, include_first_letter: bool) -> str:
    tokens = [tok for tok in (headword or "").strip().split() if tok]
    if not tokens:
        return "________________________"
    if not include_first_letter:
        return "   ".join("_" * max(6, min(14, len(tok))) for tok in tokens)
    parts: list[str] = []
    for tok in tokens:
        clean = re.sub(r"[^A-Za-z\-]", "", tok)
        if not clean:
            parts.append("______")
        elif len(clean) == 1:
            parts.append(f"{clean}_")
        else:
            parts.append(f"{clean[0]} {'_' * max(5, min(13, len(clean) - 1))}")
    return "   ".join(parts)


def _render_vocab_story_pdf(full_html: str) -> bytes:
    story = pymupdf.Story(html=full_html, user_css=_VOCAB_PDF_CSS)
    writer_buf = io.BytesIO()
    mediabox = pymupdf.paper_rect("a4")
    where = mediabox + (34, 34, -34, -40)
    writer = pymupdf.DocumentWriter(writer_buf)
    more = True
    while more:
        dev = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(dev, None)
        writer.end_page()
    writer.close()

    doc = pymupdf.open(stream=writer_buf.getvalue(), filetype="pdf")
    total_pages = len(doc)
    for page_idx, page in enumerate(doc, start=1):
        pw = float(page.rect.width)
        ph = float(page.rect.height)
        page.insert_text(
            pymupdf.Point(pw / 2.0 - 20.0, ph - 18.0),
            f"- {page_idx} / {total_pages} -",
            fontsize=8.2,
            color=(0.40, 0.45, 0.52),
        )
    try:
        doc.subset_fonts()
    except Exception:
        pass
    out_bytes = doc.tobytes(deflate=True, garbage=3)
    doc.close()
    return out_bytes


def generate_vocab_wordbook_pdf(
    items: list[UserVocabItem],
    title: str = "나만의 수능 영단어장",
    mode: str = "study",
    include_first_letter_hint: bool = True,
) -> bytes:
    """
    개인 단어장 목록으로 암기용 2단 단어장 PDF 또는 테스트지+정답지 PDF를 생성한다.
    - mode == 'study': 2단 암기용 단어장
    - mode == 'test_meaning': 영단어 → 한국어 뜻 쓰기 테스트지 + 정답지
    - mode == 'test_word': 한국어 뜻 → 영단어 스펠링 테스트지 + 정답지
    """
    if not items:
        raise ValueError("PDF로 생성할 영단어 항목이 없습니다.")

    safe_title = html.escape((title or "나만의 수능 영단어장").strip())
    today_str = datetime.now().strftime("%Y.%m.%d")

    mode_sub_map = {
        "study": "수능·모의고사 영단어 정밀 암기장",
        "test_meaning": "영단어 암기 확인 테스트 — 뜻 쓰기 시험지",
        "test_word": "영단어 암기 확인 테스트 — 영단어 스펠링 시험지",
    }
    sub_label = html.escape(mode_sub_map.get(mode, mode_sub_map["study"]))

    banner_html = f"""
    <div class="v-banner">
      <div class="v-sub">{sub_label}</div>
      <div class="v-title">{safe_title}</div>
      <table class="v-meta-tbl">
        <tr>
          <td width="34%"><b>이름:</b> ____________________</td>
          <td width="33%"><b>날짜:</b> {today_str}</td>
          <td width="33%"><b>총 문항 / 점수:</b> _____ / {len(items)}개</td>
        </tr>
      </table>
    </div>
    """

    if mode == "study":
        rows_html: list[str] = []
        for idx in range(0, len(items), 2):
            cells: list[str] = []
            for offset in (0, 1):
                pos_i = idx + offset
                if pos_i >= len(items):
                    cells.append("<td class='study-empty'></td>")
                    continue
                it = items[pos_i]
                num = pos_i + 1
                hw = html.escape(it.headword)
                ipa = f"<span class='w-ipa'>{html.escape(it.phonetic)}</span>" if it.phonetic else ""
                pos_b = f"<span class='w-pos'>{html.escape(it.pos_summary)}</span>" if it.pos_summary else ""
                lvl_b = f"<span class='w-lvl'>{html.escape(it.cefr_level)}</span>" if it.cefr_level else ""
                mean = html.escape(it.summary_ko or "-")
                syn_html = (
                    f"<div class='w-syn'><b>유의어:</b> {html.escape(', '.join(it.synonyms[:4]))}</div>"
                    if it.synonyms
                    else ""
                )
                ex_bits: list[str] = []
                if it.example_en:
                    ex_bits.append(f"<div>{html.escape(it.example_en)}</div>")
                if it.example_ko:
                    ex_bits.append(f"<div style='color:#64748B;'>{html.escape(it.example_ko)}</div>")
                ex_html = f"<div class='w-ex'>{''.join(ex_bits)}</div>" if ex_bits else ""
                memo_html = (
                    f"<div class='w-syn'><b>메모:</b> {html.escape(it.memo)}</div>"
                    if it.memo
                    else ""
                )
                cells.append(
                    f"<td class='study-cell'>"
                    f"<div><span class='w-num'>{num:02d}.</span> <span class='w-head'>{hw}</span> {ipa} {pos_b} {lvl_b}</div>"
                    f"<div class='w-mean'>{mean}</div>"
                    f"{syn_html}{memo_html}{ex_html}"
                    f"</td>"
                )
            rows_html.append(f"<tr>{''.join(cells)}</tr>")

        body_html = f"{banner_html}<table class='study-tbl'>{''.join(rows_html)}</table>"
        return _render_vocab_story_pdf(f"<html><body>{body_html}</body></html>")

    # Test modes ('test_meaning' or 'test_word') + separate Answer Key page
    is_meaning_test = mode != "test_word"
    q_col_hdr = "영단어 / 숙어" if is_meaning_test else "한국어 뜻"
    a_col_hdr = "한국어 뜻 적기" if is_meaning_test else "영단어 스펠링 쓰기"

    test_rows: list[str] = []
    for idx, it in enumerate(items, start=1):
        row_cls = " class='test-alt'" if idx % 2 == 0 else ""
        if is_meaning_test:
            ipa = f" <span class='w-ipa'>{html.escape(it.phonetic)}</span>" if it.phonetic else ""
            pos = f" <span class='w-pos'>{html.escape(it.pos_summary)}</span>" if it.pos_summary else ""
            q_cell = f"<span class='w-head' style='font-size:10pt;'>{html.escape(it.headword)}</span>{ipa}{pos}"
            a_cell = "<span class='blank-line'>________________________________________________</span>"
        else:
            pos = f"<span class='w-pos'>{html.escape(it.pos_summary)}</span> " if it.pos_summary else ""
            q_cell = f"{pos}{html.escape(it.summary_ko or '-')}"
            hint = _build_spelling_hint(it.headword, include_first_letter=include_first_letter_hint)
            a_cell = f"<span class='blank-line'>{html.escape(hint)}</span>"

        test_rows.append(
            f"<tr{row_cls}>"
            f"<td width='7%' style='font-weight:bold; color:#1E3A8A; text-align:center;'>{idx}</td>"
            f"<td width='45%'>{q_cell}</td>"
            f"<td width='48%'>{a_cell}</td>"
            f"</tr>"
        )

    test_table = (
        f"<table class='test-tbl'>"
        f"<tr><th width='7%' style='text-align:center;'>No.</th><th width='45%'>{q_col_hdr}</th><th width='48%'>{a_col_hdr}</th></tr>"
        f"{''.join(test_rows)}</table>"
    )

    # Answer key page (2-column compact table)
    half = (len(items) + 1) // 2
    ans_rows: list[str] = []
    for r_i in range(half):
        left = items[r_i]
        l_word = html.escape(left.headword)
        l_mean = html.escape(left.summary_ko or "-")
        right_cells = "<td></td><td></td><td></td>"
        if r_i + half < len(items):
            right = items[r_i + half]
            right_cells = (
                f"<td style='font-weight:bold; color:#1E3A8A; text-align:center;'>{r_i + half + 1}</td>"
                f"<td style='font-weight:bold;'>{html.escape(right.headword)}</td>"
                f"<td>{html.escape(right.summary_ko or '-')}</td>"
            )
        ans_rows.append(
            f"<tr>"
            f"<td width='6%' style='font-weight:bold; color:#1E3A8A; text-align:center;'>{r_i + 1}</td>"
            f"<td width='19%' style='font-weight:bold;'>{l_word}</td>"
            f"<td width='25%'>{l_mean}</td>"
            f"{right_cells}"
            f"</tr>"
        )

    ans_block = (
        f"<div class='ans-break'></div>"
        f"<div class='ans-hdr'>✅ {safe_title} — 정답 확인표 ({len(items)}문항)</div>"
        f"<table class='ans-tbl'>"
        f"<tr><th width='6%' style='text-align:center;'>No.</th><th width='19%'>영단어</th><th width='25%'>뜻</th>"
        f"<th width='6%' style='text-align:center;'>No.</th><th width='19%'>영단어</th><th width='25%'>뜻</th></tr>"
        f"{''.join(ans_rows)}</table>"
    )

    return _render_vocab_story_pdf(f"<html><body>{banner_html}{test_table}{ans_block}</body></html>")
