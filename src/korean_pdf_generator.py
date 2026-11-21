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

