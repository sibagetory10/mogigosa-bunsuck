from __future__ import annotations

import base64
import html
import re
from datetime import datetime
from typing import Any

from src.diagram_renderer import render_korean_diagram_png
from src.pdf.common import _build_korean_story_font_bundle, _render_story_pdf
from src.pdf.poetry_renderer import _highlight_poem_line_html, _render_poetry_work_png_pages
from src.pdf.styles import _KOREAN_ANALYSIS_CSS

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
_CIRCLED_NUMS = (
    "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩",
    "⑪", "⑫", "⑬", "⑭", "⑮", "⑯", "⑰", "⑱", "⑲", "⑳",
)


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
