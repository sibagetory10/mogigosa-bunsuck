from __future__ import annotations

import hashlib
import math
import re
import threading
from typing import TYPE_CHECKING, Any

import pymupdf

if TYPE_CHECKING:
    from src.models.english import EnglishSentenceAnalysisItem

# ---------------------------------------------------------------------------
# PDF 텍스트 추출 프로세스 내 캐시 (배치 처리 시 동일 PDF 반복 파싱 방지)
# ---------------------------------------------------------------------------
_PDF_EXTRACT_CACHE: dict[str, Any] = {}
_PDF_EXTRACT_CACHE_LOCK = threading.RLock()
_MAX_PDF_CACHE_ENTRIES = 32


def _cache_get(kind: str, pdf_bytes: bytes) -> Any | None:
    if not pdf_bytes:
        return None
    key = f"{kind}:{hashlib.sha256(pdf_bytes).hexdigest()[:16]}"
    with _PDF_EXTRACT_CACHE_LOCK:
        return _PDF_EXTRACT_CACHE.get(key)


def _cache_set(kind: str, pdf_bytes: bytes, value: Any) -> None:
    if not pdf_bytes:
        return
    key = f"{kind}:{hashlib.sha256(pdf_bytes).hexdigest()[:16]}"
    with _PDF_EXTRACT_CACHE_LOCK:
        if len(_PDF_EXTRACT_CACHE) >= _MAX_PDF_CACHE_ENTRIES:
            oldest = next(iter(_PDF_EXTRACT_CACHE))
            _PDF_EXTRACT_CACHE.pop(oldest, None)
        _PDF_EXTRACT_CACHE[key] = value


# ---------------------------------------------------------------------------
# 공통 PDF 하이픈 / 노이즈 / 다단 레이아웃 유틸리티 (AI 미사용)
# ---------------------------------------------------------------------------
_PDF_HYPHEN_CHARS_RE = re.compile(r"[\u00ad\u2010\u2011\u2012\u2212\uf02d]")
_PDF_ENDASH_ALNUM_RE = re.compile(r"(?<=[A-Za-z0-9])\u2013(?=[A-Za-z0-9])")
_CIRCLED_NUMS = ("①", "②", "③", "④", "⑤")
_CIRCLED_TO_INT = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5}
_INT_TO_CIRCLED = {1: "①", 2: "②", 3: "③", 4: "④", 5: "⑤"}

_DEFAULT_ORDER_MAP: dict[str, tuple[str, str, str]] = {
    "1": ("A", "C", "B"),
    "2": ("B", "A", "C"),
    "3": ("B", "C", "A"),
    "4": ("C", "A", "B"),
    "5": ("C", "B", "A"),
}
_DEFAULT_LONG_ORDER_MAP: dict[str, tuple[str, str, str]] = {
    "1": ("B", "D", "C"),
    "2": ("C", "B", "D"),
    "3": ("C", "D", "B"),
    "4": ("D", "B", "C"),
    "5": ("D", "C", "B"),
}


def _normalize_pdf_hyphens(text: str) -> str:
    if not text:
        return ""
    out = _PDF_HYPHEN_CHARS_RE.sub("-", text)
    out = _PDF_ENDASH_ALNUM_RE.sub("-", out)
    return out


def _detect_pdf_columns(page: pymupdf.Page) -> list[tuple[float, float]]:
    """1단 / 2단 / 3단 PDF 레이아웃을 세로 구분선 및 텍스트 분포로 판별한다."""
    w = float(page.rect.width)
    h = float(page.rect.height)
    if w <= 0 or h <= 0:
        return [(0.0, max(1.0, w))]

    dividers: list[float] = []
    for d in page.get_drawings():
        rect = d.get("rect")
        if not rect:
            continue
        rw = abs(float(rect.x1) - float(rect.x0))
        rh = abs(float(rect.y1) - float(rect.y0))
        cx = (float(rect.x0) + float(rect.x1)) / 2.0
        if (0.12 * w <= cx <= 0.88 * w) and rw <= 5.0 and rh >= 0.20 * h:
            if not any(abs(cx - existing) <= 12.0 for existing in dividers):
                dividers.append(cx)

    dividers.sort()
    if len(dividers) >= 2:
        # 3단 레이아웃 지원 (예: 일부 3단 해설지)
        d1, d2 = dividers[0], dividers[-1]
        if d2 - d1 >= 0.20 * w:
            return [(0.0, d1), (d1, d2), (d2, w)]
    if len(dividers) == 1:
        return [(0.0, dividers[0]), (dividers[0], w)]

    # 세로 구분선 미탐지 시 텍스트 라인 좌표 분포로 1단 vs 2단 판별
    mid_x = w / 2.0
    spanning = 0
    right_only = 0
    page_dict = page.get_text("dict")
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            lx0, ly0, lx1, ly1 = line["bbox"]
            if ly1 < 0.12 * h or ly0 > 0.92 * h:
                continue
            txt = "".join(sp.get("text", "") for sp in line.get("spans", [])).strip()
            if len(txt) < 8:
                continue
            if lx0 < 0.30 * w and lx1 > 0.70 * w:
                spanning += 1
            elif lx0 >= mid_x - 4.0:
                right_only += 1
    if spanning >= 4 and right_only == 0:
        return [(0.0, w)]
    return [(0.0, mid_x), (mid_x, w)]


def _has_vector_hyphen_between(
    x_gap_start: float,
    x_gap_end: float,
    line_y0: float,
    line_y1: float,
    drawings: list[dict[str, Any]] | None,
) -> bool:
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
            rcx = (float(rect.x0) + float(rect.x1)) / 2.0
            rcy = (float(rect.y0) + float(rect.y1)) / 2.0
            if x_lo <= rcx <= x_hi and y_lo <= rcy <= y_hi:
                return True
    return False


def _has_vector_blank_underline(
    x_gap_start: float,
    x_gap_end: float,
    line_y0: float,
    line_y1: float,
    drawings: list[dict[str, Any]] | None,
) -> bool:
    """문제지 빈칸(벡터 밑줄선) 감지기: 두 스팬 사이 넓은 공백 아래에 수평 선분이 있으면 True."""
    if not drawings or (x_gap_end - x_gap_start) < 22.0:
        return False
    y_lo = line_y0
    y_hi = line_y1 + 6.0
    for d in drawings:
        rect = d.get("rect")
        if not rect:
            continue
        rw = float(rect.x1 - rect.x0)
        rh = float(rect.y1 - rect.y0)
        if rw >= 18.0 and rh <= 2.8:
            rcy = (float(rect.y0) + float(rect.y1)) / 2.0
            if y_lo <= rcy <= y_hi:
                overlap = min(float(rect.x1), x_gap_end + 4.0) - max(float(rect.x0), x_gap_start - 4.0)
                if overlap >= 16.0:
                    return True
    return False


def _is_margin_noise_line(text: str, y0: float, page_h: float) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    always_noise = (
        "이 문제지에 관한 저작권",
        "한국교육과정평가원",
        "답안지의 해당란에",
    )
    if any(p in t for p in always_noise):
        return True
    if re.match(r"^(?:국어|수학|영어|한국사|사회탐구|과학탐구|직업탐구|제2외국어)\s*영역$", t):
        return True
    in_margin = (y0 < 0.095 * page_h) or (y0 > 0.895 * page_h)
    if (y0 < 0.115 * page_h or y0 > 0.885 * page_h) and re.match(r"^\[?\s*\d{1,3}\s*\]?$", t):
        return True
    if not in_margin:
        return False
    if re.match(r"^(?:\d{1,3}\s+)?(?:국어|수학|영어|한국사|사회탐구|과학탐구)(?:\s+\d{1,3})?$", t):
        return True
    if re.match(r"^(?:영역\s*\d{1,3}|\d{1,3}\s*영역)$", t):
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
        "정답 및 해설",
        "정답과 해설",
    )
    return any(p in t for p in margin_noise)


def _extract_column_logical_lines(
    page_dict: dict[str, Any],
    col_x0: float,
    col_x1: float,
    page_h: float,
    page_w: float,
    drawings: list[dict[str, Any]] | None = None,
    *,
    is_last_col: bool = False,
    detect_blank_underlines: bool = False,
) -> list[dict[str, Any]]:
    """
    단(column) 내부의 모든 텍스트 스팬을 y중심 오차 ±3.5pt 이내 동일 행으로 묶어
    x0 오름차순으로 병합한 논리 라인 리스트를 반환한다.
    """
    is_multi_col = (col_x1 - col_x0) < (0.85 * page_w)
    spans_in_col: list[dict[str, Any]] = []

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            lx0, ly0, lx1, ly1 = line["bbox"]
            if ly1 < 0.030 * page_h or ly0 > 0.955 * page_h:
                continue
            for sp in line.get("spans", []):
                stext = sp.get("text", "")
                if not stext:
                    continue
                sx0, sy0, sx1, sy1 = sp.get("bbox", (lx0, ly0, lx1, ly1))
                scx = (float(sx0) + float(sx1)) / 2.0
                if is_multi_col:
                    if is_last_col:
                        if not (col_x0 <= scx <= col_x1 + 2.0):
                            continue
                    else:
                        if not (col_x0 - 2.0 <= scx < col_x1):
                            continue
                spans_in_col.append(
                    {
                        "x0": float(sx0),
                        "y0": float(sy0),
                        "x1": float(sx1),
                        "y1": float(sy1),
                        "yc": (float(sy0) + float(sy1)) / 2.0,
                        "text": _normalize_pdf_hyphens(stext),
                    }
                )

    if not spans_in_col:
        return []

    spans_in_col.sort(key=lambda s: (round(s["yc"], 1), s["x0"]))
    rows: list[list[dict[str, Any]]] = []
    for sp in spans_in_col:
        if not rows or abs(sp["yc"] - rows[-1][0]["yc"]) > 3.5:
            rows.append([sp])
        else:
            rows[-1].append(sp)

    logical_lines: list[dict[str, Any]] = []
    for row in rows:
        row.sort(key=lambda s: s["x0"])
        ly0 = min(s["y0"] for s in row)
        ly1 = max(s["y1"] for s in row)
        lx0 = min(s["x0"] for s in row)
        lx1 = max(s["x1"] for s in row)
        parts: list[str] = []
        for idx_s, sp in enumerate(row):
            st = sp["text"]
            if idx_s > 0 and parts:
                prev_sp = row[idx_s - 1]
                gap = sp["x0"] - prev_sp["x1"]
                prev_t = parts[-1]
                if detect_blank_underlines and _has_vector_blank_underline(
                    prev_sp["x1"], sp["x0"], ly0, ly1, drawings
                ):
                    parts.append(" ____ ")
                elif (
                    0.5 <= gap <= 14.0
                    and prev_t
                    and st
                    and prev_t[-1].isalnum()
                    and st[0].isalnum()
                    and _has_vector_hyphen_between(prev_sp["x1"], sp["x0"], ly0, ly1, drawings)
                ):
                    parts.append("-")
                elif gap > 1.5 and not prev_t.endswith((" ", "\t")) and not st.startswith((" ", "\t")):
                    parts.append(" ")
            parts.append(st)

        line_text = re.sub(r"[ \t]{2,}", " ", "".join(parts)).strip()
        if not line_text or _is_margin_noise_line(line_text, ly0, page_h):
            continue
        logical_lines.append(
            {
                "x0": lx0,
                "y0": ly0,
                "x1": lx1,
                "y1": ly1,
                "text": line_text,
                "col_x0": col_x0,
                "col_w": col_x1 - col_x0,
            }
        )

    logical_lines.sort(key=lambda item: (round(item["y0"], 1), item["x0"]))
    return logical_lines


# ---------------------------------------------------------------------------
# 문제지 PDF (`source_pdf`) 원문 및 각주 결정론적 추출 (AI 0%)
# ---------------------------------------------------------------------------
_PROB_SHARED_RE = re.compile(r"^\[\s*(\d{1,2})\s*[~\-～–—]\s*(\d{1,2})\s*\]")
_PROB_Q_START_RE = re.compile(r"^(\d{1,2})\s*[.)]\s*(.*)$")


def _parse_problem_pdf_questions(problem_pdf_bytes: bytes) -> dict[str, str]:
    """문제지 PDF 전체를 1회 파싱해 문항 번호('18'..'45') 및 세트('41~42','43~45') 원문 텍스트 맵을 반환한다."""
    cached = _cache_get("prob", problem_pdf_bytes)
    if cached is not None:
        return cached

    ordered_lines: list[dict[str, Any]] = []
    with pymupdf.open(stream=problem_pdf_bytes, filetype="pdf") as doc:
        for page_idx in range(doc.page_count):
            page = doc[page_idx]
            page_h = float(page.rect.height)
            page_w = float(page.rect.width)
            page_dict = page.get_text("dict")
            drawings = page.get_drawings()
            cols = _detect_pdf_columns(page)
            for col_idx, (cx0, cx1) in enumerate(cols):
                col_lines = _extract_column_logical_lines(
                    page_dict,
                    cx0,
                    cx1,
                    page_h,
                    page_w,
                    drawings=drawings,
                    is_last_col=(col_idx == len(cols) - 1),
                    detect_blank_underlines=True,
                )
                ordered_lines.extend(col_lines)

    out_map: dict[str, str] = {}
    cur_q: int | None = None
    q_buffers: dict[int, list[str]] = {}
    shared_buffers: dict[str, list[str]] = {}
    active_shared_gid = ""
    active_shared_end = 0
    seen_qs: set[int] = set()

    for ln in ordered_lines:
        txt = ln["text"]
        col_x0 = ln["col_x0"]
        col_w = max(1.0, ln["col_w"])

        sm = _PROB_SHARED_RE.match(txt)
        if sm and ln["x0"] <= col_x0 + 0.48 * col_w:
            qs, qe = int(sm.group(1)), int(sm.group(2))
            if (qs, qe) in ((41, 42), (43, 45)):
                active_shared_gid = f"{qs}~{qe}"
                active_shared_end = qe
                cur_q = None
                shared_buffers.setdefault(active_shared_gid, []).append(txt)
            else:
                # [31~34], [36~37], [38~39] 등 발문 공통 헤더는 지문 공유가 아니므로
                # 직전 문항 버퍼로 누수되지 않도록 cur_q만 초기화하고 건너뛴다.
                cur_q = None
            continue

        qm = _PROB_Q_START_RE.match(txt)
        if qm and ln["x0"] <= col_x0 + 0.45 * col_w:
            qn = int(qm.group(1))
            if 1 <= qn <= 45 and qn not in seen_qs and (not seen_qs or qn > max(seen_qs)):
                seen_qs.add(qn)
                cur_q = qn
                if active_shared_gid and qn > active_shared_end:
                    active_shared_gid = ""
                    active_shared_end = 0
                q_buffers.setdefault(qn, []).append(txt)
                continue

        if active_shared_gid and cur_q is None:
            shared_buffers.setdefault(active_shared_gid, []).append(txt)
        elif cur_q is not None:
            q_buffers.setdefault(cur_q, []).append(txt)

    for qn, lines_list in q_buffers.items():
        out_map[str(qn)] = "\n".join(lines_list).strip()
    for gid, lines_list in shared_buffers.items():
        out_map[gid] = "\n".join(lines_list).strip()

    _cache_set("prob", problem_pdf_bytes, out_map)
    return out_map


def extract_problem_question_text_from_pdf(
    problem_pdf_bytes: bytes,
    q_num: int,
    passage_group_id: str = "",
) -> str:
    """저장된 문제지 PDF에서 해당 문항(또는 41~42/43~45 세트 지문)의 원문 텍스트를 추출한다."""
    if not problem_pdf_bytes or q_num <= 0:
        return ""
    parsed = _parse_problem_pdf_questions(problem_pdf_bytes)
    gid = (passage_group_id or "").strip()
    if gid in ("41~42", "43~45") and parsed.get(gid):
        shared_txt = parsed[gid]
        q_txt = parsed.get(str(q_num), "")
        return f"{shared_txt}\n{q_txt}".strip() if q_txt else shared_txt
    return parsed.get(str(q_num), "")


# 공식 각주 추출 정규식 (* word: 한국어 뜻)
_ENG_FOOTNOTE_RE = re.compile(
    r"(?<!\*)\*{1,4}(?!\*)\s*([A-Za-z][A-Za-z0-9'\-\s]{0,36}?)\s*:\s*([^*\n①②③④⑤]+?)"
    r"(?="
    r"\s*\*+\s*[A-Za-z]"
    r"|\s*[①②③④⑤]"
    r"|\s*\[\d점\]"
    r"|\s+(?:(?:\d{1,2}\s+)?(?:1[89]|[234]\d)\.\s*(?:[가-힣]|윗글|다음|밑줄|주어진)|\[\d{1,2}\s*[~～-]\s*\d{1,2}\])"
    r"|\s*(?:￫|↓|→|⇒|➜|\[요약문\]|\[보기\])"
    r"|\s+\(A\)\s+"
    r"|\s+(?:\d{1,2}\s+)?영어\s*영역"
    r"|\s+[A-Z][a-z]+(?:\s+[A-Za-z0-9,'’\"-]+){3,}"
    r"|$)"
)
_FOOTNOTE_TAIL_CUT_RE = re.compile(
    r"\s*(?:￫|↓|→|⇒|➜|\[요약문\]|\[보기\]|\(A\)\s+"
    r"|(?:(?:\d{1,2}\s+)?(?:1[89]|[234]\d)\.\s*(?:[가-힣]|윗글|다음|밑줄|주어진))"
    r"|\[\d{1,2}\s*[~～-]\s*\d{1,2}\]"
    r"|(?:\d{1,2}\s+)?영어\s*영역).*$"
)


def _extract_footnotes_local(text: str) -> list[str]:
    raw = _normalize_pdf_hyphens((text or "").strip())
    if not raw or "*" not in raw:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for m in _ENG_FOOTNOTE_RE.finditer(raw):
        term = re.sub(r"\s+", " ", m.group(1)).strip()
        meaning = re.sub(r"\s+", " ", m.group(2)).strip()
        meaning = _FOOTNOTE_TAIL_CUT_RE.sub("", meaning).strip()
        m_eng_tail = re.search(
            r"([가-힣][가-힣0-9\s,·()'’\"~-]*?[가-힣\)])\s+[A-Z][a-z]+(?:\s+[A-Za-z0-9,'’\"-]+){3,}",
            meaning,
        )
        if m_eng_tail:
            meaning = m_eng_tail.group(1).strip()
        meaning = re.sub(r"(?:\s+\d{1,2}){1,2}\s*$", "", meaning).strip()
        meaning = re.sub(r"\s*영어\s*영역.*$", "", meaning).strip()
        if not term or not meaning or not re.search(r"[가-힣]", meaning):
            continue
        k = term.lower()
        if k not in seen:
            seen.add(k)
            out.append(f"{term}: {meaning}")
    return out


def _strip_problem_header_and_footnotes(question_text: str) -> str:
    s = re.sub(r"\s+", " ", _normalize_pdf_hyphens((question_text or "").strip()))
    if not s:
        return ""
    if "*" in s:
        s = _ENG_FOOTNOTE_RE.sub(" ", s)
        s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^(?:\d{1,2}\s+영어\s*영역\s*|영어\s*영역\s+\d{1,2}\s*)", "", s).strip()
    s = re.sub(
        r"^\[\d{1,2}\s*[~～-]\s*\d{1,2}\][^A-Za-z]*?[가-힣][^?.]*?(?:\?|시오\.)(?:\s*\[[1-4]점\])?\s*",
        "",
        s,
        count=1,
    ).strip()
    s = re.sub(r"^(?:(?:[1-9]|1[0-2])\s+)?(?:1[89]|[234]\d)\.\s*", "", s, count=1).strip()
    s = re.sub(r"^[1-9](?:1[89]|[234]\d)\.\s*", "", s, count=1).strip()
    for _ in range(3):
        next_s = re.sub(
            r"^(?:[^A-Za-z]*?[가-힣][^?.]*?(?:\?|시오\.|한다\.)(?:\s*\[[1-4]점\])?\s*)",
            "",
            s,
            count=1,
        ).strip()
        if next_s == s:
            break
        s = next_s
    s = re.sub(r"^(?:(?:[1-9]|1[0-2])\s+)?(?:1[89]|[234]\d)\.\s*", "", s, count=1).strip()
    s = re.sub(r"^\[[1-4]점\]\s*", "", s).strip()
    s = re.sub(r"(?<=[.?!\"'”’\)])\s+(?:(?:[1-9]|1[0-2])\s+)?(?:1[89]|[234]\d)\.(?=\s|$)", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _split_circled_five(text: str) -> tuple[str, dict[int, str]]:
    positions: list[tuple[int, int]] = []
    search_from = 0
    for idx_m, sym in enumerate(_CIRCLED_NUMS, start=1):
        pos = text.find(sym, search_from)
        if pos == -1:
            break
        positions.append((idx_m, pos))
        search_from = pos + len(sym)
    if not positions:
        return text, {}
    prefix = text[: positions[0][1]].strip()
    segs: dict[int, str] = {}
    for k, (idx_m, pos) in enumerate(positions):
        start = pos + 1
        end = positions[k + 1][1] if k + 1 < len(positions) else len(text)
        segs[idx_m] = text[start:end].strip()
    return prefix, segs


def _clean_passage_surface(passage: str) -> str:
    txt = _normalize_pdf_hyphens((passage or "").strip())
    if not txt:
        return ""
    txt = re.sub(r"</?u\b[^>]*>", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"[ \t]*\(\s*[①②③④⑤]\s*\)[ \t]*", " ", txt)
    txt = re.sub(r"[ \t]*[①②③④⑤][ \t]*", " ", txt)
    txt = re.sub(r"\(\s*[a-e]\s*\)\s*", " ", txt)
    txt = re.sub(r"(?:^|(?<=\n)|(?<=[.?!\"'”’]\s))\(\s*[ABCD]\s*\)\s*", " ", txt)
    txt = re.sub(r"^(?:(?:[1-9]|1[0-2])\s+)?(?:1[89]|[234]\d)\.\s+(?=[A-Z\"'“‘(\[])", "", txt)
    txt = re.sub(r"(?<=[.?!\"'”’\)])\s+(?:(?:[1-9]|1[0-2])\s+)?(?:1[89]|[234]\d)\.(?=\s|$)", "", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    txt = re.sub(r"[ \t]{2,}", " ", txt)
    return txt.strip()


def _parse_answer_int(answer: str) -> int:
    s = (answer or "").strip()
    if s in _CIRCLED_TO_INT:
        return _CIRCLED_TO_INT[s]
    m = re.search(r"([1-5])", s)
    return int(m.group(1)) if m else 0


def _reconstruct_english_from_raw_qtext(
    raw_q_text: str,
    *,
    q_num: int = 0,
    passage_group_id: str = "",
    orig_question_type: str = "",
    answer: str = "",
) -> str:
    """문제지 원문(question_text / PDF)에서 발문·선지·기호를 제거하고 순수 영어 지문을 결정론적으로 복원한다."""
    qt = (orig_question_type or "").strip()
    gid = (passage_group_id or "").strip()
    ans_int = _parse_answer_int(answer)

    # 41~42 / 43~45 장문 세트 (41./43. 경계를 헤더 정규화 전에 먼저 절단)
    if gid == "41~42" or q_num in (41, 42):
        pre_cut = re.split(r"(?:^|\n)\s*41\.\s*", raw_q_text, maxsplit=1)[0]
        body = _strip_problem_header_and_footnotes(pre_cut)
        return _clean_passage_surface(body)

    if gid == "43~45" or q_num in (43, 44, 45):
        pre_cut = re.split(r"(?:^|\n)\s*43\.\s*", raw_q_text, maxsplit=1)[0]
        body = _strip_problem_header_and_footnotes(pre_cut)
        # Q43 정답을 아는 경우 (A)-(B)-(C)-(D) 재배열
        if q_num == 43 and ans_int:
            parts = re.split(r"(?:^|\s)\(([ABCD])\)\s+", body)
            blocks: dict[str, str] = {}
            for i_p in range(1, len(parts) - 1, 2):
                lbl = parts[i_p]
                txt = parts[i_p + 1].strip()
                if lbl in ("A", "B", "C", "D") and txt:
                    blocks[lbl] = txt
            if set(blocks.keys()) == {"A", "B", "C", "D"}:
                m_c = re.search(
                    rf"{_INT_TO_CIRCLED[ans_int]}\s*\(([BCD])\)\s*[-–—]\s*\(([BCD])\)\s*[-–—]\s*\(([BCD])\)",
                    raw_q_text,
                )
                if m_c:
                    b1, b2, b3 = m_c.group(1), m_c.group(2), m_c.group(3)
                    return _clean_passage_surface(f"{blocks['A']} {blocks[b1]} {blocks[b2]} {blocks[b3]}")
                seq = _DEFAULT_LONG_ORDER_MAP.get(str(ans_int))
                if seq:
                    return _clean_passage_surface(f"{blocks['A']} {blocks[seq[0]]} {blocks[seq[1]]} {blocks[seq[2]]}")
        return _clean_passage_surface(body)

    body = _strip_problem_header_and_footnotes(raw_q_text)
    if not body:
        return ""

    # 40번 요약문: [요약문] / ￫ / → / ↓ 이전까지만 본문
    if q_num == 40 or "요약" in qt:
        head = re.split(r"\s*(?:\[요약문\]|￫|→|↓|➜)\s*", body, maxsplit=1)[0].strip()
        head, _ = _split_circled_five(head)
        return _clean_passage_surface(head)

    # 31~34번 빈칸 추론
    if q_num in (31, 32, 33, 34) or "빈칸" in qt or re.search(r"_{3,}", body):
        raw_body, segs = _split_circled_five(body)
        if ans_int and ans_int in segs and segs[ans_int]:
            ans_phrase = re.sub(r"\s*\[[1-4]점\]\s*$", "", segs[ans_int]).strip()
            if re.search(r"_{2,}", raw_body):
                filled = re.sub(r"\s*_{2,}\s*", f" {ans_phrase} ", raw_body, count=1)
                filled = re.sub(r"\s+([.,?!;:])", r"\1", filled)
                return _clean_passage_surface(filled)
        return _clean_passage_surface(raw_body)

    # 35번 무관한 문장
    if q_num == 35 or "무관" in qt or ("흐름" in qt and "관계" in qt):
        intro, segs = _split_circled_five(body)
        if len(segs) == 5 and ans_int in segs:
            seg5_sents = split_english_sentences_clean(segs[5])
            tail_after_5 = ""
            if ans_int == 5 and len(seg5_sents) >= 2:
                tail_after_5 = " ".join(seg5_sents[1:]).strip()
            kept = [intro] if intro else []
            for k in range(1, 6):
                if k == ans_int:
                    continue
                if segs.get(k):
                    kept.append(segs[k])
            if tail_after_5:
                kept.append(tail_after_5)
            return _clean_passage_surface(" ".join(kept))
        return ""

    # 36~37번 글의 순서
    if q_num in (36, 37) or "순서" in qt:
        order_body, segs = _split_circled_five(body)
        seq: tuple[str, str, str] | None = None
        if ans_int and ans_int in segs:
            m_seq = re.search(
                r"\(?\s*([ABC])\s*\)?\s*(?:-|→|->|–|—)\s*\(?\s*([ABC])\s*\)?\s*(?:-|→|->|–|—)\s*\(?\s*([ABC])\s*\)?",
                segs[ans_int],
            )
            if m_seq:
                seq = (m_seq.group(1), m_seq.group(2), m_seq.group(3))
        if seq is None and ans_int:
            seq = _DEFAULT_ORDER_MAP.get(str(ans_int))
        if seq:
            parts = re.split(r"(?:^|\s)\(([ABC])\)\s+", order_body)
            if len(parts) >= 7:
                given_txt = parts[0].strip()
                blocks: dict[str, str] = {}
                for i_p in range(1, len(parts) - 1, 2):
                    lbl, txt = parts[i_p], parts[i_p + 1].strip()
                    if lbl in ("A", "B", "C") and txt:
                        blocks[lbl] = txt
                if given_txt and set(blocks.keys()) == {"A", "B", "C"} and set(seq) == {"A", "B", "C"}:
                    return _clean_passage_surface(f"{given_txt} {blocks[seq[0]]} {blocks[seq[1]]} {blocks[seq[2]]}")
        return ""

    # 38~39번 문장 삽입
    if q_num in (38, 39) or "삽입" in qt:
        norm_ins = re.sub(r"\(\s*([①②③④⑤])\s*\)", r"\1", body)
        prefix_raw, segs = _split_circled_five(norm_ins)
        if len(segs) == 5 and ans_int in segs:
            prefix_clean = re.sub(r"\[주어진\s*문장\]", "", prefix_raw).strip()
            pref_sents = split_english_sentences_clean(prefix_clean)
            if pref_sents:
                given_sent = pref_sents[0]
                body_intro = " ".join(pref_sents[1:]).strip()
                rebuilt_parts = [body_intro] if body_intro else []
                for k in range(1, 6):
                    if k == ans_int:
                        rebuilt_parts.append(given_sent)
                    if segs.get(k):
                        rebuilt_parts.append(segs[k])
                return _clean_passage_surface(" ".join(rebuilt_parts))
        return ""

    # 25(도표), 29(어법), 30(어휘) — 본문 내부 인라인 ①~⑤ 제거 (① 앞에서 자르지 않음)
    if q_num in (25, 29, 30) or any(k in qt for k in ("어법", "어휘", "낱말", "도표")):
        if re.search(r"\(\s*[A-C]\s*\)\s*(?:\[[^\]]+/[^\]]+\]|[A-Za-z-]+/[A-Za-z-]+)", body):
            # 선택형 [a / b] 유형은 DB에 정답 반영 완료된 passage_text 우선
            return ""
        cleaned_body = re.sub(r"\s*①\s*②\s*③\s*④\s*⑤\s*$", "", body).strip()
        return _clean_passage_surface(cleaned_body)

    # 일반 독해 (18~24, 26~28 등) — 하단 선지 ①~⑤ 앞까지
    prefix, segs = _split_circled_five(body)
    if len(segs) >= 3:
        return _clean_passage_surface(prefix)
    return _clean_passage_surface(body)


def _is_valid_english_candidate(cand: str, reference_passage: str, *, allow_shorter: bool = False) -> bool:
    c = (cand or "").strip()
    if len(c) < 60:
        return False
    if re.search(r"[가-힣]", c):
        return False
    if re.search(r"[①②③④⑤]|_{3,}", c):
        return False
    c_words = re.findall(r"[A-Za-z]{2,}", c)
    if len(c_words) < 15:
        return False
    ref = (reference_passage or "").strip()
    if not ref:
        return True
    if not allow_shorter and len(c) < int(len(ref) * 0.72):
        return False
    ref_set = {w.lower() for w in re.findall(r"[A-Za-z]{3,}", ref)}
    c_set = {w.lower() for w in re.findall(r"[A-Za-z]{3,}", c)}
    if not ref_set or not c_set:
        return False
    overlap = len(ref_set & c_set) / max(1, min(len(ref_set), len(c_set)))
    return overlap >= 0.65


def extract_english_passage_from_problem_source(
    *,
    passage_text: str = "",
    question_text: str = "",
    source_pdf_bytes: bytes | None = None,
    q_num: int = 0,
    passage_group_id: str = "",
    orig_question_type: str = "",
    answer: str = "",
) -> tuple[str, list[str]]:
    """
    DB에 저장된 영어 지문 원문(`passage_text`)을 최우선으로 사용하고,
    없거나 불완전할 때만 문제지 PDF(`source_pdf_bytes`) / `question_text`에서 보완 추출한다.
    공식 각주(`* word: 뜻`)는 문제지 PDF / `question_text`에서 함께 추출한다. (AI 미사용)
    """
    raw_q_src = (question_text or "").strip()
    if source_pdf_bytes and q_num > 0:
        try:
            pdf_q_text = extract_problem_question_text_from_pdf(
                source_pdf_bytes, q_num=q_num, passage_group_id=passage_group_id
            )
            if pdf_q_text:
                raw_q_src = pdf_q_text
        except Exception:
            pass

    footnotes = _extract_footnotes_local(raw_q_src)
    if not footnotes and question_text and raw_q_src != question_text.strip():
        footnotes = _extract_footnotes_local(question_text)

    db_passage = _clean_passage_surface(passage_text)
    if _is_valid_english_candidate(db_passage, ""):
        return db_passage, footnotes

    is_irrel = q_num == 35 or "무관" in (orig_question_type or "")
    if raw_q_src:
        cand = _reconstruct_english_from_raw_qtext(
            raw_q_src,
            q_num=q_num,
            passage_group_id=passage_group_id,
            orig_question_type=orig_question_type,
            answer=answer,
        )
        if _is_valid_english_candidate(cand, db_passage, allow_shorter=is_irrel):
            return cand, footnotes

    return db_passage, footnotes


# ---------------------------------------------------------------------------
# 해설지 PDF (`answer_pdf`) `[해석]` 섹션 결정론적 추출 (AI 0%)
# ---------------------------------------------------------------------------
_QUICK_ANS_PAIR_RE = re.compile(r"\b(?:[1-9]|[1-4]\d)\s*[①②③④⑤]")
_ANS_SUBJECT_ENG_RE = re.compile(r"(?:^|\s)(?:영어\s*영역|제\s*3\s*교시\s*영어)(?:\s|$)")
_ANS_SUBJECT_OTHER_RE = re.compile(
    r"(?:국어\s*영역|수학\s*영역|한국사\s*영역|사회\s*탐구\s*영역|과학\s*탐구\s*영역|"
    r"제\s*1\s*교시|제\s*2\s*교시|제\s*4\s*교시)"
)

_ANS_RANGE_HDR_RE = re.compile(
    r"^(?:[•▶▷◈■◆●★]?\s*)?(?:\[\s*(\d{1,2})\s*[~\-～–—]\s*(\d{1,2})\s*\]|(\d{1,2})\s*[~\-～–—]\s*(\d{1,2})\s*[.)번]?)(?:\s+(.*)|\s*$)"
)
_ANS_Q_HDR_RE = re.compile(
    r"^(?:[•▶▷◈■◆●★]?\s*)?(?:\[\s*(\d{1,2})\s*\]|(\d{1,2})\s*[.)번])(?:\s+(.*)|\s*$)"
)
_INTERP_START_RE = re.compile(
    r"^(?:[\[\<〈【\(]\s*(?:전\s*문\s*|지\s*문\s*|본\s*문\s*|우\s*리\s*말\s*)?해\s*석\s*[\]\>〉】\)]"
    r"|[■▶●◆•◈▷]\s*(?:전\s*문\s*|지\s*문\s*|본\s*문\s*)?해\s*석\b"
    r"|(?:전\s*문\s*|지\s*문\s*|본\s*문\s*)?해\s*석\s*[:：]?\s*$)"
    r"(?:\s*(.*))?$"
)
_INTERP_INLINE_START_RE = re.compile(
    r"^[\[\<〈【]\s*(?:전\s*문\s*|지\s*문\s*|본\s*문\s*|우\s*리\s*말\s*)?해\s*석\s*[\]\>〉】]\s*(.+)$"
)
_INTERP_END_RE = re.compile(
    r"^(?:[\[\<〈【\(]\s*(?:풀\s*이|해\s*설|정\s*답(?:\s*해\s*설)?|오\s*답(?:\s*피\s*하\s*기|\s*해\s*설)?|"
    r"어\s*휘|어\s*구|단\s*어|구\s*문(?:\s*풀\s*이)?|선\s*택\s*지\s*해\s*석|선\s*지\s*해\s*석|"
    r"출\s*제\s*의\s*도|소\s*재|Words\s*(?:&|and)?\s*Phrases)\s*[\]\>〉】\)]"
    r"|[■▶●◆•◈▷]\s*(?:풀\s*이|해\s*설|정\s*답|어\s*휘|어\s*구|단\s*어|구\s*문|선\s*지\s*해\s*석|출\s*제\s*의\s*도|소\s*재)\b"
    r"|(?:풀\s*이|해\s*설|정\s*답\s*해\s*설|오\s*답\s*(?:피\s*하\s*기|해\s*설)|어\s*휘|어\s*구|단\s*어|구\s*문\s*풀\s*이|선\s*택\s*지\s*해\s*석|선\s*지\s*해\s*석|출\s*제\s*의\s*도)\s*[:：]?\s*$"
    r"|(?:Words\s*(?:&|and)?\s*Phrases|Vocabulary)\s*[:：]?$)",
    re.IGNORECASE,
)
_VOCAB_LINE_RE = re.compile(r"^(?:[*•▪◦\-]\s*)?[A-Za-z][A-Za-z0-9'\-\s]{0,32}\s*[:：]\s*[가-힣]")
_INLINE_CIRCLED_TYPES_KW = ("무관", "흐름", "삽입", "어법", "어휘", "도표")


def _join_korean_wrapped_line(prev_line: str, next_line: str) -> str:
    if not prev_line:
        return next_line
    if not next_line:
        return prev_line
    if prev_line.endswith((" ", "\t")):
        return prev_line + next_line
    # (A), (B), (C), (D) 또는 ①~⑤ 새 항목 시작이면 공백 유지
    if re.match(r"^(?:\([A-D]\)|[①②③④⑤])\s*", next_line):
        return f"{prev_line} {next_line}"
    if re.match(
        r"^(?:[.,?!:;·\)\]\”\’]|(?:이다|입니다|있다|없다|한다|된다|하며|하여|하고|하면|해서|하는|되는|된|할|될|은|는|이|가|을|를|의|에|에서|에게|로|으로|와|과|도|만|까지|부터|처럼|보다|라고|이라고|라는|이라는|라|이라|며|으며|고|서|니|면|지만|으나|나|거나)\b)",
        next_line,
    ):
        return prev_line + next_line
    last_tok = prev_line.split()[-1] if prev_line.split() else ""
    first_tok = next_line.split()[0] if next_line.split() else ""
    if (
        1 <= len(last_tok) <= 2
        and 1 <= len(first_tok) <= 2
        and re.fullmatch(r"[가-힣]+", last_tok)
        and re.fullmatch(r"[가-힣]+", first_tok)
        and len(prev_line) >= 16
    ):
        return prev_line + next_line
    return f"{prev_line} {next_line}"


def _join_korean_lines(lines: list[str]) -> str:
    out = ""
    for ln in lines:
        s = re.sub(r"\s+", " ", (ln or "").strip())
        if not s:
            continue
        out = _join_korean_wrapped_line(out, s)
    return re.sub(r"[ \t]{2,}", " ", out).strip()


def _is_quick_answer_table_line(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if len(_QUICK_ANS_PAIR_RE.findall(s)) >= 2:
        return True
    # 순수 번호 또는 원문자만 나열된 행 (예: '18 19 20 21' 또는 '① ③ ④ ②')
    if re.fullmatch(r"(?:[①②③④⑤]\s*){2,}", s):
        return True
    if re.fullmatch(r"(?:\d{1,2}\s+){2,}\d{1,2}", s):
        return True
    return False


def _extract_interp_from_block(block_lines: list[str], q_key: str) -> tuple[str, dict[str, str]]:
    """단일 문항/세트 블록 라인 목록에서 [해석] 본문과 선택적 하단 선지 해석을 추출한다."""
    interp_lines: list[str] = []
    choice_lines: list[str] = []
    in_interp = False
    in_choices = False
    has_explicit_interp = False
    is_group_or_inline = q_key in ("41~42", "43~45") or q_key in ("25", "29", "30", "35", "38", "39")

    for raw_ln in block_lines:
        ln = raw_ln.strip()
        if not ln:
            continue

        # 명시적 [해석] 시작 감지
        m_inl = _INTERP_INLINE_START_RE.match(ln)
        if m_inl:
            has_explicit_interp = True
            in_interp = True
            in_choices = False
            rest = m_inl.group(1).strip()
            if rest:
                interp_lines.append(rest)
            continue
        m_st = _INTERP_START_RE.match(ln)
        if m_st:
            has_explicit_interp = True
            in_interp = True
            in_choices = False
            rest = (m_st.group(1) or "").strip()
            if rest:
                interp_lines.append(rest)
            continue

        # 섹션 종료 마커
        if _INTERP_END_RE.match(ln):
            if in_interp:
                in_interp = False
            in_choices = False
            continue

        if not in_interp:
            continue

        # 세트(41~42, 43~45) 블록 내부에서 개별 문항 '41.', '42.', '43.'이 나오면 해석 종료
        if q_key in ("41~42", "43~45") and re.match(r"^(?:4[1-5])\s*[.)]", ln):
            in_interp = False
            continue

        # 비-세트 문항 블록 내부에 41~42 / 43~45 범위 헤더가 섞여 들어온 경우 즉시 해석 종료
        if q_key not in ("41~42", "43~45") and re.match(
            r"^(?:[•▶▷◈■◆●★]?\s*)?(?:\[\s*4[13]\s*[~\-～–—]\s*4[25]\s*\]|4[13]\s*[~\-～–—]\s*4[25]\s*[.)번]?)(?:\s|$)",
            ln,
        ):
            in_interp = False
            continue

        # 각주/단어장 라인 도달 시 해석 종료
        if _VOCAB_LINE_RE.match(ln):
            in_interp = False
            continue

        # 비-인라인 문항에서 하단 선지(① ... ② ...)가 시작되면 별도 choices로 분리
        if not is_group_or_inline and re.match(r"^①\s+\S", ln) and not in_choices:
            # 단, 문장 중간에 우연히 ①이 아닌 확실한 선지 나열일 때만 분리
            in_choices = True
            choice_lines.append(ln)
            continue
        if in_choices:
            choice_lines.append(ln)
            continue

        interp_lines.append(ln)

    # 폴백: 명시적 [해석] 헤더가 전혀 없던 해설지 형식일 때
    if not has_explicit_interp:
        fallback_lines: list[str] = []
        skip_mode = False
        for raw_ln in block_lines:
            ln = raw_ln.strip()
            if not ln:
                continue
            if re.match(r"^(?:\d{1,2}\s*[.)]|\[\d{1,2}\s*[~\-～]\s*\d{1,2}\])", ln):
                # 문항 헤더 첫 줄에서 번호/정답 제거 후 남은 한글이 길면 포함
                stripped_hdr = re.sub(r"^(?:\d{1,2}\s*[.)]\s*|\[\d{1,2}\]\s*)(?:[①②③④⑤]\s*)?", "", ln).strip()
                if len(re.findall(r"[가-힣]", stripped_hdr)) >= 12 and not _INTERP_END_RE.match(stripped_hdr):
                    fallback_lines.append(stripped_hdr)
                continue
            if _INTERP_END_RE.match(ln):
                if re.search(r"출\s*제\s*의\s*도|소\s*재", ln):
                    skip_mode = False
                    continue
                skip_mode = True
                continue
            if _VOCAB_LINE_RE.match(ln):
                skip_mode = True
                continue
            if not skip_mode and len(re.findall(r"[가-힣]", ln)) >= 4:
                fallback_lines.append(ln)
        joined_fb = _join_korean_lines(fallback_lines)
        if len(re.findall(r"[가-힣]", joined_fb)) >= 25:
            return joined_fb, {}
        return "", {}

    choices_map: dict[str, str] = {}
    if choice_lines:
        c_blob = _join_korean_lines(choice_lines)
        _, segs = _split_circled_five(c_blob)
        for k, v in segs.items():
            if v:
                choices_map[str(k)] = v

    return _join_korean_lines(interp_lines), choices_map


def extract_all_english_translations_from_answer_pdf(answer_pdf_bytes: bytes) -> dict[str, dict[str, Any]]:
    """
    해설지 PDF에서 영어 문항별(`[18]`..`[45]`, `41~42`, `43~45`) `[해석]` 섹션을 추출한다.
    통합 해설지(전 과목 합본)와 빠른 정답표를 자동으로 걸러낸다.
    반환 형태: `{ '20': {'text': ..., 'choices': {...}}, '41~42': {'text': ..., 'choices': {}}, ... }`
    """
    if not answer_pdf_bytes:
        return {}
    cached = _cache_get("ans", answer_pdf_bytes)
    if cached is not None:
        return cached

    page_records: list[tuple[str, list[dict[str, Any]]]] = []
    has_other_subjects = False

    with pymupdf.open(stream=answer_pdf_bytes, filetype="pdf") as doc:
        for page_idx in range(doc.page_count):
            page = doc[page_idx]
            page_h = float(page.rect.height)
            page_w = float(page.rect.width)
            raw_page_txt = page.get_text("text") or ""
            if _ANS_SUBJECT_OTHER_RE.search(raw_page_txt):
                has_other_subjects = True

            page_dict = page.get_text("dict")
            cols = _detect_pdf_columns(page)
            col_lines_all: list[dict[str, Any]] = []
            for col_idx, (cx0, cx1) in enumerate(cols):
                col_lines_all.extend(
                    _extract_column_logical_lines(
                        page_dict,
                        cx0,
                        cx1,
                        page_h,
                        page_w,
                        drawings=None,
                        is_last_col=(col_idx == len(cols) - 1),
                        detect_blank_underlines=False,
                    )
                )
            page_records.append((raw_page_txt, col_lines_all))

    # 통합 해설지인 경우 영어 영역 페이지만 선별
    selected_lines: list[dict[str, Any]] = []
    if has_other_subjects:
        in_eng_section = False
        for raw_txt, lines in page_records:
            head_txt = raw_txt[:400]
            has_eng_hdr = bool(_ANS_SUBJECT_ENG_RE.search(head_txt) or _ANS_SUBJECT_ENG_RE.search(raw_txt))
            has_other_hdr = bool(_ANS_SUBJECT_OTHER_RE.search(head_txt))
            has_interp = bool(re.search(r"[\[\<]\s*(?:전\s*문\s*)?해\s*석\s*[\]\>]", raw_txt))
            if has_eng_hdr:
                in_eng_section = True
            elif has_other_hdr and not has_eng_hdr:
                in_eng_section = False
            if in_eng_section or has_eng_hdr or has_interp:
                selected_lines.extend(lines)
    else:
        for _, lines in page_records:
            selected_lines.extend(lines)

    # 문항 블록 분할 (빠른 정답표 제외 & 중복 시 [해석] 본문 우선)
    blocks_by_key: dict[str, list[list[str]]] = {}
    cur_key = ""
    cur_buf: list[str] = []
    last_q_num = 0

    def _flush_block() -> None:
        nonlocal cur_key, cur_buf
        if cur_key and cur_buf:
            blocks_by_key.setdefault(cur_key, []).append(list(cur_buf))
        cur_buf = []

    for ln in selected_lines:
        txt = ln["text"]
        if _is_quick_answer_table_line(txt):
            continue

        rm = _ANS_RANGE_HDR_RE.match(txt)
        if rm:
            s_str = rm.group(1) or rm.group(3)
            e_str = rm.group(2) or rm.group(4)
            if s_str and e_str:
                qs, qe = int(s_str), int(e_str)
                if (qs, qe) in ((41, 42), (43, 45)):
                    _flush_block()
                    cur_key = f"{qs}~{qe}"
                    last_q_num = qs
                    rest = (rm.group(5) or "").strip()
                    if rest:
                        cur_buf.append(rest)
                    continue
                if ln["x0"] <= ln["col_x0"] + 0.50 * max(1.0, ln["col_w"]):
                    # [31~34], [36~37], [38~39] 공통 발문 배너는 무시
                    if 18 <= qs <= qe <= 45:
                        continue

        qm = _ANS_Q_HDR_RE.match(txt)
        if qm and ln["x0"] <= ln["col_x0"] + 0.45 * max(1.0, ln["col_w"]):
            qn = int(qm.group(1) or qm.group(2))
            if 18 <= qn <= 45:
                # 41~42 / 43~45 블록 내부에서 첫 번째 하위 문항 전이면 해석이 아직 안 끝났을 수 있으나,
                # 보통 [41~42] [해석] -> 41. [풀이] -> 42. [풀이] 순서이므로 41/42/43은 새 문항 시작 허용
                if qn >= last_q_num or (last_q_num >= 40 and qn <= 45) or cur_key in ("41~42", "43~45"):
                    _flush_block()
                    cur_key = str(qn)
                    last_q_num = qn
                    rest = (qm.group(3) or "").strip()
                    if rest:
                        cur_buf.append(rest)
                    continue

        if cur_key:
            cur_buf.append(txt)

    _flush_block()

    out: dict[str, dict[str, Any]] = {}
    for q_key, candidate_blocks in blocks_by_key.items():
        best_text = ""
        best_choices: dict[str, str] = {}
        best_score = -1
        for blk in candidate_blocks:
            txt, ch_map = _extract_interp_from_block(blk, q_key)
            kor_cnt = len(re.findall(r"[가-힣]", txt))
            if kor_cnt > best_score:
                best_score = kor_cnt
                best_text = txt
                best_choices = ch_map
        if best_text:
            out[q_key] = {"text": best_text, "choices": best_choices}

    _cache_set("ans", answer_pdf_bytes, out)
    return out


def _normalize_extracted_translation_by_type(
    raw_kor: str,
    *,
    q_num: int = 0,
    orig_question_type: str = "",
    answer: str = "",
    choices_map: dict[str, str] | None = None,
    order_choices: dict[str, str] | None = None,
) -> str:
    """해설지에서 추출한 한국어 [해석]을 문항 유형에 맞게 결정론적으로 정리한다 (AI 미사용)."""
    txt = re.sub(r"\s+", " ", (raw_kor or "").strip())
    if not txt:
        return ""

    qt = (orig_question_type or "").strip()
    ans_int = _parse_answer_int(answer)

    # 40번 요약문: 요약문 해석('→', '->', '[요약문]', '요약문:', '(A)...(B)' 등) 이하 일체 제거
    if q_num == 40 or "요약" in qt:
        txt = re.split(
            r"\s*(?:"
            r"[\[\<〈【]\s*요\s*약(?:\s*문)?(?:\s*해\s*석)?\s*[\]\>〉】]"
            r"|[■▶●◆•◈▷▼▽]\s*요\s*약(?:\s*문)?(?:\s*해\s*석)?"
            r"|(?:요약문(?:\s*해석)?|요약)\s*[:：]"
            r"|￫|→|↓|➜|⇒|⇨|➡|➔|➝|➞|->"
            r"|[\ue000-\uf8ff]"
            r")\s*",
            txt,
            maxsplit=1,
        )[0].strip()
        m_ab_sent = re.search(r"(?:^|(?<=[.?!])\s+)(?=[^.?!]*\(A\)[^.?!]*\(B\))", txt)
        if m_ab_sent:
            txt = txt[: m_ab_sent.start()].strip()

    # 31~34번 빈칸: 해석에 ____가 남아 있고 선지 한국어 해석이 있으면 채움
    if (q_num in (31, 32, 33, 34) or "빈칸" in qt) and re.search(r"_{2,}", txt):
        if choices_map and ans_int and str(ans_int) in choices_map:
            ans_kor = choices_map[str(ans_int)].strip()
            txt = re.sub(r"\s*_{2,}\s*", f" {ans_kor} ", txt, count=1)

    # 35번 무관한 문장: 괄호 쳐진 무관한 문장 해석((③ ...), (완결 해석 문장.)) 제거 및 정답 번호 문장 제외
    if q_num == 35 or "무관" in qt or ("흐름" in qt and "관계" in qt):
        # 1) 괄호 안에 ①~⑤가 포함된 해석 또는 괄호 쳐진 완결 한국어 해석 문장 제거
        txt = re.sub(r"\(\s*[^()]*?[①②③④⑤][^()]*?\s*\)", " ", txt)
        txt = re.sub(r"\(\s*[^()]*?(?:[.?!]|다\.|요\.|[가-힣]{8,})[^()]*?\s*\)", " ", txt)
        txt = re.sub(r"[ \t]{2,}", " ", txt).strip()
        intro, segs = _split_circled_five(txt)
        if len(segs) == 5 and ans_int in segs:
            kept = [intro] if intro else []
            for k in range(1, 6):
                if k == ans_int:
                    continue
                if segs.get(k):
                    kept.append(segs[k])
            txt = " ".join(kept)

    # 36~37번 글의 순서 / 43~45번 장문 순서
    if q_num in (36, 37, 43, 44, 45) or "순서" in qt:
        label_matches = list(re.finditer(r"(?:^|\s)\(([ABCD])\)\s+", txt))
        labels_found = [m.group(1) for m in label_matches]
        if labels_found == ["A", "B", "C"] and ans_int:
            # 알파벳 정순 (A)(B)(C)로 인쇄된 경우에만 정답 순서로 재배열
            seq: tuple[str, str, str] | None = None
            if order_choices and str(ans_int) in order_choices:
                m_sq = re.search(r"\(([ABC])\).*\(([ABC])\).*\(([ABC])\)", order_choices[str(ans_int)])
                if m_sq:
                    seq = (m_sq.group(1), m_sq.group(2), m_sq.group(3))
            if seq is None:
                seq = _DEFAULT_ORDER_MAP.get(str(ans_int))
            if seq:
                parts = re.split(r"(?:^|\s)\(([ABC])\)\s+", txt)
                if len(parts) >= 7:
                    given_k = parts[0].strip()
                    blocks = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
                    if given_k and set(blocks.keys()) == {"A", "B", "C"} and set(seq) == {"A", "B", "C"}:
                        txt = f"{given_k} {blocks[seq[0]]} {blocks[seq[1]]} {blocks[seq[2]]}"
        elif labels_found == ["A", "B", "C", "D"] and q_num == 43 and ans_int:
            l_seq = _DEFAULT_LONG_ORDER_MAP.get(str(ans_int))
            if l_seq:
                parts = re.split(r"(?:^|\s)\(([ABCD])\)\s+", txt)
                if len(parts) >= 9:
                    blocks = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
                    if set(blocks.keys()) == {"A", "B", "C", "D"}:
                        txt = f"{blocks['A']} {blocks[l_seq[0]]} {blocks[l_seq[1]]} {blocks[l_seq[2]]}"
        # 이미 정답 순서로 인쇄돼 있거나 재배열 완료 후 (A)~(D) 라벨 제거
        txt = re.sub(r"(?:^|(?<=\s))\(\s*[ABCD]\s*\)\s*", " ", txt)

    # 38~39번 문장 삽입
    if q_num in (38, 39) or "삽입" in qt:
        norm_ins = re.sub(r"\(\s*([①②③④⑤])\s*\)", r"\1", txt)
        pref, segs = _split_circled_five(norm_ins)
        if len(segs) == 5 and ans_int in segs and pref:
            pref_clean = re.sub(r"\[주어진\s*문장\]", "", pref).strip()
            pref_sents = split_korean_sentences_clean(pref_clean)
            if pref_sents:
                given_k = pref_sents[0]
                intro_k = " ".join(pref_sents[1:]).strip()
                parts_k = [intro_k] if intro_k else []
                for k in range(1, 6):
                    if k == ans_int:
                        parts_k.append(given_k)
                    if segs.get(k):
                        parts_k.append(segs[k])
                txt = " ".join(parts_k)
        txt = re.sub(r"\[주어진\s*문장\]", " ", txt)

    # 공통 잔여 마커 정리 (①~⑤, (a)~(e))
    txt = re.sub(r"[ \t]*\(\s*[①②③④⑤]\s*\)[ \t]*", " ", txt)
    txt = re.sub(r"[ \t]*[①②③④⑤][ \t]*", " ", txt)
    txt = re.sub(r"\(\s*[a-e]\s*\)\s*", " ", txt)
    txt = re.sub(r"\s+([.,?!;:])", r"\1", txt)
    return re.sub(r"[ \t]{2,}", " ", txt).strip()


def extract_english_question_translation_from_pdf(
    answer_pdf_bytes: bytes,
    q_num: int,
    passage_group_id: str = "",
    orig_question_type: str = "",
    answer: str = "",
    order_choices: dict[str, str] | None = None,
) -> str:
    """저장된 해설지 PDF에서 특정 문항(또는 41~42/43~45 세트)의 한국어 해석 전문을 추출·정리한다."""
    if not answer_pdf_bytes or q_num <= 0:
        return ""
    all_trans = extract_all_english_translations_from_answer_pdf(answer_pdf_bytes)
    if not all_trans:
        return ""

    gid = (passage_group_id or "").strip()
    entry: dict[str, Any] | None = None
    if gid in ("41~42", "43~45") and gid in all_trans:
        entry = all_trans[gid]
    elif str(q_num) in all_trans:
        entry = all_trans[str(q_num)]
    elif q_num in (41, 42) and "41~42" in all_trans:
        entry = all_trans["41~42"]
    elif q_num in (43, 44, 45) and "43~45" in all_trans:
        entry = all_trans["43~45"]

    if not entry:
        return ""
    return _normalize_extracted_translation_by_type(
        str(entry.get("text") or ""),
        q_num=q_num,
        orig_question_type=orig_question_type,
        answer=answer,
        choices_map=entry.get("choices") or {},
        order_choices=order_choices,
    )


# ---------------------------------------------------------------------------
# 영문 / 국문 문장 분할 및 DP 1:1 문장 정렬 (AI 0%)
# ---------------------------------------------------------------------------
_ENG_ABBREV_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Mt|vs|etc|e\.g|i\.e|et\s+al|U\.S|U\.K|"
    r"Jan|Feb|Mar|Apr|Aug|Sept?|Oct|Nov|Dec|a\.m|p\.m|Ph\.D|Inc|Corp|Ltd|Co|No|Fig|Vol|pp?|ca)\."
    r"|(?<=\s)[A-Z]\.(?=\s+[A-Z])",
    re.IGNORECASE,
)
_ENG_SPEECH_ATTRIB_RE = re.compile(
    r"^(?:he|she|they|we|I|you|it|[A-Z][a-z]+)\s+"
    r"(?:said|asked|replied|answered|shouted|whispered|murmured|added|explained|noted|thought|cried|exclaimed|continued|wrote)\b"
)


def split_english_sentences_clean(text: str) -> list[str]:
    """약어·소수점·따옴표/괄호 내부를 보호하면서 영어 지문을 문장 리스트로 분할한다."""
    cleaned_lines: list[str] = []
    for raw_ln in _normalize_pdf_hyphens(text or "").splitlines():
        ln = raw_ln.strip()
        if not ln:
            continue
        if re.search(r"[가-힣]", ln) and not re.search(r"[A-Za-z]{4,}", ln):
            continue
        if ln.startswith("*") and ":" in ln:
            continue
        if re.match(r"^(?:(?:[1-9]|1[0-2])\s+)?(?:1[89]|[234]\d)\.\s*$", ln):
            continue
        cleaned_lines.append(ln)

    joined = re.sub(r"\s+", " ", " ".join(cleaned_lines)).strip()
    if not joined:
        return []

    stashed: list[str] = []

    def _stash(m: re.Match[str]) -> str:
        idx = len(stashed)
        stashed.append(m.group(0))
        return f"@@ES_{idx}@@"

    masked = _ENG_ABBREV_RE.sub(_stash, joined)
    masked = re.sub(r"\b\d+\.\d+\b", _stash, masked)
    masked = re.sub(r"\.{3,}", _stash, masked)

    # 균형 잡힌 괄호 (...) 내부 마침표/물음표 보호 (최대 260자 이내 닫힘)
    masked = re.sub(r"\([^()]{1,260}\)", _stash, masked)

    # 균형 잡힌 큰따옴표("..." / “...”) 내부 마침표 보호 — 단, 문장 전체가 따옴표가 아닌 인용구 내 구두점만 마스킹
    def _stash_quote_punct(m: re.Match[str]) -> str:
        inner = m.group(0)
        if len(inner) > 300:
            return inner
        # 인용부호 끝이 종결부호(.?!)이고 바로 뒤에 전달절이 오지 않으면 문장 경계로 작동하도록 바깥에서 처리
        return _stash(m) if not re.search(r"[.?!][\"”]\s*$", inner) else inner

    masked = re.sub(r'["“][^"”]{1,300}["”]', _stash_quote_punct, masked)

    raw_sents = re.split(
        r"(?<=[.?!][\"'”’\)])\s+(?=[A-Z0-9\"'“‘\(])|(?<=[.?!])\s+(?=[A-Z0-9\"'“‘\(])",
        masked,
    )

    restored_list: list[str] = []
    for seg in raw_sents:
        r = seg
        # 다중 중첩 스태시 복원
        for _ in range(3):
            if "@@ES_" not in r:
                break
            for idx, orig in enumerate(stashed):
                r = r.replace(f"@@ES_{idx}@@", orig)
        r = r.strip()
        if r:
            restored_list.append(r)

    # 대화 전달절 병합: '"Wait!" John shouted.'가 2문장으로 쪼개졌으면 병합
    merged: list[str] = []
    for s in restored_list:
        if merged and merged[-1].endswith(('"', "”")) and _ENG_SPEECH_ATTRIB_RE.match(s):
            merged[-1] = f"{merged[-1]} {s}"
        else:
            merged.append(s)
    return merged


def split_korean_sentences_clean(text: str) -> list[str]:
    """소수점·영문 약어·인용부호·괄호 내부를 보호하면서 한국어 해석을 문장 단위로 분할한다."""
    s = re.sub(r"\s+", " ", (text or "").strip())
    if not s:
        return []

    stashed: list[str] = []

    def _stash(m: re.Match[str]) -> str:
        idx = len(stashed)
        stashed.append(m.group(0))
        return f"@@KS_{idx}@@"

    masked = _ENG_ABBREV_RE.sub(_stash, s)
    masked = re.sub(r"\b\d+\.\d+\b", _stash, masked)
    masked = re.sub(r"\.{3,}", _stash, masked)
    masked = re.sub(r"\([^()]{1,220}\)", _stash, masked)

    def _stash_kor_quote(m: re.Match[str]) -> str:
        q_str = m.group(0)
        after = m.string[m.end() :]
        # '"잠깐만!"이라고' 처럼 닫는 따옴표 바로 뒤에 공백 없이 한글 조사가 이어지면 통째로 보호
        if after and re.match(r"^[가-힣]", after):
            return _stash(m)
        # '그는 "지금 행동하라." 다음 날 ...' 처럼 인용부호 끝이 종결부호이고 뒤에 공백이 오면 인용 끝에서 분할 허용
        if re.search(r"[.?!][”\"」]\s*$", q_str):
            return q_str
        return _stash(m)

    masked = re.sub(r"[“\"「][^”\"」]{1,260}[”\"」]", _stash_kor_quote, masked)

    raw_sents = re.split(r"(?<=[.?!][\"'”’\)\]])\s+(?=\S)|(?<=[.?!])\s+(?=\S)", masked)
    out: list[str] = []
    for seg in raw_sents:
        r = seg
        for _ in range(3):
            if "@@KS_" not in r:
                break
            for idx, orig in enumerate(stashed):
                r = r.replace(f"@@KS_{idx}@@", orig)
        r = r.strip()
        if r:
            out.append(r)
    return out


def _anchor_bonus(eng_txt: str, kor_txt: str) -> float:
    """숫자, 물음표/느낌표, 영문 고유명사/기호 일치 시 DP 정렬 비용 감소 보너스를 계산한다."""
    bonus = 0.0
    e_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", eng_txt))
    k_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", kor_txt))
    if e_nums and k_nums:
        if e_nums & k_nums:
            bonus -= 0.45 * len(e_nums & k_nums)
        else:
            bonus += 0.25

    if eng_txt.rstrip().endswith("?") and "?" in kor_txt:
        bonus -= 0.35
    if eng_txt.rstrip().endswith("!") and "!" in kor_txt:
        bonus -= 0.25
    return bonus


def _dp_partition_many_to_few(
    many: list[str],
    few: list[str],
    *,
    is_many_kor: bool,
) -> list[str]:
    """
    연속된 `many` 문장들을 `few` 개수의 그룹으로 100% 무손실 분할한다 (Gale-Church 길이 비율 + 앵커 DP).
    """
    n = len(many)
    m = len(few)
    if m <= 0:
        return []
    if n <= m:
        return list(many)

    many_lens = [max(1, len(re.sub(r"\s+", "", x))) for x in many]
    few_lens = [max(1, len(re.sub(r"\s+", "", x))) for x in few]
    total_many = sum(many_lens)
    total_few = sum(few_lens)
    ratio = total_many / max(1, total_few)
    max_grp = max(3, math.ceil(n / m) + 1)

    inf = 1e18
    dp = [[inf] * (n + 1) for _ in range(m + 1)]
    parent = [[-1] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0.0

    for i in range(1, m + 1):
        rem_groups = m - i
        j_min = i
        j_max = n - rem_groups
        f_txt = few[i - 1]
        f_len = few_lens[i - 1]
        expected_len = max(4.0, f_len * ratio)

        for j in range(j_min, j_max + 1):
            k_lo = max(i - 1, j - max_grp)
            k_hi = j - 1
            for k in range(k_lo, k_hi + 1):
                if dp[i - 1][k] >= inf:
                    continue
                grp_len = sum(many_lens[k:j])
                len_cost = abs(grp_len - expected_len) / expected_len
                # 여러 문장 묶음일 때 세미콜론/콜론/대시 힌트가 있으면 패널티 완화
                count_pen = 0.08 * ((j - k) - 1)
                if (j - k) > 1 and any(ch in f_txt for ch in (";", ":", "—")):
                    count_pen *= 0.35
                grp_txt = " ".join(many[k:j])
                anch = _anchor_bonus(f_txt, grp_txt) if is_many_kor else _anchor_bonus(grp_txt, f_txt)
                cost = dp[i - 1][k] + len_cost + count_pen + anch
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    parent[i][j] = k

    if dp[m][n] >= inf:
        # 안전 폴백: 균등 분할
        groups: list[str] = []
        base, extra = divmod(n, m)
        cur = 0
        for idx_g in range(m):
            sz = base + (1 if idx_g < extra else 0)
            groups.append(" ".join(many[cur : cur + sz]))
            cur += sz
        return groups

    cuts: list[int] = [n]
    cur_j = n
    for i in range(m, 0, -1):
        cur_j = parent[i][cur_j]
        cuts.append(cur_j)
    cuts.reverse()
    return [" ".join(many[cuts[i] : cuts[i + 1]]) for i in range(m)]


def align_english_korean_sentences(
    eng_sents: list[str],
    kor_sents: list[str],
) -> list[tuple[str, str]]:
    """
    영어 원문 문장 리스트와 해설지 한국어 문장 리스트를 AI 없이 1:1 매칭한다.
    - 문장 수 동일: 즉시 1:1 zip
    - 국문 > 영문: DP로 연속 국문 문장을 영문 문장 수만큼 그룹화
    - 국문 < 영문 (충분히 분리된 경우): DP로 인접 영문 문장을 국문 수에 맞게 그룹화
    - 국문 미제공/단일 블록: 영문 문장 원형 보존 (오매칭 방지)
    """
    if not eng_sents:
        return []
    if not kor_sents:
        return [(e, "") for e in eng_sents]
    if len(eng_sents) == len(kor_sents):
        return list(zip(eng_sents, kor_sents))
    if len(eng_sents) == 1:
        return [(eng_sents[0], " ".join(kor_sents))]

    if len(kor_sents) > len(eng_sents):
        grouped_kor = _dp_partition_many_to_few(kor_sents, eng_sents, is_many_kor=True)
        return list(zip(eng_sents, grouped_kor))

    # len(kor_sents) < len(eng_sents)
    if len(kor_sents) >= max(2, int(len(eng_sents) * 0.5)):
        grouped_eng = _dp_partition_many_to_few(eng_sents, kor_sents, is_many_kor=False)
        return list(zip(grouped_eng, kor_sents))

    # 해설 추출이 너무 적거나 1줄뿐인 경우: 영문 문장 그대로 유지
    return [(e, "") for e in eng_sents]


def build_english_sentence_items(
    clean_english: str,
    korean_translation: str = "",
) -> list[EnglishSentenceAnalysisItem]:
    """깨끗한 영어 원문과 한국어 해석을 문장별 EnglishSentenceAnalysisItem 리스트로 조립한다 (AI 0%)."""
    from src.models.english import EnglishSentenceAnalysisItem

    eng_sents = split_english_sentences_clean(clean_english)
    if not eng_sents and (clean_english or "").strip():
        eng_sents = [clean_english.strip()]
    kor_sents = split_korean_sentences_clean(korean_translation) if (korean_translation or "").strip() else []
    aligned = align_english_korean_sentences(eng_sents, kor_sents)

    items: list[EnglishSentenceAnalysisItem] = []
    for idx, (e_txt, k_txt) in enumerate(aligned, start=1):
        items.append(
            EnglishSentenceAnalysisItem(
                sentence_num=idx,
                is_topic_sentence=False,
                original_english=e_txt,
                chunked_english=e_txt,
                korean_translation=k_txt,
                syntax_notes=[],
                original_blank_phrase="",
                is_inserted_sentence=False,
            )
        )
    return items
