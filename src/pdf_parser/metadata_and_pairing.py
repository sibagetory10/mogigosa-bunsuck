from __future__ import annotations

import os
import re
import unicodedata
from typing import Sequence

import pymupdf

from src.models.question import normalize_exam_year

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
