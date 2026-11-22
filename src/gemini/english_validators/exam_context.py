from __future__ import annotations

import difflib
import re
from typing import Any

from src.gemini.english_validators.transform_helpers import (
    _ORDER_SEQ_RE,
    _normalize_pdf_hyphens,
    _preprocess_english_source_passage,
    _split_english_sentences,
    _strip_stray_exam_qnum_noise,
)
from src.gemini.english_validators.transform_validate import (
    _has_exam_structure_markers,
    _has_incomplete_english_blank,
    _needs_english_passage_restoration,
)
from src.gemini.usage import _CIRCLED_DIGIT_TO_NUM, _normalize_extracted_answer
from src.text_formatter import normalize_english_exam_text

# 기출 원문(question_text) 기반 무손실 각주·빈칸·무관한 문장·순서 추출 및 보정
# ---------------------------------------------------------------------------
_CIRCLED_NUM_LIST = ("①", "②", "③", "④", "⑤")
_NUM_TO_CIRCLED_MAP = {"1": "①", "2": "②", "3": "③", "4": "④", "5": "⑤"}

_ENG_FOOTNOTE_EXTRACT_RE = re.compile(
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


def _extract_english_footnotes(question_text: str) -> list[str]:
    """PyMuPDF 추출 문항 텍스트(단일 줄/다중 줄 모두 지원)에서 문제지 공식 각주(`* word: 뜻`)를 추출한다."""
    raw = _normalize_pdf_hyphens((question_text or "").strip())
    if not raw or "*" not in raw:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for m in _ENG_FOOTNOTE_EXTRACT_RE.finditer(raw):
        term = re.sub(r"\s+", " ", m.group(1)).strip()
        meaning = re.sub(r"\s+", " ", m.group(2)).strip()
        meaning = _FOOTNOTE_TAIL_CUT_RE.sub("", meaning).strip()
        # 한국어 각주 뜻 뒤에 4단어 이상 영문 요약문/다음 지문이 붙은 경우 한국어까지만 절단
        m_eng_tail = re.search(r"([가-힣][가-힣0-9\s,·()'’\"~-]*?[가-힣\)])\s+[A-Z][a-z]+(?:\s+[A-Za-z0-9,'’\"-]+){3,}", meaning)
        if m_eng_tail:
            meaning = m_eng_tail.group(1).strip()
        # 끝부분에 붙은 시험지 쪽번호(예: '5 8', '6') 제거
        meaning = re.sub(r"(?:\s+\d{1,2}){1,2}\s*$", "", meaning).strip()
        meaning = re.sub(r"\s*영어\s*영역.*$", "", meaning).strip()
        if not term or not meaning or not re.search(r"[가-힣]", meaning):
            continue
        key = term.lower()
        if key not in seen:
            seen.add(key)
            out.append(f"{term}: {meaning}")
    return out


def _strip_english_question_header_and_footnotes(question_text: str) -> str:
    """question_text에서 앞부분 한국어 발문·문항번호·배점과 각주·쪽번호만 정밀 제거하고 본문(+선지)을 100% 보존한다."""
    s = re.sub(r"\s+", " ", _normalize_pdf_hyphens((question_text or "").strip()))
    if not s:
        return ""
    # 1) 각주는 전체 뒷부분을 자르지 않고 매칭된 각주 구간만 인라인 제거 (본문-각주-선지 순서에서도 ①~⑤ 선지 보존)
    if "*" in s:
        s = _ENG_FOOTNOTE_EXTRACT_RE.sub(" ", s)
        s = re.sub(r"\s+", " ", s).strip()

    # 2) 앞부분 헤더 단계별 순차 제거 (백트래킹 없이 36., 38., 7 39., 739. 완벽 제거)
    s = re.sub(r"^(?:\d{1,2}\s+영어\s*영역\s*|영어\s*영역\s+\d{1,2}\s*)", "", s).strip()
    s = re.sub(
        r"^\[\d{1,2}\s*[~～-]\s*\d{1,2}\][^A-Za-z]*?[가-힣][^?.]*?(?:\?|시오\.)(?:\s*\[[1-4]점\])?\s*",
        "",
        s,
        count=1,
    ).strip()
    s = re.sub(r"^(?:(?:[1-9]|1[0-2])\s+)?(?:1[89]|[234]\d)\.\s*", "", s, count=1).strip()
    s = re.sub(r"^[1-9](?:1[89]|[234]\d)\.\s*", "", s, count=1).strip()
    s = re.sub(
        r"^(?:[^A-Za-z]*?[가-힣][^?.]*?(?:\?|시오\.)(?:\s*\[[1-4]점\])?\s*)",
        "",
        s,
        count=1,
    ).strip()
    s = re.sub(r"^(?:(?:[1-9]|1[0-2])\s+)?(?:1[89]|[234]\d)\.\s*", "", s, count=1).strip()
    s = re.sub(r"^[1-9](?:1[89]|[234]\d)\.\s*", "", s, count=1).strip()
    s = re.sub(r"^\[[1-4]점\]\s*", "", s).strip()
    s = re.sub(r"(?:\s+\d{1,2}\s+영어\s*영역|\s+영어\s*영역\s+\d{1,2})\s*$", "", s).strip()
    s = _strip_stray_exam_qnum_noise(s)
    return s


def _split_available_circled_choices(text: str) -> tuple[str, dict[int, str]]:
    """⑤가 칼럼 경계로 잘렸더라도 존재하는 ①~⑤ 마커를 분할해 (앞부분, {번호: 선지})를 반환한다."""
    positions: list[tuple[int, int]] = []
    search_from = 0
    for idx_m, sym in enumerate(_CIRCLED_NUM_LIST, start=1):
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


def _split_numbered_five_markers(text: str) -> tuple[str, dict[int, str]] | None:
    """텍스트에서 오름차순으로 등장하는 ①~⑤ 5개 마커를 분할해 (앞부분, {1: seg1, ..., 5: seg5})를 반환한다."""
    positions: list[tuple[int, int]] = []
    search_from = 0
    for idx_m, sym in enumerate(_CIRCLED_NUM_LIST, start=1):
        pos = text.find(sym, search_from)
        if pos == -1:
            return None
        positions.append((idx_m, pos))
        search_from = pos + len(sym)
    prefix = text[: positions[0][1]].strip()
    segs: dict[int, str] = {}
    for k, (idx_m, pos) in enumerate(positions):
        start = pos + 1
        end = positions[k + 1][1] if k + 1 < len(positions) else len(text)
        segs[idx_m] = text[start:end].strip()
    return prefix, segs


def _is_clean_english_reconstruction(candidate: str, reference_passage: str) -> bool:
    """재구성된 영어 지문이 한국어 누출·줄바꿈 하이픈 깨짐 없이 reference_passage와 같은 지문인지 검증한다."""
    cand = (candidate or "").strip()
    if len(cand) < 80:
        return False
    if re.search(r"[가-힣]", cand):
        return False
    if re.search(r"[①②③④⑤]|_{3,}", cand):
        return False
    # PDF 줄바꿈 하이픈 깨짐('com- munity')이 많으면 안전상 자동 덮어쓰기 보류
    if len(re.findall(r"[A-Za-z]-\s+[a-z]", cand)) >= 1:
        return False
    ref = (reference_passage or "").strip()
    if not ref:
        return True
    ref_toks = set(re.findall(r"[a-z]{3,}", ref.lower()))
    cand_toks = set(re.findall(r"[a-z]{3,}", cand.lower()))
    if not ref_toks or not cand_toks:
        return False
    overlap = len(ref_toks & cand_toks) / max(1, min(len(ref_toks), len(cand_toks)))
    return overlap >= 0.72


def _extract_english_word_tokens(text: str) -> list[str]:
    """구문 표기(/, [], ())를 제거한 순수 영문 단어 토큰 목록을 추출한다."""
    s = (text or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[/\[\]\(\)]", " ", s)
    return re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*", s)


def _resolve_english_exam_point_title(orig_question_type: str = "", exam_point_note: str = "") -> str:
    """문항 유형(및 노트 내용)에 맞는 기출 포인트 박스 동적 제목을 반환한다."""
    qt = (orig_question_type or "").strip()
    note = (exam_point_note or "").strip()
    if "빈칸" in qt or "[기출 빈칸" in note:
        return "기출 포인트 (원문 빈칸)"
    if "무관" in qt or ("흐름" in qt and "관계" in qt) or "[제거된 무관한 문장" in note:
        return "기출 포인트 (제거된 무관한 문장)"
    if "순서" in qt or "[정답 순서" in note:
        return "기출 포인트 (글의 순서)"
    if "삽입" in qt or "[삽입 문장" in note:
        return "기출 포인트 (문장 삽입)"
    if "어법" in qt:
        return "기출 포인트 (어법)"
    if "어휘" in qt or "낱말" in qt:
        return "기출 포인트 (어휘)"
    if "함축" in qt:
        return "기출 포인트 (함축 의미)"
    if "요약" in qt:
        return "기출 포인트 (요약문)"
    return "기출 포인트"


_HYPHEN_SUFFIX_LEVEL_RE = re.compile(
    r"\b(individual|policy|community|system|surface|macro|micro|entry|high|low|top|mid|ground|"
    r"state|local|national|global|school|family|group|team|firm|farm|field|gene|cell|brain|city|street|grassroots)level\b",
    re.IGNORECASE,
)
_DROPPED_HYPHEN_COMPOUNDS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(decision)making\b", re.I), r"\1-making"),
    (re.compile(r"\b(problem)solving\b", re.I), r"\1-solving"),
    (re.compile(r"\b(long)term\b", re.I), r"\1-term"),
    (re.compile(r"\b(short)term\b", re.I), r"\1-term"),
    (re.compile(r"\b(real)world\b", re.I), r"\1-world"),
    (re.compile(r"\b(large)scale\b", re.I), r"\1-scale"),
    (re.compile(r"\b(small)scale\b", re.I), r"\1-scale"),
    (re.compile(r"\b(self)esteem\b", re.I), r"\1-esteem"),
    (re.compile(r"\b(self)control\b", re.I), r"\1-control"),
    (re.compile(r"\b(well)known\b", re.I), r"\1-known"),
    (re.compile(r"\b(cross)cultural\b", re.I), r"\1-cultural"),
    (re.compile(r"\b(cost)effective\b", re.I), r"\1-effective"),
    (re.compile(r"\b(evidence)based\b", re.I), r"\1-based"),
)


def _preserve_hyphens_from_reference(target_text: str, ref_text: str) -> str:
    """ref_text에 존재하는 하이픈 포함 영숫자 토큰(16-18, individual-level 등)이 target_text에서 붙어버렸으면 원래 하이픈을 복원한다."""
    out = _normalize_pdf_hyphens((target_text or "").strip())
    ref = _normalize_pdf_hyphens((ref_text or "").strip())
    if not out or not ref or "-" not in ref:
        return out
    hyphen_tokens = re.findall(r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b", ref)
    seen: set[str] = set()
    for h_tok in hyphen_tokens:
        joined = h_tok.replace("-", "")
        if len(joined) < 3 or joined.lower() in seen:
            continue
        seen.add(joined.lower())
        if re.search(rf"\b{re.escape(joined)}\b", out, flags=re.IGNORECASE):
            is_num_range = bool(re.match(r"^\d+-\d+$", h_tok))

            def _repl(m: re.Match[str], orig_h: str = h_tok, num_rng: bool = is_num_range) -> str:
                if num_rng:
                    pre = out[max(0, m.start() - 12) : m.start()].lower()
                    if re.search(r"\b(?:in|since|around|year|c\.)\s*$", pre):
                        return m.group(0)
                matched = m.group(0)
                if matched.isupper() and not orig_h.isupper():
                    return orig_h.upper()
                if matched[0].isupper() and orig_h[0].islower():
                    return orig_h[0].upper() + orig_h[1:]
                return orig_h
            out = re.sub(rf"\b{re.escape(joined)}\b", _repl, out, flags=re.IGNORECASE)
    return out


def _repair_pdf_dropped_hyphens(text: str, explanation: str = "") -> str:
    """PDF 추출 과정에서 하이픈이 빠진 숫자 범위(16~18 -> 16-18) 및 명백한 비단어 복합어를 안전하게 복원한다."""
    out = _normalize_pdf_hyphens((text or "").strip())
    if not out:
        return ""
    exp = (explanation or "").strip()
    if exp:
        for m_rng in re.finditer(r"(?<!\d)(\d{1,3})\s*[~∼\-–]\s*(\d{1,3})(?!\d)", exp):
            n1, n2 = m_rng.group(1), m_rng.group(2)
            joined_num = f"{n1}{n2}"
            hyphen_num = f"{n1}-{n2}"
            if int(n1) < int(n2) and re.search(rf"\b{re.escape(joined_num)}\b", out) and hyphen_num not in out:
                def _repl_num(m: re.Match[str], rep: str = hyphen_num) -> str:
                    pre = out[max(0, m.start() - 12) : m.start()].lower()
                    if re.search(r"\b(?:in|since|around|year|c\.)\s*$", pre):
                        return m.group(0)
                    return rep
                out = re.sub(rf"\b{re.escape(joined_num)}\b", _repl_num, out)

    out = _HYPHEN_SUFFIX_LEVEL_RE.sub(lambda m: f"{m.group(1)}-level", out)
    for pat, repl in _DROPPED_HYPHEN_COMPOUNDS:
        out = pat.sub(repl, out)
    return out


def _infer_blank_phrase_by_diff(raw_body_with_blank: str, completed_passage: str) -> str:
    """
    PDF 텍스트 추출 과정에서 선지 분할이 깨졌거나 빈칸이 벡터 선(공백)으로만 남았을 때,
    원본 빈칸 문장과 복원된 완성 지문(`completed_passage`)의 토큰 diff로
    원래 빈칸 구절(`original_blank_phrase`)을 결정론적으로 역추출한다.
    """
    raw_b = (raw_body_with_blank or "").strip()
    comp_p = (completed_passage or "").strip()
    if not raw_b or not comp_p:
        return ""

    if re.search(r"_{2,}", raw_b):
        raw_masked = re.sub(r"_{2,}", " __BLANK_SENTINEL__ ", raw_b)
        raw_sents = _split_english_sentences(raw_masked)
        blank_sent = next((s for s in raw_sents if "__BLANK_SENTINEL__" in s), "")
        if not blank_sent:
            return ""
        left_part, right_part = blank_sent.split("__BLANK_SENTINEL__", 1)
    else:
        raw_sents = _split_english_sentences(raw_b)
        blank_sent = next(
            (s for s in raw_sents if _has_incomplete_english_blank(s) or re.search(r"[A-Za-z][ \t]{2,}[.?!]", s)),
            "",
        )
        if not blank_sent:
            return ""
        gap_m = re.search(r"[ \t]{2,}", blank_sent)
        if gap_m:
            left_part = blank_sent[: gap_m.start()]
            right_part = blank_sent[gap_m.end() :]
        else:
            left_part = blank_sent
            right_part = ""

    left_toks = [t.lower() for t in _extract_english_word_tokens(left_part)]
    right_toks = [t.lower() for t in _extract_english_word_tokens(right_part)]
    anchor_set = set(left_toks + right_toks)
    if not anchor_set:
        return ""

    comp_sents = _split_english_sentences(comp_p)
    best_sent = ""
    best_score = 0.0
    for cs in comp_sents:
        cs_toks = [t.lower() for t in _extract_english_word_tokens(cs)]
        if not cs_toks:
            continue
        cs_set = set(cs_toks)
        overlap = len(anchor_set & cs_set) / max(1, len(anchor_set))
        if overlap > best_score:
            best_score = overlap
            best_sent = cs
    if not best_sent or best_score < 0.45:
        return ""

    word_re = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")
    cs_matches = list(word_re.finditer(best_sent))
    cs_low = [m.group(0).lower().replace("’", "'") for m in cs_matches]
    if not cs_matches:
        return ""

    start_tok_idx = 0
    if left_toks:
        for k in range(min(4, len(left_toks)), 0, -1):
            pat = left_toks[-k:]
            for idx in range(len(cs_low) - k + 1):
                if cs_low[idx : idx + k] == pat:
                    start_tok_idx = idx + k
                    break
            if start_tok_idx > 0:
                break

    end_tok_idx = len(cs_matches)
    if right_toks:
        for k in range(min(4, len(right_toks)), 0, -1):
            pat = right_toks[:k]
            for idx in range(start_tok_idx, len(cs_low) - k + 1):
                if cs_low[idx : idx + k] == pat:
                    end_tok_idx = idx
                    break
            if end_tok_idx < len(cs_matches):
                break

    if 0 <= start_tok_idx < end_tok_idx <= len(cs_matches):
        span_len = end_tok_idx - start_tok_idx
        if 1 <= span_len <= 24 and span_len < len(cs_matches):
            phrase = best_sent[cs_matches[start_tok_idx].start() : cs_matches[end_tok_idx - 1].end()].strip()
            return phrase
    return ""


def _reconcile_words_against_question_body(
    passage_text: str,
    question_text: str,
    orig_question_type: str = "",
) -> str:
    """
    question_text에서 발문·각주·선지·원문자(①~⑤)만 제거한 원문 본문과 passage_text를 단어 단위로 대조하여:
    - 어휘(30번)/어법(29번)에서 원문자 제거 중 본동사 등이 통째로 삭제된 순수 누락(delete)을 즉시 복구한다.
    - 비변형/빈칸 문항에서 'at one' -> 'at once'처럼 1~2단어가 원문과 다르게 변조된 부분을 복구한다.
    """
    raw_p = (passage_text or "").strip()
    raw_q = (question_text or "").strip()
    if not raw_p or not raw_q:
        return raw_p
    qt = (orig_question_type or "").strip()
    if any(k in qt for k in ("순서", "삽입", "무관", "흐름")):
        return raw_p

    q_stripped = _strip_english_question_header_and_footnotes(raw_q)
    if not q_stripped:
        return raw_p
    is_grammar_or_vocab = ("어법" in qt) or ("어휘" in qt) or ("낱말" in qt) or ("42" in qt)
    if is_grammar_or_vocab:
        # 어법(29)·어휘(30)는 본문 내부에 ①~⑤가 인라인 삽입되어 있으므로 ① 앞에서 자르면 본문이 날아간다.
        q_body = re.sub(r"\s*①\s*②\s*③\s*④\s*⑤\s*$", "", q_stripped)
    else:
        split_res = _split_available_circled_choices(q_stripped)
        q_body = split_res[0] if len(split_res[1]) >= 2 else q_stripped
    # 본문 내 어법/어휘 밑줄 번호 ①~⑤ 또는 (a)~(e) 기호만 제거 (바로 뒤 영단어는 100% 보존)
    q_body = re.sub(r"[①②③④⑤]", " ", q_body)
    q_body = re.sub(r"\(\s*[a-e]\s*\)\s*", " ", q_body)
    q_body = _strip_stray_exam_qnum_noise(re.sub(r"\s+", " ", q_body).strip())
    if len(q_body) < 60 or re.search(r"[가-힣]", q_body):
        return raw_p

    word_re = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")
    p_matches = list(word_re.finditer(raw_p))
    q_matches = list(word_re.finditer(q_body))
    if len(p_matches) < 15 or len(q_matches) < 15:
        return raw_p

    p_low = [m.group(0).lower().replace("’", "'") for m in p_matches]
    q_low = [m.group(0).lower().replace("’", "'") for m in q_matches]
    sm = difflib.SequenceMatcher(None, p_low, q_low)
    if sm.ratio() < 0.72:
        return raw_p

    edits: list[tuple[int, int, str]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        p_span_len = i2 - i1
        q_span_len = j2 - j1
        if tag == "insert" and p_span_len == 0 and 1 <= q_span_len <= 2:
            # raw_p에서 단어가 통째로 삭제된 경우 (예: 'the subsidy [distorts] the economy')
            ins_txt = q_body[q_matches[j1].start() : q_matches[j2 - 1].end()]
            if i1 == 0:
                edits.append((0, 0, f"{ins_txt} "))
            else:
                prev_end = p_matches[i1 - 1].end()
                edits.append((prev_end, prev_end, f" {ins_txt}"))
        elif (
            not is_grammar_or_vocab
            and tag == "replace"
            and 1 <= p_span_len <= 2
            and 1 <= q_span_len <= 2
        ):
            # 하이픈 차이('16-18' vs '1618')는 건드리지 않고, 실제 단어 변조('at once' vs 'at one')만 복구
            p_joined = "".join(p_low[i1:i2]).replace("-", "")
            q_joined = "".join(q_low[j1:j2]).replace("-", "")
            if p_joined == q_joined:
                continue
            # 빈칸 복원 구절 오인 방지: 원문 question_text에 빈칸(____)이 있던 자리 옆은 건드리지 않음
            q_ctx_before = q_body[max(0, q_matches[j1].start() - 15) : q_matches[j1].start()]
            if "_" in q_ctx_before:
                continue
            repl_txt = q_body[q_matches[j1].start() : q_matches[j2 - 1].end()]
            edits.append((p_matches[i1].start(), p_matches[i2 - 1].end(), repl_txt))

    if not edits:
        return raw_p
    out = raw_p
    for st_i, en_i, rep_s in reversed(edits):
        out = f"{out[:st_i]}{rep_s}{out[en_i:]}"
    out = re.sub(r"[ \t]{2,}", " ", out).strip()
    return out


def _extract_english_exam_context(
    question_text: str,
    orig_question_type: str = "",
    answer: str = "",
    passage_text: str = "",
) -> dict[str, Any]:
    """
    원본 PDF 텍스트(question_text)와 정답(answer)을 대조해
    각주(footnotes), 원래 빈칸 구절(original_blank_phrase), 제거된 무관한 문장(irrelevant_sentence),
    글의 순서 정답 배열(order_answer_seq)·단락별 원문(order_blocks), 문장 삽입 원문/위치(inserted_sentence),
    그리고 원문 변조·누락 시 복구 가능한 무손실 기준 지문(reconstructed_clean_passage)을 결정론적으로 추출한다.
    """
    ctx: dict[str, Any] = {
        "footnotes": _extract_english_footnotes(question_text),
        "original_blank_phrase": "",
        "irrelevant_sentence": "",
        "irrelevant_answer_sym": "",
        "order_answer_seq": "",
        "order_given_text": "",
        "order_blocks": {},
        "inserted_sentence": "",
        "inserted_answer_sym": "",
        "reconstructed_clean_passage": "",
        "choices": {},
    }
    raw_q = (question_text or "").strip()
    if not raw_q:
        return ctx

    body_and_choices = _strip_english_question_header_and_footnotes(raw_q)
    if not body_and_choices:
        return ctx

    q_type = (orig_question_type or "").strip()
    ans_key = _normalize_extracted_answer(answer)
    ans_int = int(ans_key) if ans_key in ("1", "2", "3", "4", "5") else 0

    # 1) 빈칸 추론 (31~34번)
    is_blank = ("빈칸" in q_type) or bool(re.search(r"_{2,}", body_and_choices))
    if is_blank:
        raw_body, segs = _split_available_circled_choices(body_and_choices)
        raw_body = _strip_stray_exam_qnum_noise(raw_body)
        if segs:
            choices_map = {str(k): v for k, v in segs.items() if v}
            ctx["choices"] = choices_map
            if ans_int and str(ans_int) in choices_map:
                ans_phrase = choices_map[str(ans_int)].strip()
                ans_phrase = re.sub(r"(?:\s+\d{1,2}){1,2}\s*$", "", ans_phrase).strip()
                # 만약 마지막 선지에 다음 문항/영역 헤더가 붙었으면 제거
                ans_phrase = re.sub(r"\s+\d{1,2}\s+영어\s*영역.*$", "", ans_phrase).strip()
                ctx["original_blank_phrase"] = ans_phrase
                if ans_phrase:
                    filled = ""
                    if re.search(r"_{2,}", raw_body):
                        filled = re.sub(r"\s*_{2,}\s*", f" {ans_phrase} ", raw_body, count=1)
                    else:
                        # PDF에서 빈칸이 벡터 밑줄선으로 그려져 '____' 텍스트 없이 dangling 문장으로 끝난 경우 복원
                        body_sents = _split_english_sentences(raw_body)
                        incomplete_idxs = [
                            i_s for i_s, s_txt in enumerate(body_sents)
                            if _has_incomplete_english_blank(s_txt) or re.search(r"[A-Za-z][ \t]{2,}[.?!]", s_txt)
                        ]
                        if len(incomplete_idxs) == 1:
                            t_i = incomplete_idxs[0]
                            target_s = body_sents[t_i]
                            gap_m = re.search(r"[ \t]{2,}(?=[.?!])", target_s)
                            if gap_m:
                                body_sents[t_i] = f"{target_s[:gap_m.start()].rstrip()} {ans_phrase}{target_s[gap_m.end():]}"
                            else:
                                m_end_p = re.search(r"([.?!][\"'”’\)]*)$", target_s.rstrip())
                                if m_end_p:
                                    body_sents[t_i] = f"{target_s[:m_end_p.start()].rstrip()} {ans_phrase}{m_end_p.group(1)}"
                                else:
                                    body_sents[t_i] = f"{target_s.rstrip()} {ans_phrase}."
                            filled = " ".join(body_sents)
                    if filled:
                        filled = re.sub(r"\s+([.,?!;:])", r"\1", filled)
                        filled = re.sub(r"\s+", " ", filled).strip()
                        filled = _preserve_hyphens_from_reference(filled, passage_text)
                        if _is_clean_english_reconstruction(filled, passage_text) and not _has_incomplete_english_blank(filled):
                            ctx["reconstructed_clean_passage"] = filled
        # 폴백: ①~⑤ 선지 추출이 깨졌더라도 빈칸 문장과 완성 지문의 diff로 정답 구절 역추출
        if not ctx["original_blank_phrase"] and passage_text:
            inferred_bp = _infer_blank_phrase_by_diff(raw_body, passage_text)
            if inferred_bp:
                ctx["original_blank_phrase"] = inferred_bp
        return ctx

    # 2) 무관한 문장 (35번)
    is_irrelevant = ("무관" in q_type) or ("흐름" in q_type and "관계" in raw_q)
    if is_irrelevant:
        split_res = _split_numbered_five_markers(body_and_choices)
        if split_res is not None:
            intro, segs = split_res
            if ans_int and ans_int in segs:
                seg5_sents = _split_english_sentences(segs[5])
                tail_after_5 = ""
                if ans_int == 5 and len(seg5_sents) >= 2:
                    irrel_sent = seg5_sents[0]
                    tail_after_5 = " ".join(seg5_sents[1:]).strip()
                else:
                    irrel_sent = segs[ans_int].strip()
                ctx["irrelevant_sentence"] = _strip_stray_exam_qnum_noise(irrel_sent)
                ctx["irrelevant_answer_sym"] = _NUM_TO_CIRCLED_MAP.get(str(ans_int), "")
                kept_parts: list[str] = []
                if intro:
                    kept_parts.append(intro)
                for k in range(1, 6):
                    if k == ans_int:
                        continue
                    if segs.get(k):
                        kept_parts.append(segs[k])
                if tail_after_5:
                    kept_parts.append(tail_after_5)
                rebuilt = _strip_stray_exam_qnum_noise(re.sub(r"\s+", " ", " ".join(kept_parts)).strip())
                rebuilt = _preserve_hyphens_from_reference(rebuilt, passage_text)
                if _is_clean_english_reconstruction(rebuilt, passage_text):
                    ctx["reconstructed_clean_passage"] = rebuilt
        return ctx

    # 3) 글의 순서 (36~37번)
    is_order = "순서" in q_type
    if is_order:
        default_order_choices = {
            "1": "(A) - (C) - (B)",
            "2": "(B) - (A) - (C)",
            "3": "(B) - (C) - (A)",
            "4": "(C) - (A) - (B)",
            "5": "(C) - (B) - (A)",
        }
        order_body = body_and_choices
        m_seq = None
        split_res = _split_numbered_five_markers(body_and_choices)
        if split_res is not None:
            order_body, segs = split_res
            choices_map = {str(k): v for k, v in segs.items() if v}
            ctx["choices"] = choices_map
            m_seq = _ORDER_SEQ_RE.search(choices_map.get(str(ans_int), "")) if ans_int else None
        if m_seq is None and ans_int and str(ans_int) in default_order_choices:
            m_seq = _ORDER_SEQ_RE.search(default_order_choices[str(ans_int)])

        if m_seq is not None:
            seq = (m_seq.group(1), m_seq.group(2), m_seq.group(3))
            ctx["order_answer_seq"] = f"({seq[0]}) - ({seq[1]}) - ({seq[2]})"
            parts = re.split(r"(?:^|\s)\(([ABC])\)\s+", order_body)
            if len(parts) >= 7:
                given_txt = _strip_stray_exam_qnum_noise(parts[0].strip())
                blocks: dict[str, str] = {}
                for i_p in range(1, len(parts) - 1, 2):
                    lbl = parts[i_p]
                    txt = _strip_stray_exam_qnum_noise(parts[i_p + 1].strip())
                    if lbl in ("A", "B", "C") and txt:
                        blocks[lbl] = txt
                if given_txt and set(blocks.keys()) == {"A", "B", "C"} and set(seq) == {"A", "B", "C"}:
                    ctx["order_given_text"] = _preserve_hyphens_from_reference(given_txt, passage_text)
                    ctx["order_blocks"] = {
                        k: _preserve_hyphens_from_reference(v, passage_text)
                        for k, v in blocks.items()
                    }
                    rebuilt = re.sub(
                        r"\s+",
                        " ",
                        " ".join([given_txt, blocks[seq[0]], blocks[seq[1]], blocks[seq[2]]]),
                    ).strip()
                    rebuilt = _preserve_hyphens_from_reference(rebuilt, passage_text)
                    if _is_clean_english_reconstruction(rebuilt, passage_text):
                        ctx["reconstructed_clean_passage"] = rebuilt
        return ctx

    # 4) 문장 삽입 (38~39번)
    is_insertion = "삽입" in q_type
    if is_insertion:
        norm_ins = re.sub(r"\(\s*([①②③④⑤])\s*\)", r"\1", body_and_choices)
        split_res = _split_numbered_five_markers(norm_ins)
        if split_res is not None:
            prefix_raw, segs = split_res
            prefix_clean = _strip_stray_exam_qnum_noise(re.sub(r"\[주어진\s*문장\]", "", prefix_raw).strip())
            pref_sents = _split_english_sentences(prefix_clean)
            if pref_sents:
                given_sent = pref_sents[0]
                body_intro = " ".join(pref_sents[1:]).strip()
                if passage_text and len(pref_sents) >= 2:
                    p_sents = _split_english_sentences(passage_text)
                    if p_sents:
                        first_p_toks = set(t.lower() for t in _extract_english_word_tokens(p_sents[0]))
                        for cand_i, ps in enumerate(pref_sents):
                            if cand_i == 0:
                                continue
                            ps_toks = set(t.lower() for t in _extract_english_word_tokens(ps))
                            if ps_toks and len(first_p_toks & ps_toks) / max(1, len(first_p_toks | ps_toks)) >= 0.65:
                                given_sent = " ".join(pref_sents[:cand_i]).strip()
                                body_intro = " ".join(pref_sents[cand_i:]).strip()
                                break
                ctx["inserted_sentence"] = _preserve_hyphens_from_reference(given_sent, passage_text)
                if ans_int:
                    ctx["inserted_answer_sym"] = _NUM_TO_CIRCLED_MAP.get(str(ans_int), "")
                    rebuilt_parts: list[str] = []
                    if body_intro:
                        rebuilt_parts.append(body_intro)
                    for k in range(1, 6):
                        if k == ans_int:
                            rebuilt_parts.append(given_sent)
                        if segs.get(k):
                            rebuilt_parts.append(segs[k])
                    rebuilt = _strip_stray_exam_qnum_noise(re.sub(r"\s+", " ", " ".join(rebuilt_parts)).strip())
                    rebuilt = _preserve_hyphens_from_reference(rebuilt, passage_text)
                    if _is_clean_english_reconstruction(rebuilt, passage_text):
                        ctx["reconstructed_clean_passage"] = rebuilt
        return ctx

    return ctx


def _reconcile_english_passage_with_question_text(
    passage_text: str,
    question_text: str,
    orig_question_type: str = "",
    answer: str = "",
    explanation: str = "",
) -> tuple[str, dict[str, Any]]:
    """
    저장된 passage_text를 원본 question_text(+정답)와 결정론적으로 대조하여,
    원문 단어 변조·구절 누락·정상 문장 통누락이 감지되면 무손실 원본으로 자동 복구한다.
    이미 완성된 지문을 불완전한 재조합본으로 덮어쓰지 않도록 안전 가드를 적용하고 하이픈·쪽번호도 보정한다.
    """
    raw_p = _repair_pdf_dropped_hyphens(_strip_stray_exam_qnum_noise((passage_text or "").strip()), explanation=explanation)
    exam_ctx = _extract_english_exam_context(
        question_text=question_text,
        orig_question_type=orig_question_type,
        answer=answer,
        passage_text=raw_p,
    )
    rebuilt = (exam_ctx.get("reconstructed_clean_passage") or "").strip()
    if rebuilt:
        rebuilt = _repair_pdf_dropped_hyphens(_strip_stray_exam_qnum_noise(rebuilt), explanation=explanation)
        rebuilt = _preserve_hyphens_from_reference(rebuilt, raw_p)
        exam_ctx["reconstructed_clean_passage"] = rebuilt

    out_p = raw_p
    if rebuilt and not _has_incomplete_english_blank(rebuilt):
        if not raw_p or _needs_english_passage_restoration(raw_p, orig_question_type):
            out_p = rebuilt
        else:
            p_toks = [t.lower() for t in _extract_english_word_tokens(raw_p)]
            r_toks = [t.lower() for t in _extract_english_word_tokens(rebuilt)]
            bp = (exam_ctx.get("original_blank_phrase") or "").strip()
            bp_toks = [t.lower() for t in _extract_english_word_tokens(bp)]
            bp_lost = False
            if bp_toks:
                bp_str = " ".join(bp_toks)
                if bp_str in " ".join(p_toks) and bp_str not in " ".join(r_toks):
                    bp_lost = True
            is_irrel = "무관" in (orig_question_type or "") or ("흐름" in (orig_question_type or ""))
            shrunk_dangerously = (not is_irrel) and (len(r_toks) < len(p_toks) - 3)
            paren_lost = rebuilt.count("(") > raw_p.count("(") or rebuilt.count(")") > raw_p.count(")")
            if p_toks and r_toks and not bp_lost and not shrunk_dangerously and (p_toks != r_toks or paren_lost):
                p_sents = _split_english_sentences(raw_p)
                r_sents = _split_english_sentences(rebuilt)
                sm = difflib.SequenceMatcher(None, p_toks, r_toks)
                if sm.ratio() >= 0.78 or (len(r_sents) == len(p_sents) + 1 and sm.ratio() >= 0.70):
                    out_p = rebuilt

    # 어법/어휘 본동사 삭제('distorts') 및 비변형 문항 단어 변조('at once' -> 'at one') 결정론적 복구
    out_p = _reconcile_words_against_question_body(out_p, question_text, orig_question_type=orig_question_type)
    out_p = _repair_pdf_dropped_hyphens(_strip_stray_exam_qnum_noise(out_p), explanation=explanation)
    return out_p, exam_ctx
