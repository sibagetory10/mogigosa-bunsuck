from __future__ import annotations

import difflib
import re
from typing import Any

from src.gemini.english_validators.exam_context import (
    _CIRCLED_NUM_LIST,
    _extract_english_word_tokens,
    _resolve_english_exam_point_title,
)
from src.gemini.english_validators.transform_helpers import (
    _normalize_pdf_hyphens,
    _sentence_has_english_verb,
    _split_english_sentences,
)
from src.models import (
    EnglishPassageAnalysisOutput,
    EnglishPassageSummaryDiagramOutput,
    EnglishSentenceAnalysisItem,
)
from src.text_formatter import normalize_english_exam_text, normalize_english_korean_translation

# 영어 정밀 지문 분석지(EnglishPassageAnalysisOutput) 결정론적 검증·자동 교정기
# ---------------------------------------------------------------------------
_PARALLEL_SUBJ_BRACKET_RE = re.compile(
    r"\[((?:they|we|he|she|it|you|I|people|scientists|researchers|clinicians|humans)\s+[^\[\]]+?)\]\s*"
    r"(?:/\s*)?\b(and|or|but)\b\s*(?:/\s*)?\[([^\[\]]+?)\]",
    re.IGNORECASE,
)
_WH_PAREN_CLAUSE_RE = re.compile(
    r"\(\s*(why|how|what|whether|where|when|who|whom|whose)\s+([^()]+)\)",
    re.IGNORECASE,
)
_WEAK_TOPIC_SENT_RE = re.compile(
    r"^(?:The\s+same\s+is\s+true|This\s+is\s+(?:also\s+)?true|So\s+too|Not\s+so|"
    r"However,|Yet,|Still,|For\s+example,|For\s+instance,|In\s+contrast,|On\s+the\s+other\s+hand,)",
    re.IGNORECASE,
)
# 일본어(히라가나/가타카나/반각가타카나), 키릴, 아랍, 태국, 데바나가리 및 한국어 문맥 내 의도치 않은 비한글 외국 문자 감지
_FOREIGN_SCRIPT_RE = re.compile(
    r"[\u3040-\u309f\u30a0-\u30ff\uff65-\uff9f\u0400-\u04ff\u0600-\u06ff\u0e00-\u0e7f\u0900-\u097f]"
)
_UNEXPECTED_CJK_HANJA_RE = re.compile(r"[\u4e00-\u9fff]{2,}")


_FULLWIDTH_BRACKET_TRANS = str.maketrans("［］（）", "[]()")
_DISALLOWED_CURLY_RE = re.compile(r"[{｝}｛]")


def _sanitize_disallowed_syntax_braces(source_sent: str, chunked_sent: str) -> str:
    """
    원문(`source_sent`)에 중괄호 `{}`가 없으면 출력에서 모든 `{}`(및 전각 중괄호)를 제거하고,
    전각 대괄호/소괄호(`［］（）`)를 반각(`[]()`)으로 정규화한다.
    """
    chk = (chunked_sent or "").translate(_FULLWIDTH_BRACKET_TRANS)
    src = source_sent or ""
    if "{" not in src and "}" not in src:
        chk = _DISALLOWED_CURLY_RE.sub("", chk)
    chk = re.sub(r"[ \t]{2,}", " ", chk).strip()
    chk = re.sub(r"\s+([.,?!;:])", r"\1", chk)
    return chk


def _has_balanced_syntax_brackets(text: str, *, allow_curly: bool = False) -> bool:
    """대괄호 []와 소괄호 ()의 열림/닫힘 짝, 중첩 순서 및 불허용 중괄호({})·전각 괄호 유무를 검사한다."""
    if not allow_curly and any(ch in text for ch in ("{", "}", "｛", "｝")):
        return False
    if any(ch in text for ch in ("［", "］", "（", "）")):
        return False
    stack: list[str] = []
    pairs = {"]": "[", ")": "("}
    if allow_curly:
        pairs["}"] = "{"
    opens = set(pairs.values())
    for ch in text:
        if ch in opens:
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return not stack


def _strip_spurious_quotes_against_source(source_sent: str, chunked_sent: str) -> str:
    """
    원문 영어 문장(`source_sent`)에 없는 임의 큰따옴표(`"`, `“`, `”`)가
    빈칸·강조 표기 목적으로 `chunked_sent`에 추가된 경우 결정론적으로 제거한다.
    """
    src = (source_sent or "").strip()
    chk = (chunked_sent or "").strip()
    if not chk:
        return chk
    src_q_cnt = sum(src.count(q) for q in ('"', "“", "”"))
    chk_q_cnt = sum(chk.count(q) for q in ('"', "“", "”"))
    if chk_q_cnt <= src_q_cnt:
        return chk
    if src_q_cnt == 0:
        cleaned = re.sub(r'["“”]', "", chk)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
        cleaned = re.sub(r"\s+([.,?!;:])", r"\1", cleaned)
        return cleaned
    # 원문에 따옴표가 일부 있는 경우, 원문에 없는 위치의 쌍따옴표 쌍만 제거
    src_quoted = {
        re.sub(r"\s+", " ", m.group(1)).strip().lower()
        for m in re.finditer(r'["“]([^"”]+)["”]', src)
    }

    def _drop_if_spurious(m: re.Match[str]) -> str:
        inner = m.group(1)
        norm_inner = re.sub(r"[/\[\]\(\)]", " ", inner)
        norm_inner = re.sub(r"\s+", " ", norm_inner).strip().lower()
        if any(norm_inner in sq or sq in norm_inner for sq in src_quoted if sq):
            return m.group(0)
        return inner

    cleaned = re.sub(r'["“]([^"”]+)["”]', _drop_if_spurious, chk)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned


def _balance_added_markup_chars(text: str) -> str:
    """
    임시 센티넬(⟨⟩)로 보호된 원문 괄호는 유지하고,
    새로 추가된 []/() 중 짝이 맞지 않거나 교차 중첩(`[... ( ... ] ...)`)된 기호만 제거한다.
    """
    chars = list(text)
    remove_idx: set[int] = set()
    stack: list[tuple[str, int]] = []
    match_open = {"]": "[", ")": "("}
    for idx, ch in enumerate(chars):
        if ch in ("[", "("):
            stack.append((ch, idx))
        elif ch in ("]", ")"):
            want = match_open[ch]
            # 스택 맨 위와 일치하면 정상 팝
            if stack and stack[-1][0] == want:
                stack.pop()
            else:
                # 교차 중첩 검사: 스택 안쪽에 want가 있으면 그 사이의 미닫힘 괄호들을 제거하고 want를 닫음
                found_pos = -1
                for k in range(len(stack) - 1, -1, -1):
                    if stack[k][0] == want:
                        found_pos = k
                        break
                if found_pos != -1:
                    while len(stack) - 1 > found_pos:
                        _, bad_i = stack.pop()
                        remove_idx.add(bad_i)
                    stack.pop()
                else:
                    remove_idx.add(idx)
    for _, unmatched_i in stack:
        remove_idx.add(unmatched_i)

    cleaned = "".join(ch for i, ch in enumerate(chars) if i not in remove_idx)
    cleaned = re.sub(r"\[\s*\]|\(\s*\)", "", cleaned)
    cleaned = re.sub(r"(?:\s*/\s*){2,}", " / ", cleaned)
    cleaned = re.sub(r"^\s*/\s*|\s*/\s*$", "", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([.,?!;:])", r"\1", cleaned)
    return cleaned


def _normalize_syntax_bracket_consistency(source_sent: str, chunked_sent: str) -> str:
    """
    병렬 동사구 주어 묶음(`[they accept ...] and [move on]`) 및
    원문 자체 괄호가 아닌 간접의문문 소괄호(`(why ... V ...)`)를 결정론적으로 교정해
    명사절/관계절/부사절=`[ ]`, 구 수식어=`( )` 표기 일관성을 맞춘다.
    """
    src = (source_sent or "").strip()
    chk = (chunked_sent or "").strip()
    if not chk:
        return chk

    # 1) 주어 공유 동사구 병렬에서 주어까지 앞 대괄호에 묶은 오류 자동 교정
    def _fix_parallel(m: re.Match[str]) -> str:
        left_inner = m.group(1).strip()
        conj = m.group(2)
        right_inner = m.group(3).strip()
        right_words = _extract_english_word_tokens(right_inner)
        if right_words and right_words[0].lower() not in (
            "they", "we", "he", "she", "it", "you", "i", "that", "which", "who", "what", "when", "if", "because", "although"
        ):
            return f"{left_inner} / {conj} / {right_inner}"
        return m.group(0)

    chk = _PARALLEL_SUBJ_BRACKET_RE.sub(_fix_parallel, chk)

    # 2) 원문 자체 괄호가 아닌 간접의문문(주어+동사 포함 wh-절)을 소괄호 () → 대괄호 []로 교정
    src_parens = {m.group(0).lower() for m in re.finditer(r"\([^()]+\)", src)}

    def _fix_wh_paren(m: re.Match[str]) -> str:
        if m.group(0).lower() in src_parens:
            return m.group(0)
        inner = f"{m.group(1)} {m.group(2)}"
        if len(_extract_english_word_tokens(inner)) >= 4 and _sentence_has_english_verb(inner):
            return f"[{m.group(1)} {m.group(2).strip()}]"
        return m.group(0)

    chk = _WH_PAREN_CLAUSE_RE.sub(_fix_wh_paren, chk)
    chk = re.sub(r"(?:\s*/\s*){2,}", " / ", chk)
    chk = re.sub(r"[ \t]{2,}", " ", chk).strip()
    return chk


def _project_chunk_markup_onto_source(source_sent: str, chunked_sent: str) -> str:
    """
    AI의 chunked_english에 단어 변조·누락·괄호 깨짐이 있더라도,
    100% 정확한 source_sent 단어·원문 괄호·구두점 위에 유효한 구문 표기(/, [], ())만 안전하게 투영한다.
    """
    src = (source_sent or "").strip()
    chk = _sanitize_disallowed_syntax_braces(src, (chunked_sent or "").strip())
    chk = _strip_spurious_quotes_against_source(src, chk)
    if not src:
        return chk
    if not chk:
        return src

    chk = _normalize_syntax_bracket_consistency(src, chk)
    allow_curly = "{" in src or "}" in src

    # 이미 원문과 단어 100% 일치 + 괄호 균형 + 원문 자체 괄호 보존이면 정리 후 반환
    src_toks = _extract_english_word_tokens(src)
    chk_toks = _extract_english_word_tokens(chk)
    if (
        [t.lower() for t in src_toks] == [t.lower() for t in chk_toks]
        and _has_balanced_syntax_brackets(chk, allow_curly=allow_curly)
        and not re.search(r"\[\s*\]|\(\s*\)|/\s*/", chk)
        and src.count("(") <= chk.count("(")
        and src.count(")") <= chk.count(")")
    ):
        return chk

    word_re = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")
    src_matches = list(word_re.finditer(src))
    chk_matches = list(word_re.finditer(chk))
    if not src_matches or not chk_matches:
        return src

    src_low = [m.group(0).lower().replace("’", "'") for m in src_matches]
    chk_low = [m.group(0).lower().replace("’", "'") for m in chk_matches]
    sm = difflib.SequenceMatcher(None, src_low, chk_low)
    if sm.ratio() < 0.55:
        return src

    # 원문 자체 괄호 ( )를 센티넬 ⟨ ⟩ 로 보호
    src_gaps: list[str] = []
    prev_end = 0
    for m in src_matches:
        g = src[prev_end : m.start()].replace("(", "⟨").replace(")", "⟩")
        src_gaps.append(g)
        prev_end = m.end()
    src_gaps.append(src[prev_end:].replace("(", "⟨").replace(")", "⟩"))

    chk_gaps: list[str] = []
    c_prev = 0
    for m in chk_matches:
        chk_gaps.append(chk[c_prev : m.start()])
        c_prev = m.end()
    chk_gaps.append(chk[c_prev:])

    n_src = len(src_matches)
    gap_close: list[str] = [""] * (n_src + 1)
    gap_slash: list[bool] = [False] * (n_src + 1)
    gap_open: list[str] = [""] * (n_src + 1)

    def _split_gap_markup(g_txt: str) -> tuple[str, bool, str]:
        if "/" in g_txt:
            left_s, right_s = g_txt.split("/", 1)
            closes = "".join(re.findall(r"[\]\)]", left_s))
            opens = "".join(re.findall(r"[\[\(]", right_s))
            return closes, True, opens
        closes = "".join(re.findall(r"[\]\)]", g_txt))
        opens = "".join(re.findall(r"[\[\(]", g_txt))
        return closes, False, opens

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            continue
        block_len = i2 - i1
        _, has_sl0, op0 = _split_gap_markup(chk_gaps[j1])
        if op0 and "⟨" not in src_gaps[i1]:
            gap_open[i1] = op0
        if has_sl0 and i1 > 0:
            gap_slash[i1] = True
        for offset in range(1, block_len):
            si = i1 + offset
            cj = j1 + offset
            cl, has_sl, op = _split_gap_markup(chk_gaps[cj])
            if cl and "⟩" not in src_gaps[si]:
                gap_close[si] = cl
            if has_sl:
                gap_slash[si] = True
            if op and "⟨" not in src_gaps[si]:
                gap_open[si] = op
        cl_end, has_sl_end, _ = _split_gap_markup(chk_gaps[j2])
        if cl_end and "⟩" not in src_gaps[i2]:
            gap_close[i2] = cl_end
        if has_sl_end and i2 < n_src:
            gap_slash[i2] = True

    pieces: list[str] = []
    for idx_w, m in enumerate(src_matches):
        g = src_gaps[idx_w]
        if idx_w == 0:
            pieces.append(g + gap_open[0])
        else:
            m_punc = re.match(r"^([,.;:?!\"'”’⟩]*)(.*)$", g, flags=re.S)
            pre_p = m_punc.group(1) if m_punc else ""
            post_p = m_punc.group(2) if m_punc else g
            if gap_slash[idx_w]:
                # 원문 괄호(⟨), 따옴표(“), 대시(—) 등 post_p의 비공백 문자를 절대 유실하지 않고 슬래시 뒤에 보존
                post_nonspace = post_p.lstrip()
                mid = f" / {post_nonspace}" if post_nonspace else " / "
            else:
                mid = post_p if post_p else " "
            pieces.append(f"{gap_close[idx_w]}{pre_p}{mid}{gap_open[idx_w]}")
        pieces.append(m.group(0))

    last_g = src_gaps[n_src]
    m_end = re.match(r"^([⟩\"'”’]*)([.?!,;:]?)(.*)$", last_g, flags=re.S)
    if m_end:
        pieces.append(f"{m_end.group(1)}{gap_close[n_src]}{m_end.group(2)}{m_end.group(3)}")
    else:
        pieces.append(f"{gap_close[n_src]}{last_g}")

    balanced = _balance_added_markup_chars("".join(pieces))
    restored = balanced.replace("⟨", "(").replace("⟩", ")")
    restored = _sanitize_disallowed_syntax_braces(src, restored)
    restored = _normalize_syntax_bracket_consistency(src, restored)
    if (
        [t.lower() for t in _extract_english_word_tokens(restored)] == src_low
        and _has_balanced_syntax_brackets(restored, allow_curly=allow_curly)
        and src.count("(") <= restored.count("(")
        and src.count(")") <= restored.count(")")
    ):
        return restored
    return src


def _check_chunked_english_fidelity(source_sent: str, chunked_sent: str) -> list[str]:
    """chunked_english가 source_sent의 모든 영어 단어를 100% 그대로 보존하고 괄호 짝이 맞는지 검사한다."""
    issues: list[str] = []
    src = (source_sent or "").strip()
    chk = (chunked_sent or "").strip()
    if not src or not chk:
        return ["문장이 비어 있습니다."]

    src_toks = _extract_english_word_tokens(src)
    chk_toks = _extract_english_word_tokens(chk)
    src_low = [t.lower() for t in src_toks]
    chk_low = [t.lower() for t in chk_toks]
    if src_low != chk_low:
        sm = difflib.SequenceMatcher(None, src_low, chk_low)
        diffs: list[str] = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            s_part = " ".join(src_toks[i1:i2]) or "(없음)"
            c_part = " ".join(chk_toks[j1:j2]) or "(없음)"
            diffs.append(f"원문 '{s_part}' → 출력 '{c_part}'")
            if len(diffs) >= 3:
                break
        issues.append(f"원문 단어 변조/누락 감지 ({'; '.join(diffs)})")

    if not _has_balanced_syntax_brackets(chk):
        issues.append("대괄호 `[ ]` 또는 소괄호 `( )`의 짝이 맞지 않습니다.")

    if re.search(r"\[\s*\]|\(\s*\)|/\s*/", chk):
        issues.append("빈 괄호 `[]`/`()` 또는 중복 슬래시 `/ /`가 포함되어 있습니다.")

    return issues


def _collect_all_analysis_korean_blobs(out: EnglishPassageAnalysisOutput) -> str:
    """분석지의 한국어 텍스트 영역(주제·흐름·도식·해석)을 모두 합쳐 교차 검사용 문자열을 만든다."""
    parts: list[str] = [
        out.title or "",
        out.core_theme or "",
        out.structure_flow or "",
        out.exam_point_title or "",
        out.exam_point_note or "",
    ]
    vd = out.visual_diagram
    if vd is not None:
        parts.append(vd.title or "")
        parts.append(vd.center_label or "")
        parts.append(vd.quadrant_x_label or "")
        parts.append(vd.quadrant_y_label or "")
        parts.append(vd.takeaway or "")
        parts.extend(vd.relation_labels or [])
        for nd in vd.nodes or []:
            parts.append(nd.title or "")
            parts.append(nd.subtitle or "")
            parts.extend(nd.details or [])
        for mr in vd.matrix_rows or []:
            parts.append(mr.criterion or "")
            parts.extend(mr.values or [])
        gs = vd.graph_spec
        if gs is not None:
            parts.append(gs.x_label or "")
            parts.append(gs.y_label or "")
            parts.append(gs.highlight_point or "")
            for cv in gs.curves or []:
                parts.append(cv.label or "")
                parts.append(cv.annotation or "")
    for sa in out.sentence_analyses or []:
        parts.append(sa.korean_translation or "")
    return "\n".join(parts)


def _detect_foreign_script_contamination(text: str) -> list[str]:
    """한국어 출력 필드에 일본어(히라가나·가타카나), 키릴, 아랍, 태국, 의도치 않은 한자 등 외국 문자 혼입을 감지한다."""
    if not text:
        return []
    found: list[str] = []
    m_for = _FOREIGN_SCRIPT_RE.search(text)
    if m_for:
        st_i = max(0, m_for.start() - 6)
        en_i = min(len(text), m_for.end() + 6)
        found.append(f"외국 문자({text[st_i:en_i].strip()!r})")
    m_cjk = _UNEXPECTED_CJK_HANJA_RE.search(text)
    if m_cjk:
        found.append(f"한자/중국어({m_cjk.group(0)!r})")
    return found


def _validate_english_analysis_output(
    out: EnglishPassageAnalysisOutput,
    source_sents: Sequence[str],
    *,
    clean_passage: str = "",
    exam_context: dict[str, Any] | None = None,
) -> list[str]:
    """
    LLM 재시도가 반드시 필요한 **치명적 오류(critical issues)**만 선별해 반환한다.
    (미세 단어 변조, 괄호 짝 어긋남, 원문 따옴표 삽입, 약한 주제문 플래그 등 파이썬 후처리로
    0.01초 만에 100% 교정 가능한 항목은 재시도 트리거에서 제외해 로딩 시간과 토큰 소모를 최소화한다.)
    """
    _ = clean_passage
    critical_issues: list[str] = []
    ctx = exam_context or {}
    sents = list(source_sents or [])
    analyses = sorted(list(out.sentence_analyses or []), key=lambda x: x.sentence_num)

    # 1. [치명] 문장 수 불일치 (정상 문장 누락·병합·초과)
    if sents and len(analyses) != len(sents):
        critical_issues.append(
            f"문장 수 불일치: 원문은 총 {len(sents)}문장이나 분석 결과는 {len(analyses)}문장입니다. "
            "1번부터 마지막 문장까지 단 한 문장도 누락·병합·추가하지 말고 정확히 1:1로 분석하세요."
        )

    # 2. [치명] 토큰 투영으로 복구 불가능한 수준의 심각한 문장 변조/환각 (유사도 < 0.55) 또는 해석 누락
    compare_n = min(len(sents), len(analyses))
    for idx in range(compare_n):
        src_toks = [t.lower() for t in _extract_english_word_tokens(sents[idx])]
        chk_toks = [t.lower() for t in _extract_english_word_tokens(analyses[idx].chunked_english)]
        if src_toks:
            ratio = difflib.SequenceMatcher(None, src_toks, chk_toks).ratio()
            if ratio < 0.55:
                critical_issues.append(
                    f"{idx + 1}번 문장이 원문과 심각하게 다릅니다(유사도 {ratio:.2f}). 원문 그대로 1:1 분석하세요."
                )
                break
        if not (analyses[idx].korean_translation or "").strip():
            critical_issues.append(f"{idx + 1}번 문장의 한국어 해석이 비어 있습니다.")
            break

    # 3. [치명] 한국어 출력 필드 외국어(일본어 가나·키릴·아랍·의도치 않은 중국어 한자 등) 혼입 차단
    kor_blob = _collect_all_analysis_korean_blobs(out)
    foreign_hits = _detect_foreign_script_contamination(kor_blob)
    if foreign_hits:
        critical_issues.append(
            f"한국어 출력 영역에 비한글 외국 문자({', '.join(foreign_hits)})가 혼입되었습니다. "
            "모든 해석·주제·논리 흐름·도식은 순수 한국어(필요 시 원문 영어 용어만 병기)로 작성하세요."
        )

    # 3-2. [치명] 단독 부정('not merely / not simply / not just' + 'but' 없음)을 '~일 뿐만 아니라'로 정반대 오역하는 오류 차단
    for idx in range(compare_n):
        src_s_low = sents[idx].lower()
        kor_t = (analyses[idx].korean_translation or "").strip()
        if (
            re.search(r"\bnot\s+(?:merely|simply|just)\b", src_s_low)
            and not re.search(r"\bbut\b", src_s_low)
            and "뿐만 아니라" in kor_t
        ):
            critical_issues.append(
                f"{idx + 1}번 문장의 `not merely/simply/just`는 뒤에 `but`이 없는 단독 부정('단순히 ~인 것이 아니다')입니다. "
                "`~일 뿐만 아니라`로 정반대 오역하지 말고 부정 의미를 살려 정확히 번역하세요."
            )
            break

    # 4. [범용] 문제지 공식 각주(* word: 뜻) 뜻 반영 여부 검증
    footnotes = list(ctx.get("footnotes") or [])
    for fn in footnotes:
        if ":" not in fn:
            continue
        eng_term, kor_mean = [x.strip() for x in fn.split(":", 1)]
        eng_stem = re.sub(r"(?:e|s|es|ed|ing)$", "", eng_term.split()[0].lower())
        if len(eng_stem) < 4:
            eng_stem = eng_term.split()[0].lower()
        kor_tokens = [
            tok for tok in re.findall(r"[가-힣]{2,}", kor_mean)
            if tok not in ("하다", "되다", "이다", "있는", "없는", "것을", "으로", "에서")
        ]
        if not kor_tokens:
            continue
        primary_kor = kor_tokens[0]
        for idx in range(compare_n):
            if eng_stem in sents[idx].lower():
                if primary_kor not in (analyses[idx].korean_translation or ""):
                    critical_issues.append(
                        f"각주 뜻 미반영 ({idx + 1}번 문장): 문제지 각주 `{eng_term}: {kor_mean}`의 뜻('{primary_kor}')을 "
                        f"{idx + 1}번 문장 해석에 정확히 반영하세요."
                    )
                break

    return critical_issues


def _enforce_english_analysis_token_guard(
    out: EnglishPassageAnalysisOutput,
    source_sents: Sequence[str] | None = None,
) -> EnglishPassageAnalysisOutput:
    """
    최종 저장·렌더링 직전 모든 출력 경로에서 호출하는 최종 토큰 일치 & 괄호 균형 가드.
    chunked_english에서 구문 기호(/, [], ())를 제거한 토큰 시퀀스가 원문 문장과 단 1단어라도 다르거나
    괄호 짝이 맞지 않거나 불허용 중괄호({})가 있으면 강제로 원문 무손실 투영/폴백하여 변조된 문장이 절대 통과하지 못하게 한다.
    """
    sents = list(source_sents or [])
    for idx, sa in enumerate(out.sentence_analyses or []):
        chk = (sa.chunked_english or "").strip()
        if idx < len(sents) and sents[idx].strip():
            src_s = sents[idx].strip()
            allow_curly = "{" in src_s or "}" in src_s
            src_low = [t.lower() for t in _extract_english_word_tokens(src_s)]
            chk_clean = _sanitize_disallowed_syntax_braces(src_s, chk)
            chk_clean = _strip_spurious_quotes_against_source(src_s, chk_clean)
            chk_low = [t.lower() for t in _extract_english_word_tokens(chk_clean)]
            if (
                chk_low != src_low
                or not _has_balanced_syntax_brackets(chk_clean, allow_curly=allow_curly)
                or src_s.count("(") > chk_clean.count("(")
                or src_s.count(")") > chk_clean.count(")")
            ):
                chk_clean = _project_chunk_markup_onto_source(src_s, chk_clean)
            # 최종 2차 확인: 그래도 1단어라도 다르거나 괄호 불균형이면 원문 그대로 강제 폴백
            if (
                [t.lower() for t in _extract_english_word_tokens(chk_clean)] != src_low
                or not _has_balanced_syntax_brackets(chk_clean, allow_curly=allow_curly)
            ):
                chk_clean = src_s
            sa.chunked_english = chk_clean
        else:
            # 원문 문장 리스트가 없는 렌더링 폴백 경로에서도 최소한 중괄호 제거 + 괄호 짝 무결성 보장
            if chk:
                chk_clean = _sanitize_disallowed_syntax_braces("", chk)
                if not _has_balanced_syntax_brackets(chk_clean, allow_curly=False):
                    chk_clean = _balance_added_markup_chars(chk_clean)
                if not _has_balanced_syntax_brackets(chk_clean, allow_curly=False):
                    chk_clean = re.sub(r"[\[\]\(\){}]", "", chk_clean)
                    chk_clean = re.sub(r"[ \t]{2,}", " ", chk_clean).strip()
                sa.chunked_english = chk_clean
    return out


def _repair_english_analysis_output(
    out: EnglishPassageAnalysisOutput,
    source_sents: Sequence[str],
    *,
    clean_passage: str = "",
    orig_question_type: str = "",
    exam_context: dict[str, Any] | None = None,
) -> EnglishPassageAnalysisOutput:
    """
    결정론적 파이썬 후처리 파이프라인:
    1) 문장 1..N 정확 매칭 + 원문 따옴표 제거 + 구문 괄호 일관성/짝 복구 + 토큰 일치 최종 가드
    2) 단순 전환 단문에 붙은 약한 ★ 주제문 자동 해제
    3) 문항 유형별 기출 포인트(빈칸 원위치·무관한 문장·글의 순서·문장 삽입) 및 동적 제목 보강
    4) 공식 각주 목록 보강
    """
    _ = clean_passage
    ctx = exam_context or {}
    sents = list(source_sents or [])
    raw_list = sorted(list(out.sentence_analyses or []), key=lambda x: x.sentence_num)

    # 1. 문장 1..N 정확 매칭 및 원문 100% 무손실 투영
    repaired_sents: list[EnglishSentenceAnalysisItem] = []
    if sents:
        used_indices: set[int] = set()
        for idx_s, src_s in enumerate(sents):
            src_tok_set = set(t.lower() for t in _extract_english_word_tokens(src_s))
            matched_sa: EnglishSentenceAnalysisItem | None = None
            if idx_s < len(raw_list) and idx_s not in used_indices:
                cand_tok_set = set(t.lower() for t in _extract_english_word_tokens(raw_list[idx_s].chunked_english))
                jac = len(src_tok_set & cand_tok_set) / max(1, len(src_tok_set | cand_tok_set))
                if jac >= 0.45:
                    matched_sa = raw_list[idx_s]
                    used_indices.add(idx_s)
            if matched_sa is None:
                best_j = -1
                best_jac = 0.0
                for j, cand in enumerate(raw_list):
                    if j in used_indices:
                        continue
                    cand_tok_set = set(t.lower() for t in _extract_english_word_tokens(cand.chunked_english))
                    jac = len(src_tok_set & cand_tok_set) / max(1, len(src_tok_set | cand_tok_set))
                    if jac > best_jac:
                        best_jac = jac
                        best_j = j
                if best_j != -1 and best_jac >= 0.45:
                    matched_sa = raw_list[best_j]
                    used_indices.add(best_j)

            if matched_sa is not None:
                safe_chunk = _project_chunk_markup_onto_source(src_s, matched_sa.chunked_english)
                repaired_sents.append(
                    EnglishSentenceAnalysisItem(
                        sentence_num=idx_s + 1,
                        is_topic_sentence=bool(matched_sa.is_topic_sentence),
                        chunked_english=safe_chunk,
                        korean_translation=normalize_english_korean_translation(matched_sa.korean_translation),
                        syntax_notes=[],
                        original_blank_phrase=(matched_sa.original_blank_phrase or "").strip(),
                        is_inserted_sentence=bool(getattr(matched_sa, "is_inserted_sentence", False)),
                    )
                )
            else:
                repaired_sents.append(
                    EnglishSentenceAnalysisItem(
                        sentence_num=idx_s + 1,
                        is_topic_sentence=False,
                        chunked_english=src_s,
                        korean_translation="",
                        syntax_notes=[],
                        original_blank_phrase="",
                        is_inserted_sentence=False,
                    )
                )
    else:
        for idx_r, sa in enumerate(raw_list, start=1):
            sa.sentence_num = idx_r
            sa.korean_translation = normalize_english_korean_translation(sa.korean_translation)
            sa.syntax_notes = []
            repaired_sents.append(sa)

    # 2. 단순 전환 단문에 붙은 ★ 주제문 정리 (다른 실질 주제문이 이미 있을 때 자동 해제)
    strong_topics = [
        sa for sa in repaired_sents
        if sa.is_topic_sentence
        and len(_extract_english_word_tokens(sa.chunked_english)) > 6
        and not (_WEAK_TOPIC_SENT_RE.search(sa.chunked_english.strip()) and len(_extract_english_word_tokens(sa.chunked_english)) <= 9)
    ]
    if strong_topics:
        for sa in repaired_sents:
            if not sa.is_topic_sentence:
                continue
            w_cnt = len(_extract_english_word_tokens(sa.chunked_english))
            if w_cnt <= 6 or (_WEAK_TOPIC_SENT_RE.search(sa.chunked_english.strip()) and w_cnt <= 9):
                sa.is_topic_sentence = False

    out.sentence_analyses = repaired_sents

    # 3. 문항 유형별 기출 포인트 결정론적 보강 및 교차 배지 오염 차단 (빈칸·무관한 문장·글의 순서·문장 삽입)
    qt_str = (orig_question_type or "").strip()
    blank_phrase = (ctx.get("original_blank_phrase") or "").strip()
    ins_sent = (ctx.get("inserted_sentence") or "").strip()
    is_blank_q = ("빈칸" in qt_str) or bool(blank_phrase)
    is_insert_q = ("삽입" in qt_str) or bool(ins_sent)

    if not is_blank_q:
        for sa in out.sentence_analyses:
            sa.original_blank_phrase = ""
    elif blank_phrase:
        for sa in out.sentence_analyses:
            sa.original_blank_phrase = ""
        bp_toks = [t.lower() for t in _extract_english_word_tokens(blank_phrase)]
        blank_sent_num = 0
        for idx_s, sa in enumerate(out.sentence_analyses):
            s_ref = sents[idx_s] if idx_s < len(sents) else sa.chunked_english
            s_toks = [t.lower() for t in _extract_english_word_tokens(s_ref)]
            matched_bp = False
            if bp_toks and len(bp_toks) <= len(s_toks):
                for start_i in range(len(s_toks) - len(bp_toks) + 1):
                    if s_toks[start_i : start_i + len(bp_toks)] == bp_toks:
                        matched_bp = True
                        break
            if matched_bp:
                sa.original_blank_phrase = blank_phrase
                blank_sent_num = sa.sentence_num
                break
        if blank_sent_num and blank_phrase.lower() not in (out.exam_point_note or "").lower():
            c_badge = _CIRCLED_NUM_LIST[blank_sent_num - 1] if 1 <= blank_sent_num <= 5 else f"{blank_sent_num}번"
            prefix = f"[기출 빈칸 원위치 ({c_badge} 문장)] {blank_phrase}"
            out.exam_point_note = f"{prefix}\n{out.exam_point_note}".strip() if out.exam_point_note else prefix

    irrel_sent = (ctx.get("irrelevant_sentence") or "").strip()
    irrel_sym = (ctx.get("irrelevant_answer_sym") or "").strip()
    if irrel_sent:
        if irrel_sent[:35].lower() not in (out.exam_point_note or "").lower():
            prefix = f"[제거된 무관한 문장 (정답 {irrel_sym or '해당 문항'})] {irrel_sent}"
            out.exam_point_note = f"{prefix}\n{out.exam_point_note}".strip() if out.exam_point_note else prefix

    order_seq = (ctx.get("order_answer_seq") or "").strip()
    if order_seq:
        if order_seq not in (out.exam_point_note or ""):
            prefix = f"[정답 순서] {order_seq}"
            out.exam_point_note = f"{prefix}\n{out.exam_point_note}".strip() if out.exam_point_note else prefix

    ins_sym = (ctx.get("inserted_answer_sym") or "").strip()
    if not is_insert_q:
        for sa in out.sentence_analyses:
            sa.is_inserted_sentence = False
    elif ins_sent:
        for sa in out.sentence_analyses:
            sa.is_inserted_sentence = False
        ins_toks = set(t.lower() for t in _extract_english_word_tokens(ins_sent))
        for idx_s, sa in enumerate(out.sentence_analyses):
            s_ref = sents[idx_s] if idx_s < len(sents) else sa.chunked_english
            s_toks = set(t.lower() for t in _extract_english_word_tokens(s_ref))
            if ins_toks and s_toks and len(ins_toks & s_toks) / max(1, len(ins_toks | s_toks)) >= 0.72:
                sa.is_inserted_sentence = True
                break
        if ins_sent[:35].lower() not in (out.exam_point_note or "").lower():
            pos_txt = f"정답 {ins_sym}번 위치" if ins_sym else "정답 위치"
            prefix = f"[삽입 문장 ({pos_txt})] {ins_sent}"
            out.exam_point_note = f"{prefix}\n{out.exam_point_note}".strip() if out.exam_point_note else prefix

    # 동적 기출 포인트 제목 확정
    if (out.exam_point_note or "").strip():
        out.exam_point_title = _resolve_english_exam_point_title(orig_question_type, out.exam_point_note)
    else:
        out.exam_point_title = ""

    # 4. 공식 각주 어휘 보강
    extracted_fn = list(ctx.get("footnotes") or [])
    if extracted_fn:
        out.footnotes = extracted_fn

    # 5. 최종 토큰 일치 & 괄호 균형 가드 강제 적용
    out = _enforce_english_analysis_token_guard(out, sents)
    return out


def _validate_english_summary_diagram_output(
    out: EnglishPassageSummaryDiagramOutput,
    *,
    require_diagram: bool = True,
) -> list[str]:
    """
    AI가 생성한 영어 지문 제목 · 핵심 주제 · 논리 전개 흐름 · 구조 도식 스펙을 검증한다.
    (문장별 원문/해석은 AI가 생성하지 않으므로 검증 대상에서 제외)
    """
    issues: list[str] = []
    if not (out.title or "").strip():
        issues.append("`title`(지문 핵심 제목)이 비어 있습니다.")
    if not (out.core_theme or "").strip():
        issues.append("`core_theme`(핵심 주제)이 비어 있습니다.")
    if not (out.structure_flow or "").strip():
        issues.append("`structure_flow`(논리 전개 흐름)이 비어 있습니다.")

    vd = out.visual_diagram
    if require_diagram:
        has_diag = vd is not None and bool(
            getattr(vd, "nodes", None)
            or getattr(vd, "graph_spec", None)
            or getattr(vd, "emotion_points", None)
        )
        if not has_diag:
            issues.append("`visual_diagram`(이해용 구조 도식)이 비어 있습니다. nodes 또는 graph_spec을 채워 주세요.")

    text_parts: list[str] = [
        out.title or "",
        out.core_theme or "",
        out.structure_flow or "",
    ]
    if vd is not None:
        text_parts.extend(
            [
                vd.title or "",
                vd.center_label or "",
                vd.quadrant_x_label or "",
                vd.quadrant_y_label or "",
                vd.takeaway or "",
                *(vd.relation_labels or []),
            ]
        )
        for nd in vd.nodes or []:
            text_parts.append(nd.title or "")
            text_parts.append(nd.subtitle or "")
            text_parts.extend(nd.details or [])
        for mr in vd.matrix_rows or []:
            text_parts.append(mr.criterion or "")
            text_parts.extend(mr.values or [])
        gs = vd.graph_spec
        if gs is not None:
            text_parts.extend([gs.x_label or "", gs.y_label or "", gs.highlight_point or ""])
            for cv in gs.curves or []:
                text_parts.extend([cv.label or "", cv.annotation or ""])

    foreign_hits = _detect_foreign_script_contamination("\n".join(text_parts))
    if foreign_hits:
        issues.append(
            f"한국어 출력 영역에 비한글 외국 문자({', '.join(foreign_hits)})가 혼입되었습니다. "
            "모든 제목·주제·논리 흐름·도식은 순수 한국어(필요 시 원문 영어 용어만 병기)로 작성하세요."
        )
    return issues


