from __future__ import annotations

import difflib
import random
import re
from typing import Any, Sequence

from src.gemini.core import (
    _CIRCLED_DIGIT_TO_NUM,
    _normalize_extracted_answer,
)
from src.models import (
    EnglishPassageAnalysisOutput,
    EnglishSentenceAnalysisItem,
    RewrittenQuestion,
)
from src.text_formatter import (
    normalize_english_exam_text,
    normalize_english_korean_translation,
    strip_english_embedded_choices,
)


def _build_english_transform_rule(
    target_type: str,
    blank_unit: str = "구 (3~6단어)",
    vocab_style: str = "밑줄형 (①~⑤)",
) -> tuple[str, str]:
    """입력 토큰 절감을 위해 지정된 target_type 1개 유형의 핵심 규칙만 압축해 반환한다."""
    t = (target_type or "").strip()

    if "함축" in t:
        return (
            "함축적 의미 추론",
            "- 발문: `밑줄 친 부분이 다음 글에서 의미하는 바로 가장 적절한 것은?`\n"
            "- 출력 절감 규칙: `rewritten_stem`에 지문 전체를 쓰지 말고, 원문에서 비유·핵심 의미를 담은 **유일한 연속 구절 1개(target_phrase)**만 원문 철자 그대로 반환하세요(시스템이 `<u>...</u>` 자동 치환).\n"
            "- 선지: 밑줄 표현의 문맥상 함축 의미를 담은 **영어 선지 5개**(번호 제외).",
        )
    if "요지" in t:
        return (
            "요지 파악",
            "- 발문: `다음 글의 요지로 가장 적절한 것은?`\n"
            "- 출력 절감 규칙: 지문은 원문 그대로 쓰이므로 `rewritten_stem`은 **빈 문자열 `\"\"`**로 반환하세요.\n"
            "- 선지: 평가원 표준 **한국어 완결 문장 선지 5개**(번호 제외).",
        )
    if "주제" in t:
        return (
            "주제 파악",
            "- 발문: `다음 글의 주제로 가장 적절한 것은?`\n"
            "- 출력 절감 규칙: 지문은 원문 그대로 쓰이므로 `rewritten_stem`은 **빈 문자열 `\"\"`**로 반환하세요.\n"
            "- 선지: 평가원 표준 **영어 명사구 선지 5개**(번호 제외).",
        )
    if "제목" in t:
        return (
            "제목 추론",
            "- 발문: `다음 글의 제목으로 가장 적절한 것은?`\n"
            "- 출력 절감 규칙: 지문은 원문 그대로 쓰이므로 `rewritten_stem`은 **빈 문자열 `\"\"`**로 반환하세요.\n"
            "- 선지: 평가원 표준 **영어 제목 선지 5개**(번호 제외, 첫 글자 대문자 타이틀 형식).",
        )
    if "어법" in t:
        return (
            "어법상 틀린 것 (밑줄 ①~⑤)",
            "- 발문: `다음 글의 밑줄 친 부분 중, 어법상 틀린 것은?`\n"
            "- 출력 절감 규칙: `rewritten_stem`에 지문 전체를 쓰지 말고, 원문 등장 순서대로 문법 포인트 5곳을 아래 5줄 형식으로만 반환하세요(시스템이 원문에 `① <u>...</u>`~`⑤ <u>...</u>`를 자동 조립).\n"
            "  `1: 원문구절1`\n  `2: 원문구절2`\n  `3: 원문구절3 -> 틀린어법구절3`\n  `4: 원문구절4`\n  `5: 원문구절5`\n"
            "- 규칙: 5개 구절은 원문에서 유일하게 찾을 수 있도록 1~3단어로 적고, 정확히 1줄(정답)에만 `-> 틀린어법`을 적어 실제 비문(수일치·분사 능/수동·관계사/접속사 what/that·병렬·대동사 do/be 중 1가지)으로 변형. 나머지 4줄은 화살표 없이 원문 그대로 유지.\n"
            "- 금지: 원문 정상 표현을 틀렸다고 우기기, 해설 자체 모순('타당하다/적절하다'), 의미만 반대인 반의어 교체, 단순 시제·진행형(`is occurring`↔`occurs`, `signaled`↔`signals`) 변형 절대 금지.\n"
            "- 선지: `[\"①\", \"②\", \"③\", \"④\", \"⑤\"]` 5개.",
        )
    if "어휘" in t:
        if "선택형" in vocab_style or "(A)" in vocab_style:
            return (
                "문맥상 어휘 적절성 (선택형 (A)(B)(C))",
                "- 발문: `(A), (B), (C)의 각 네모 안에서 문맥에 맞는 낱말로 가장 적절한 것은?`\n"
                "- 규칙: 기존 본문 내 핵심 어휘 3곳을 `(A) [w1 / w2]`, `(B) [w3 / w4]`, `(C) [w5 / w6]`(원문 단어와 반의어 쌍)으로 표시. 지문 끝에 새 문장 추가·어색한 연어(`severe encouragement`)·어법 변형(`makes`↔`making`) 절대 금지. 반어법·풍자·부정어(`not`/`never`/`hardly`) 절의 단어는 교체 대상에서 제외.\n"
                "- 선지: `\"(A) w1 — (B) w3 — (C) w5\"` 형식 5개.",
            )
        return (
            "문맥상 어휘 적절성 (밑줄 ①~⑤)",
            "- 발문: `다음 글의 밑줄 친 부분 중, 문맥상 낱말의 쓰임이 적절하지 않은 것은?`\n"
            "- 출력 절감 규칙: `rewritten_stem`에 지문 전체를 쓰지 말고, 원문 등장 순서대로 핵심 어휘 5곳을 아래 5줄 형식으로만 반환하세요(시스템이 원문에 `① <u>...</u>`~`⑤ <u>...</u>`를 자동 조립).\n"
            "  `1: 원문단어1`\n  `2: 원문단어2 -> 반의어2`\n  `3: 원문단어3`\n  `4: 원문단어4`\n  `5: 원문단어5`\n"
            "- 규칙: 정확히 1줄(정답)에만 `-> 반의어`를 적어 문법 형태·품사가 100% 동일한 자연스러운 반의어로 교체. 나머지 4줄은 화살표 없이 원문 단어 그대로 유지.\n"
            "- 금지: 어법 오류(`makes`→`making` 등)·어색한 연어 생성 금지. 반어법·풍자 표현이나 부정어(`not`/`never`/`hardly`)가 걸린 절의 단어는 반의어 교체 대상에서 제외.\n"
            "- 선지: 밑줄 친 5개 어휘 리스트(정답 위치는 교체된 반의어).",
        )
    if "빈칸" in t:
        b_u = (blank_unit or "").strip()
        if b_u.startswith("절") or ("절(" in b_u or "절 (" in b_u):
            unit_desc = "주어+술어 포함 **절(6~12단어)**"
            short_u = "절"
        elif b_u.startswith("단어") or ("단어" in b_u and not b_u.startswith("구")):
            unit_desc = "핵심 **단어(1~2단어)**"
            short_u = "단어"
        else:
            unit_desc = "핵심 **구(3~6단어)**"
            short_u = "구"
        return (
            f"빈칸 추론 ({short_u})",
            "- 발문: `다음 빈칸에 들어갈 말로 가장 적절한 것은?`\n"
            f"- 출력 절감 규칙: `rewritten_stem`에 지문 전체를 쓰지 말고, 주제문/핵심 구간에서 빈칸으로 뚫을 **{unit_desc} 분량의 유일한 원문 구절 1개(target_phrase)**만 원문 철자 그대로 반환하세요(시스템이 `___________`로 자동 치환).\n"
            "- 제약: 빈칸 앞뒤 2단어와 정답 선지 시작/끝 2단어 중복 금지, 문장 내 동사 통째 삭제 금지, `(①)~(⑤)` 기호 금지.\n"
            f"- 선지: {unit_desc} 길이·품사에 맞는 **영어 선지 5개**(정답 1개 + 매력적 오답 4개).",
        )
    if "무관" in t:
        return (
            "전체 흐름과 무관한 문장",
            "- 발문: `다음 글에서 전체 흐름과 관계없는 문장은?`\n"
            "- 출력 절감 규칙: `rewritten_stem`에 지문 전체를 쓰지 말고, 아래 2줄 형식으로만 반환하세요(시스템이 원문 사이에 `①~⑤`와 새 문장을 자동 조립).\n"
            "  `INSERT_AT: 3` (2~4 중 삽입할 번호 1개)\n"
            "  `SENTENCE: <원문 키워드만 공유하고 전체 논리 흐름과 어긋나는 새 영어 문장 1개>`\n"
            "- 선지: `[\"①\", \"②\", \"③\", \"④\", \"⑤\"]` 5개.",
        )
    if "순서" in t:
        return (
            "글의 순서 ((A)-(B)-(C))",
            "- 발문: `주어진 글 다음에 이어질 글의 순서로 가장 적절한 것은?`\n"
            "- 출력 절감 규칙: 원문 각 문장에 `[S0]`, `[S1]`, ... 번호가 붙어 있습니다. `rewritten_stem`에 지문 전체를 쓰지 말고, `[주어진 글]`과 3개 단락의 시작 문장 번호 3개만 아래 1줄 형식으로 반환하세요(시스템이 원문을 자르고 오순 셔플과 정답을 자동 확정).\n"
            "  `SPLIT: s1, s2, s3` (예: `SPLIT: 1, 3, 5` → 주어진글=S0, 1번덩어리=S1~S2, 2번덩어리=S3~S4, 3번덩어리=S5~끝; `0 < s1 < s2 < s3 < 전체문장수`)\n"
            "- 선지: `[\"(A) - (C) - (B)\", \"(B) - (A) - (C)\", \"(B) - (C) - (A)\", \"(C) - (A) - (B)\", \"(C) - (B) - (A)\"]` 5개.",
        )
    if "삽입" in t:
        return (
            "주어진 문장 삽입",
            "- 발문: `글의 흐름으로 보아, 주어진 문장이 들어가기에 가장 적절한 곳은?`\n"
            "- 출력 절감 규칙: 원문이 6문장 이상이면 `rewritten_stem`에 지문 전체를 쓰지 말고 원문에서 뽑아낼 **핵심 문장 1개만 원문 그대로** 반환하세요(시스템이 `( ① )~( ⑤ )`와 정답을 자동 배치). 5문장 이하일 때만 `[주어진 문장]`과 본문 `( ① )~( ⑤ )`를 출력하세요.\n"
            "- 선지: `[\"①\", \"②\", \"③\", \"④\", \"⑤\"]` 5개.",
        )
    # 기본: 요약문 완성
    return (
        "요약문 완성 ((A), (B))",
        "- 발문: `다음 글의 내용을 한 문장으로 요약하고자 한다. 빈칸 (A), (B)에 들어갈 말로 가장 적절한 것은?`\n"
        "- 출력 절감 규칙: `rewritten_stem`에 지문 본문·`[선지]`·①~⑤를 절대 쓰지 말고, 오직 **`(A) ________`와 `(B) ________`가 포함된 영어 요약문 1문장만** 반환하세요(원문은 시스템이 자동 결합).\n"
        "- 제약: `(A)`, `(B)` 정답 어휘는 원문과 동일 단어/동일 어근 노출 금지(패러프레이징 필수).\n"
        "- 선지: `\"(A) word1 — (B) word2\"` 형식 5개.",
    )


_AI_META_MONOLOGUE_RE = re.compile(
    r"(?:wait,\s*let\s*me|let['’]?s\s+(?:rewrite|apply|check|change|fix|try|revise|make|replace|keep)|"
    r"keep\s+the\s+original\s+text|actually,\s*let['’]?s|i\s+will\s+(?:rewrite|replace|use)|"
    r"here\s+is\s+the\s+revised|note:\s*i\s+(?:changed|replaced)|"
    r"\[\s*정답[^\]]*\]|예시\s*교정|변형\s*필요|정답\s*처리|수정\s*필요|비문으로\s*만듦)",
    re.IGNORECASE,
)
_DANGLING_TO_END_RE = re.compile(
    r"\b(?:has|have|had|want|wants|wanted|able|unable|need|needs|needed|"
    r"ought|used|going|supposed|expected|likely|unlikely|tend|tends|tended|"
    r"try|tries|tried|fail|fails|failed|seem|seems|seemed|appear|appears|appeared)\s+to\s*\.",
    re.IGNORECASE,
)
_ORDER_SEQ_RE = re.compile(
    r"\(?\s*([ABC])\s*\)?\s*(?:-|→|->|–|—)\s*\(?\s*([ABC])\s*\)?\s*(?:-|→|->|–|—)\s*\(?\s*([ABC])\s*\)?"
)
_SENT_ABBREV_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Mt|vs|etc|e\.g|i\.e|U\.S|U\.K|No|Fig|Vol|p|pp)\.",
    re.IGNORECASE,
)


def _preprocess_english_source_passage(passage: str) -> str:
    """변형 출제 전 원문 지문에 남아 있는 기존 기출 기호((A)(B)(C), (①)~(⑤), <u> 등)를 제거해 순수 영문 지문으로 정규화한다."""
    txt = (passage or "").strip()
    if not txt:
        return ""
    txt = re.sub(r"</?u\b[^>]*>", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"[ \t]*\(\s*[①②③④⑤]\s*\)[ \t]*", " ", txt)
    txt = re.sub(r"[ \t]*[①②③④⑤][ \t]*", " ", txt)
    # 문장/단락 시작의 기출 순서 라벨 (A), (B), (C), (D) 제거 (문맥상 괄호 약어는 보존)
    txt = re.sub(r"(?:^|(?<=\n)|(?<=[.?!\"'”’]\s))\(\s*[ABCD]\s*\)\s*", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    txt = re.sub(r"[ \t]{2,}", " ", txt)
    return txt.strip()


_COMMON_ENG_VERBS: frozenset[str] = frozenset(
    {
        "is", "are", "was", "were", "am", "be", "been", "being",
        "has", "have", "had", "do", "does", "did",
        "can", "could", "will", "would", "shall", "should", "may", "might", "must",
        "make", "makes", "made", "take", "takes", "took", "give", "gives", "gave",
        "show", "shows", "showed", "get", "gets", "got", "go", "goes", "went",
        "come", "comes", "came", "see", "sees", "saw", "know", "knows", "knew",
        "think", "thinks", "thought", "use", "uses", "used", "find", "finds", "found",
        "tell", "tells", "told", "ask", "asks", "asked", "work", "works", "worked",
        "seem", "seems", "seemed", "feel", "feels", "felt", "try", "tries", "tried",
        "leave", "leaves", "left", "call", "calls", "called", "keep", "keeps", "kept",
        "let", "lets", "begin", "begins", "began", "help", "helps", "helped",
        "turn", "turns", "turned", "start", "starts", "started", "play", "plays", "played",
        "run", "runs", "ran", "move", "moves", "moved", "like", "likes", "liked",
        "live", "lives", "lived", "believe", "believes", "believed", "hold", "holds", "held",
        "bring", "brings", "brought", "happen", "happens", "happened", "write", "writes", "wrote",
        "provide", "provides", "provided", "sit", "sits", "sat", "stand", "stands", "stood",
        "lose", "loses", "lost", "pay", "pays", "paid", "meet", "meets", "met",
        "include", "includes", "included", "continue", "continues", "continued",
        "learn", "learns", "learned", "change", "changes", "changed", "lead", "leads", "led",
        "understand", "understands", "understood", "watch", "watches", "watched",
        "follow", "follows", "followed", "stop", "stops", "stopped", "create", "creates", "created",
        "speak", "speaks", "spoke", "read", "reads", "allow", "allows", "allowed",
        "add", "adds", "added", "spend", "spends", "spent", "grow", "grows", "grew",
        "open", "opens", "opened", "offer", "offers", "offered", "remember", "remembers",
        "consider", "considers", "considered", "appear", "appears", "appeared",
        "serve", "serves", "served", "send", "sends", "sent", "expect", "expects", "expected",
        "build", "builds", "built", "stay", "stays", "stayed", "fall", "falls", "fell",
        "reach", "reaches", "reached", "remain", "remains", "remained", "suggest", "suggests",
        "raise", "raises", "raised", "pass", "passes", "passed", "require", "requires", "required",
        "report", "reports", "reported", "decide", "decides", "decided", "mean", "means", "meant",
        "become", "becomes", "became", "cause", "causes", "caused", "enable", "enables", "enabled",
        "reduce", "reduces", "reduced", "increase", "increases", "increased", "improve", "improves",
        "affect", "affects", "affected", "depend", "depends", "depended", "tend", "tends", "tended",
        "fail", "fails", "failed", "exist", "exists", "existed", "occur", "occurs", "occurred",
        "need", "needs", "needed", "want", "wants", "wanted", "aim", "aims", "aimed",
        "reveal", "reveals", "revealed", "explain", "explains", "explained",
        "cannot", "can't", "don't", "doesn't", "didn't", "won't", "wouldn't", "isn't", "aren't",
    }
)


def _sentence_has_english_verb(sentence_without_blank: str) -> bool:
    """빈칸을 제외한 문장에 영문 동사/조동사가 최소 1개 이상 남아있는지 판별한다."""
    toks = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", (sentence_without_blank or "").lower())
    for tok in toks:
        if tok in _COMMON_ENG_VERBS:
            return True
        if len(tok) >= 5 and tok.endswith(("ed", "ize", "ise", "ify", "ate")):
            return True
        if len(tok) >= 5 and tok.endswith("ing") and tok not in ("thing", "morning", "evening", "nothing", "something", "anything", "everything", "meaning", "feeling", "during"):
            return True
    return False


def _is_order_abc_canonical(stem: str, source_passage: str) -> bool:
    """(A), (B), (C) 단락이 원문 등장 순서((A)-(B)-(C) 정순) 그대로 배열되었는지 판별한다."""
    src_sents = _split_english_sentences(source_passage)
    if len(src_sents) < 3:
        return False

    after_given = re.split(r"\[주어진\s*글\]", stem or "", maxsplit=1)[-1]
    parts = re.split(r"(?:^|\n|\s)\(([ABC])\)\s+", after_given)
    chunk_map: dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):
        lbl = parts[i]
        txt = parts[i + 1].strip()
        if lbl in ("A", "B", "C") and txt:
            chunk_map[lbl] = txt

    if len(chunk_map) != 3:
        return False

    src_tok_sets = [set(re.findall(r"[a-z0-9]+", s.lower())) for s in src_sents]
    matched_idx: dict[str, int] = {}
    for lbl in ("A", "B", "C"):
        c_sents = _split_english_sentences(chunk_map[lbl])
        first_s = c_sents[0] if c_sents else chunk_map[lbl]
        c_set = set(re.findall(r"[a-z0-9]+", first_s.lower()))
        if not c_set:
            return False
        best_i = -1
        best_sc = 0.0
        for i, s_set in enumerate(src_tok_sets):
            if not s_set:
                continue
            sc = len(c_set & s_set) / max(1, len(c_set | s_set))
            if sc > best_sc:
                best_sc = sc
                best_i = i
        if best_i == -1 or best_sc < 0.45:
            return False
        matched_idx[lbl] = best_i

    return matched_idx["A"] < matched_idx["B"] < matched_idx["C"]


def _split_english_sentences(text: str) -> list[str]:
    """약어·소수점을 보존하면서 영어 지문을 문장 단위 리스트로 분할한다."""
    cleaned_lines: list[str] = []
    for raw_ln in (text or "").splitlines():
        ln = raw_ln.strip()
        if not ln:
            continue
        if re.search(r"[가-힣]", ln) and not re.search(r"[A-Za-z]{4,}", ln):
            continue
        if ln.startswith("*") and ":" in ln:
            continue
        cleaned_lines.append(ln)
    joined = re.sub(r"\s+", " ", " ".join(cleaned_lines)).strip()
    if not joined:
        return []

    stashed: list[str] = []

    def _stash(m: re.Match[str]) -> str:
        idx = len(stashed)
        stashed.append(m.group(0))
        return f"@@SENTTOK_{idx}@@"

    masked = _SENT_ABBREV_RE.sub(_stash, joined)
    masked = re.sub(r"\b\d+\.\d+\b", _stash, masked)
    raw_sents = re.split(
        r"(?<=[.?!][\"'”’\)])\s+(?=[A-Z\"'“‘\(])|(?<=[.?!])\s+(?=[A-Z\"'“‘\(])",
        masked,
    )
    out: list[str] = []
    for s in raw_sents:
        restored = s
        for idx, orig in enumerate(stashed):
            restored = restored.replace(f"@@SENTTOK_{idx}@@", orig)
        restored = restored.strip()
        if restored:
            out.append(restored)
    return out


def _extract_given_sentence_from_insertion_stem(stem: str) -> str:
    """문장 삽입 변형 출력에서 [주어진 문장] 박스의 추출 문장만 발췌한다."""
    parts = re.split(r"\[주어진\s*문장\]", stem or "", maxsplit=1)
    if len(parts) < 2:
        return ""
    after_hdr = parts[1].strip()
    # 빈 줄 이전 또는 첫 ( ① ) 직전까지를 주어진 문장으로 간주
    block = re.split(r"\n\s*\n", after_hdr, maxsplit=1)[0].strip()
    block = re.split(r"\(\s*①\s*\)", block, maxsplit=1)[0].strip()
    sents = _split_english_sentences(block)
    return sents[0] if sents else block


def _rebuild_insertion_stem_from_source(rw: RewrittenQuestion, source_passage: str) -> bool:
    """
    원문 문장이 6문장 이상인 경우, AI가 고른 주어진 문장을 source_passage에서 찾아
    5개 문장 경계에 ( ① )~( ⑤ )를 파이썬으로 직접 배치하고 정확한 preserved_answer를 계산한다.
    """
    src_sents = _split_english_sentences(source_passage)
    if len(src_sents) < 6:
        return False

    given_cand = _extract_given_sentence_from_insertion_stem(rw.rewritten_stem or "")
    if not given_cand:
        return False

    def _norm_toks(s: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", (s or "").lower())

    g_toks = _norm_toks(given_cand)
    if len(g_toks) < 3:
        return False
    g_set = set(g_toks)

    best_idx = -1
    best_score = 0.0
    for idx_s, s in enumerate(src_sents):
        s_toks = _norm_toks(s)
        if not s_toks:
            continue
        if s_toks == g_toks:
            best_idx = idx_s
            best_score = 1.0
            break
        s_set = set(s_toks)
        jaccard = len(g_set & s_set) / max(1, len(g_set | s_set))
        if jaccard > best_score:
            best_score = jaccard
            best_idx = idx_s

    # 첫 문장(idx 0)은 통상 도입문이므로 최소 0.72 이상 일치하고 idx >= 1일 때 기계적 재조립 수행
    if best_idx < 1 or best_score < 0.72:
        return False

    given_exact = src_sents[best_idx]
    remaining = [s for i, s in enumerate(src_sents) if i != best_idx]
    if len(remaining) < 5:
        return False

    # remaining의 문장 경계 gap k: 1 <= k <= len(remaining) (k는 remaining[k-1] 뒤 위치)
    true_gap = best_idx  # match_idx >= 1 이므로 remaining[best_idx - 1] 바로 뒤
    max_gap = len(remaining) - 1 if true_gap <= len(remaining) - 1 else len(remaining)
    # 5개 연속 경계 [g_start .. g_start + 4] 안에 true_gap이 포함되도록 설정
    g_start = max(1, min(true_gap - 2, max_gap - 4))
    if true_gap < g_start:
        g_start = true_gap
    if true_gap > g_start + 4:
        g_start = true_gap - 4

    markers = ["( ① )", "( ② )", "( ③ )", "( ④ )", "( ⑤ )"]
    chunks: list[str] = []
    for pos, sent in enumerate(remaining):
        chunks.append(sent)
        gap_idx = pos + 1
        if g_start <= gap_idx <= g_start + 4:
            chunks.append(markers[gap_idx - g_start])

    main_body = " ".join(chunks).strip()
    rw.rewritten_stem = (
        "글의 흐름으로 보아, 주어진 문장이 들어가기에 가장 적절한 곳은?\n"
        "[주어진 문장]\n"
        f"{given_exact}\n\n"
        f"{main_body}"
    )
    rw.rewritten_choices = ["①", "②", "③", "④", "⑤"]
    ans_num = str((true_gap - g_start) + 1)
    rw.preserved_answer = ans_num
    # 파이썬이 확정한 정답 번호와 해설 내 위치 번호가 어긋나지 않도록 동기화
    num_to_circle = {"1": "①", "2": "②", "3": "③", "4": "④", "5": "⑤"}
    c_sym = num_to_circle.get(ans_num, "①")
    if rw.pdf_grounded_explanation:
        exp_sync = rw.pdf_grounded_explanation
        exp_sync = re.sub(
            r"(정답(?:은|:)?\s*)(?:[1-5]\s*번|[①②③④⑤](?:\s*번)?)",
            rf"\g<1>{c_sym}번",
            exp_sync,
        )
        exp_sync = re.sub(
            r"(?:[1-5]\s*번|[①②③④⑤](?:\s*번)?)\s*(위치|자리|에\s*들어가|에\s*가장\s*적절)",
            rf"{c_sym}번 \g<1>",
            exp_sync,
        )
        rw.pdf_grounded_explanation = exp_sync
    return True


def _count_english_sentences(text: str) -> int:
    """영어 지문 본문의 문장 수를 약어/소수점을 제외하고 근사 계산한다."""
    cleaned_lines: list[str] = []
    for raw_ln in (text or "").splitlines():
        ln = raw_ln.strip()
        if not ln:
            continue
        if re.search(r"[가-힣]", ln) and not re.search(r"[A-Za-z]{4,}", ln):
            continue
        if ln.startswith("*") and ":" in ln:
            continue
        cleaned_lines.append(ln)
    body = " ".join(cleaned_lines)
    body = re.sub(r"[①②③④⑤]", " ", body)
    body = _SENT_ABBREV_RE.sub("ABBR", body)
    body = re.sub(r"\b\d+\.\d+\b", "NUM", body)
    endings = re.findall(r"[.?!]+(?=\s|$|[\"'”’\)])", body)
    return len(endings) if endings else (1 if re.search(r"[A-Za-z]{3,}", body) else 0)


def _reconcile_english_insertion_answer(rw: RewrittenQuestion) -> None:
    """문장 삽입 유형에서 해설에 명시된 위치 번호(①~⑤ / N번)와 preserved_answer가 어긋나면 자동 보정한다."""
    exp = rw.pdf_grounded_explanation or ""
    if not exp:
        return
    m_exp = re.search(
        r"(?:정답(?:은|:)?|위치(?:는|가)?|삽입(?:하기에)?)\s*(?:은|는|:)?\s*(?:([1-5])\s*번|([①②③④⑤]))",
        exp,
    )
    found_num = ""
    if m_exp:
        found_num = m_exp.group(1) or _CIRCLED_DIGIT_TO_NUM.get(m_exp.group(2) or "", "")
    else:
        circled = list(dict.fromkeys(re.findall(r"[①②③④⑤]", exp)))
        if len(circled) == 1:
            found_num = _CIRCLED_DIGIT_TO_NUM.get(circled[0], "")
    if found_num in ("1", "2", "3", "4", "5") and found_num != _normalize_extracted_answer(rw.preserved_answer):
        rw.preserved_answer = found_num


def _reconcile_english_summary_answer(rw: RewrittenQuestion) -> None:
    """요약문 완성 유형에서 해설의 (A)/(B) 정답 어휘 쌍 또는 명시 번호가 preserved_answer와 어긋나면 자동 보정한다."""
    choices = rw.rewritten_choices or []
    exp = rw.pdf_grounded_explanation or ""
    if len(choices) != 5 or not exp:
        return

    # 1) 해설에 (A) wordA ... (B) wordB 쌍이 명시된 경우 선지와 대조
    m_ab = re.search(
        r"\(A\)\s*([A-Za-z][A-Za-z\-']*)[^A-Za-z]{1,20}\(B\)\s*([A-Za-z][A-Za-z\-']*)",
        exp,
        flags=re.IGNORECASE,
    )
    if m_ab:
        w_a, w_b = m_ab.group(1).lower(), m_ab.group(2).lower()
        matched_indices: list[int] = []
        for idx_c, ch in enumerate(choices):
            m_ch = re.search(
                r"\(A\)\s*([A-Za-z][A-Za-z\-']*)[^A-Za-z]{1,20}\(B\)\s*([A-Za-z][A-Za-z\-']*)",
                ch or "",
                flags=re.IGNORECASE,
            )
            if m_ch and m_ch.group(1).lower() == w_a and m_ch.group(2).lower() == w_b:
                matched_indices.append(idx_c + 1)
        if len(matched_indices) == 1:
            rw.preserved_answer = str(matched_indices[0])
            return

    # 2) 해설에 정답 번호가 명시된 경우 보정
    m_num = re.search(
        r"(?:정답(?:은|:)?)\s*(?:([1-5])\s*번|([①②③④⑤]))",
        exp,
    )
    if m_num:
        found_num = m_num.group(1) or _CIRCLED_DIGIT_TO_NUM.get(m_num.group(2) or "", "")
        if found_num in ("1", "2", "3", "4", "5"):
            rw.preserved_answer = found_num


def _reconcile_english_order_answer(rw: RewrittenQuestion) -> None:
    """글의 순서 유형에서 해설에 명시된 최종 (A)-(B)-(C) 논리 순서와 preserved_answer 선지 인덱스가 어긋나면 자동 보정한다."""
    choices = rw.rewritten_choices or []
    exp = rw.pdf_grounded_explanation or ""
    if len(choices) != 5 or not exp:
        return

    choice_tuples: list[tuple[str, str, str] | None] = []
    for ch in choices:
        letters = re.findall(r"\(([ABC])\)|\b([ABC])\b", ch or "")
        seq = [a or b for a, b in letters]
        choice_tuples.append(tuple(seq) if len(seq) == 3 and set(seq) == {"A", "B", "C"} else None)

    matches = [
        (m.start(), (m.group(1), m.group(2), m.group(3)))
        for m in _ORDER_SEQ_RE.finditer(exp)
        if set((m.group(1), m.group(2), m.group(3))) == {"A", "B", "C"}
    ]
    if not matches:
        return

    # '정답'/'순서'/'따라서' 근처의 순서 표기를 우선 선택하고, 없으면 유일/첫 매칭 사용
    target_seq: tuple[str, str, str] | None = None
    for pos, seq in matches:
        ctx = exp[max(0, pos - 24) : pos + 24]
        if any(kw in ctx for kw in ("정답", "순서", "따라서", "결론", "적절")):
            target_seq = seq
            break
    if target_seq is None:
        distinct = {seq for _, seq in matches}
        if len(distinct) == 1:
            target_seq = matches[0][1]
        else:
            return

    ans = _normalize_extracted_answer(rw.preserved_answer)
    cur_idx = int(ans) - 1 if ans in ("1", "2", "3", "4", "5") else -1
    if 0 <= cur_idx < len(choice_tuples) and choice_tuples[cur_idx] == target_seq:
        return

    for idx_c, c_seq in enumerate(choice_tuples):
        if c_seq == target_seq:
            rw.preserved_answer = str(idx_c + 1)
            return


_REPEATABLE_PREPS: frozenset[str] = frozenset(
    {"from", "to", "in", "on", "of", "with", "for", "by", "at", "into", "as", "about", "through"}
)


def _has_order_scrambled_sentences(stem: str, source_passage: str) -> bool:
    """(A), (B), (C) 단락 각각이 원문의 연속된 문장 묶음으로 구성됐는지 확인해 비연속 문장 뒤섞임을 감지한다."""
    src_sents = _split_english_sentences(source_passage)
    if len(src_sents) < 4:
        return False
    after_given = re.split(r"\[주어진\s*글\]", stem or "", maxsplit=1)[-1]
    parts = re.split(r"(?:^|\n|\s)\(([ABC])\)\s+", after_given)
    chunk_map: dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):
        lbl = parts[i]
        txt = parts[i + 1].strip()
        if lbl in ("A", "B", "C") and txt:
            chunk_map[lbl] = txt
    if len(chunk_map) != 3:
        return False

    src_sets = [set(re.findall(r"[a-z0-9]+", s.lower())) for s in src_sents]
    for lbl in ("A", "B", "C"):
        c_sents = _split_english_sentences(chunk_map[lbl])
        if len(c_sents) <= 1:
            continue
        idxs: list[int] = []
        for cs in c_sents:
            c_set = set(re.findall(r"[a-z0-9]+", cs.lower()))
            if not c_set:
                continue
            best_i, best_sc = -1, 0.0
            for i, s_set in enumerate(src_sets):
                if not s_set:
                    continue
                sc = len(c_set & s_set) / max(1, len(c_set | s_set))
                if sc > best_sc:
                    best_sc, best_i = sc, i
            if best_i != -1 and best_sc >= 0.55:
                idxs.append(best_i)
        if len(idxs) >= 2:
            for k in range(len(idxs) - 1):
                if idxs[k + 1] != idxs[k] + 1:
                    return True
    return False


def _is_irrelevant_answer_from_source(stem: str, ans_num: str, source_passage: str) -> bool:
    """무관한 문장 유형에서 정답 번호에 해당하는 문장이 원문에 이미 존재하는 문장인지 검사한다."""
    if ans_num not in ("1", "2", "3", "4", "5") or not source_passage:
        return False
    sym = {"1": "①", "2": "②", "3": "③", "4": "④", "5": "⑤"}[ans_num]
    m_sym = re.search(rf"{re.escape(sym)}\s*([^\n①②③④⑤]+)", stem or "")
    if not m_sym:
        return False
    ans_toks = set(re.findall(r"[a-z0-9]+", m_sym.group(1).lower()))
    if len(ans_toks) < 4:
        return False
    for s in _split_english_sentences(source_passage):
        s_toks = set(re.findall(r"[a-z0-9]+", s.lower()))
        if not s_toks:
            continue
        if len(ans_toks & s_toks) / max(1, len(ans_toks | s_toks)) >= 0.78:
            return True
    return False


def _ensure_korean_instruction_prefix(stem: str, canonical_type: str, vocab_style: str) -> str:
    """생성된 영어 변형 지문 첫 줄에 표준 한국어 발문이 누락된 경우 자동으로 부착한다."""
    s = (stem or "").strip()
    if not s:
        return s
    first_line = s.splitlines()[0].strip()
    if re.search(r"[가-힣]", first_line) and ("?" in first_line or "시오" in first_line or "것은" in first_line or "곳은" in first_line):
        return s

    t = canonical_type or ""
    if "함축" in t:
        inst = "밑줄 친 부분이 다음 글에서 의미하는 바로 가장 적절한 것은?"
    elif "요지" in t:
        inst = "다음 글의 요지로 가장 적절한 것은?"
    elif "주제" in t:
        inst = "다음 글의 주제로 가장 적절한 것은?"
    elif "제목" in t:
        inst = "다음 글의 제목으로 가장 적절한 것은?"
    elif "어법" in t:
        inst = "다음 글의 밑줄 친 부분 중, 어법상 틀린 것은?"
    elif "어휘" in t:
        inst = (
            "(A), (B), (C)의 각 네모 안에서 문맥에 맞는 낱말로 가장 적절한 것은?"
            if ("선택형" in t or "선택형" in vocab_style or "(A)" in vocab_style)
            else "다음 글의 밑줄 친 부분 중, 문맥상 낱말의 쓰임이 적절하지 않은 것은?"
        )
    elif "빈칸" in t:
        inst = "다음 빈칸에 들어갈 말로 가장 적절한 것은?"
    elif "무관" in t:
        inst = "다음 글에서 전체 흐름과 관계없는 문장은?"
    elif "순서" in t:
        inst = "주어진 글 다음에 이어질 글의 순서로 가장 적절한 것은?"
    elif "삽입" in t:
        inst = "글의 흐름으로 보아, 주어진 문장이 들어가기에 가장 적절한 곳은?"
    else:
        inst = "다음 글의 내용을 한 문장으로 요약하고자 한다. 빈칸 (A), (B)에 들어갈 말로 가장 적절한 것은?"
    return f"{inst}\n{s}"


def _validate_english_transformed_output(
    rw: RewrittenQuestion,
    target_type: str,
    vocab_style: str,
    source_passage: str = "",
) -> list[str]:
    """생성된 영어 변형 문항의 필수 태그·메타 독백·어휘 실제 교체·미완성 문장 여부를 검증한다."""
    issues: list[str] = []
    stem = rw.rewritten_stem or ""
    t = (target_type or "").strip()

    # 1. AI 영문/한글 독백(메타 발화) 혼입 차단
    all_text_fields = [stem, *(rw.rewritten_choices or []), rw.pdf_grounded_explanation or ""]
    for fld in all_text_fields:
        m_meta = _AI_META_MONOLOGUE_RE.search(fld or "")
        if m_meta:
            issues.append(
                f"AI 내부 사고/자기 수정 메타 문구({m_meta.group(0)!r})가 출력에 혼입되었습니다. 메타 독백 없이 순수 문제와 해설만 반환하세요."
            )
            break

    # 1-b. 영문 지문 본문(발문·섹션 헤더·각주 제외)에 한글 코멘트가 섞였는지 차단
    for ln_raw in stem.splitlines():
        ln_s = ln_raw.strip()
        if not ln_s:
            continue
        if re.search(r"가장\s*적절한\s*(?:것은|곳은)\??$", ln_s):
            continue
        if re.match(r"^\[(?:주어진\s*글|주어진\s*문장|요약문)\]$", ln_s):
            continue
        if ln_s.startswith("*"):
            continue
        if re.search(r"[가-힣]{2,}", ln_s) and re.search(r"[A-Za-z]{4,}", ln_s):
            issues.append(
                f"영문 지문 본문 안에 한글 설명/메모({ln_s[:40]!r})가 섞였습니다. 지문 본문은 순수 영어만 남기세요."
            )
            break

    # 4. 문장 끝 단어 탈락 방지 ('has to.', 'able to.' 등)
    m_to = _DANGLING_TO_END_RE.search(stem)
    if m_to:
        issues.append(
            f"지문 내에 부정사 뒤 동사원형이 탈락한 미완성 문장({m_to.group(0)!r})이 감지되었습니다. 빠진 단어를 복원해 완전한 문장으로 작성하세요."
        )

    ans = _normalize_extracted_answer(rw.preserved_answer)
    if ans not in ("1", "2", "3", "4", "5"):
        issues.append(f"정답(preserved_answer)이 '1'~'5' 중 하나여야 합니다 (현재: {rw.preserved_answer!r}).")

    if len(rw.rewritten_choices or []) != 5:
        issues.append(f"선택지(rewritten_choices)는 정확히 5개여야 합니다 (현재: {len(rw.rewritten_choices or [])}개).")

    u_matches = re.findall(r"<u\b[^>]*>(.*?)</u>", stem, flags=re.I | re.S)
    u_count = len(u_matches)
    if "어휘" in t and source_passage:
        src_cnt = _count_english_sentences(source_passage)
        gen_cnt = _count_english_sentences(stem)
        if src_cnt >= 3 and gen_cnt != src_cnt:
            issues.append(
                f"문맥상 어휘 유형은 원문 문장 수({src_cnt}개)와 동일해야 합니다 (현재 {gen_cnt}문장). "
                "지문 끝에 새 문장을 추가하거나 기존 문장을 삭제하지 말고 원문 본문 단어만 교체하세요."
            )

    if "어법" in t or ("어휘" in t and "선택형" not in vocab_style and "(A)" not in vocab_style):
        missing_markers = [mk for mk in ("①", "②", "③", "④", "⑤") if stem.count(mk) != 1]
        if u_count != 5 or missing_markers:
            issues.append(
                f"지문(rewritten_stem) 안에 `① <u>...</u>` ~ `⑤ <u>...</u>` 밑줄 5개가 모두 정확히 1개씩 포함되어야 합니다 "
                f"(현재 `<u>` {u_count}개, 누락/중복 번호: {missing_markers})."
            )
        elif "어법" in t:
            exp_txt = rw.pdf_grounded_explanation or ""
            if re.search(r"(?:의미상|의미적|문맥상\s*반대|반의어|부정어\s*(?:not|no)\s*(?:이|가)?\s*빠)", exp_txt):
                issues.append(
                    "어법 문제 해설에 어휘 뜻풀이('의미상'/'의미적'/'반의어'/'부정어 누락')가 포함되었습니다. 어휘 문제와 혼동하지 말고 명백한 문법 비문 1개만 출제하세요."
                )
            elif re.search(r"(?:과거형\s*대신\s*현재형|현재형\s*대신\s*과거형|시제\s*흐름과\s*일치)", exp_txt):
                issues.append(
                    "어법 문제에서 단순 과거/현재 시제 차이를 오류로 출제했습니다. 문맥상 가능한 시제 변경 대신 수일치·능동/수동·관계사·분사·준동사 등 명백한 통사 오류 1개만 출제하세요."
                )
            elif re.search(r"(?:타당|적절|올바르|맞)(?:하다|다|습니다|음)(?!\s*(?:않|못))", exp_txt) and not re.search(
                r"(?:적절하지\s*않|올바르지\s*않|타당하지\s*않|맞지\s*않)", exp_txt
            ):
                issues.append(
                    "어법 문제 해설이 정답 밑줄을 '적절하다/타당하다'고 모순되게 설명했습니다. 명백한 문법 오류 1개만 출제하고 왜 틀렸는지 설명하세요."
                )
        elif "어휘" in t and ans in ("1", "2", "3", "4", "5"):
            # 2. 문맥상 어휘 적절성: 해설/치환목록에 적힌 교체 단어가 실제 지문 정답 밑줄에 반영되었는지 확인
            u_words = [re.sub(r"<[^>]+>", "", w).strip() for w in u_matches]
            ans_word = u_words[int(ans) - 1]
            ans_alpha = re.sub(r"[^a-zA-Z\-']", "", ans_word).lower()
            for entry in rw.replaced_terms_map or []:
                m_map = re.search(r"([A-Za-z][A-Za-z\-']*)\s*(?:->|→)\s*([A-Za-z][A-Za-z\-']*)", entry or "")
                if m_map:
                    orig_w, new_w = m_map.group(1).lower(), m_map.group(2).lower()
                    if orig_w != new_w and ans_alpha == orig_w:
                        issues.append(
                            f"문맥상 어휘 문제에서 정답 {ans}번 밑줄({ans_word!r})이 원문 그대로 남아 있고 "
                            f"교체 어휘({new_w!r})로 바뀌지 않았습니다. 지문 본문의 정답 밑줄을 실제로 교체하세요."
                        )
                        break
            if source_passage and ans_alpha:
                # 3. 문맥상 어휘 적절성: 5개 밑줄 단어 중 정확히 1개(정답 위치)만 원문과 다른 단어인지 검증
                not_in_src_indices: list[int] = []
                for idx_u, w in enumerate(u_words[:5]):
                    cleaned_w = re.sub(r"[^a-zA-Z\-]", "", w).lower()
                    if cleaned_w and not re.search(
                        r"\b" + re.escape(cleaned_w) + r"\b",
                        source_passage,
                        flags=re.IGNORECASE,
                    ):
                        not_in_src_indices.append(idx_u)
                if not not_in_src_indices:
                    issues.append(
                        f"문맥상 어휘 문제에서 5개 밑줄 단어({u_words[:5]})가 모두 원문 그대로입니다. "
                        f"반드시 정답 {ans}번 위치의 단어를 문맥상 부적절한 반의어로 지문 본문에서 직접 교체하세요."
                    )
                elif len(not_in_src_indices) == 1 and not_in_src_indices[0] != int(ans) - 1:
                    issues.append(
                        f"문맥상 어휘 문제의 정답 번호({ans}번)와 실제 지문에서 교체된 밑줄 위치({not_in_src_indices[0] + 1}번)가 일치하지 않습니다."
                    )
    elif "함축" in t:
        if u_count < 1:
            issues.append("함축적 의미 추론 지문에는 밑줄 친 표현 `<u>...</u>`가 최소 1개 포함되어야 합니다.")
    elif "빈칸" in t:
        if "____" not in stem:
            issues.append("빈칸 추론 지문에는 `___________` 빈칸 표기가 반드시 포함되어야 합니다.")
        else:
            # 2. 빈칸 포함 문장에서 빈칸을 제외했을 때 영문 동사가 하나도 남지 않는 불완전 문장 차단
            m_blk = re.search(r"_{4,}", stem)
            if m_blk:
                pre_txt = stem[: m_blk.start()]
                post_txt = stem[m_blk.end() :]
                s_start = max(pre_txt.rfind("\n"), pre_txt.rfind(". "), pre_txt.rfind("? "), pre_txt.rfind("! "))
                sent_left = pre_txt[s_start + 1 :] if s_start != -1 else pre_txt
                m_end = re.search(r"[.?!](?=\s|$|[\"'”’\)])|\n", post_txt)
                sent_right = post_txt[: m_end.start()] if m_end else post_txt
                rem_sentence = f"{sent_left} {sent_right}".strip()
                if not _sentence_has_english_verb(rem_sentence):
                    issues.append(
                        f"빈칸이 포함된 문장({rem_sentence!r})에서 빈칸을 제외하면 영문 동사가 전혀 남지 않습니다. "
                        "문장 전체(동사 포함)를 비우지 말고 주어·동사가 유지된 상태에서 핵심 목적어/보어/구만 빈칸으로 비우세요."
                    )

            # 3. 빈칸 앞뒤 구절/전치사 중복 차단
            if m_blk and ans in ("1", "2", "3", "4", "5") and len(rw.rewritten_choices or []) >= int(ans):
                before_toks = re.findall(r"[a-zA-Z\-']+", stem[: m_blk.start()].lower())
                after_toks = re.findall(r"[a-zA-Z\-']+", stem[m_blk.end() :].lower())
                ans_toks = re.findall(r"[a-zA-Z\-']+", (rw.rewritten_choices[int(ans) - 1] or "").lower())
                if len(ans_toks) >= 2:
                    if len(before_toks) >= 2 and before_toks[-2:] == ans_toks[:2]:
                        issues.append(
                            f"빈칸 직전 구절({' '.join(before_toks[-2:])})과 정답 선지의 시작 2단어가 중복됩니다. 중복을 제거해 다시 생성하세요."
                        )
                    elif len(after_toks) >= 2 and after_toks[:2] == ans_toks[-2:]:
                        issues.append(
                            f"빈칸 직후 구절({' '.join(after_toks[:2])})과 정답 선지의 끝 2단어가 중복됩니다. 중복을 제거해 다시 생성하세요."
                        )
                if after_toks and after_toks[0] in _REPEATABLE_PREPS and after_toks[0] in ans_toks[-4:]:
                    issues.append(
                        f"빈칸 바로 뒤 전치사({after_toks[0]!r})가 정답 선지 끝부분에도 중복 포함되어 비문이 됩니다. 전치사를 겹치지 않게 다시 생성하세요."
                    )
                elif before_toks and before_toks[-1] in (_REPEATABLE_PREPS | {"a", "an", "the"}) and ans_toks and ans_toks[0] == before_toks[-1]:
                    issues.append(
                        f"빈칸 바로 앞 단어({before_toks[-1]!r})가 정답 선지 시작 단어와 중복됩니다. 중복을 제거해 다시 생성하세요."
                    )
                elif (
                    before_toks
                    and before_toks[-1] in {"that", "because", "although", "though", "since", "when", "while", "if", "unless"}
                    and ans_toks
                    and ans_toks[0] in {"without", "with", "in", "on", "at", "by", "for", "from", "through"}
                ):
                    issues.append(
                        f"접속사({before_toks[-1]!r}) 바로 뒤 빈칸에 전치사구({ans_toks[0]!r} ...)만 들어가면 비문이 됩니다. 주어+동사를 포함한 절 또는 동사구로 출제하세요."
                    )
                post_raw = stem[m_blk.end() :]
                if re.match(r"^[ \t]+[A-Z]", post_raw) and not re.search(r"[.?!]$", pre_txt.rstrip()):
                    issues.append(
                        "문장 전체를 빈칸으로 비웠으나 빈칸 뒤에 마침표(.)가 누락되어 다음 문장과 붙었습니다. 문장 종결 부호를 유지하거나 문장 일부만 빈칸 처리하세요."
                    )
        if re.search(r"\(\s*[①②③④⑤]\s*\)|[①②③④⑤]", stem):
            issues.append(
                "빈칸 추론 지문에는 문장 삽입 번호 기호 `( ① )~( ⑤ )` 또는 `①~⑤`가 포함되면 안 됩니다. 모두 제거하세요."
            )
    elif "무관" in t:
        for c_num in ("①", "②", "③", "④", "⑤"):
            if c_num not in stem:
                issues.append(f"무관한 문장 본문에 번호 `{c_num}`가 누락되었습니다.")
                break
        if source_passage:
            src_cnt = _count_english_sentences(source_passage)
            gen_cnt = _count_english_sentences(stem)
            if src_cnt >= 3 and gen_cnt != src_cnt + 1:
                issues.append(
                    f"전체 흐름과 무관한 문장은 원문 문장({src_cnt}개)을 삭제하지 않고 무관한 새 문장 1개만 삽입해 "
                    f"총 {src_cnt + 1}문장이 되어야 합니다 (현재 {gen_cnt}문장). 원문을 보존하고 새 문장 1개를 삽입하세요."
                )
            elif _is_irrelevant_answer_from_source(stem, ans, source_passage):
                issues.append(
                    f"무관한 문장의 정답 {ans}번 문장이 원문 문장 그대로입니다. 원문 문장을 오답으로 우기지 말고 흐름과 무관한 새 문장을 삽입하세요."
                )
    elif "순서" in t:
        for mk in ("(A)", "(B)", "(C)"):
            if mk not in stem:
                issues.append(f"글의 순서 지문에 `{mk}` 단락 표기가 누락되었습니다.")
        if "[주어진 글]" not in stem and "주어진 글" not in stem:
            issues.append("글의 순서 지문에 `[주어진 글]` 섹션 헤더가 누락되었습니다.")
        if source_passage:
            # 4. 글의 순서 원문 문장 보존 검증 ([주어진 글] + (A) + (B) + (C) 합계 == 원문 문장 수)
            order_clean = re.sub(r"\[주어진\s*글\]", " ", stem)
            order_clean = re.sub(r"\(\s*[ABC]\s*\)", " ", order_clean)
            src_cnt = _count_english_sentences(source_passage)
            gen_cnt = _count_english_sentences(order_clean)
            if src_cnt >= 3 and gen_cnt != src_cnt:
                issues.append(
                    f"글의 순서 문제에서 원문 문장 수({src_cnt}개)와 [주어진 글]+(A)+(B)+(C) 합계 문장 수({gen_cnt}개)가 다릅니다. "
                    "원문 문장을 누락하거나 새 문장을 임의로 추가하지 말고 원문 그대로 배치하세요."
                )
            elif _is_order_abc_canonical(stem, source_passage):
                issues.append(
                    "글의 순서 문제에서 (A), (B), (C)가 원문 등장 순서 그대로((A)-(B)-(C) 정순) 배열되었습니다. "
                    "수능 5지선다에는 (A)-(B)-(C)가 없으므로 반드시 (B)-(A)-(C), (C)-(B)-(A) 등 단락 순서를 섞어 다시 출제하세요."
                )
            elif _has_order_scrambled_sentences(stem, source_passage):
                issues.append(
                    "글의 순서 문제에서 (A), (B), (C) 단락 내부에 원문의 비연속 문장이 뒤섞여 합쳐졌습니다. 각 단락은 원문에서 연속된 문장 묶음으로만 구성하세요."
                )
    elif "삽입" in t:
        if "[주어진 문장]" not in stem and "주어진 문장" not in stem:
            issues.append("문장 삽입 지문에 `[주어진 문장]` 섹션 헤더가 누락되었습니다.")
        for c_num in ("①", "②", "③", "④", "⑤"):
            if c_num not in stem:
                issues.append(f"문장 삽입 본문에 위치 기호 `{c_num}`가 누락되었습니다.")
                break
        # ( ① )~( ⑤ ) 마커가 문장 경계(. ? ! 뒤)가 아닌 주어·동사 한가운데 삽입되었는지 검증
        ins_body = re.split(r"\[주어진\s*문장\]", stem, maxsplit=1)[-1]
        ins_body_alpha = re.findall(r"[A-Za-z]+", ins_body)
        if len(ins_body_alpha) < 25:
            issues.append(
                f"문장 삽입 본문 영문 단어가 {len(ins_body_alpha)}개뿐입니다. [주어진 문장] 외에 전체 영어 지문 본문을 빠짐없이 포함하세요."
            )
        for m_ins in re.finditer(r"\(?\s*([①②③④⑤])\s*\)", ins_body):
            before = ins_body[: m_ins.start()].rstrip()
            if before and not re.search(r"[.?!][\"'”’\)]*$", before):
                issues.append(
                    f"문장 삽입 기호 `{m_ins.group(0).strip()}`가 문장 종결 부호(. ? !) 뒤의 문장 경계가 아닌 문장 중간에 배치되었습니다. 반드시 문장과 문장 사이 경계로 옮기세요."
                )
                break
    elif "요약" in t:
        if "[요약문]" not in stem and "요약문" not in stem:
            issues.append("요약문 완성 지문에 `[요약문]` 섹션 헤더가 누락되었습니다.")
        sum_section = re.split(r"\[요약문\]", stem, maxsplit=1)[-1]
        if not re.search(r"\(A\)\s*_{2,}", sum_section) or not re.search(r"\(B\)\s*_{2,}", sum_section):
            issues.append("요약문 섹션 안에 `(A) ________`와 `(B) ________` 빈칸 표기가 모두 포함되어야 합니다.")

    return issues


def _assemble_english_slim_stem(
    rw: RewrittenQuestion,
    canonical_type: str,
    clean_passage: str,
) -> None:
    """
    지문 전체 재출력 생략 유형(요지·주제·제목·요약문·문장 삽입·빈칸·함축)에서
    AI가 반환한 경량 결과(빈 문자열 / 요약 1문장 / 추출 1문장 / target_phrase)를
    원문(clean_passage)에 직접 결합·치환해 완성된 rewritten_stem을 조립한다.
    이미 전체 지문 형태로 출력된 경우(폴백 재시도 등)는 그대로 보존한다.
    """
    raw_stem = (rw.rewritten_stem or "").strip()
    t = canonical_type or ""

    if "요지" in t:
        if len(raw_stem) < max(60, int(len(clean_passage) * 0.55)):
            rw.rewritten_stem = f"다음 글의 요지로 가장 적절한 것은?\n{clean_passage}"
        return
    if "주제" in t:
        if len(raw_stem) < max(60, int(len(clean_passage) * 0.55)):
            rw.rewritten_stem = f"다음 글의 주제로 가장 적절한 것은?\n{clean_passage}"
        return
    if "제목" in t:
        if len(raw_stem) < max(60, int(len(clean_passage) * 0.55)):
            rw.rewritten_stem = f"다음 글의 제목으로 가장 적절한 것은?\n{clean_passage}"
        return

    if "요약" in t:
        # 이미 원문+요약문 전체가 들어있으면 그대로 두고, 요약문 1문장만 반환된 경우 원문과 결합
        if len(raw_stem) < max(120, int(len(clean_passage) * 0.65)):
            s_line = strip_english_embedded_choices(normalize_english_exam_text(raw_stem))
            s_line = re.sub(r"^[^\n]*가장\s*적절한\s*것은\??\s*", "", s_line).strip()
            s_line = re.sub(r"^\[?요약문\]?\s*", "", s_line).strip()
            if s_line:
                rw.rewritten_stem = (
                    "다음 글의 내용을 한 문장으로 요약하고자 한다. 빈칸 (A), (B)에 들어갈 말로 가장 적절한 것은?\n"
                    f"{clean_passage}\n\n"
                    "[요약문]\n"
                    f"{s_line}"
                )
        return

    if "삽입" in t:
        # 6문장 이상에서 추출 문장 1개만 반환된 경우 [주어진 문장] 헤더로 감싸 기계적 마커 배치로 넘김
        if (
            len(_split_english_sentences(clean_passage)) >= 6
            and "[주어진 문장]" not in raw_stem
            and len(raw_stem) < max(80, int(len(clean_passage) * 0.55))
        ):
            ext_s = re.sub(r"^[^\n]*가장\s*적절한\s*곳은\??\s*", "", raw_stem).strip()
            if ext_s:
                rw.rewritten_stem = f"[주어진 문장]\n{ext_s}"
        return

    if "빈칸" in t or "함축" in t:
        is_full_stem = len(raw_stem) >= max(80, int(len(clean_passage) * 0.6)) and (
            ("빈칸" in t and "____" in raw_stem) or ("함축" in t and "<u" in raw_stem.lower())
        )
        if is_full_stem or not raw_stem:
            return

        phrase = re.sub(r"</?u\b[^>]*>", "", raw_stem, flags=re.IGNORECASE).strip()
        phrase = re.sub(r"^[^\n]*가장\s*적절한\s*것은\??\s*", "", phrase).strip()
        phrase = phrase.strip("`\"'“”‘’")
        if not phrase:
            return

        exact_cnt = clean_passage.count(phrase)
        matched_span: tuple[int, int, str] | None = None
        if exact_cnt == 1:
            pos = clean_passage.index(phrase)
            matched_span = (pos, pos + len(phrase), phrase)
        elif exact_cnt == 0:
            ci_matches = list(re.finditer(re.escape(phrase), clean_passage, flags=re.IGNORECASE))
            if len(ci_matches) == 1:
                m_ci = ci_matches[0]
                matched_span = (m_ci.start(), m_ci.end(), m_ci.group(0))

        if matched_span is not None:
            st_i, en_i, orig_seg = matched_span
            if "빈칸" in t:
                replaced_body = f"{clean_passage[:st_i]}___________{clean_passage[en_i:]}"
                rw.rewritten_stem = f"다음 빈칸에 들어갈 말로 가장 적절한 것은?\n{replaced_body}"
            else:
                replaced_body = f"{clean_passage[:st_i]}<u>{orig_seg}</u>{clean_passage[en_i:]}"
                rw.rewritten_stem = f"밑줄 친 부분이 다음 글에서 의미하는 바로 가장 적절한 것은?\n{replaced_body}"
        return

    if "순서" in t:
        # 이미 완성된 (A)(B)(C) 전체 지문이 반환된 경우(폴백 재시도 등)는 그대로 보존
        if "(A)" in raw_stem and "(B)" in raw_stem and "(C)" in raw_stem and len(raw_stem) >= max(100, int(len(clean_passage) * 0.6)):
            return
        m_sp = re.search(r"SPLIT\s*:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", raw_stem, flags=re.IGNORECASE)
        sents = _split_english_sentences(clean_passage)
        if m_sp and len(sents) >= 4:
            s1, s2, s3 = int(m_sp.group(1)), int(m_sp.group(2)), int(m_sp.group(3))
            if 0 < s1 < s2 < s3 < len(sents):
                intro = " ".join(sents[:s1]).strip()
                orig_blocks = [
                    " ".join(sents[s1:s2]).strip(),
                    " ".join(sents[s2:s3]).strip(),
                    " ".join(sents[s3:]).strip(),
                ]
                if all(orig_blocks):
                    # 5지선다 오순 조합: (A)-(C)-(B)=1, (B)-(A)-(C)=2, (B)-(C)-(A)=3, (C)-(A)-(B)=4, (C)-(B)-(A)=5
                    # 각 조합에서 (A), (B), (C)에 어떤 원문 덩어리(0,1,2)를 넣으면 정답이 해당 순서가 되는지 매핑
                    order_specs = [
                        ("1", "(A) - (C) - (B)", (0, 2, 1)),  # A=0, B=2, C=1 -> 원문순서 0,1,2 = (A)-(C)-(B)
                        ("2", "(B) - (A) - (C)", (1, 0, 2)),  # A=1, B=0, C=2 -> 원문순서 0,1,2 = (B)-(A)-(C)
                        ("3", "(B) - (C) - (A)", (2, 0, 1)),  # A=2, B=0, C=1 -> 원문순서 0,1,2 = (B)-(C)-(A)
                        ("4", "(C) - (A) - (B)", (1, 2, 0)),  # A=1, B=2, C=0 -> 원문순서 0,1,2 = (C)-(A)-(B)
                        ("5", "(C) - (B) - (A)", (2, 1, 0)),  # A=2, B=1, C=0 -> 원문순서 0,1,2 = (C)-(B)-(A)
                    ]
                    ans_num, ans_seq_str, perm = random.choice(order_specs)
                    block_a = orig_blocks[perm[0]]
                    block_b = orig_blocks[perm[1]]
                    block_c = orig_blocks[perm[2]]
                    rw.rewritten_stem = (
                        "주어진 글 다음에 이어질 글의 순서로 가장 적절한 것은?\n"
                        "[주어진 글]\n"
                        f"{intro}\n\n"
                        f"(A) {block_a}\n\n"
                        f"(B) {block_b}\n\n"
                        f"(C) {block_c}"
                    )
                    rw.rewritten_choices = [
                        "(A) - (C) - (B)",
                        "(B) - (A) - (C)",
                        "(B) - (C) - (A)",
                        "(C) - (A) - (B)",
                        "(C) - (B) - (A)",
                    ]
                    rw.preserved_answer = ans_num
                    if rw.pdf_grounded_explanation:
                        rw.pdf_grounded_explanation = _ORDER_SEQ_RE.sub(ans_seq_str, rw.pdf_grounded_explanation)
                    else:
                        rw.pdf_grounded_explanation = f"주어진 글 뒤에 논리적 흐름에 따라 {ans_seq_str} 순서로 이어지는 것이 자연스럽습니다."
        return

    if "무관" in t:
        if all(mk in raw_stem for mk in ("①", "②", "③", "④", "⑤")) and len(raw_stem) >= max(100, int(len(clean_passage) * 0.6)):
            return
        m_pos = re.search(r"INSERT_AT\s*:\s*([1-5])", raw_stem, flags=re.IGNORECASE)
        m_sent = re.search(r"SENTENCE\s*:\s*(.+)", raw_stem, flags=re.IGNORECASE | re.DOTALL)
        sents = _split_english_sentences(clean_passage)
        if m_sent and len(sents) >= 5:
            ins_sent = re.sub(r"\s+", " ", m_sent.group(1)).strip().strip("\"'")
            if ins_sent and not re.search(r"[.?!]$", ins_sent):
                ins_sent += "."
            if len(re.findall(r"[A-Za-z]+", ins_sent)) >= 5:
                insert_pos = int(m_pos.group(1)) if m_pos else random.choice([2, 3, 4])
                insert_pos = max(2, min(4, insert_pos))
                intro_cnt = 2 if len(sents) >= 6 else 1
                intro_part = " ".join(sents[:intro_cnt]).strip()
                body_orig = sents[intro_cnt:]
                numbered_orig = body_orig[:4]
                tail_orig = body_orig[4:]
                merged_five: list[str] = []
                orig_iter = iter(numbered_orig)
                for slot in range(1, 6):
                    if slot == insert_pos:
                        merged_five.append(ins_sent)
                    else:
                        try:
                            merged_five.append(next(orig_iter))
                        except StopIteration:
                            break
                if len(merged_five) == 5:
                    syms = ["①", "②", "③", "④", "⑤"]
                    marked_str = " ".join(f"{syms[i]} {merged_five[i]}" for i in range(5))
                    tail_str = (" " + " ".join(tail_orig)) if tail_orig else ""
                    rw.rewritten_stem = (
                        "다음 글에서 전체 흐름과 관계없는 문장은?\n"
                        f"{intro_part} {marked_str}{tail_str}".strip()
                    )
                    rw.rewritten_choices = list(syms)
                    rw.preserved_answer = str(insert_pos)
                    c_sym = syms[insert_pos - 1]
                    if rw.pdf_grounded_explanation:
                        rw.pdf_grounded_explanation = re.sub(
                            r"(?:[1-5]\s*번|[①②③④⑤](?:\s*번)?)",
                            f"{c_sym}번",
                            rw.pdf_grounded_explanation,
                            count=1,
                        )
        return

    if ("어법" in t or ("어휘" in t and "선택형" not in t and "(A)" not in t)) and not (
        "<u" in raw_stem.lower() and len(raw_stem) >= max(100, int(len(clean_passage) * 0.6))
    ):
        # 5줄 슬림 포맷 파싱: "1: orig" 또는 "3: orig -> replaced"
        spec_lines: dict[int, tuple[str, str]] = {}
        for ln in raw_stem.splitlines():
            m_ln = re.match(r"^\s*(?:([1-5])|([①②③④⑤]))\s*[:.)]\s*(.+?)\s*$", ln)
            if not m_ln:
                continue
            num_str = m_ln.group(1) or _CIRCLED_DIGIT_TO_NUM.get(m_ln.group(2) or "", "")
            if not num_str:
                continue
            idx_n = int(num_str)
            body_spec = m_ln.group(3).strip()
            if "->" in body_spec or "→" in body_spec:
                parts = re.split(r"\s*(?:->|→)\s*", body_spec, maxsplit=1)
                orig_p = parts[0].strip().strip("`\"'“”‘’")
                repl_p = parts[1].strip().strip("`\"'“”‘’")
            else:
                orig_p = body_spec.strip("`\"'“”‘’")
                repl_p = orig_p
            if orig_p:
                spec_lines[idx_n] = (orig_p, repl_p)

        if len(spec_lines) == 5:
            syms = ["①", "②", "③", "④", "⑤"]
            cursor = 0
            out_parts: list[str] = []
            display_words: list[str] = []
            changed_indices: list[int] = []
            matched_all = True
            for i in range(1, 6):
                orig_p, repl_p = spec_lines[i]
                # 커서 이후에서 단어 경계 우선 매칭, 실패 시 단순 부분 문자열 매칭
                seg_after = clean_passage[cursor:]
                m_word = re.search(rf"(?<![A-Za-z]){re.escape(orig_p)}(?![A-Za-z])", seg_after)
                if m_word is None:
                    m_word = re.search(re.escape(orig_p), seg_after, flags=re.IGNORECASE)
                if m_word is None:
                    matched_all = False
                    break
                abs_st = cursor + m_word.start()
                abs_en = cursor + m_word.end()
                out_parts.append(clean_passage[cursor:abs_st])
                shown = repl_p if repl_p != orig_p else clean_passage[abs_st:abs_en]
                out_parts.append(f"{syms[i - 1]} <u>{shown}</u>")
                display_words.append(shown)
                if repl_p.lower() != orig_p.lower():
                    changed_indices.append(i)
                cursor = abs_en

            if matched_all and len(changed_indices) == 1:
                out_parts.append(clean_passage[cursor:])
                assembled_body = "".join(out_parts)
                instr = (
                    "다음 글의 밑줄 친 부분 중, 어법상 틀린 것은?"
                    if "어법" in t
                    else "다음 글의 밑줄 친 부분 중, 문맥상 낱말의 쓰임이 적절하지 않은 것은?"
                )
                rw.rewritten_stem = f"{instr}\n{assembled_body}"
                rw.preserved_answer = str(changed_indices[0])
                if "어법" in t:
                    rw.rewritten_choices = list(syms)
                else:
                    rw.rewritten_choices = display_words


_INCOMPLETE_BLANK_RE = re.compile(
    r"_{2,}|"
    r"\[\s+\]|\(\s+\)|"
    r"\(\s*[A-C]\s*\)\s*(?:\[[^\]]+/[^\]]+\]|[A-Za-z-]+/[A-Za-z-]+)|"
    r"[A-Za-z][ \t]{2,}[.?!](?=\s|$|[\"'”’\)])|"
    r",\s*[.?!](?=\s|$|[\"'”’\)])|"
    r"\b(?:isn't\s+much|aren't\s+many|has\s+to|have\s+to|had\s+to|want\s+to|need\s+to|"
    r"able\s+to|going\s+to|ought\s+to|used\s+to|the|a|an)\s*[.?!]"
    r"(?=\s|$|[\"'”’\)])",
    re.IGNORECASE,
)

_EXAM_STRUCTURE_MARKER_RE = re.compile(
    r"[①②③④⑤]|"
    r"</?u\b[^>]*>|"
    r"\[(?:주어진\s*글|주어진\s*문장)\]|"
    r"(?:^|\n|(?<=[.?!\"'”’\)]\s))\(\s*[ABC]\s*\)\s+(?=[A-Z\"'“‘\(])",
    re.IGNORECASE,
)

_RESTORE_TARGET_TYPE_KEYWORDS = (
    "어법",
    "어휘",
    "빈칸",
    "무관",
    "흐름",
    "삽입",
    "순서",
    "낱말",
)


def _has_exam_structure_markers(passage: str) -> bool:
    """영어 지문에 미정리 기출 마커(①~⑤, (A)(B)(C), <u>, [주어진 글/문장])가 남아 있는지 판별한다."""
    if not passage:
        return False
    return bool(_EXAM_STRUCTURE_MARKER_RE.search(passage))


def _has_incomplete_english_blank(passage: str) -> bool:
    """원문 영어 지문 자체가 빈칸·선택형 기출이어서 비문/누락 문장이 있는지 판별한다."""
    if not passage:
        return False
    if _INCOMPLETE_BLANK_RE.search(passage):
        return True
    for sent in _split_english_sentences(passage):
        words = re.findall(r"[A-Za-z]+", sent)
        if len(words) >= 5 and not _sentence_has_english_verb(sent):
            # 일반 3인칭 단수 동사(-s/-es)까지 오탐하지 않도록 추가 확인
            if not any(len(w) >= 4 and w.lower().endswith(("s", "ed", "ing")) for w in words):
                return True
    return False


def _needs_english_passage_restoration(passage: str, orig_question_type: str = "") -> bool:
    """
    기출 지문에 실제 미복원 흔적(빈칸 `____`, `①~⑤`, `(A)(B)(C)`, `<u>`, 불완전 문장)이 있을 때만
    원문 복원/교정이 필요한 것으로 판별한다.
    이미 탭 1에서 복원이 완료된 깨끗한 지문은 재복원(원문 변조·문장 삭제 위험)하지 않는다.
    """
    _ = orig_question_type
    if not passage or not passage.strip():
        return False
    if _has_incomplete_english_blank(passage):
        return True
    if _has_exam_structure_markers(passage):
        return True
    return False


# ---------------------------------------------------------------------------
# 기출 원문(question_text) 기반 무손실 각주·빈칸·무관한 문장·순서 추출 및 보정
# ---------------------------------------------------------------------------
_CIRCLED_NUM_LIST = ("①", "②", "③", "④", "⑤")
_NUM_TO_CIRCLED_MAP = {"1": "①", "2": "②", "3": "③", "4": "④", "5": "⑤"}

_ENG_FOOTNOTE_EXTRACT_RE = re.compile(
    r"\*{1,2}\s*([A-Za-z][A-Za-z0-9'\-\s]{0,32}?)\s*:\s*([^*\n①②③④⑤]+?)"
    r"(?=\s*\*{1,2}\s*[A-Za-z]|\s*[①②③④⑤]|\s*\[\d점\]|$)"
)


def _extract_english_footnotes(question_text: str) -> list[str]:
    """PyMuPDF 추출 문항 텍스트(단일 줄/다중 줄 모두 지원)에서 문제지 공식 각주(`* word: 뜻`)를 추출한다."""
    raw = (question_text or "").strip()
    if not raw or "*" not in raw:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for m in _ENG_FOOTNOTE_EXTRACT_RE.finditer(raw):
        term = re.sub(r"\s+", " ", m.group(1)).strip()
        meaning = re.sub(r"\s+", " ", m.group(2)).strip()
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
    """question_text에서 앞부분 한국어 발문·배점과 뒷부분 각주·쪽번호를 제거한 본문(+선지) 영역을 반환한다."""
    s = re.sub(r"\s+", " ", (question_text or "").strip())
    if not s:
        return ""
    # 끝부분 각주(* word: ...) 제거
    m_fn = re.search(r"\s*\*{1,2}\s*[A-Za-z][A-Za-z0-9'\-\s]{0,32}?\s*:\s*[^*]*[가-힣]", s)
    if m_fn:
        s = s[: m_fn.start()].strip()
    # 앞부분 문항 번호 + 한국어 발문 + [3점] 제거
    s = re.sub(
        r"^(?:\[\d{1,2}\s*[~～-]\s*\d{1,2}\][^\n]*?(?:물음에\s*답하시오\.|것은\?)\s*)?"
        r"(?:\d{1,2}\.\s*)?"
        r"(?:[^A-Za-z]*?[가-힣][^?.]*?(?:\?|시오\.)(?:\s*\[[1-4]점\])?)\s*",
        "",
        s,
        count=1,
    ).strip()
    # 혹시 남은 [3점] 및 쪽번호/영역 헤더 제거
    s = re.sub(r"^\[[1-4]점\]\s*", "", s).strip()
    s = re.sub(r"(?:\s+\d{1,2}\s+영어\s*영역|\s+영어\s*영역\s+\d{1,2})\s*$", "", s).strip()
    return s


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


def _infer_blank_phrase_by_diff(raw_body_with_blank: str, completed_passage: str) -> str:
    """
    PDF 텍스트 추출 과정에서 ①~⑤ 선지 분할이 깨졌을 때,
    원본 빈칸 문장(`____`)과 복원된 완성 지문(`completed_passage`)의 토큰 diff로
    원래 빈칸 구절(`original_blank_phrase`)을 결정론적으로 역추출한다.
    """
    raw_b = (raw_body_with_blank or "").strip()
    comp_p = (completed_passage or "").strip()
    if not raw_b or not comp_p or not re.search(r"_{3,}", raw_b):
        return ""

    # 빈칸이 포함된 원문 문장 추출
    raw_masked = re.sub(r"_{3,}", " __BLANK_SENTINEL__ ", raw_b)
    raw_sents = _split_english_sentences(raw_masked)
    blank_sent = next((s for s in raw_sents if "__BLANK_SENTINEL__" in s), "")
    if not blank_sent:
        return ""

    left_part, right_part = blank_sent.split("__BLANK_SENTINEL__", 1)
    left_toks = [t.lower() for t in _extract_english_word_tokens(left_part)]
    right_toks = [t.lower() for t in _extract_english_word_tokens(right_part)]
    anchor_set = set(left_toks + right_toks)
    if not anchor_set:
        return ""

    # 완성 지문에서 빈칸 문장과 가장 일치하는 문장 탐색
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
    if not best_sent or best_score < 0.55:
        return ""

    # 완성 문장에서 left_toks 끝 ~ right_toks 시작 사이 원문 슬라이스 추출
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
        # 문장 전체가 아니라 합리적 빈칸 길이(1~22단어)일 때만 채택
        if 1 <= span_len <= 22 and span_len < len(cs_matches):
            phrase = best_sent[cs_matches[start_tok_idx].start() : cs_matches[end_tok_idx - 1].end()].strip()
            return phrase
    return ""


def _extract_english_exam_context(
    question_text: str,
    orig_question_type: str = "",
    answer: str = "",
    passage_text: str = "",
) -> dict[str, Any]:
    """
    원본 PDF 텍스트(question_text)와 정답(answer)을 대조해
    각주(footnotes), 원래 빈칸 구절(original_blank_phrase), 제거된 무관한 문장(irrelevant_sentence),
    글의 순서 정답 배열(order_answer_seq), 문장 삽입 원문/위치(inserted_sentence),
    그리고 원문 변조·누락 시 복구 가능한 무손실 기준 지문(reconstructed_clean_passage)을 결정론적으로 추출한다.
    """
    ctx: dict[str, Any] = {
        "footnotes": _extract_english_footnotes(question_text),
        "original_blank_phrase": "",
        "irrelevant_sentence": "",
        "irrelevant_answer_sym": "",
        "order_answer_seq": "",
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
    is_blank = ("빈칸" in q_type) or bool(re.search(r"_{3,}", body_and_choices))
    if is_blank:
        raw_body = body_and_choices
        split_res = _split_numbered_five_markers(body_and_choices)
        if split_res is not None:
            raw_body, segs = split_res
            choices_map = {str(k): v for k, v in segs.items() if v}
            ctx["choices"] = choices_map
            if ans_int and str(ans_int) in choices_map:
                ans_phrase = choices_map[str(ans_int)].strip()
                ans_phrase = re.sub(r"(?:\s+\d{1,2}){1,2}\s*$", "", ans_phrase).strip()
                ctx["original_blank_phrase"] = ans_phrase
                if re.search(r"_{3,}", raw_body) and ans_phrase:
                    filled = re.sub(r"\s*_{3,}\s*", f" {ans_phrase} ", raw_body, count=1)
                    filled = re.sub(r"\s+([.,?!;:])", r"\1", filled)
                    filled = re.sub(r"\s+", " ", filled).strip()
                    if _is_clean_english_reconstruction(filled, passage_text):
                        ctx["reconstructed_clean_passage"] = filled
        # 폴백: ①~⑤ 선지 추출이 깨졌더라도 빈칸 문장(____)과 완성 지문의 diff로 정답 구절 역추출
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
                ctx["irrelevant_sentence"] = irrel_sent
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
                rebuilt = re.sub(r"\s+", " ", " ".join(kept_parts)).strip()
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
            # (A), (B), (C) 단락 분할
            parts = re.split(r"(?:^|\s)\(([ABC])\)\s+", order_body)
            if len(parts) >= 7:
                given_txt = parts[0].strip()
                blocks: dict[str, str] = {}
                for i_p in range(1, len(parts) - 1, 2):
                    lbl = parts[i_p]
                    txt = parts[i_p + 1].strip()
                    if lbl in ("A", "B", "C") and txt:
                        blocks[lbl] = txt
                if given_txt and set(blocks.keys()) == {"A", "B", "C"} and set(seq) == {"A", "B", "C"}:
                    rebuilt = re.sub(
                        r"\s+",
                        " ",
                        " ".join([given_txt, blocks[seq[0]], blocks[seq[1]], blocks[seq[2]]]),
                    ).strip()
                    if _is_clean_english_reconstruction(rebuilt, passage_text):
                        ctx["reconstructed_clean_passage"] = rebuilt
        return ctx

    # 4) 문장 삽입 (38~39번)
    is_insertion = "삽입" in q_type
    if is_insertion:
        # ( ① ) ~ ( ⑤ ) 또는 ① ~ ⑤ 위치 마커 분할
        norm_ins = re.sub(r"\(\s*([①②③④⑤])\s*\)", r"\1", body_and_choices)
        split_res = _split_numbered_five_markers(norm_ins)
        if split_res is not None:
            prefix_raw, segs = split_res
            prefix_clean = re.sub(r"\[주어진\s*문장\]", "", prefix_raw).strip()
            pref_sents = _split_english_sentences(prefix_clean)
            if pref_sents:
                # 주어진 문장은 보통 박스 안의 첫 1문장(또는 완성 지문에서 뒤쪽 위치에 등장하는 앞 문장)
                given_sent = pref_sents[0]
                body_intro = " ".join(pref_sents[1:]).strip()
                # 완성 지문과 대조해 주어진 문장 경계 확인 (주어진 문장이 2문장인 드문 케이스 대비)
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
                ctx["inserted_sentence"] = given_sent
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
                    rebuilt = re.sub(r"\s+", " ", " ".join(rebuilt_parts)).strip()
                    if _is_clean_english_reconstruction(rebuilt, passage_text):
                        ctx["reconstructed_clean_passage"] = rebuilt
        return ctx

    return ctx


def _reconcile_english_passage_with_question_text(
    passage_text: str,
    question_text: str,
    orig_question_type: str = "",
    answer: str = "",
) -> tuple[str, dict[str, Any]]:
    """
    저장된 passage_text를 원본 question_text(+정답)와 결정론적으로 대조하여,
    원문 단어 변조·구절 누락·정상 문장 통누락이 감지되면 무손실 원본으로 자동 복구한다.
    원문 자체에 소괄호 `( ... )`가 포함된 경우에도 괄호와 전후 단어를 100% 보존한다.
    """
    raw_p = (passage_text or "").strip()
    exam_ctx = _extract_english_exam_context(
        question_text=question_text,
        orig_question_type=orig_question_type,
        answer=answer,
        passage_text=raw_p,
    )
    rebuilt = (exam_ctx.get("reconstructed_clean_passage") or "").strip()
    if not rebuilt:
        return raw_p, exam_ctx

    if not raw_p or _needs_english_passage_restoration(raw_p, orig_question_type):
        return rebuilt, exam_ctx

    p_toks = [t.lower() for t in _extract_english_word_tokens(raw_p)]
    r_toks = [t.lower() for t in _extract_english_word_tokens(rebuilt)]
    if not p_toks or not r_toks:
        return raw_p, exam_ctx

    # 원문과 토큰 시퀀스가 다르거나 원문 자체 괄호 구절이 유실된 경우 무손실 PDF 기준본으로 복구
    paren_lost = rebuilt.count("(") > raw_p.count("(") or rebuilt.count(")") > raw_p.count(")")
    if p_toks != r_toks or paren_lost:
        p_sents = _split_english_sentences(raw_p)
        r_sents = _split_english_sentences(rebuilt)
        sm = difflib.SequenceMatcher(None, p_toks, r_toks)
        if sm.ratio() >= 0.78 or (len(r_sents) == len(p_sents) + 1 and sm.ratio() >= 0.70):
            return rebuilt, exam_ctx

    return raw_p, exam_ctx


# ---------------------------------------------------------------------------
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


def _has_balanced_syntax_brackets(text: str) -> bool:
    """대괄호 []와 소괄호 ()의 열림/닫힘 짝과 중첩 순서가 정확히 맞는지 검사한다."""
    stack: list[str] = []
    pairs = {"]": "[", ")": "("}
    for ch in text:
        if ch in ("[", "("):
            stack.append(ch)
        elif ch in ("]", ")"):
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
    chk = _strip_spurious_quotes_against_source(src, (chunked_sent or "").strip())
    if not src:
        return chk
    if not chk:
        return src

    chk = _normalize_syntax_bracket_consistency(src, chk)

    # 이미 원문과 단어 100% 일치 + 괄호 균형 + 원문 자체 괄호 보존이면 정리 후 반환
    src_toks = _extract_english_word_tokens(src)
    chk_toks = _extract_english_word_tokens(chk)
    if (
        [t.lower() for t in src_toks] == [t.lower() for t in chk_toks]
        and _has_balanced_syntax_brackets(chk)
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
    restored = _normalize_syntax_bracket_consistency(src, restored)
    if (
        [t.lower() for t in _extract_english_word_tokens(restored)] == src_low
        and _has_balanced_syntax_brackets(restored)
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
    괄호 짝이 맞지 않으면 강제로 원문 무손실 투영/폴백하여 변조된 문장이 절대 통과하지 못하게 한다.
    """
    sents = list(source_sents or [])
    for idx, sa in enumerate(out.sentence_analyses or []):
        chk = (sa.chunked_english or "").strip()
        if idx < len(sents) and sents[idx].strip():
            src_s = sents[idx].strip()
            src_low = [t.lower() for t in _extract_english_word_tokens(src_s)]
            chk_clean = _strip_spurious_quotes_against_source(src_s, chk)
            chk_low = [t.lower() for t in _extract_english_word_tokens(chk_clean)]
            if (
                chk_low != src_low
                or not _has_balanced_syntax_brackets(chk_clean)
                or src_s.count("(") > chk_clean.count("(")
                or src_s.count(")") > chk_clean.count(")")
            ):
                chk_clean = _project_chunk_markup_onto_source(src_s, chk_clean)
            # 최종 2차 확인: 그래도 1단어라도 다르거나 괄호 불균형이면 원문 그대로 강제 폴백
            if (
                [t.lower() for t in _extract_english_word_tokens(chk_clean)] != src_low
                or not _has_balanced_syntax_brackets(chk_clean)
            ):
                chk_clean = src_s
            sa.chunked_english = chk_clean
        else:
            # 원문 문장 리스트가 없는 렌더링 폴백 경로에서도 최소한 괄호 짝 무결성은 보장
            if chk and not _has_balanced_syntax_brackets(chk):
                sa.chunked_english = _balance_added_markup_chars(chk)
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

    # 3. 문항 유형별 기출 포인트 결정론적 보강 (빈칸·무관한 문장·글의 순서·문장 삽입)
    blank_phrase = (ctx.get("original_blank_phrase") or "").strip()
    if blank_phrase:
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

    ins_sent = (ctx.get("inserted_sentence") or "").strip()
    ins_sym = (ctx.get("inserted_answer_sym") or "").strip()
    if ins_sent:
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


