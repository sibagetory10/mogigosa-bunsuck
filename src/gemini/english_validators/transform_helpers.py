from __future__ import annotations

import difflib
import re
from typing import Any, Sequence

from src.gemini.usage import _CIRCLED_DIGIT_TO_NUM, _normalize_extracted_answer
from src.models import RewrittenQuestion
from src.text_formatter import normalize_english_exam_text, strip_english_embedded_choices

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


_PDF_HYPHEN_CHARS_RE = re.compile(r"[\u00ad\u2010\u2011\u2012\u2212\uf02d]")
_PDF_ENDASH_ALNUM_RE = re.compile(r"(?<=[A-Za-z0-9])\u2013(?=[A-Za-z0-9])")
_DETERMINER_SET: frozenset[str] = frozenset(
    {"the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their", "some", "any", "no", "each", "every"}
)


def _normalize_pdf_hyphens(text: str) -> str:
    """PDF 추출 텍스트의 소프트 하이픈·PUA 하이픈·유니코드 대시를 ASCII '-'로 정규화한다."""
    if not text:
        return ""
    out = _PDF_HYPHEN_CHARS_RE.sub("-", text)
    out = _PDF_ENDASH_ALNUM_RE.sub("-", out)
    return out


def _strip_stray_exam_qnum_noise(text: str) -> str:
    """영어 본문 앞뒤·문장 끝에 혼입된 기출 문항번호(36., 38.) 및 쪽번호(7 39., 739.)를 제거한다."""
    s = (text or "").strip()
    if not s:
        return ""
    # 맨 앞: '7 39. ', '739. ', '36. ' (단독 문항번호 18~45 뒤 영문 대문자/따옴표/괄호 시작)
    s = re.sub(r"^(?:(?:[1-9]|1[0-2])\s+)?(?:1[89]|[234]\d)\.\s+(?=[A-Z\"'“‘(\[])", "", s)
    s = re.sub(r"^[1-9](?:1[89]|[234]\d)\.\s+(?=[A-Z\"'“‘(\[])", "", s)
    s = re.sub(r"^(?:1[89]|[234]\d)\.\s*$", "", s)
    # 문장 종결 부호(. ? !) 뒤 또는 문자열 끝에 붙은 기출 문항·쪽번호 잔여물('... 38.', '... 7 39.', '... 739.')
    s = re.sub(r"(?<=[.?!\"'”’\)])\s+(?:(?:[1-9]|1[0-2])\s+)?(?:1[89]|[234]\d)\.(?=\s|$)", "", s)
    s = re.sub(r"(?<=[.?!\"'”’\)])\s+[1-9](?:1[89]|[234]\d)\.(?=\s|$)", "", s)
    # 문장 사이에 단독 문항번호가 낀 경우 ('... end. 38. Next ...')
    s = re.sub(r"(?<=[.?!])\s+(?:(?:[1-9]|1[0-2])\s+)?(?:1[89]|[234]\d)\.\s+(?=[A-Z\"'“‘(\[])", " ", s)
    s = re.sub(r"(?<=[.?!])\s+[1-9](?:1[89]|[234]\d)\.\s+(?=[A-Z\"'“‘(\[])", " ", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def _preprocess_english_source_passage(passage: str) -> str:
    """변형 출제 전 원문 지문에 남아 있는 기존 기출 기호((A)(B)(C), (①)~(⑤), <u> 등)와 혼입된 문항·쪽번호를 제거해 순수 영문 지문으로 정규화한다."""
    txt = _normalize_pdf_hyphens((passage or "").strip())
    if not txt:
        return ""
    txt = re.sub(r"</?u\b[^>]*>", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"[ \t]*\(\s*[①②③④⑤]\s*\)[ \t]*", " ", txt)
    txt = re.sub(r"[ \t]*[①②③④⑤][ \t]*", " ", txt)
    # 문장/단락 시작의 기출 순서 라벨 (A), (B), (C), (D) 제거 (문맥상 괄호 약어는 보존)
    txt = re.sub(r"(?:^|(?<=\n)|(?<=[.?!\"'”’]\s))\(\s*[ABCD]\s*\)\s*", " ", txt)
    txt = _strip_stray_exam_qnum_noise(txt)
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
    for idx_t, tok in enumerate(toks):
        if tok in _COMMON_ENG_VERBS:
            return True
        if len(tok) >= 5 and tok.endswith(("ed", "ize", "ise", "ify", "ate")):
            return True
        if (
            len(tok) >= 5
            and tok.endswith("ing")
            and tok not in ("thing", "morning", "evening", "nothing", "something", "anything", "everything", "meaning", "feeling", "during")
        ):
            # 관사·지시사·소유격 바로 뒤의 '-ing'는 동명사/명사구(예: 'This intertwining')이므로 정동사에서 제외
            if idx_t > 0 and toks[idx_t - 1] in _DETERMINER_SET:
                continue
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
    joined = _strip_stray_exam_qnum_noise(re.sub(r"\s+", " ", " ".join(cleaned_lines)).strip())
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
