from __future__ import annotations

import difflib
import hashlib
import random
import re
from typing import Any, Sequence

from src.gemini.usage import _CIRCLED_DIGIT_TO_NUM, _normalize_extracted_answer
from src.models import (
    EnglishGistTransformSpec,
    EnglishInsertionTransformSpec,
    EnglishIrrelevantTransformSpec,
    EnglishOrderTransformSpec,
    EnglishSpanTargetTransformSpec,
    EnglishSummaryPairItem,
    EnglishSummaryTransformSpec,
    EnglishUnderlineFiveItem,
    EnglishUnderlineFiveTransformSpec,
    EnglishVocabChoiceTransformSpec,
    EnglishVocabChoiceTripleItem,
    RewrittenQuestion,
)
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
        "test", "tests", "tested", "study", "studies", "studied", "form", "forms", "formed",
        "act", "acts", "acted", "shape", "shapes", "shaped", "drive", "drives", "drove",
        "link", "links", "linked", "view", "views", "viewed", "note", "notes", "noted",
        "lack", "lacks", "lacked", "face", "faces", "faced", "pose", "poses", "posed",
        "seek", "seeks", "sought", "yield", "yields", "yielded", "rely", "relies", "relied",
        "apply", "applies", "applied", "imply", "implies", "implied", "vary", "varies", "varied",
        "adapt", "adapts", "adapted", "adopt", "adopts", "adopted", "alter", "alters", "altered",
        "define", "defines", "defined", "assume", "assumes", "assumed", "argue", "argues", "argued",
        "claim", "claims", "claimed", "state", "states", "stated", "prove", "proves", "proved",
        "focus", "focuses", "focused", "result", "results", "resulted", "share", "shares", "shared",
        "value", "values", "valued", "limit", "limits", "limited", "force", "forces", "forced",
        "place", "places", "placed", "point", "points", "pointed", "refer", "refers", "referred",
        "omit", "omits", "omitted", "emerge", "emerges", "emerged", "revise", "revises", "revised",
        "Free", "frees", "freed", "weaken", "weakens", "weakened", "invest", "invests", "invested",
        "emphasize", "emphasizes", "emphasized", "maintain", "maintains", "maintained",
        "produce", "produces", "produced", "develop", "develops", "developed", "support", "supports",
        "promote", "promotes", "promoted", "prevent", "prevents", "prevented", "protect", "protects",
        "reflect", "reflects", "reflected", "represent", "represents", "determine", "determines",
        "influence", "influences", "influenced", "encourage", "encourages", "discourage", "discourages",
        "recognize", "recognizes", "identify", "identifies", "achieve", "achieves", "generate", "generates",
        "enhance", "enhances", "impair", "impairs", "alleviate", "alleviates", "contribute", "contributes",
        "function", "functions", "operate", "operates", "respond", "responds", "react", "reacts",
        "cannot", "can't", "don't", "doesn't", "didn't", "won't", "wouldn't", "isn't", "aren't",
    }
)


def _sentence_has_english_verb(sentence_without_blank: str) -> bool:
    """빈칸을 제외한 문장에 영문 정동사(본동사/조동사)가 최소 1개 이상 남아있는지 판별한다 (단독 -ing 동명사/분사는 정동사에서 제외)."""
    toks = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", (sentence_without_blank or "").lower())
    for idx_t, tok in enumerate(toks):
        if tok in _COMMON_ENG_VERBS:
            return True
        if len(tok) >= 5 and tok.endswith(("ed", "ize", "ise", "ify", "ate")):
            # 관사/전치사/형용사 바로 뒤에서 명사를 수식하는 과거분사(예: 'for more complex analytical ...', 'under strict idealized equations') 오탐 방지
            if idx_t > 0 and toks[idx_t - 1] in (_DETERMINER_SET | _REPEATABLE_PREPS):
                if idx_t + 1 < len(toks) and toks[idx_t + 1].endswith(("s", "ion", "ions", "ment", "ments", "ty", "ties", "ness")):
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


# ---------------------------------------------------------------------------
# Transform Engine v2: 유형별 전용 Pydantic 스키마 매핑 & 100% 파이썬 결정론적 조립기
# ---------------------------------------------------------------------------
_NUM_TO_CIRCLED_SYM: dict[str, str] = {"1": "①", "2": "②", "3": "③", "4": "④", "5": "⑤"}


def _resolve_english_transform_schema_and_rule(
    target_type: str,
    blank_unit: str = "구 (3~6단어)",
    vocab_style: str = "밑줄형 (①~⑤)",
) -> tuple[str, type[Any], str]:
    """목표 변형 유형을 (정규화 유형명, 전용 Pydantic 스키마 클래스, 핵심 출제 지침)으로 매핑한다."""
    t = (target_type or "").strip()

    if "함축" in t:
        return (
            "함축적 의미 추론",
            EnglishSpanTargetTransformSpec,
            "- 목표: 원문에서 비유적·상징적·핵심적 함축 의미를 담은 연속 영어 구절 1곳(`target_span`)과 그 문장 번호(`target_sentence_index`)를 선정하세요.\n"
            "- `target_span`: 원문 철자 100% 그대로 2~8단어 연속 구절만 적으세요 (`<u>` 태그 없이 순수 텍스트).\n"
            "- `correct_choice`: 밑줄 구절이 문맥상 의미하는 바를 패러프레이징한 영어 정답 선지 1개.\n"
            "- `distractor_choices`: 정답과 문법 형태·길이가 비슷하지만 글의 핵심 논지와 반대되거나 지엽적인 매력적 영어 오답 선지 4개 (밑줄 원문 단어를 그대로 반복하지 마세요).",
        )
    if "요지" in t:
        return (
            "요지 파악",
            EnglishGistTransformSpec,
            "- 목표: 글 전체의 핵심 주장·요지를 가장 정확히 나타낸 **한국어 완결 문장 선지 5개**(`choices`, 번호 ①~⑤ 제외)와 정답 번호(`correct_index`: 1~5)를 작성하세요.\n"
            "- 오답 선지 4개는 본문 키워드를 일부 포함하되 인과 반전·과장·반대 주장으로 구성하세요.",
        )
    if "주제" in t:
        return (
            "주제 파악",
            EnglishGistTransformSpec,
            "- 목표: 글 전체의 중심 주제를 가장 정확히 나타낸 평가원 표준 **영어 명사구 선지 5개**(`choices`, 번호 ①~⑤ 제외)와 정답 번호(`correct_index`: 1~5)를 작성하세요.",
        )
    if "제목" in t:
        return (
            "제목 추론",
            EnglishGistTransformSpec,
            "- 목표: 글 전체의 핵심을 함축한 평가원 표준 **영어 제목 선지 5개**(`choices`, 번호 제외, 주요 단어 첫 글자 대문자 타이틀 형식)와 정답 번호(`correct_index`: 1~5)를 작성하세요.",
        )
    if "어법" in t:
        return (
            "어법상 틀린 것 (밑줄 ①~⑤)",
            EnglishUnderlineFiveTransformSpec,
            "- 목표: 원문 등장 순서대로 서로 다른 문장(또는 절)에서 어법 판단 포인트 5곳(`items` 정확히 5개)을 선정하세요.\n"
            "- 각 항목에는 문장 번호(`sentence_index`), 원문 구절(`original_span`, 1~3단어), 시험지 표시 구절(`displayed_span`), 정답 여부(`is_error_target`)를 적으세요.\n"
            "- **핵심 규칙**: 5개 중 **정확히 1개만** `is_error_target=true`로 설정하고 `displayed_span`을 명백한 문법 비문(① 주어-본동사 수일치, ② 정동사 자리 준동사 변형, ③ 분사 능동 `-ing` ↔ 수동 `-ed`, ④ 관계대명사 `which` ↔ 관계부사 `where`/접속사 `that` ↔ `what`, ⑤ 대동사 `do/does/did` ↔ `is/are/was/were` 중 택1)으로 바꾸세요.\n"
            "- 나머지 4개(`is_error_target=false`)는 반드시 `displayed_span`을 `original_span`과 100% 동일하게 유지하세요.\n"
            "- 절대 금지: 원문 정상 표현을 그대로 두고 틀렸다고 우기기, 의미만 반대인 반의어 교체, 단순 시제(과거↔현재) 변경 금지.",
        )
    if "어휘" in t:
        if "선택형" in vocab_style or "(A)" in vocab_style:
            return (
                "문맥상 어휘 적절성 (선택형 (A)(B)(C))",
                EnglishVocabChoiceTransformSpec,
                "- 목표: 원문 등장 순서대로 문맥 흐름 판단의 핵심이 되는 영단어 3곳(`items` 정확히 3개: (A), (B), (C))을 선정하세요.\n"
                "- 각 항목에는 문장 번호(`sentence_index`), 원문 단어(`original_word`), 문맥상 정반대 의미이면서 **품사·시제·수일치·어미 형태가 100% 동일한 자연스러운 반의어**(`antonym_word`)를 적으세요.\n"
                "- 금지: 어법 변형(`makes`↔`making`)이나 어색한 단어 조합 금지. 부정어(`not`/`never`/`hardly`) 바로 뒤에 걸려 반의어 교체 시 문맥이 모호해지는 단어는 피하세요.\n"
                "- 5지선다 조합과 정답 번호는 파이썬이 자동 생성하므로 `items` 3개와 `explanation`만 정확히 작성하세요.",
            )
        return (
            "문맥상 어휘 적절성 (밑줄 ①~⑤)",
            EnglishUnderlineFiveTransformSpec,
            "- 목표: 원문 등장 순서대로 문맥상 핵심 어휘(동사·형용사·부사·명사) 5곳(`items` 정확히 5개)을 선정하세요.\n"
            "- **핵심 규칙**: 5개 중 **정확히 1개만** `is_error_target=true`로 설정하고, `displayed_span`을 `original_span`과 품사·어미 형태가 100% 동일한 **문맥상 정반대 어휘(반의어)**로 교체하세요.\n"
            "- 나머지 4개(`is_error_target=false`)는 반드시 `displayed_span`을 `original_span`과 100% 동일한 원문 단어로 유지하세요.\n"
            "- 절대 금지: 정답 항목의 `displayed_span`을 원문 단어 그대로 두는 행위, 어법 변형(`make`→`making`) 금지.",
        )
    if "빈칸" in t:
        b_u = (blank_unit or "").strip()
        if b_u.startswith("절") or ("절(" in b_u or "절 (" in b_u):
            unit_desc = "핵심 **절 또는 긴 동사구(5~11단어, 단 문장 전체가 아닌 접속사절/보어절/목적어절)**"
            short_u = "절"
        elif b_u.startswith("단어") or ("단어" in b_u and not b_u.startswith("구")):
            unit_desc = "핵심 **단어(1~2단어 명사/형용사/동사원형)**"
            short_u = "단어"
        else:
            unit_desc = "핵심 **구(3~6단어 명사구/동사구/전치사구/보어구)**"
            short_u = "구"
        return (
            f"빈칸 추론 ({short_u})",
            EnglishSpanTargetTransformSpec,
            f"- 목표: 주제문 또는 핵심 논리 귀결 문장에서 빈칸으로 뚫을 **{unit_desc}** 1곳(`target_span`)과 해당 문장 번호(`target_sentence_index`)를 선정하세요.\n"
            "- **핵심 규칙**:\n"
            "  1) `target_span`은 해당 문장 안에 실제로 존재하는 연속 구절을 원문 철자 100% 그대로 적으세요.\n"
            "  2) 문장의 주어+본동사를 통째로 비워 빈칸 제외한 문장에 동사가 하나도 남지 않게 하지 마세요 (주어와 본동사/조동사는 문장에 남기고 목적어·보어·핵심 구절을 빈칸으로 선정).\n"
            "  3) `correct_choice`는 빈칸에 들어갈 정답 선지(원문 `target_span` 또는 문법적으로 100% 호환되는 패러프레이징)이며, 빈칸 직전/직후 단어와 절대 중복되지 않아야 합니다.\n"
            f"  4) `distractor_choices`는 `correct_choice`와 품사·문법 구조·길이가 동일한 매력적 영어 오답 선지 4개입니다.",
        )
    if "무관" in t:
        return (
            "전체 흐름과 무관한 문장",
            EnglishIrrelevantTransformSpec,
            "- 목표: 원문 문장은 단 한 문장도 삭제·변형하지 말고, `insert_position`(2, 3, 4 중 하나)에 삽입할 **흐름과 무관한 새 영어 문장 1개**(`unrelated_sentence`)를 작성하세요.\n"
            "- `unrelated_sentence`는 원문의 중심 소재 단어(키워드)만 표면적으로 포함하되, 글의 핵심 논지·방향과 전혀 다른 측면(예: 역사적 기원, 경제적 비용, 물리적 구조 등)을 서술하여 명확히 흐름에서 벗어나야 합니다.",
        )
    if "순서" in t:
        return (
            "글의 순서 ((A)-(B)-(C))",
            EnglishOrderTransformSpec,
            "- 목표: `[S0]`~`[Sn-1]`로 번호가 매겨진 원문 문장들을 `[주어진 글]` + `1번 단락` + `2번 단락` + `3번 단락`의 4덩어리로 나누는 분할 시작 인덱스 3개 `split_sentence_indices: [s1, s2, s3]`를 지정하세요.\n"
            "- 규칙: 반드시 `0 < s1 < s2 < s3 < 전체 문장 수`여야 하며, 각 단락 시작 문장에 지시어(`this`, `these`, `such`, `they`), 연결어(`however`, `for example`, `therefore`), 대명사 등 명확한 순서 단서가 걸리도록 분할하세요.",
        )
    if "삽입" in t:
        return (
            "주어진 문장 삽입",
            EnglishInsertionTransformSpec,
            "- 목표: `[S1]`~`[Sn-1]` 중 `[주어진 문장]` 박스로 발췌할 핵심 문장 번호 1개(`extracted_sentence_index`, 반드시 `1` 이상)를 선정하세요.\n"
            "- 규칙: 첫 도입문(`0`번)은 절대 선택하지 마세요. 연결어(`However`, `Instead`, `For example`, `Thus`, `In contrast`)나 지시어(`This`, `These`, `Such + 명사`)를 포함하여 앞뒤 문장과의 논리적 단절·연결 고리가 가장 뚜렷한 문장을 고르세요.",
        )
    return (
        "요약문 완성 ((A), (B))",
        EnglishSummaryTransformSpec,
        "- 목표: 지문 전체의 핵심 내용을 한 문장으로 압축한 영어 요약문(`summary_template`)과 빈칸 `(A)`, `(B)`에 들어갈 정답 어휘(`correct_word_a`, `correct_word_b`) 및 매력적 오답 어휘 쌍 4개(`distractor_pairs`)를 작성하세요.\n"
        "- **핵심 규칙**:\n"
        "  1) `summary_template` 안에는 반드시 `(A) ________`와 `(B) ________` 표기가 모두 포함되어야 합니다 (지문 본문이나 선지를 넣지 말고 오직 요약문 1문장만 작성).\n"
        "  2) `correct_word_a`, `correct_word_b`는 각 1단어(품사 일치)로 작성하되 원문 핵심어를 적절히 패러프레이징하세요.\n"
        "  3) `distractor_pairs`는 4개의 오답 `(word_a, word_b)` 쌍으로, 반의어·논리 역전 어휘를 조합하세요.",
    )


def _split_single_compound_sentence(sent: str) -> list[str]:
    """문장 수가 부족한 지문에서 세미콜론·대시·등위접속사절 경계를 기준으로 긴 복합문 1개를 2개 문장으로 안전 분할한다."""
    s = (sent or "").strip()
    if len(re.findall(r"[A-Za-z]+", s)) < 14:
        return [s]

    # 1순위: 세미콜론(;) 또는 긴 대시( — / -- )
    for pat in (r";\s+", r"\s+(?:—|--)\s+"):
        for m in re.finditer(pat, s):
            left = s[: m.start()].strip()
            right = s[m.end() :].strip()
            left_w = re.findall(r"[A-Za-z]+", left)
            right_w = re.findall(r"[A-Za-z]+", right)
            if (
                len(left_w) >= 5
                and len(right_w) >= 5
                and _sentence_has_english_verb(left)
                and _sentence_has_english_verb(right)
                and left.count("(") == left.count(")")
                and left.count('"') % 2 == 0
            ):
                if not re.search(r"[.?!]$", left):
                    left += "."
                right_cap = right[0].upper() + right[1:] if right else right
                return [left, right_cap]

    # 2순위: ', and / , but / , yet / , so / , for / , while / , whereas / , because / , although / , since ' + 절
    conj_re = re.compile(
        r",\s+(?=(?:and|but|yet|so|for|while|whereas|although|though|because|since)\b\s+[A-Za-z])",
        re.IGNORECASE,
    )
    for m in conj_re.finditer(s):
        left = s[: m.start()].strip()
        right = s[m.end() :].strip()
        left_w = re.findall(r"[A-Za-z]+", left)
        right_w = re.findall(r"[A-Za-z]+", right)
        if (
            len(left_w) >= 5
            and len(right_w) >= 5
            and _sentence_has_english_verb(left)
            and _sentence_has_english_verb(right)
            and left.count("(") == left.count(")")
            and left.count('"') % 2 == 0
        ):
            if not re.search(r"[.?!]$", left):
                left += "."
            # 등위/종속접속사로 시작하는 절을 독립 문장으로 바꿀 때 연결 부사로 자연스럽게 보정
            right_norm = re.sub(r"^and\b\s+", "", right, flags=re.IGNORECASE)
            right_norm = re.sub(r"^(?:but|yet)\b\s+", "However, ", right_norm, flags=re.IGNORECASE)
            right_norm = re.sub(r"^so\b\s+", "Therefore, ", right_norm, flags=re.IGNORECASE)
            right_norm = re.sub(r"^(?:because|since|for)\b\s+", "This is because ", right_norm, flags=re.IGNORECASE)
            right_norm = re.sub(r"^(?:although|though|while|whereas)\b\s+", "However, ", right_norm, flags=re.IGNORECASE)
            right_cap = right_norm[0].upper() + right_norm[1:] if right_norm else right_norm
            return [left, right_cap]

    return [s]


def _expand_compound_sentences_for_structure(sents: list[str], min_target_count: int) -> list[str]:
    """순서(4문장)·무관(5문장)·삽입(6문장) 출제에 문장 수가 부족할 때 긴 복합문을 절 경계에서 안전 분할한다."""
    cur = [s.strip() for s in sents if s and s.strip()]
    if len(cur) >= min_target_count or not cur:
        return cur

    for _ in range(min_target_count - len(cur) + 2):
        if len(cur) >= min_target_count:
            break
        # 가장 단어 수가 많은 문장부터 분할 시도
        ranked_indices = sorted(
            range(len(cur)),
            key=lambda i: len(re.findall(r"[A-Za-z]+", cur[i])),
            reverse=True,
        )
        split_done = False
        for idx in ranked_indices:
            pieces = _split_single_compound_sentence(cur[idx])
            if len(pieces) > 1:
                cur = cur[:idx] + pieces + cur[idx + 1 :]
                split_done = True
                break
        if not split_done:
            break
    return cur


def _prepare_indexed_sentences_for_transform(
    clean_passage: str,
    canonical_type: str = "",
) -> tuple[str, list[str], list[tuple[int, int]]]:
    """
    원문 영어 지문을 문장 리스트와 각 문장의 캐릭터 오프셋 `[(start, end), ...]`으로 정규화한다.
    순서/무관/삽입 유형에서 문장 수가 부족한 복합문 지문은 절 경계 안전 분할을 적용한다.
    """
    sents = _split_english_sentences(clean_passage)
    t = canonical_type or ""
    if "삽입" in t and len(sents) < 6:
        sents = _expand_compound_sentences_for_structure(sents, 6)
    elif "무관" in t and len(sents) < 5:
        sents = _expand_compound_sentences_for_structure(sents, 5)
    elif "순서" in t and len(sents) < 4:
        sents = _expand_compound_sentences_for_structure(sents, 4)

    if not sents:
        norm = re.sub(r"\s+", " ", (clean_passage or "").strip())
        return norm, ([norm] if norm else []), ([(0, len(norm))] if norm else [])

    spans: list[tuple[int, int]] = []
    parts: list[str] = []
    cursor = 0
    for idx, s in enumerate(sents):
        s_clean = re.sub(r"\s+", " ", s.strip())
        if idx > 0:
            parts.append(" ")
            cursor += 1
        st = cursor
        parts.append(s_clean)
        cursor += len(s_clean)
        spans.append((st, cursor))
    norm_passage = "".join(parts)
    return norm_passage, [norm_passage[a:b] for a, b in spans], spans


def _clean_candidate_span_text(raw_span: str) -> str:
    s = (raw_span or "").strip()
    s = re.sub(r"</?u\b[^>]*>", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"^\s*(?:[1-5①②③④⑤]|\([ABC1-5]\))\s*[:.)]?\s*", "", s).strip()
    s = s.strip("`\"'“”‘’[]")
    s = re.sub(r"^_{2,}\s*|\s*_{2,}$", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _overlaps_any(st: int, en: int, exclude_ranges: Sequence[tuple[int, int]]) -> bool:
    for ex_s, ex_e in exclude_ranges:
        if max(st, ex_s) < min(en, ex_e):
            return True
    return False


def _find_span_in_window(
    passage: str,
    win_start: int,
    win_end: int,
    target_span: str,
    exclude_ranges: Sequence[tuple[int, int]] = (),
) -> tuple[int, int, str] | None:
    """지정된 구간 `[win_start, win_end]` 안에서 `target_span`과 매칭되는 원문 `(abs_start, abs_end, matched_text)`를 찾는다."""
    seg = passage[win_start:win_end]
    if not seg or not target_span:
        return None

    # 1. 단어 경계 정확 매칭 -> 대소문자 무시 매칭
    boundary_pat = rf"(?<![A-Za-z]){re.escape(target_span)}(?![A-Za-z])"
    for flags in (0, re.IGNORECASE):
        for m in re.finditer(boundary_pat, seg, flags=flags):
            abs_s, abs_e = win_start + m.start(), win_start + m.end()
            if not _overlaps_any(abs_s, abs_e, exclude_ranges):
                return (abs_s, abs_e, passage[abs_s:abs_e])

    # 2. 구두점/하이픈/공백 유연 정규식 매칭
    tokens = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", target_span)
    if tokens:
        flex_pat = r"(?<![A-Za-z])" + r"[\s\-,;:'\"“”‘’()]+".join(re.escape(tk) for tk in tokens) + r"(?![A-Za-z])"
        for m in re.finditer(flex_pat, seg, flags=re.IGNORECASE):
            abs_s, abs_e = win_start + m.start(), win_start + m.end()
            if not _overlaps_any(abs_s, abs_e, exclude_ranges):
                return (abs_s, abs_e, passage[abs_s:abs_e])

    # 3. 토큰 슬라이딩 윈도우 퍼지 매칭 (2단어 이상 구절에서 사소한 관사/철자 차이 보정)
    if len(tokens) >= 2:
        seg_tok_matches = list(re.finditer(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", seg))
        seg_toks_low = [m.group(0).lower() for m in seg_tok_matches]
        target_low = [t.lower() for t in tokens]
        k = len(target_low)
        best_ratio = 0.0
        best_span: tuple[int, int, str] | None = None
        for w_len in (k, max(1, k - 1), k + 1):
            if w_len > len(seg_tok_matches):
                continue
            for i in range(len(seg_tok_matches) - w_len + 1):
                abs_s = win_start + seg_tok_matches[i].start()
                abs_e = win_start + seg_tok_matches[i + w_len - 1].end()
                if _overlaps_any(abs_s, abs_e, exclude_ranges):
                    continue
                cand_low = seg_toks_low[i : i + w_len]
                ratio = difflib.SequenceMatcher(None, target_low, cand_low).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_span = (abs_s, abs_e, passage[abs_s:abs_e])
        if best_span is not None and best_ratio >= 0.78:
            return best_span

    return None


def _locate_span_in_indexed_passage(
    passage: str,
    sent_spans: Sequence[tuple[int, int]],
    sentence_index: int,
    raw_span: str,
    min_cursor: int = 0,
    exclude_ranges: Sequence[tuple[int, int]] = (),
) -> tuple[int, int, str] | None:
    """
    AI가 지정한 `sentence_index` 문장 내부를 1순위로 탐색하고,
    인접 문장(`sentence_index ± 1`) → 전체 지문 순으로 원문 스팬을 결정론적으로 위치시킨다.
    """
    clean_span = _clean_candidate_span_text(raw_span)
    if not clean_span or not passage:
        return None

    # 끝의 마침표/쉼표가 불필요하게 붙은 후보도 함께 준비
    span_variants = [clean_span]
    stripped_punct = clean_span.strip(".,;!?")
    if stripped_punct and stripped_punct != clean_span:
        span_variants.append(stripped_punct)

    # 1순위: 지정된 sentence_index 문장 내부
    if 0 <= sentence_index < len(sent_spans):
        s_st, s_en = sent_spans[sentence_index]
        eff_st = max(s_st, min_cursor)
        if eff_st < s_en:
            for sp_v in span_variants:
                hit = _find_span_in_window(passage, eff_st, s_en, sp_v, exclude_ranges)
                if hit is not None:
                    return hit

    # 2순위: 인접 문장 (AI가 문장 번호를 ±1 오차로 지정한 경우)
    for delta in (-1, 1, -2, 2):
        nb_idx = sentence_index + delta
        if 0 <= nb_idx < len(sent_spans):
            s_st, s_en = sent_spans[nb_idx]
            eff_st = max(s_st, min_cursor)
            if eff_st < s_en:
                for sp_v in span_variants:
                    hit = _find_span_in_window(passage, eff_st, s_en, sp_v, exclude_ranges)
                    if hit is not None:
                        return hit

    # 3순위: min_cursor 이후 전체 지문 탐색
    if min_cursor < len(passage):
        for sp_v in span_variants:
            hit = _find_span_in_window(passage, min_cursor, len(passage), sp_v, exclude_ranges)
            if hit is not None:
                return hit

    # 4순위: min_cursor 무시하고 전체 지문에서 비중복 위치 탐색 (순서 뒤바뀐 항목을 사후 정렬하기 위함)
    if min_cursor > 0:
        for sp_v in span_variants:
            hit = _find_span_in_window(passage, 0, len(passage), sp_v, exclude_ranges)
            if hit is not None:
                return hit

    return None


def _sync_explanation_answer_number(explanation: str, ans_num: str) -> str:
    """파이썬이 확정한 정답 번호(`1`~`5`)와 해설 내 정답 번호 표기를 일치시킨다."""
    exp = (explanation or "").strip()
    if ans_num not in _NUM_TO_CIRCLED_SYM:
        return exp
    c_sym = _NUM_TO_CIRCLED_SYM[ans_num]
    if not exp:
        return f"정답은 {c_sym}번입니다."
    exp = re.sub(
        r"(정답\s*(?:은|은\s*바로|:)?\s*)(?:[1-5]\s*번|[①②③④⑤](?:\s*번)?)",
        rf"\g<1>{c_sym}번",
        exp,
    )
    return exp


def _clean_choice_item(text: str) -> str:
    s = re.sub(r"\s+", " ", (text or "").strip())
    s = re.sub(r"^(?:[①②③④⑤]|[1-5][.)])\s*", "", s).strip()
    return s


def _trim_blank_span_to_preserve_verb(
    passage: str,
    st_i: int,
    en_i: int,
    blank_unit: str,
) -> tuple[int, int, str]:
    """
    빈칸 추론에서 선택된 구절(`passage[st_i:en_i]`)을 비웠을 때 문장에 영문 동사가 하나도 남지 않는 경우,
    구절 앞쪽의 주어/본동사/조동사를 문장에 남겨 문장 동사가 보존되도록 빈칸 범위를 자동 트리밍한다.
    """
    orig_seg = passage[st_i:en_i]
    pre_txt = passage[:st_i]
    post_txt = passage[en_i:]
    s_start = max(pre_txt.rfind("\n"), pre_txt.rfind(". "), pre_txt.rfind("? "), pre_txt.rfind("! "))
    sent_left = pre_txt[s_start + 1 :] if s_start != -1 else pre_txt
    m_end = re.search(r"[.?!](?=\s|$|[\"'”’\)])|\n", post_txt)
    sent_right = post_txt[: m_end.start()] if m_end else post_txt
    rem_sentence = f"{sent_left} {sent_right}".strip()

    if _sentence_has_english_verb(rem_sentence):
        return st_i, en_i, orig_seg

    # 빈칸 내부에서 첫 동사(조동사/본동사)를 찾아 그 동사까지 문장에 남기고 뒤쪽 구절만 빈칸으로 뚫는다.
    tok_matches = list(re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?", orig_seg))
    if len(tok_matches) >= 3:
        for idx_m, m_tk in enumerate(tok_matches[:-1]):
            w_low = m_tk.group(0).lower()
            is_v = w_low in _COMMON_ENG_VERBS or (len(w_low) >= 5 and w_low.endswith(("ed", "ize", "ise", "ify", "ate")))
            if is_v:
                # 조동사 + 본동사 연쇄(예: 'can make', 'will become')면 다음 동사까지 포함
                cut_tok_idx = idx_m
                if idx_m + 1 < len(tok_matches) - 1 and tok_matches[idx_m + 1].group(0).lower() in _COMMON_ENG_VERBS:
                    cut_tok_idx = idx_m + 1
                next_tok_start = tok_matches[cut_tok_idx + 1].start()
                new_st = st_i + next_tok_start
                new_seg = passage[new_st:en_i].strip()
                if len(re.findall(r"[A-Za-z]+", new_seg)) >= 1:
                    return new_st, en_i, passage[new_st:en_i]

    return st_i, en_i, orig_seg


def _dedup_and_trim_blank_choice_overlap(
    passage: str,
    st_i: int,
    en_i: int,
    choice_text: str,
) -> str:
    """빈칸 앞뒤 단어/전치사와 선지 시작/끝 단어가 중복되는 경우 선지에서 중복 경계어를 자동 제거한다."""
    c = _clean_choice_item(choice_text).strip(".,;!?")
    if not c:
        return c
    before_toks = re.findall(r"[a-zA-Z\-']+", passage[:st_i].lower())
    after_toks = re.findall(r"[a-zA-Z\-']+", passage[en_i:].lower())
    c_words = c.split()
    c_toks = [re.sub(r"[^a-zA-Z\-']", "", w).lower() for w in c_words]

    # 앞쪽 2단어 또는 직전 1단어 중복 제거
    if len(c_words) > 2 and len(before_toks) >= 2 and before_toks[-2:] == c_toks[:2]:
        c_words = c_words[2:]
        c_toks = c_toks[2:]
    elif len(c_words) > 1 and before_toks and c_toks and before_toks[-1] == c_toks[0]:
        c_words = c_words[1:]
        c_toks = c_toks[1:]

    # 뒤쪽 2단어 또는 직후 1단어 중복 제거
    if len(c_words) > 2 and len(after_toks) >= 2 and after_toks[:2] == c_toks[-2:]:
        c_words = c_words[:-2]
        c_toks = c_toks[:-2]
    elif len(c_words) > 1 and after_toks and c_toks and after_toks[0] == c_toks[-1]:
        c_words = c_words[:-1]

    return " ".join(c_words).strip()


def _deterministic_rng(seed_text: str) -> random.Random:
    seed_int = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:12], 16)
    return random.Random(seed_int)


def _assemble_rewritten_question_from_spec(
    spec: Any,
    canonical_type: str,
    clean_passage: str,
    sent_list: Sequence[str],
    sent_spans: Sequence[tuple[int, int]],
    blank_unit: str = "구 (3~6단어)",
    vocab_style: str = "밑줄형 (①~⑤)",
    attempt_seed: int = 0,
) -> RewrittenQuestion:
    """
    유형별 전용 Pydantic 스키마(`spec`)를 원문(`clean_passage`)과 결정론적으로 결합하여
    완성된 `RewrittenQuestion` 객체를 조립한다.
    """
    t = (canonical_type or "").strip()
    rng = _deterministic_rng(f"{clean_passage[:200]}||{t}||{attempt_seed}")
    key_terms = [str(k).strip() for k in (getattr(spec, "key_terms", None) or []) if str(k).strip()][:3]
    raw_exp = (getattr(spec, "explanation", "") or "").strip()

    # 1. 요지 / 주제 / 제목 (EnglishGistTransformSpec)
    if isinstance(spec, EnglishGistTransformSpec):
        instr = _ensure_korean_instruction_prefix(clean_passage, t, vocab_style).splitlines()[0]
        raw_choices = [_clean_choice_item(c) for c in (spec.choices or []) if _clean_choice_item(c)]
        while len(raw_choices) < 5:
            raw_choices.append(f"Option {len(raw_choices) + 1}")
        choices_5 = raw_choices[:5]
        ans_idx = min(5, max(1, int(spec.correct_index or 1)))
        ans_str = str(ans_idx)
        return RewrittenQuestion(
            rewritten_stem=f"{instr}\n{clean_passage}",
            rewritten_choices=choices_5,
            replaced_terms_map=key_terms,
            pdf_grounded_explanation=_sync_explanation_answer_number(raw_exp, ans_str),
            preserved_answer=ans_str,
            self_verification_note=f"{t} 결정론적 조립 완료",
            target_question_type=t,
            subject_domain="영어",
        )

    # 2. 빈칸 추론 / 함축적 의미 추론 (EnglishSpanTargetTransformSpec)
    if isinstance(spec, EnglishSpanTargetTransformSpec):
        hit = _locate_span_in_indexed_passage(
            clean_passage,
            sent_spans,
            int(spec.target_sentence_index or 0),
            spec.target_span,
        )
        if hit is None and spec.correct_choice:
            hit = _locate_span_in_indexed_passage(
                clean_passage,
                sent_spans,
                int(spec.target_sentence_index or 0),
                spec.correct_choice,
            )
        if hit is None:
            raise ValueError(f"원문에서 목표 구절({spec.target_span!r})을 찾을 수 없습니다.")

        st_i, en_i, orig_seg = hit
        if "빈칸" in t:
            st_i, en_i, orig_seg = _trim_blank_span_to_preserve_verb(clean_passage, st_i, en_i, blank_unit)
            replaced_body = f"{clean_passage[:st_i]}___________{clean_passage[en_i:]}"
            stem = f"다음 빈칸에 들어갈 말로 가장 적절한 것은?\n{replaced_body}"
            corr_txt = _dedup_and_trim_blank_choice_overlap(
                clean_passage,
                st_i,
                en_i,
                spec.correct_choice or orig_seg,
            ) or orig_seg.strip(".,;!?")
            dist_list = [
                _dedup_and_trim_blank_choice_overlap(clean_passage, st_i, en_i, d)
                for d in (spec.distractor_choices or [])
            ]
        else:
            replaced_body = f"{clean_passage[:st_i]}<u>{orig_seg}</u>{clean_passage[en_i:]}"
            stem = f"밑줄 친 부분이 다음 글에서 의미하는 바로 가장 적절한 것은?\n{replaced_body}"
            corr_txt = _clean_choice_item(spec.correct_choice).strip(".")
            dist_list = [_clean_choice_item(d).strip(".") for d in (spec.distractor_choices or [])]

        unique_dists: list[str] = []
        seen_low = {corr_txt.lower()}
        for d in dist_list:
            if d and d.lower() not in seen_low:
                seen_low.add(d.lower())
                unique_dists.append(d)
        if len(unique_dists) < 4:
            raise ValueError(f"중복되지 않는 오답 선지 4개가 필요합니다 (현재 {len(unique_dists)}개).")

        tagged_choices = [(True, corr_txt)] + [(False, d) for d in unique_dists[:4]]
        rng.shuffle(tagged_choices)
        final_choices = [txt for _, txt in tagged_choices]
        ans_num = str(next(i + 1 for i, (is_c, _) in enumerate(tagged_choices) if is_c))

        return RewrittenQuestion(
            rewritten_stem=stem,
            rewritten_choices=final_choices,
            replaced_terms_map=key_terms,
            pdf_grounded_explanation=_sync_explanation_answer_number(raw_exp, ans_num),
            preserved_answer=ans_num,
            self_verification_note=f"{t} 구절 매칭 완료",
            target_question_type=t,
            subject_domain="영어",
        )

    # 3. 어법 밑줄 ①~⑤ / 어휘 밑줄 ①~⑤ (EnglishUnderlineFiveTransformSpec)
    if isinstance(spec, EnglishUnderlineFiveTransformSpec):
        raw_items = list(spec.items or [])
        if len(raw_items) != 5:
            raise ValueError(f"밑줄 항목(items)은 정확히 5개여야 합니다 (현재 {len(raw_items)}개).")

        matched_records: list[dict[str, Any]] = []
        used_ranges: list[tuple[int, int]] = []
        min_cur = 0
        for item in raw_items:
            orig_s = _clean_candidate_span_text(item.original_span)
            disp_s = _clean_candidate_span_text(item.displayed_span) or orig_s
            hit = _locate_span_in_indexed_passage(
                clean_passage,
                sent_spans,
                int(item.sentence_index or 0),
                orig_s,
                min_cursor=min_cur,
                exclude_ranges=used_ranges,
            )
            if hit is None:
                raise ValueError(f"밑줄 원문 구절({orig_s!r}, 문장 S{item.sentence_index})을 원문에서 찾을 수 없습니다.")
            st_i, en_i, exact_orig = hit
            used_ranges.append((st_i, en_i))
            min_cur = en_i
            is_changed = disp_s.lower() != exact_orig.lower() and disp_s.lower() != orig_s.lower()
            matched_records.append(
                {
                    "start": st_i,
                    "end": en_i,
                    "exact_orig": exact_orig,
                    "disp": disp_s if is_changed else exact_orig,
                    "is_changed": is_changed,
                    "is_target_flag": bool(item.is_error_target),
                }
            )

        # 지문 등장 순서(start 오름차순)로 자동 정렬하여 ①~⑤ 번호 역전 원천 방지
        matched_records.sort(key=lambda r: r["start"])

        changed_slots = [idx for idx, r in enumerate(matched_records) if r["is_changed"]]
        flagged_slots = [idx for idx, r in enumerate(matched_records) if r["is_target_flag"]]

        if len(changed_slots) == 1:
            ans_slot = changed_slots[0]
        elif len(changed_slots) > 1 and len(flagged_slots) == 1 and flagged_slots[0] in changed_slots:
            # 정답 플래그가 찍힌 1개만 변형 유지하고 나머지는 원문으로 자동 복구
            ans_slot = flagged_slots[0]
            for idx, r in enumerate(matched_records):
                if idx != ans_slot:
                    r["disp"] = r["exact_orig"]
                    r["is_changed"] = False
        else:
            raise ValueError(
                f"5개 밑줄 중 정확히 1개만 변형(오답/반의어)되어야 합니다 (현재 변형된 항목 수: {len(changed_slots)}개)."
            )

        syms = ["①", "②", "③", "④", "⑤"]
        out_parts: list[str] = []
        display_words: list[str] = []
        cursor = 0
        for idx, rec in enumerate(matched_records):
            st_i, en_i = rec["start"], rec["end"]
            out_parts.append(clean_passage[cursor:st_i])
            shown = rec["disp"] if idx == ans_slot else rec["exact_orig"]
            out_parts.append(f"{syms[idx]} <u>{shown}</u>")
            display_words.append(shown)
            cursor = en_i
        out_parts.append(clean_passage[cursor:])
        assembled_body = "".join(out_parts)

        is_grammar = "어법" in t
        instr = (
            "다음 글의 밑줄 친 부분 중, 어법상 틀린 것은?"
            if is_grammar
            else "다음 글의 밑줄 친 부분 중, 문맥상 낱말의 쓰임이 적절하지 않은 것은?"
        )
        ans_num = str(ans_slot + 1)
        ans_rec = matched_records[ans_slot]
        change_map_entry = f"{ans_rec['exact_orig']} → {ans_rec['disp']}"
        merged_terms = [change_map_entry] + [k for k in key_terms if "→" in k and k != change_map_entry]

        return RewrittenQuestion(
            rewritten_stem=f"{instr}\n{assembled_body}",
            rewritten_choices=list(syms) if is_grammar else display_words,
            replaced_terms_map=merged_terms[:3],
            pdf_grounded_explanation=_sync_explanation_answer_number(raw_exp, ans_num),
            preserved_answer=ans_num,
            self_verification_note=f"{t} 5개 밑줄 조립 완료 (정답 {ans_num}번: {change_map_entry})",
            target_question_type=t,
            subject_domain="영어",
        )

    # 4. 선택형 어휘 (A)(B)(C) (EnglishVocabChoiceTransformSpec)
    if isinstance(spec, EnglishVocabChoiceTransformSpec):
        raw_items = list(spec.items or [])
        if len(raw_items) != 3:
            raise ValueError(f"선택형 어휘 항목(items)은 정확히 3개((A),(B),(C))여야 합니다 (현재 {len(raw_items)}개).")

        matched_triples: list[dict[str, Any]] = []
        used_ranges = []
        min_cur = 0
        for item in raw_items:
            orig_w = _clean_candidate_span_text(item.original_word)
            ant_w = _clean_candidate_span_text(item.antonym_word)
            if not orig_w or not ant_w or orig_w.lower() == ant_w.lower():
                raise ValueError(f"선택형 어휘 쌍({orig_w!r} / {ant_w!r})이 유효하지 않거나 서로 동일합니다.")
            hit = _locate_span_in_indexed_passage(
                clean_passage,
                sent_spans,
                int(item.sentence_index or 0),
                orig_w,
                min_cursor=min_cur,
                exclude_ranges=used_ranges,
            )
            if hit is None:
                raise ValueError(f"선택형 어휘 원문 단어({orig_w!r}, 문장 S{item.sentence_index})를 원문에서 찾을 수 없습니다.")
            st_i, en_i, exact_orig = hit
            # 원문 단어가 대문자로 시작하면 반의어도 대문자로, 소문자면 소문자로 케이스 통일
            if exact_orig and exact_orig[0].isupper():
                ant_norm = ant_w[0].upper() + ant_w[1:]
                orig_norm = exact_orig
            else:
                ant_norm = ant_w[0].lower() + ant_w[1:] if ant_w else ant_w
                orig_norm = exact_orig
            used_ranges.append((st_i, en_i))
            min_cur = en_i
            matched_triples.append(
                {
                    "start": st_i,
                    "end": en_i,
                    "orig": orig_norm,
                    "ant": ant_norm,
                }
            )

        matched_triples.sort(key=lambda r: r["start"])
        labels = ["(A)", "(B)", "(C)"]
        out_parts = []
        cursor = 0
        for idx, rec in enumerate(matched_triples):
            st_i, en_i = rec["start"], rec["end"]
            out_parts.append(clean_passage[cursor:st_i])
            pair = [rec["orig"], rec["ant"]]
            rng.shuffle(pair)
            out_parts.append(f"{labels[idx]} [{pair[0]} / {pair[1]}]")
            cursor = en_i
        out_parts.append(clean_passage[cursor:])
        assembled_body = "".join(out_parts)

        wA_c, wA_w = matched_triples[0]["orig"], matched_triples[0]["ant"]
        wB_c, wB_w = matched_triples[1]["orig"], matched_triples[1]["ant"]
        wC_c, wC_w = matched_triples[2]["orig"], matched_triples[2]["ant"]

        # 평가원 표준 5지선다 조합 구성 (정답 1개 + 매력적 오답 조합 4개)
        combo_tuples = [
            (True, (wA_c, wB_c, wC_c)),
            (False, (wA_c, wB_w, wC_c)),
            (False, (wA_c, wB_w, wC_w)),
            (False, (wA_w, wB_c, wC_w)),
            (False, (wA_w, wB_w, wC_c)),
        ]
        rng.shuffle(combo_tuples)
        # (A) 첫 단어 기준으로 정렬하여 실제 수능 선지 배열처럼 깔끔하게 정렬
        combo_tuples.sort(key=lambda x: (x[1][0].lower(), x[1][1].lower(), x[1][2].lower()))
        choices_5 = [f"(A) {a} — (B) {b} — (C) {c}" for _, (a, b, c) in combo_tuples]
        ans_num = str(next(i + 1 for i, (is_c, _) in enumerate(combo_tuples) if is_c))

        return RewrittenQuestion(
            rewritten_stem=(
                "(A), (B), (C)의 각 네모 안에서 문맥에 맞는 낱말로 가장 적절한 것은?\n"
                f"{assembled_body}"
            ),
            rewritten_choices=choices_5,
            replaced_terms_map=[
                f"(A) {wA_c} ↔ {wA_w}",
                f"(B) {wB_c} ↔ {wB_w}",
                f"(C) {wC_c} ↔ {wC_w}",
            ],
            pdf_grounded_explanation=_sync_explanation_answer_number(raw_exp, ans_num),
            preserved_answer=ans_num,
            self_verification_note=f"선택형 어휘 (A)(B)(C) 조립 완료 (정답 {ans_num}번)",
            target_question_type="문맥상 어휘 적절성 (선택형 (A)(B)(C))",
            subject_domain="영어",
        )

    # 5. 글의 순서 (EnglishOrderTransformSpec)
    if isinstance(spec, EnglishOrderTransformSpec):
        sents = list(sent_list)
        n_sents = len(sents)
        if n_sents < 4:
            raise ValueError(f"글의 순서 유형은 최소 4문장 이상인 지문에서만 출제할 수 있습니다 (현재 {n_sents}문장).")

        raw_splits = sorted({int(x) for x in (spec.split_sentence_indices or []) if 0 < int(x) < n_sents})
        if len(raw_splits) >= 3:
            s1, s2, s3 = raw_splits[0], raw_splits[1], raw_splits[2]
        else:
            # AI 인덱스가 부정확한 경우 문장 수에 비례해 균등 4분할 자동 계산
            s1 = max(1, n_sents // 4)
            s2 = max(s1 + 1, (n_sents * 2) // 4)
            s3 = max(s2 + 1, min(n_sents - 1, (n_sents * 3) // 4))

        if not (0 < s1 < s2 < s3 < n_sents):
            s1, s2, s3 = 1, 2, 3

        intro = " ".join(sents[:s1]).strip()
        orig_blocks = [
            " ".join(sents[s1:s2]).strip(),
            " ".join(sents[s2:s3]).strip(),
            " ".join(sents[s3:]).strip(),
        ]
        order_specs = [
            ("1", "(A) - (C) - (B)", (0, 2, 1)),
            ("2", "(B) - (A) - (C)", (1, 0, 2)),
            ("3", "(B) - (C) - (A)", (2, 0, 1)),
            ("4", "(C) - (A) - (B)", (1, 2, 0)),
            ("5", "(C) - (B) - (A)", (2, 1, 0)),
        ]
        ans_num, ans_seq_str, perm = rng.choice(order_specs)
        block_a = orig_blocks[perm[0]]
        block_b = orig_blocks[perm[1]]
        block_c = orig_blocks[perm[2]]
        stem = (
            "주어진 글 다음에 이어질 글의 순서로 가장 적절한 것은?\n"
            "[주어진 글]\n"
            f"{intro}\n\n"
            f"(A) {block_a}\n\n"
            f"(B) {block_b}\n\n"
            f"(C) {block_c}"
        )
        choices_5 = [
            "(A) - (C) - (B)",
            "(B) - (A) - (C)",
            "(B) - (C) - (A)",
            "(C) - (A) - (B)",
            "(C) - (B) - (A)",
        ]
        exp_out = _ORDER_SEQ_RE.sub(ans_seq_str, raw_exp) if raw_exp else ""
        if ans_seq_str not in exp_out:
            exp_out = f"주어진 글에 이어 논리적 흐름상 {ans_seq_str} 순서로 전개되는 것이 가장 적절합니다. {exp_out}".strip()
        exp_out = _sync_explanation_answer_number(exp_out, ans_num)

        return RewrittenQuestion(
            rewritten_stem=stem,
            rewritten_choices=choices_5,
            replaced_terms_map=key_terms,
            pdf_grounded_explanation=exp_out,
            preserved_answer=ans_num,
            self_verification_note=f"글의 순서 분할(S{s1},S{s2},S{s3}) 및 오순 셔플 완료 (정답 {ans_num}번)",
            target_question_type="글의 순서 ((A)-(B)-(C))",
            subject_domain="영어",
        )

    # 6. 주어진 문장 삽입 (EnglishInsertionTransformSpec)
    if isinstance(spec, EnglishInsertionTransformSpec):
        sents = list(sent_list)
        n_sents = len(sents)
        if n_sents < 6:
            raise ValueError(f"문장 삽입 유형은 최소 6문장(또는 6절) 이상인 지문에서만 출제할 수 있습니다 (현재 {n_sents}문장).")

        ext_idx = int(spec.extracted_sentence_index or 2)
        ext_idx = max(1, min(n_sents - 1, ext_idx))
        given_exact = sents[ext_idx]
        remaining = [s for i, s in enumerate(sents) if i != ext_idx]

        true_gap = ext_idx
        max_gap = len(remaining) - 1 if true_gap <= len(remaining) - 1 else len(remaining)
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
        ans_num = str((true_gap - g_start) + 1)
        c_sym = _NUM_TO_CIRCLED_SYM.get(ans_num, "③")
        exp_out = _sync_explanation_answer_number(raw_exp, ans_num)
        exp_out = re.sub(
            r"(?:[1-5]\s*번|[①②③④⑤](?:\s*번)?)\s*(위치|자리|에\s*들어가|에\s*가장\s*적절)",
            rf"{c_sym}번 \g<1>",
            exp_out,
        )

        return RewrittenQuestion(
            rewritten_stem=(
                "글의 흐름으로 보아, 주어진 문장이 들어가기에 가장 적절한 곳은?\n"
                "[주어진 문장]\n"
                f"{given_exact}\n\n"
                f"{main_body}"
            ),
            rewritten_choices=["①", "②", "③", "④", "⑤"],
            replaced_terms_map=key_terms,
            pdf_grounded_explanation=exp_out,
            preserved_answer=ans_num,
            self_verification_note=f"문장 삽입(S{ext_idx} 추출) 조립 완료 (정답 {ans_num}번)",
            target_question_type="주어진 문장 삽입",
            subject_domain="영어",
        )

    # 7. 전체 흐름과 무관한 문장 (EnglishIrrelevantTransformSpec)
    if isinstance(spec, EnglishIrrelevantTransformSpec):
        sents = list(sent_list)
        n_sents = len(sents)
        if n_sents < 5:
            raise ValueError(f"무관한 문장 유형은 최소 5문장 이상인 지문에서만 출제할 수 있습니다 (현재 {n_sents}문장).")

        ins_sent = re.sub(r"\s+", " ", (spec.unrelated_sentence or "").strip()).strip("\"'“”‘’")
        ins_sent = re.sub(r"^(?:[①②③④⑤]|[1-5][.)])\s*", "", ins_sent).strip()
        if not ins_sent or len(re.findall(r"[A-Za-z]+", ins_sent)) < 5:
            raise ValueError("생성된 무관한 문장(unrelated_sentence)이 너무 짧거나 비어 있습니다.")
        if not re.search(r"[.?!]$", ins_sent):
            ins_sent += "."

        insert_pos = max(2, min(4, int(spec.insert_position or 3)))
        intro_cnt = 2 if n_sents >= 6 else 1
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
                merged_five.append(next(orig_iter))

        syms = ["①", "②", "③", "④", "⑤"]
        marked_str = " ".join(f"{syms[i]} {merged_five[i]}" for i in range(5))
        tail_str = (" " + " ".join(tail_orig)) if tail_orig else ""
        ans_num = str(insert_pos)
        c_sym = syms[insert_pos - 1]
        exp_out = _sync_explanation_answer_number(raw_exp, ans_num)
        exp_out = re.sub(
            r"(?:[1-5]\s*번|[①②③④⑤](?:\s*번)?)\s*(문장)",
            rf"{c_sym}번 \g<1>",
            exp_out,
            count=1,
        )

        return RewrittenQuestion(
            rewritten_stem=(
                "다음 글에서 전체 흐름과 관계없는 문장은?\n"
                f"{intro_part} {marked_str}{tail_str}".strip()
            ),
            rewritten_choices=list(syms),
            replaced_terms_map=key_terms,
            pdf_grounded_explanation=exp_out,
            preserved_answer=ans_num,
            self_verification_note=f"무관한 문장 {ans_num}번 삽입 완료",
            target_question_type="전체 흐름과 무관한 문장",
            subject_domain="영어",
        )

    # 8. 요약문 완성 (A), (B) (EnglishSummaryTransformSpec)
    if isinstance(spec, EnglishSummaryTransformSpec):
        s_line = strip_english_embedded_choices(normalize_english_exam_text(spec.summary_template or ""))
        s_line = re.sub(r"^[^\n]*가장\s*적절한\s*것은\??\s*", "", s_line).strip()
        s_line = re.sub(r"^\[?요약문\]?\s*", "", s_line).strip()
        # (A), (B) 뒤에 밑줄이 누락되었거나 짧은 경우 표준 '(A) ________', '(B) ________'로 정규화
        s_line = re.sub(r"\(\s*A\s*\)(?:\s*_{1,20})?", "(A) ________", s_line)
        s_line = re.sub(r"\(\s*B\s*\)(?:\s*_{1,20})?", "(B) ________", s_line)
        if "(A) ________" not in s_line or "(B) ________" not in s_line:
            raise ValueError("요약문 템플릿(summary_template)에 `(A) ________`와 `(B) ________`가 포함되어야 합니다.")

        def _clean_sum_word(w: str) -> str:
            w_c = re.sub(r"^\(\s*[AB]\s*\)\s*", "", (w or "").strip()).strip(".,;:\"'“”‘’[]()")
            return re.sub(r"\s+", " ", w_c)

        corr_a = _clean_sum_word(spec.correct_word_a)
        corr_b = _clean_sum_word(spec.correct_word_b)
        if not corr_a or not corr_b:
            raise ValueError("요약문 정답 어휘(correct_word_a, correct_word_b)가 비어 있습니다.")

        pairs: list[tuple[bool, tuple[str, str]]] = [(True, (corr_a, corr_b))]
        seen_pairs = {(corr_a.lower(), corr_b.lower())}
        pool_a = [corr_a]
        pool_b = [corr_b]
        for dp in spec.distractor_pairs or []:
            da = _clean_sum_word(dp.word_a)
            db = _clean_sum_word(dp.word_b)
            if da and db:
                if da.lower() not in {x.lower() for x in pool_a}:
                    pool_a.append(da)
                if db.lower() not in {x.lower() for x in pool_b}:
                    pool_b.append(db)
                key_ab = (da.lower(), db.lower())
                if key_ab not in seen_pairs:
                    seen_pairs.add(key_ab)
                    pairs.append((False, (da, db)))

        # 오답 쌍이 4개 미만이면 교차 조합으로 보충
        for ca in pool_a:
            for cb in pool_b:
                if len(pairs) >= 5:
                    break
                key_ab = (ca.lower(), cb.lower())
                if key_ab not in seen_pairs:
                    seen_pairs.add(key_ab)
                    pairs.append((False, (ca, cb)))

        if len(pairs) < 5:
            raise ValueError(f"요약문 5지선다를 구성할 고유 어휘 쌍이 부족합니다 (현재 {len(pairs)}개).")

        five_pairs = pairs[:5]
        rng.shuffle(five_pairs)
        five_pairs.sort(key=lambda x: (x[1][0].lower(), x[1][1].lower()))
        choices_5 = [f"(A) {a} — (B) {b}" for _, (a, b) in five_pairs]
        ans_num = str(next(i + 1 for i, (is_c, _) in enumerate(five_pairs) if is_c))

        exp_out = _sync_explanation_answer_number(raw_exp, ans_num)
        if corr_a.lower() not in exp_out.lower() or corr_b.lower() not in exp_out.lower():
            exp_out = f"(A)에는 '{corr_a}', (B)에는 '{corr_b}'가 들어가는 {_NUM_TO_CIRCLED_SYM[ans_num]}번이 가장 적절합니다. {exp_out}".strip()

        return RewrittenQuestion(
            rewritten_stem=(
                "다음 글의 내용을 한 문장으로 요약하고자 한다. 빈칸 (A), (B)에 들어갈 말로 가장 적절한 것은?\n"
                f"{clean_passage}\n\n"
                "[요약문]\n"
                f"{s_line}"
            ),
            rewritten_choices=choices_5,
            replaced_terms_map=[f"(A) {corr_a}", f"(B) {corr_b}", *key_terms][:3],
            pdf_grounded_explanation=exp_out,
            preserved_answer=ans_num,
            self_verification_note=f"요약문 (A) {corr_a} / (B) {corr_b} 조립 완료 (정답 {ans_num}번)",
            target_question_type="요약문 완성 ((A), (B))",
            subject_domain="영어",
        )

    raise TypeError(f"지원하지 않는 영어 변형 스키마 타입입니다: {type(spec)!r}")
