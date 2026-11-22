from __future__ import annotations

import html as _html
import re
from typing import Any, Sequence

_SUB_MAP = str.maketrans("0123456789nmk", "₀₁₂₃₄₅₆₇₈₉ₙₘₖ")
_SUP_MAP = str.maketrans("0123456789+-nmk", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿᵐᵏ")

_UNICODE_SUB_REV = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₙₘₖ", "0123456789nmk")
_UNICODE_SUP_REV = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿᵐᵏ", "0123456789+-nmk")

_COMMON_FORMULAS_RE = re.compile(
    r"(?<![A-Za-z])(H2O2|H2O|CO2|SO2|SO3|SO4|NO2|NO3|N2O|N2|O2|O3|H2|F2|Cl2|Br2|I2|S8|P4|NH3|NH4|CH4|CO3|HCO3|PO4|CH3COO)(?![A-Za-z0-9])"
)
# 원소기호+숫자 뒤에 바로 다른 대문자 원소나 닫는 괄호가 이어지는 화학식 (예: CH3COOH, Na2CO3, H2SO4)
_INLINE_CHEM_SUB_RE = re.compile(r"([A-Z][a-z]?)(\d+)(?=(?:[A-Z\(]|$))")
# 괄호 닫힘 뒤 숫자 (예: B(OH)2, Ca(NO3)2)
_PAREN_SUB_RE = re.compile(r"\)(\d+)")
# 명시적 아래첨자 표기: _2, _n, _(2), _{n}
_EXPLICIT_SUB_RE = re.compile(r"_\{([0-9nmk]+)\}|_\(([0-9nmk]+)\)|_([0-9nmk])")
# 명시적 위첨자 표기: ^(n-), ^(2+), ^2-, ^+
_EXPLICIT_SUP_RE = re.compile(
    r"\^\{([0-9nmk]*[+\-]|[0-9]+)\}|\^\(([0-9nmk]*[+\-]|[0-9]+)\)|\^([0-9nmk]*[+\-])"
)
# 이온 전하 표기: H+, OH-, Na+, Ca2+, Fe3+, NH4+, CH3COO-, A2-, X- (뒤에 한국어 조사 가능)
_ION_CHARGE_RE = re.compile(
    r"((?:OH|NH[4₄]|NO[3₃]|SO[4₄]|CO[3₃]|PO[4₄]|CH[3₃]COO|[A-Z][a-z]?[0-9₀-₉]*\)?))([2-4])?([+\-])(?![A-Za-z0-9\(])"
)
# 물리/화학 시간·속도 첨자 변수: t1, t2, t3, v1, v2
_VAR_SUB_RE = re.compile(r"(?<![A-Za-z])([tv])([1-9])(?![0-9A-Za-z])")

_VISUAL_REF_RE = re.compile(
    r"(?:그림\s*(?:은|에서|과|와|을|를|속|\(|\[)|"
    r"그래프\s*(?:는|에서|과|와|를|\(|\[)|"
    r"표\s*(?:는|에서|과|와|를|\(|\[)|"
    r"다음\s*자료\s*(?:는|에서)|"
    r"모식도|실험\s*장치|회로\s*(?:는|에서|와|과))"
)

_CANONICAL_DATA_TITLES: dict[str, str] = {
    "<자료>": "< 자 료 >",
    "<조건>": "< 조 건 >",
    "<실험>": "< 실 험 >",
    "<실험 과정>": "< 실 험 과 정 >",
    "<실험 결과>": "< 실 험 결 과 >",
}

_SECTION_TAG_RE = re.compile(
    r"(?<![가-힣A-Za-z0-9])[<〈\[]\s*(자료|조\s*건|실\s*험\s*과\s*정|실\s*험\s*결\s*과|실\s*험|보\s*기)\s*[>〉\]](?![가-힣])[ \t]*"
)
_INLINE_BULLET_RE = re.compile(r"(?<![\n \t])[ \t]+(?=[◦•○●▪▸▹▶※])")
_INLINE_STEP_KOR_RE = re.compile(r"(?<=[.?!:;>\]])\s+(?=\((?:가|나|다|라|마)\)\s)")
_INLINE_DAN_RE = re.compile(r"(?<=[.?!\)])\s+(?=\(?단,\s)")
_INLINE_LEADIN_Q_RE = re.compile(
    r"(?<=[.?!\)])\s+(?=(?:이에\s*대한|위\s*(?:자료|실험|그림|표|그래프)에\s*대한|이\s*자료에\s*대한))"
)
_BOGI_MARKER_INLINE_RE = re.compile(r"(?<![\n \t])[ \t]+(?=[ㄱㄴㄷㄹㅁ][.)]\s)")
_BOGI_LINE_RE = re.compile(r"^([ㄱㄴㄷㄹㅁ])[.)]\s*(.+)$")
_QUESTION_END_RE = re.compile(
    r"(?:것은\s*\??|고른\s*것은\s*\??|고르시오\.?|구하시오\.?|구하면\s*\??|값은\s*\??|쓰시오\.?)$"
)

_DEFAULT_HIGHER_TERMS: tuple[str, ...] = (
    "몰 농도",
    "몰농도",
    "몰랄 농도",
    "몰랄농도",
    "화학식량",
    "분자량",
    "원자량",
    "몰질량",
    "몰 부피",
    "몰부피",
    "몰분율",
    "몰수",
    "엔탈피",
    "엔트로피",
    "자유 에너지",
    "평형 상수",
    "반응 지수",
    "이온화 상수",
    "완충 용액",
    "mol",
    "M",
)


def format_chemical_scripts(text: str) -> str:
    """생 화학식·이온 전하·변수 첨자를 읽기 쉬운 유니코드 위·아래첨자로 변환한다."""
    if not text:
        return ""

    out = text
    # 1) 명시적 ^(위첨자) 및 _(아래첨자) 먼저 변환
    out = _EXPLICIT_SUP_RE.sub(
        lambda m: (m.group(1) or m.group(2) or m.group(3)).translate(_SUP_MAP),
        out,
    )
    out = _EXPLICIT_SUB_RE.sub(
        lambda m: (m.group(1) or m.group(2) or m.group(3)).translate(_SUB_MAP),
        out,
    )

    # 2) 다원자·이원자 대표 분자식 및 괄호 뒤 숫자 (예: H2O, CO2, B(OH)2)
    out = _COMMON_FORMULAS_RE.sub(
        lambda m: re.sub(r"\d+", lambda dm: dm.group(0).translate(_SUB_MAP), m.group(1)),
        out,
    )
    out = _PAREN_SUB_RE.sub(lambda m: ")" + m.group(1).translate(_SUB_MAP), out)

    # 3) 연속 원소 결합 화학식 내부 숫자 (예: CH3COOH, H2SO4, Na2CO3)
    # 단, 문장 내 'A1', 'B2' 같은 단독 표본/변수 기호는 뒤이어 다른 원소 대문자가 올 때만 변환
    def _replace_compound_formula(match: re.Match[str]) -> str:
        tok = match.group(0)
        if not re.search(r"[A-Z][a-z]?\d+[A-Z\(]", tok):
            return tok
        return _INLINE_CHEM_SUB_RE.sub(
            lambda m: m.group(1) + m.group(2).translate(_SUB_MAP),
            tok,
        )

    out = re.sub(r"\b(?:[A-Z][a-z]?\d*|\([A-Za-z0-9₀-₉]+\)\d*){2,}", _replace_compound_formula, out)

    # 4) 원소/작용기 뒤 이온 전하 (예: H+, OH-, Na+, Ca2+, CH3COO-)
    def _replace_ion(m: re.Match[str]) -> str:
        base, mult, sign = m.group(1), m.group(2) or "", m.group(3)
        if not base:
            return m.group(0)
        return base + (mult + sign).translate(_SUP_MAP)

    out = _ION_CHARGE_RE.sub(_replace_ion, out)

    # 5) 시점/속도 변수 첨자 (예: t1, t2, t3)
    out = _VAR_SUB_RE.sub(lambda m: m.group(1) + m.group(2).translate(_SUB_MAP), out)
    return out


def unicode_scripts_to_html_tags(escaped_html: str) -> str:
    """
    HTML 이스케이프된 문자열의 유니코드 위·아래첨자를 PDF(PyMuPDF htmlbox) 전용
    <sub> / <sup> 태그로 변환해 폰트 글리프 누락(두부 깨짐)을 방지한다.
    """
    if not escaped_html:
        return ""
    out = re.sub(
        r"[₀₁₂₃₄₅₆₇₈₉ₙₘₖ]+",
        lambda m: f"<sub>{m.group(0).translate(_UNICODE_SUB_REV)}</sub>",
        escaped_html,
    )
    out = re.sub(
        r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿᵐᵏ]+",
        lambda m: f"<sup>{m.group(0).translate(_UNICODE_SUP_REV)}</sup>",
        out,
    )
    return out


def _unescape_newlines(text: str) -> str:
    # LaTeX 수식 명령(\neq, \nu, \nabla 등)은 보존하면서 리터럴 \n 이스케이프는 실제 줄바꿈으로 복원한다.
    norm = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\\r\\n", "\n")
    return re.sub(
        r"\\n(?!(?:eq|u|abla|atural|otin|ull|ewline|oindent)\b)(?!(?:e|i|ot)\b)",
        "\n",
        norm,
    )


def _unescape_newlines_english(text: str) -> str:
    """영어 지문·발문용: 수식 명령 예외 없이 모든 리터럴 \\n / \\r\\n 문자열을 실제 줄바꿈으로 치환한다."""
    return (
        (text or "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\\r\\n", "\n")
        .replace("\\n", "\n")
    )


def normalize_exam_layout_text(text: str) -> str:
    """<자료>·<조건>·<실험 과정>·<실험 결과>·<보기>·불릿·(단, ...)·ㄱ/ㄴ/ㄷ 항목의 줄바꿈을 강제한다."""
    if not text:
        return ""
    cleaned = format_chemical_scripts(_unescape_newlines(text))

    def _norm_tag(m: re.Match[str]) -> str:
        raw = re.sub(r"\s+", "", m.group(1))
        if raw == "자료":
            return "\n\n<자료>\n"
        if raw == "조건":
            return "\n\n<조건>\n"
        if raw == "실험과정":
            return "\n\n<실험 과정>\n"
        if raw == "실험결과":
            return "\n\n<실험 결과>\n"
        if raw == "실험":
            return "\n\n<실험>\n"
        return "\n\n<보기>\n"

    cleaned = _SECTION_TAG_RE.sub(_norm_tag, cleaned)
    cleaned = _INLINE_STEP_KOR_RE.sub("\n", cleaned)
    cleaned = _INLINE_BULLET_RE.sub("\n", cleaned)
    cleaned = _INLINE_DAN_RE.sub("\n", cleaned)
    cleaned = _INLINE_LEADIN_Q_RE.sub("\n\n", cleaned)
    cleaned = _BOGI_MARKER_INLINE_RE.sub("\n", cleaned)

    lines = [re.sub(r"^([ㄱㄴㄷㄹㅁ])\)\s+", r"\1. ", ln.strip()) for ln in cleaned.split("\n")]

    # <보기> 헤더 없이 질문 뒤에 ㄱ. ㄴ. 목록이 바로 이어지면 자동으로 <보기> 섹션을 삽입
    if "<보기>" not in lines:
        has_g = any(ln.startswith("ㄱ. ") for ln in lines)
        has_n = any(ln.startswith("ㄴ. ") for ln in lines)
        if has_g and has_n:
            patched_lines: list[str] = []
            inserted = False
            for idx_l, ln in enumerate(lines):
                if not inserted and ln.startswith("ㄱ. "):
                    prev_nonempty = next((lines[j] for j in range(idx_l - 1, -1, -1) if lines[j]), "")
                    if prev_nonempty and (_QUESTION_END_RE.search(prev_nonempty) or "?" in prev_nonempty):
                        if patched_lines and patched_lines[-1] != "":
                            patched_lines.append("")
                        patched_lines.append("<보기>")
                        inserted = True
                patched_lines.append(ln)
            lines = patched_lines

    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def format_explanation_text(text: str, for_markdown: bool = False) -> str:
    """맞춤 해설·단계별 풀이 텍스트의 번호·기호·접속어 줄바꿈을 강제하고 마크다운 줄바꿈을 보존한다."""
    if not text:
        return ""
    out = format_chemical_scripts(_unescape_newlines(text))
    out = re.sub(r"(?<=[.?!:;\)])\s+(?=\d{1,2}\.\s+[가-힣A-Za-z\(\[])", "\n", out)
    out = re.sub(r"(?<![\n \t])[ \t]+(?=[◦•○●▪※▶])", "\n", out)
    out = re.sub(r"(?<=[.?!:;])[ \t]+(?=[①②③④⑤])", "\n", out)
    out = re.sub(r"(?<![\n \t])[ \t]+(?=[ㄱㄴㄷㄹㅁ][.)]\s)", "\n", out)
    out = re.sub(r"(?<=[.?!\)])\s+(?=(?:따라서|결론적으로|그러므로|정답은|\[참고)\s?)", "\n", out)
    out = "\n".join(ln.rstrip() for ln in out.split("\n"))
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    if for_markdown:
        out = re.sub(r"(?<!\n)\n(?!\n)", "  \n", out)
    return out


def split_stem_blocks(text: str) -> list[tuple[str, Any]]:
    """
    정돈된 발문 텍스트를 순차 블록 리스트로 분해한다:
      - ('text', str)
      - ('data', {'title': str, 'body': str})
      - ('bogi', list[tuple[str, str]])
    """
    norm = normalize_exam_layout_text(text)
    if not norm:
        return []

    blocks: list[tuple[str, Any]] = []
    mode = "text"
    data_title = "< 자 료 >"
    buf_lines: list[str] = []
    bogi_items: list[tuple[str, str]] = []

    def _flush() -> None:
        nonlocal buf_lines, bogi_items
        if mode == "text":
            joined = "\n".join(buf_lines).strip()
            if joined:
                blocks.append(("text", joined))
            buf_lines = []
        elif mode == "data":
            joined = "\n".join(buf_lines).strip()
            if joined:
                blocks.append(("data", {"title": data_title, "body": joined}))
            buf_lines = []
        elif mode == "bogi":
            if bogi_items:
                blocks.append(("bogi", list(bogi_items)))
            elif buf_lines:
                blocks.append(("text", "\n".join(buf_lines).strip()))
            bogi_items = []
            buf_lines = []

    for raw_ln in norm.split("\n"):
        ln = raw_ln.strip()
        if ln in _CANONICAL_DATA_TITLES:
            _flush()
            mode = "data"
            data_title = _CANONICAL_DATA_TITLES[ln]
            continue
        if ln == "<보기>":
            _flush()
            mode = "bogi"
            continue

        if mode == "bogi":
            bm = _BOGI_LINE_RE.match(ln)
            if bm:
                bogi_items.append((f"{bm.group(1)}.", bm.group(2).strip()))
                continue
            if not ln:
                if bogi_items:
                    continue
            elif _QUESTION_END_RE.search(ln) or re.match(r"^\(?단,\s", ln):
                _flush()
                mode = "text"
                buf_lines.append(ln)
                continue
            elif bogi_items:
                prev_m, prev_t = bogi_items[-1]
                bogi_items[-1] = (prev_m, f"{prev_t} {ln}".strip())
                continue
            else:
                buf_lines.append(ln)
        elif mode == "data":
            if not ln:
                if buf_lines and buf_lines[-1] != "":
                    buf_lines.append("")
                continue
            if _QUESTION_END_RE.search(ln) and (
                "이에 대한" in ln or "고른 것" in ln or "옳은 것" in ln or len(buf_lines) >= 1
            ):
                if len(ln) <= 105 and ("설명" in ln or "고른 것" in ln or "옳은 것" in ln or ln.endswith("?")):
                    _flush()
                    mode = "text"
                    buf_lines.append(ln)
                    continue
            buf_lines.append(ln)
        else:
            if not ln:
                if buf_lines and buf_lines[-1] != "":
                    buf_lines.append("")
            else:
                buf_lines.append(ln)

    _flush()
    return blocks


def has_visual_reference(text: str) -> bool:
    """지문에 그림·그래프·표 등 원본 시각 자료를 참조하는 표현이 있는지 판별한다."""
    return bool(_VISUAL_REF_RE.search(text or ""))


def _norm_vocab_key(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").strip()).lower()


def _is_term_in_text(term: str, text: str) -> bool:
    clean_t = (term or "").strip()
    if not clean_t or not text:
        return False

    if clean_t == "M":
        # 몰 농도 단위 M: 'Mg', 'Mn', 'mol', 'mL', 'MAX' 등과 혼동하지 않도록 대문자 단독/수치 뒤만 검사
        return bool(re.search(r"(?<![A-Za-z])M(?![A-Za-z])|(?<=\d)M(?![A-Za-z])", text))

    if re.fullmatch(r"[A-Za-z]+", clean_t):
        esc = re.escape(clean_t)
        return bool(re.search(rf"(?<![A-Za-z]){esc}(?![A-Za-z])|(?<=\d){esc}(?![A-Za-z])", text, re.I))

    # 한글/혼합 용어는 내부 공백 차이를 허용(예: '몰 농도' == '몰농도')
    parts = [re.escape(p) for p in re.split(r"\s+", clean_t) if p]
    if not parts:
        return False
    pattern = r"\s*".join(parts)
    if re.match(r"^[A-Za-z]", clean_t):
        pattern = r"(?<![A-Za-z])" + pattern
    if re.search(r"[A-Za-z]$", clean_t):
        pattern = pattern + r"(?![A-Za-z])"
    return bool(re.search(pattern, text, re.I))


def find_forbidden_terms_in_text(
    texts: Sequence[str],
    out_of_scope_terms: Sequence[str],
    allowed_vocab: Sequence[str],
) -> list[str]:
    """
    개조된 발문·선지 텍스트에 개념서 허용 범위 밖의 금지 용어/단위가 남아있는지 검사해 잔존 목록을 반환한다.
    """
    allowed_keys = {_norm_vocab_key(v) for v in allowed_vocab if (v or "").strip()}
    # 허용 용어가 괄호 부가설명을 달고 있을 때 핵심 토큰도 허용으로 인정 (예: "질량(g)" -> "질량")
    for v in allowed_vocab:
        base_v = re.sub(r"\([^)]*\)", "", v or "").strip()
        if base_v:
            allowed_keys.add(_norm_vocab_key(base_v))

    candidates: list[str] = []
    seen_c: set[str] = set()
    for t in [*(out_of_scope_terms or []), *_DEFAULT_HIGHER_TERMS]:
        clean = (t or "").strip()
        if not clean:
            continue
        k = _norm_vocab_key(clean)
        # 화이트리스트(pdf_vocabulary)에 명시적으로 허용된 용어는 금지 목록에서 제외
        if k in allowed_keys:
            continue
        # out_of_scope_terms 외의 기본 상위 용어 중 너무 짧은 소문자/일반어는 제외
        if clean not in (out_of_scope_terms or []) and len(k) < 2 and clean != "M":
            continue
        if k not in seen_c:
            seen_c.add(k)
            candidates.append(clean)

    combined = "\n".join(t for t in texts if t)
    found: list[str] = []
    seen_found: set[str] = set()
    for term in candidates:
        if _is_term_in_text(term, combined):
            fk = _norm_vocab_key(term)
            if fk not in seen_found:
                seen_found.add(fk)
                found.append(term)
    return found


_ENG_U_TAG_RE = re.compile(r"<u\b[^>]*>(.*?)</u>", re.IGNORECASE | re.DOTALL)
_ENG_BLANK_RE = re.compile(r"_{4,}")
_ENG_GIVEN_TAG_RE = re.compile(
    r"(?<![가-힣A-Za-z0-9])[<\[\〈]\s*(주어진\s*문장|주어진\s*글|요\s*약\s*문|요\s*약|본\s*문|지\s*문)\s*[>\]\〉][ \t]*"
)
_ENG_ORDER_INLINE_RE = re.compile(r"(?<=[.?!\"'”’\)])\s+(?=\([ABC]\)\s+(?!\[)[A-Za-z0-9\"'“‘])")
_ENG_ORDER_LINE_RE = re.compile(r"^\(([ABC])\)\s+(?!\[)(.+)$")
_ENG_FOOTNOTE_LINE_RE = re.compile(r"^\*{1,2}\s*[A-Za-z][A-Za-z\s\-']*:")
_ENG_EMBEDDED_CHOICES_HDR_RE = re.compile(r"^[<\[\〈]\s*(?:선\s*지|선\s*택\s*지)\s*[>\]\〉]\s*$")
_ENG_CHOICE_LINE_RE = re.compile(r"^[①②③④⑤]\s*(?:\(A\)|.+[—\-].+\(B\)|\S+)")
_ENG_MD_TABLE_SEP_RE = re.compile(r"^\|[\s:\-\|]+\|$")
_KOR_INSTRUCTION_RE = re.compile(
    r"(?:다음\s*글|다음\s*빈칸|주어진\s*(?:글|문장)|밑줄\s*친|빈칸에\s*들어갈|흐름으로\s*보아|흐름과\s*관계|요약하고자|가장\s*적절한\s*것은|어법상|문맥상|글의\s*(?:순서|요지|주제|제목))"
)
# 한국어 발문(예: '다음 글의 제목으로 가장 적절한 것은?') 뒤에 줄바꿈 없이 바로 영어 지문/기호가 이어지는 경우 분리용 정규식
_KOR_INST_PREFIX_SPLIT_RE = re.compile(
    r"^([^\n]*?[가-힣][^\n]*?(?:\?|시오\.)(?:\s*[\[\(]\d+\s*점[\]\)])?)[ \t]+(?=[A-Za-z\"'“‘\(\[<①-⑤_])"
)


def render_english_inline_html(text: str) -> str:
    """
    영어 지문/선지 내 <u>...</u> 밑줄과 _____ 빈칸을 XSS 없이 안전한 HTML로 변환한다.
    화학식 위·아래첨자 변환(format_chemical_scripts)은 호출하지 않아 영문 하이픈·약어를 보호한다.
    """
    if not text:
        return ""

    text = _unescape_newlines_english(text)
    u_segments: list[str] = []

    def _stash_u(m: re.Match[str]) -> str:
        idx = len(u_segments)
        u_segments.append(m.group(1))
        return f"@@ENG_U_{idx}@@"

    stashed = _ENG_U_TAG_RE.sub(_stash_u, text)
    escaped = _html.escape(stashed)

    for idx, inner in enumerate(u_segments):
        safe_inner = _html.escape(inner)
        escaped = escaped.replace(
            f"@@ENG_U_{idx}@@",
            f"<u class='eng-u'>{safe_inner}</u>",
        )

    escaped = _ENG_BLANK_RE.sub(
        "<span class='eng-blank'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>",
        escaped,
    )
    escaped = re.sub(r"([①②③④⑤])", r"<b>\1</b>", escaped)
    return escaped.replace("\n", "<br/>")


def normalize_english_exam_text(text: str) -> str:
    """영어 변형 문제 발문·지문의 한국어 발문 뒤 줄바꿈, [주어진 문장], [주어진 글], (A)/(B)/(C), [요약문] 줄바꿈을 정규화한다."""
    if not text:
        return ""
    cleaned = _unescape_newlines_english(text).strip()

    # 첫 줄에서 한국어 발문(? 또는 ~시오.)과 영어 본문이 줄바꿈 없이 한 줄에 붙어 있는 경우 강제 줄바꿈 분리
    m_inst = _KOR_INST_PREFIX_SPLIT_RE.match(cleaned)
    if m_inst and _KOR_INSTRUCTION_RE.search(m_inst.group(1)):
        cleaned = f"{m_inst.group(1).strip()}\n\n{cleaned[m_inst.end():].lstrip()}"

    def _norm_eng_tag(m: re.Match[str]) -> str:
        raw = re.sub(r"\s+", "", m.group(1))
        if raw == "주어진문장":
            return "\n\n[주어진 문장]\n"
        if raw == "주어진글":
            return "\n\n[주어진 글]\n"
        if raw in ("본문", "지문"):
            return "\n\n"
        return "\n\n[요약문]\n"

    cleaned = _ENG_GIVEN_TAG_RE.sub(_norm_eng_tag, cleaned)
    cleaned = _ENG_ORDER_INLINE_RE.sub("\n", cleaned)
    lines = [ln.strip() for ln in cleaned.split("\n")]
    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def split_english_stem_blocks(text: str) -> list[tuple[str, Any]]:
    """
    영어 변형 발문을 시맨틱 블록 리스트로 분해한다:
      - ('instruction', str)
      - ('given_box', {'title': str, 'body': str})
      - ('passage', str)
      - ('order_chunks', list[tuple[str, str]])
      - ('summary_box', str)
      - ('footnote', str)
    """
    norm = normalize_english_exam_text(text)
    if not norm:
        return []

    blocks: list[tuple[str, Any]] = []
    mode = "passage"
    box_title = ""
    buf_lines: list[str] = []
    order_items: list[tuple[str, str]] = []
    footnote_lines: list[str] = []

    def _flush() -> None:
        nonlocal buf_lines, order_items
        if mode == "given_box":
            joined = "\n".join(buf_lines).strip()
            if joined:
                blocks.append(("given_box", {"title": box_title, "body": joined}))
            buf_lines = []
        elif mode == "summary_box":
            joined = "\n".join(buf_lines).strip()
            if joined:
                blocks.append(("summary_box", joined))
            buf_lines = []
        elif mode == "order_chunks":
            if order_items:
                blocks.append(("order_chunks", list(order_items)))
            order_items = []
            buf_lines = []
        else:
            joined = "\n".join(buf_lines).strip()
            if joined:
                blocks.append(("passage", joined))
            buf_lines = []

    raw_lines = norm.split("\n")
    idx = 0
    # 첫 문단이 한국어 발문이면 ('instruction', ...) 블록으로 분리 (영어 지문이 한 줄에 붙어 있어도 발문만 분리)
    inst_lines: list[str] = []
    while idx < len(raw_lines):
        cand = raw_lines[idx].strip()
        if not cand:
            if inst_lines:
                idx += 1
                break
            idx += 1
            continue
        has_korean = bool(re.search(r"[가-힣]", cand))
        if not inst_lines and has_korean and (_KOR_INSTRUCTION_RE.search(cand) or cand.endswith("?")):
            m_inline = _KOR_INST_PREFIX_SPLIT_RE.match(cand)
            if m_inline:
                inst_lines.append(m_inline.group(1).strip())
                raw_lines[idx] = cand[m_inline.end():].strip()
            else:
                inst_lines.append(cand)
                idx += 1
            break
        break

    if inst_lines:
        blocks.append(("instruction", "\n".join(inst_lines)))

    for raw_ln in raw_lines[idx:]:
        ln = raw_ln.strip()
        if not ln:
            if mode in ("given_box", "summary_box") and buf_lines:
                _flush()
                mode = "passage"
            elif mode == "passage" and buf_lines and buf_lines[-1] != "":
                buf_lines.append("")
            continue

        # [주어진 문장] 박스 뒤에 빈 줄 없이 바로 삽입 위치(①, ②)가 포함된 본문 줄이 이어지면 박스 종료 후 본문으로 전환
        if (
            mode == "given_box"
            and box_title == "주어진 문장"
            and buf_lines
            and ("①" in ln or "②" in ln)
        ):
            _flush()
            mode = "passage"

        if _ENG_FOOTNOTE_LINE_RE.match(ln):
            footnote_lines.append(ln)
            continue

        # AI가 rewritten_stem 내부에 [선지] / [선택지] 블록이나 요약문 (A)-(B) 마크다운 표를 중복 삽입한 경우 본문에서 제거
        if _ENG_EMBEDDED_CHOICES_HDR_RE.match(ln):
            _flush()
            break
        if mode == "summary_box" and buf_lines and (
            (ln.startswith("|") and ("(A)" in ln or "(B)" in ln or _ENG_MD_TABLE_SEP_RE.match(ln)))
            or _ENG_CHOICE_LINE_RE.match(ln)
        ):
            _flush()
            break

        if ln in ("[주어진 문장]", "[주어진 글]"):
            _flush()
            mode = "given_box"
            box_title = ln.strip("[]")
            continue

        if ln == "[요약문]":
            _flush()
            mode = "summary_box"
            continue

        om = _ENG_ORDER_LINE_RE.match(ln)
        if om and mode != "summary_box":
            if mode != "order_chunks":
                _flush()
                mode = "order_chunks"
            order_items.append((f"({om.group(1)})", om.group(2).strip()))
            continue

        if mode == "order_chunks":
            if order_items:
                prev_mk, prev_txt = order_items[-1]
                order_items[-1] = (prev_mk, f"{prev_txt} {ln}".strip())
            else:
                buf_lines.append(ln)
            continue

        buf_lines.append(ln)

    _flush()
    if footnote_lines:
        blocks.append(("footnote", "\n".join(footnote_lines)))
    return blocks


def strip_english_embedded_choices(stem_text: str) -> str:
    """rewritten_stem 끝에 AI가 중복 삽입한 [선지] 블록이나 요약문 (A)-(B) 선택지 목록을 제거한다."""
    if not stem_text:
        return ""
    lines = normalize_english_exam_text(stem_text).split("\n")
    out: list[str] = []
    in_summary = False
    for idx_l, raw_ln in enumerate(lines):
        ln = raw_ln.strip()
        if _ENG_EMBEDDED_CHOICES_HDR_RE.match(ln):
            break
        if ln == "[요약문]":
            in_summary = True
            out.append(raw_ln)
            continue
        if in_summary and ln.startswith("|") and ("(A)" in ln or "(B)" in ln or _ENG_MD_TABLE_SEP_RE.match(ln)):
            break
        if in_summary and _ENG_CHOICE_LINE_RE.match(ln):
            break
        # 본문 맨 끝에 ① ~ ⑤ 5개 선지가 통째로 붙어 나온 경우 제거
        if ln.startswith("①") and idx_l + 4 < len(lines):
            tail_syms = [lines[idx_l + k].strip()[:1] for k in range(5)]
            if tail_syms == ["①", "②", "③", "④", "⑤"]:
                break
        out.append(raw_ln)
    return "\n".join(out).strip()


_ENG_KOR_CHUNK_MARKER_RE = re.compile(r"(?:^|\s)/(?:\s|$)|\[[^\]]+\]")


def normalize_english_korean_translation(text: str) -> str:
    """영어 정밀 지문 분석지의 한국어 해석에서 직독직해용 구문 기호(/, [], ())를 제거해 깔끔한 일반 해석으로 정규화한다."""
    s = (text or "").strip()
    if not s:
        return ""
    has_chunk_markers = bool(_ENG_KOR_CHUNK_MARKER_RE.search(s))
    # 끊어읽기 슬래시 제거 (단, 단어 병기용 'A/B'는 유지)
    s = re.sub(r"\s+/\s+", " ", s)
    s = re.sub(r"^\s*/\s*|\s*/\s*$", " ", s)
    # 절 묶음 대괄호 제거
    s = s.replace("[", "").replace("]", "")
    # 직독직해 기호가 감지된 경우 수식어구 묶음 소괄호도 제거
    if has_chunk_markers:
        s = s.replace("((", "").replace("))", "")
        s = s.replace("(", "").replace(")", "")
    s = re.sub(r"[ \t]+", " ", s).strip()
    s = re.sub(r"\s+([.,?!;:])", r"\1", s)
    return s

