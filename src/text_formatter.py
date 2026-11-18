from __future__ import annotations

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

_SECTION_TAG_RE = re.compile(r"([<〈\[]\s*(?:자료|보\s*기)\s*[>〉\]])")
_BOGI_MARKER_INLINE_RE = re.compile(r"(?<!\n)[ \t]+(?=([ㄱㄴㄷㄹㅁ])\.\s)")
_BOGI_LINE_RE = re.compile(r"^([ㄱㄴㄷㄹㅁ])\.\s*(.+)$")
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


def normalize_exam_layout_text(text: str) -> str:
    """<자료>, <보기>, ㄱ/ㄴ/ㄷ 항목의 줄바꿈을 강제해 시험지 가독성을 정돈한다."""
    if not text:
        return ""
    cleaned = format_chemical_scripts(text).replace("\r\n", "\n").replace("\r", "\n")

    def _norm_tag(m: re.Match[str]) -> str:
        raw = re.sub(r"\s+", "", m.group(1))
        if "자료" in raw:
            return "\n\n<자료>\n"
        return "\n\n<보기>\n"

    cleaned = _SECTION_TAG_RE.sub(_norm_tag, cleaned)
    cleaned = _BOGI_MARKER_INLINE_RE.sub("\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def split_stem_blocks(text: str) -> list[tuple[str, Any]]:
    """
    정돈된 발문 텍스트를 순차 블록 리스트로 분해한다:
      - ('text', str)
      - ('data', str)
      - ('bogi', list[tuple[str, str]])
    """
    norm = normalize_exam_layout_text(text)
    if not norm:
        return []

    blocks: list[tuple[str, Any]] = []
    mode = "text"
    buf_lines: list[str] = []
    bogi_items: list[tuple[str, str]] = []

    def _flush() -> None:
        nonlocal buf_lines, bogi_items, mode
        if mode == "text":
            joined = "\n".join(buf_lines).strip()
            if joined:
                blocks.append(("text", joined))
            buf_lines = []
        elif mode == "data":
            joined = "\n".join(buf_lines).strip()
            if joined:
                blocks.append(("data", joined))
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
        if ln == "<자료>":
            _flush()
            mode = "data"
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
                    # 빈 줄 뒤에 <보기> 바깥 질문이 이어질 수 있으므로 유지하되 다음 비어있지 않은 줄에서 판별
                    continue
            elif _QUESTION_END_RE.search(ln):
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
            if not ln and buf_lines:
                # 빈 줄 이후 질문 문장이 나오면 자료 블록 종료 여부는 다음 줄로 미루되 빈 줄 유지
                buf_lines.append("")
                continue
            if _QUESTION_END_RE.search(ln) and (
                "이에 대한 설명" in ln or "고른 것" in ln or "옳은 것" in ln or len(buf_lines) >= 1
            ):
                if len(ln) <= 95 and ("설명" in ln or "고른 것" in ln or "옳은 것" in ln or ln.endswith("?")):
                    _flush()
                    mode = "text"
                    buf_lines.append(ln)
                    continue
            if ln:
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
