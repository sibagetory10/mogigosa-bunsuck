"""Deterministic English passage sanitization, sentence indexing, and conditional restoration."""

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass
from typing import Any

from google.genai import types
from pydantic import BaseModel, Field

from src.config import FLASH_LITE_MODEL
from src.gemini.core import _build_client, _generate_structured_with_fallback

_RESTORE_CACHE_LOCK = threading.RLock()
_RESTORE_CACHE: dict[str, str] = {}

_FOOTNOTE_RE = re.compile(r"^\*{1,2}\s*[A-Za-z][A-Za-z\s\-']*:\s*.+$")
_KOR_INST_HEAD_RE = re.compile(
    r"^\s*(?:\[\d{1,2}\s*[~\-]\s*\d{1,2}\]\s*)?"
    r"(?:다음\s*(?:글|빈칸)|주어진\s*(?:글|문장)|밑줄\s*친|글의\s*(?:흐름|순서|요지|주제|제목)|"
    r"\(A\)\s*,\s*\(B\)\s*,\s*\(C\)\s*의)"
    r"[^\n]*(?:\?|시오\.|하시오\.)(?:\s*[\[\(]\d+\s*점[\]\)])?\s*"
)
_CHOICE_LINE_RE = re.compile(r"^\s*[①②③④⑤]\s+")
_CIRCLED_WITH_PARENS_RE = re.compile(r"\(\s*[①②③④⑤]\s*\)\s*")
_CIRCLED_MARKER_RE = re.compile(r"[①②③④⑤]\s*")
_ORDER_TAG_RE = re.compile(r"(?:^|\n)\s*\([ABC]\)\s+")
_U_TAG_RE = re.compile(r"</?u\b[^>]*>", re.IGNORECASE)
_INCOMPLETE_BLANK_RE = re.compile(r"_{3,}")

# Abbreviations whose trailing period does NOT terminate a sentence.
_ABBREV_PLACEHOLDER = "\u2024"  # ONE DOT LEADER (safe internal sentinel)
_ABBREV_RE = re.compile(
    r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Mt|vs|etc|e\.g|i\.e|Fig|No|Vol|Eq|Ch|Sec|p|pp|U\.S|U\.K|U\.N|a\.m|p\.m|Jan|Feb|Mar|Apr|Aug|Sept|Oct|Nov|Dec)\.",
    re.IGNORECASE,
)
_DECIMAL_RE = re.compile(r"(\d)\.(\d)")
_INITIAL_RE = re.compile(r"\b([A-Z])\.(?=\s+[A-Z])")
_SENT_SPLIT_RE = re.compile(r"(?<=[.?!])(?:[\"”’'\)\]]+)?\s+(?=[A-Z\"“‘'\(\[])")


@dataclass(frozen=True)
class IndexedPassage:
    """Clean passage split into indexed sentences plus preserved footnotes."""

    clean_text: str
    sentences: tuple[str, ...]
    footnotes: tuple[str, ...]

    def indexed_prompt_block(self) -> str:
        """Render `[S0] ...` lines for coordinate-only LLM prompting."""
        return "\n".join(f"[S{i}] {s}" for i, s in enumerate(self.sentences))


def needs_passage_restoration(raw_passage: str, orig_question_type: str = "") -> bool:
    """Return True only when the stored source passage is incomplete (blank/order/summary/choice brackets)."""
    txt = (raw_passage or "").strip()
    if len(txt) < 80:
        return True
    if _INCOMPLETE_BLANK_RE.search(txt):
        return True
    if "[주어진 글]" in txt or "[주어진 문장]" in txt or "[요약문]" in txt:
        return True
    if "(A)" in txt and "(B)" in txt and "(C)" in txt:
        return True
    if re.search(r"\[[A-Za-z\-']+\s*/\s*[A-Za-z\-']+\]", txt):
        return True
    _ = orig_question_type
    return False


def sanitize_english_passage_deterministically(raw_text: str) -> tuple[str, list[str]]:
    """Strip exam instructions, choices, HTML underlines, and markers while extracting footnotes."""
    txt = (raw_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not txt:
        return "", []

    txt = _KOR_INST_HEAD_RE.sub("", txt, count=1).strip()

    lines = [ln.strip() for ln in txt.split("\n")]
    kept_lines: list[str] = []
    footnotes: list[str] = []

    for ln in lines:
        if not ln:
            continue
        if ln in ("[선지]", "[선택지]", "[요약문]"):
            break
        if ln.startswith("[") and ln.endswith("]") and any(k in ln for k in ("선지", "요약")):
            break
        if _CHOICE_LINE_RE.match(ln) and len(ln) < 120:
            # Stop if trailing choice list starts
            if kept_lines:
                break
        if _FOOTNOTE_RE.match(ln):
            if ln not in footnotes:
                footnotes.append(ln)
            continue
        kept_lines.append(ln)

    body = " ".join(kept_lines)
    body = _U_TAG_RE.sub("", body)
    body = _CIRCLED_WITH_PARENS_RE.sub(" ", body)
    body = _CIRCLED_MARKER_RE.sub("", body)
    body = re.sub(r"[ \t]{2,}", " ", body).strip()
    return body, footnotes


def split_english_sentences(text: str) -> list[str]:
    """Deterministically split an English paragraph into sentences while protecting abbreviations and decimals."""
    clean = re.sub(r"\s+", " ", (text or "").strip())
    if not clean:
        return []

    protected = _DECIMAL_RE.sub(rf"\1{_ABBREV_PLACEHOLDER}\2", clean)
    protected = _ABBREV_RE.sub(lambda m: m.group(0).replace(".", _ABBREV_PLACEHOLDER), protected)
    protected = _INITIAL_RE.sub(rf"\1{_ABBREV_PLACEHOLDER}", protected)

    raw_parts = _SENT_SPLIT_RE.split(protected)
    restored = [p.replace(_ABBREV_PLACEHOLDER, ".").strip() for p in raw_parts if p.strip()]

    merged: list[str] = []
    for seg in restored:
        word_count = len(re.findall(r"[A-Za-z]+", seg))
        if merged and (word_count < 3 or not re.search(r"[A-Za-z]", seg)):
            merged[-1] = f"{merged[-1]} {seg}".strip()
        else:
            merged.append(seg)
    return merged


def build_indexed_passage(raw_passage: str) -> IndexedPassage:
    """Sanitize passage text and split into IndexedPassage(sentences, footnotes)."""
    body, footnotes = sanitize_english_passage_deterministically(raw_passage)
    sentences = split_english_sentences(body)
    clean_text = " ".join(sentences).strip()
    return IndexedPassage(
        clean_text=clean_text,
        sentences=tuple(sentences),
        footnotes=tuple(footnotes),
    )


def check_type_feasibility(indexed: IndexedPassage, family: str) -> str | None:
    """Zero-cost pre-flight structural check before any LLM call."""
    n = len(indexed.sentences)
    if n < 2 or len(indexed.clean_text) < 80:
        return "지문 길이나 문장 수가 너무 적어 수능형 변형문제를 출제할 수 없습니다."
    if family == "order" and n < 4:
        return f"글의 순서 유형은 최소 4문장 이상이 필요합니다. (현재 {n}문장)"
    if family == "irrelevant" and n < 5:
        return f"전체 흐름과 무관한 문장 유형은 최소 5문장 이상이 필요합니다. (현재 {n}문장)"
    if family == "insertion" and n < 6:
        return f"주어진 문장 삽입 유형은 ①~⑤ 위치 확보를 위해 최소 6문장 이상이 필요합니다. (현재 {n}문장)"
    if family == "underline_5" and len(re.findall(r"[A-Za-z]+", indexed.clean_text)) < 35:
        return "어법/밑줄 어휘 5개 타깃을 배치하기에 지문 어휘 수가 부족합니다."
    return None


class _RestoredPassageSchema(BaseModel):
    restored_passage: str = Field(..., min_length=60)


def restore_passage_if_needed(
    api_key: str,
    model_name: str,
    raw_passage: str,
    base_question: Any = None,
    source_pdf_bytes: bytes | None = None,
) -> str:
    """Restore an incomplete exam passage (blank/order/choice brackets) into clean coherent prose."""
    _ = source_pdf_bytes
    raw = (raw_passage or "").strip()
    if not raw:
        return ""
    orig_type = (
        (getattr(base_question, "english_question_type", "") if base_question is not None else "")
        or (getattr(base_question, "unit_name", "") if base_question is not None else "")
    )
    if not needs_passage_restoration(raw, orig_type):
        body, fns = sanitize_english_passage_deterministically(raw)
        return (body + ("\n" + "\n".join(fns) if fns else "")).strip()

    q_id = str(getattr(base_question, "id", "") or "")
    cache_k = hashlib.sha256(f"{q_id}||{raw}".encode("utf-8")).hexdigest()[:24]
    with _RESTORE_CACHE_LOCK:
        if cache_k in _RESTORE_CACHE:
            return _RESTORE_CACHE[cache_k]

    if not (api_key or "").strip():
        body, fns = sanitize_english_passage_deterministically(raw)
        return (body + ("\n" + "\n".join(fns) if fns else "")).strip()

    q_text = str(getattr(base_question, "question_text", "") or "").strip()
    ans = str(getattr(base_question, "answer", "") or "").strip()
    exp = str(getattr(base_question, "official_explanation", "") or "").strip()

    prompt = (
        "Restore the incomplete CSAT English exam passage below into a single clean, complete, "
        "grammatically coherent English paragraph (preserving any '* word: 한국어 뜻' footnotes at the bottom).\n"
        "- Fill any blank (_____) or restore [주어진 문장] / (A)(B)(C) order / [A / B] choice brackets using the official answer and explanation.\n"
        "- Do NOT alter any existing sentences outside the incomplete slot.\n"
        "- Remove Korean question instructions and ①~⑤ choice lists.\n\n"
        f"[Raw Passage]\n{raw}\n\n"
        f"[Full Question Text]\n{q_text[:2500]}\n\n"
        f"[Official Answer]: {ans}\n"
        f"[Official Explanation]: {exp[:1200]}\n"
    )
    client = _build_client(api_key)
    res = _generate_structured_with_fallback(
        client=client,
        model_name=FLASH_LITE_MODEL,
        contents=prompt,
        schema_cls=_RestoredPassageSchema,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_RestoredPassageSchema,
            temperature=0.1,
        ),
        fast_mode=True,
    )
    body, fns = sanitize_english_passage_deterministically(res.restored_passage)
    cleaned = (body + ("\n" + "\n".join(fns) if fns else "")).strip()
    if cleaned:
        with _RESTORE_CACHE_LOCK:
            _RESTORE_CACHE[cache_k] = cleaned
    return cleaned or raw
