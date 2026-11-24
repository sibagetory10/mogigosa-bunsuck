"""Zero-cost Python discourse cohesion analyzer for Order ((A)-(B)-(C)) and Insertion items."""

from __future__ import annotations

import re
from typing import Sequence

from src.english_v3.preprocessor import IndexedPassage

_STRONG_CONNECTORS: tuple[str, ...] = (
    "on the other hand",
    "in contrast",
    "by contrast",
    "as a result",
    "for this reason",
    "for example",
    "for instance",
    "in other words",
    "in addition",
    "at the same time",
    "in this case",
    "in that case",
    "in the process",
    "in doing so",
    "in fact",
    "that is",
    "however",
    "nevertheless",
    "nonetheless",
    "instead",
    "conversely",
    "therefore",
    "consequently",
    "accordingly",
    "moreover",
    "furthermore",
    "similarly",
    "likewise",
    "meanwhile",
    "otherwise",
    "indeed",
    "still",
    "thus",
    "hence",
    "yet",
)

_DEMONSTRATIVE_RE = re.compile(
    r"\b(?:this|these|that|those|such|the\s+same|another|both)\b",
    re.IGNORECASE,
)
_PRONOUN_START_RE = re.compile(
    r"^(?:he|she|they|their|his|her|its|it)\b",
    re.IGNORECASE,
)
_QA_ANSWER_START_RE = re.compile(
    r"^(?:no|yes|the\s+answer|because|not\s+only|it|they)\b",
    re.IGNORECASE,
)
_THE_NOUN_RE = re.compile(r"\bthe\s+([a-z]{4,})\b", re.IGNORECASE)
_STOP_CONTENT = frozenset({
    "this", "that", "these", "those", "with", "from", "into", "over", "under",
    "about", "after", "before", "same", "other", "first", "second", "third",
    "last", "next", "only", "very", "more", "most", "some", "many", "such",
    "there", "their", "them", "they", "have", "been", "were", "will", "would",
    "could", "should", "what", "when", "where", "which", "while", "whose",
})


def _first_n_words(sentence: str, n: int = 7) -> str:
    words = re.findall(r"[A-Za-z'\-]+", sentence or "")
    return " ".join(words[:n]).lower()


def _content_nouns(sentence: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z]{4,}", (sentence or "").lower())
        if w not in _STOP_CONTENT
    }


def is_boundary_inside_open_quote(sentences: Sequence[str], boundary_idx: int) -> bool:
    """Return True if splitting before sentences[boundary_idx] cuts inside an unclosed quotation."""
    if boundary_idx <= 0 or boundary_idx >= len(sentences):
        return False
    prefix = " ".join(sentences[:boundary_idx])
    # Straight double quotes
    if prefix.count('"') % 2 == 1:
        return True
    # Curly double quotes
    if prefix.count("“") > prefix.count("”"):
        return True
    # Curly single-quote dialogue/block quotes (excluding apostrophes like don't / Smith's)
    open_sq = len(re.findall(r"(?:^|[\s(\[:])‘", prefix))
    close_sq = len(re.findall(r"’(?:$|[\s.,?!;:\)\]])", prefix))
    if open_sq > close_sq:
        return True
    # ASCII single-quote block quotes preceded by space/colon/comma
    open_asq = len(re.findall(r"(?:^|[\s(\[:])'(?=[A-Za-z])", prefix))
    close_asq = len(re.findall(r"[.?!,]'(?=$|[\s.,?!;:\)\]])", prefix))
    if open_asq > close_asq:
        return True
    return False


def analyze_sentence_cohesion(sentences: Sequence[str], idx: int) -> tuple[int, list[str]]:
    """Score how strongly sentences[idx] is anchored to sentences[idx - 1]."""
    if idx <= 0 or idx >= len(sentences):
        return 0, []

    cur = sentences[idx].strip()
    prev = sentences[idx - 1].strip()
    head7 = _first_n_words(cur, 7)
    score = 0
    clues: list[str] = []

    for conn in _STRONG_CONNECTORS:
        if re.search(rf"(?:^|\b){re.escape(conn)}\b", head7):
            score += 5
            clues.append(f"연결어({conn})")
            break

    if _DEMONSTRATIVE_RE.search(head7):
        score += 5
        clues.append("지시어(this/that/these/such)")

    if prev.rstrip("\"'”’) ").endswith("?") and _QA_ANSWER_START_RE.match(head7):
        score += 6
        clues.append("질문→응답")

    if _PRONOUN_START_RE.match(head7):
        # Avoid false-positive cleft/weather "It is ... that/to ..."
        if not re.match(r"^it\s+(?:is|was|seems|appears)\b", head7):
            score += 3
            clues.append("대명사")

    prev_nouns = _content_nouns(prev)
    for m in _THE_NOUN_RE.finditer(head7):
        noun = m.group(1).lower()
        if noun in prev_nouns or noun.rstrip("s") in prev_nouns:
            score += 3
            clues.append(f"정관사(the {noun})")
            break

    return score, clues


def rank_order_splits(
    indexed: IndexedPassage,
    exclude: set[tuple[int, int, int]] | None = None,
) -> list[tuple[tuple[int, int, int], int, str]]:
    """Enumerate and rank all valid (c1, c2, c3) order splits with 1 <= c1 < c2 < c3 < N."""
    sentences = indexed.sentences
    n = len(sentences)
    if n < 4:
        return []
    excluded = exclude or set()
    ranked: list[tuple[tuple[int, int, int], int, str]] = []

    for c1 in range(1, n - 2):
        for c2 in range(c1 + 1, n - 1):
            for c3 in range(c2 + 1, n):
                triple = (c1, c2, c3)
                if triple in excluded:
                    continue

                if (
                    is_boundary_inside_open_quote(sentences, c1)
                    or is_boundary_inside_open_quote(sentences, c2)
                    or is_boundary_inside_open_quote(sentences, c3)
                ):
                    total = -100
                    desc = "인용문 내부 절단 패널티"
                else:
                    s1, cl1 = analyze_sentence_cohesion(sentences, c1)
                    s2, cl2 = analyze_sentence_cohesion(sentences, c2)
                    s3, cl3 = analyze_sentence_cohesion(sentences, c3)
                    total = s1 + s2 + s3

                    # Penalize isolating a 1-sentence chunk that has zero cohesive anchor
                    if c2 - c1 == 1 and s1 == 0:
                        total -= 6
                    if c3 - c2 == 1 and s2 == 0:
                        total -= 6
                    if n - c3 == 1 and s3 == 0:
                        total -= 6

                    # Slight preference for 1~2 sentence [주어진 글]
                    if c1 in (1, 2):
                        total += 2

                    clue_parts = [
                        f"S{c1}({','.join(cl1) or '무단서'})",
                        f"S{c2}({','.join(cl2) or '무단서'})",
                        f"S{c3}({','.join(cl3) or '무단서'})",
                    ]
                    desc = " / ".join(clue_parts)

                ranked.append((triple, total, desc))

    ranked.sort(key=lambda x: (-x[1], x[0]))
    return ranked


def rank_insertion_targets(
    indexed: IndexedPassage,
    exclude: set[int] | None = None,
) -> list[tuple[int, int, str]]:
    """Rank valid extracted_sentence_idx candidates (preferring internal 1 <= idx <= N-2)."""
    sentences = indexed.sentences
    n = len(sentences)
    if n < 6:
        return []
    excluded = exclude or set()
    ranked: list[tuple[int, int, str]] = []

    internal_candidates = list(range(1, n - 1))
    candidates = internal_candidates
    # Only include n - 1 as last resort when n == 6 and all internal ones were excluded
    if not [c for c in internal_candidates if c not in excluded] and (n - 1) not in excluded:
        candidates = [n - 1]

    for idx in candidates:
        if idx in excluded:
            continue
        if is_boundary_inside_open_quote(sentences, idx) or is_boundary_inside_open_quote(sentences, idx + 1):
            ranked.append((idx, -100, "인용문 내부 절단 패널티"))
            continue

        back_score, back_clues = analyze_sentence_cohesion(sentences, idx)
        fwd_score = 0
        fwd_clues: list[str] = []
        if idx + 1 < n:
            fs, fc = analyze_sentence_cohesion(sentences, idx + 1)
            if fs > 0:
                fwd_score += 3
                fwd_clues = fc

        # Prefer middle positions slightly over position 1 when scores tie
        pos_bonus = 1 if 2 <= idx <= n - 2 else 0
        total = back_score + fwd_score + pos_bonus
        desc = f"S{idx}: 앞단서({','.join(back_clues) or '없음'}) / 뒷단서({','.join(fwd_clues) or '없음'})"
        ranked.append((idx, total, desc))

    ranked.sort(key=lambda x: (-x[1], x[0]))
    return ranked
