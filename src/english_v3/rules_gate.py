"""Gate 1: Zero-cost deterministic Python hard-rule validator for English Engine v3."""

from __future__ import annotations

import re
import statistics
from typing import Any

from src.english_v3.assembler import AssemblyMetadata
from src.english_v3.preprocessor import IndexedPassage
from src.models.concept import RewrittenQuestion
from src.storage.english_pool_store import _is_valid_cached_english_question

_FUNCTION_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "nor", "so", "yet", "for",
    "in", "on", "at", "to", "from", "by", "with", "of", "as", "into",
    "onto", "upon", "about", "above", "below", "between", "through",
    "during", "before", "after", "without", "within", "among", "against",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "have", "has", "had", "having",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    "it", "its", "they", "them", "their", "theirs", "this", "that", "these", "those",
    "he", "him", "his", "she", "her", "hers", "we", "us", "our", "you", "your",
    "who", "whom", "whose", "which", "what", "when", "where", "why", "how",
    "not", "no", "more", "most", "less", "least", "very", "much", "many",
    "some", "any", "each", "every", "all", "both", "such", "own", "other", "another",
    "than", "then", "also", "only", "just", "even", "still", "already",
})

_META_LEAK_RE = re.compile(r"\[S\d+\]|(?:^|\s)(?:정답|오답)\s*[:：]|[\u4e00-\u9fff]")
_TAG_STRIP_RE = re.compile(r"</?u\b[^>]*>|\(\s*[①②③④⑤]\s*\)|[①②③④⑤]\s*|\([ABC]\)\s*")


def _content_tokens(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z]+", (text or "").lower())
    return [w for w in words if w not in _FUNCTION_WORDS and len(w) >= 2]


def _word_ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _verify_source_invariance(indexed: IndexedPassage, rw: RewrittenQuestion, meta: AssemblyMetadata) -> list[str]:
    """Ensure every non-target sentence from IndexedPassage survives verbatim in rewritten_stem."""
    violations: list[str] = []
    # Strip assembly markup tags and collapse whitespace to check verbatim survival
    norm_stem = _TAG_STRIP_RE.sub(" ", rw.rewritten_stem or "")
    norm_stem = re.sub(r"\s+", " ", norm_stem).strip()

    for sent in meta.non_target_sentences:
        norm_s = re.sub(r"\s+", " ", sent).strip()
        if norm_s and norm_s not in norm_stem:
            violations.append(f"원문 불변성 위반: 비타깃 문장이 변형되거나 누락됨 ({norm_s[:45]}...)")
            break
    return violations


def _verify_blank_paraphrase(rw: RewrittenQuestion, meta: AssemblyMetadata) -> list[str]:
    """Block verbatim copy-paste in blank-filling answers and adjacent word duplication."""
    violations: list[str] = []
    if "빈칸" not in meta.target_type or not meta.exact_target_span:
        return violations

    ans_idx = int(rw.preserved_answer) - 1
    if not (0 <= ans_idx < len(rw.rewritten_choices)):
        return ["정답 번호가 선지 범위를 벗어났습니다."]

    ans_text = rw.rewritten_choices[ans_idx]
    orig_content = _content_tokens(meta.exact_target_span)
    ans_content = _content_tokens(ans_text)

    if orig_content and ans_content:
        if "단어" in meta.blank_unit:
            for ow in orig_content:
                for aw in ans_content:
                    if ow == aw or (len(ow) >= 5 and len(aw) >= 5 and ow[:5] == aw[:5]):
                        violations.append(
                            f"빈칸 정답 패러프레이징 위반(단어): 원문 핵심어 '{ow}'가 정답 '{aw}'에 그대로 재사용됨"
                        )
                        break
        else:
            orig_all = re.findall(r"[A-Za-z]+", meta.exact_target_span.lower())
            ans_all = re.findall(r"[A-Za-z]+", ans_text.lower())
            if _word_ngrams(orig_all, 4) & _word_ngrams(ans_all, 4):
                violations.append("빈칸 정답 패러프레이징 위반: 원문 4단어 이상 연속 구절을 그대로 복사함")
            elif len(orig_content) >= 2 and len(ans_content) >= 2:
                inter = set(orig_content) & set(ans_content)
                union = set(orig_content) | set(ans_content)
                jaccard = len(inter) / max(1, len(union))
                if jaccard > 0.55:
                    violations.append(
                        f"빈칸 정답 패러프레이징 부족: 핵심 내용어 중복률 과다 (Jaccard={jaccard:.2f} > 0.55)"
                    )

    # Check adjacent word duplication across all 5 choices (e.g., "to _____ " + choice starting with "to ")
    prev_low = (meta.blank_prev_word or "").lower()
    next_low = (meta.blank_next_word or "").lower()
    for i, ch in enumerate(rw.rewritten_choices, start=1):
        words = re.findall(r"[A-Za-z']+", ch.lower())
        if not words:
            continue
        if prev_low and words[0] == prev_low and prev_low in _FUNCTION_WORDS:
            violations.append(f"{i}번 선지 첫 단어 '{words[0]}'가 빈칸 앞 단어와 중복됩니다.")
        if next_low and words[-1] == next_low and next_low in _FUNCTION_WORDS:
            violations.append(f"{i}번 선지 끝 단어 '{words[-1]}'가 빈칸 뒤 단어와 중복됩니다.")

    return violations


def _verify_choice_symmetry(rw: RewrittenQuestion, meta: AssemblyMetadata) -> list[str]:
    """Block duplicate choices, meta leaks, and extreme answer-length outliers."""
    violations: list[str] = []
    choices = list(rw.rewritten_choices or [])
    if len(choices) != 5:
        return [f"선지 개수는 5개여야 합니다 (현재 {len(choices)}개)."]

    normed = [re.sub(r"\s+", " ", c.strip().lower()) for c in choices]
    if len(set(normed)) != 5:
        violations.append("5개 선지 중 중복된 선지가 존재합니다.")

    if _META_LEAK_RE.search(rw.rewritten_stem or ""):
        violations.append("지문 본문에 내부 인덱스([S0]) 또는 한자/메타 노이즈가 노출되었습니다.")
    for c in choices:
        if _META_LEAK_RE.search(c):
            violations.append(f"선지에 메타 노이즈/한자가 포함됨: {c}")

    if meta.family in ("gist", "span_inference"):
        is_korean = "요지" in meta.target_type
        if not is_korean:
            for c in choices:
                if re.search(r"[가-힣]", c):
                    violations.append(f"영문 선지에 한국어가 혼입되었습니다: {c}")

        ans_idx = int(rw.preserved_answer) - 1
        if 0 <= ans_idx < 5:
            lengths = [
                len(re.sub(r"\s+", "", c)) if is_korean else len(re.findall(r"[A-Za-z']+", c))
                for c in choices
            ]
            ans_len = lengths[ans_idx]
            dist_lens = [lengths[i] for i in range(5) if i != ans_idx]
            med = statistics.median(dist_lens)
            if med >= 4:
                if ans_len > max(dist_lens) and ans_len > med * 1.45:
                    violations.append(
                        f"선지 길이 편차 위반: 정답({ans_len})만 오답 중앙값({med:.1f}) 대비 지나치게 깁니다."
                    )
                elif ans_len < min(dist_lens) and ans_len < med * 0.55:
                    violations.append(
                        f"선지 길이 편차 위반: 정답({ans_len})만 오답 중앙값({med:.1f}) 대비 지나치게 짧습니다."
                    )

    return violations


def _verify_mutations(meta: AssemblyMetadata) -> list[str]:
    """Ensure grammar/vocab mutations actually differ from the original span."""
    violations: list[str] = []
    for orig, mut in meta.mutated_pairs:
        if re.sub(r"\s+", " ", orig.strip().lower()) == re.sub(r"\s+", " ", mut.strip().lower()):
            violations.append(f"변형 오류: 원문 '{orig}'과 변형 표현 '{mut}'이 동일합니다.")
        if "<u" in mut.lower() or re.search(r"[①②③④⑤가-힣]", mut):
            violations.append(f"변형 표현에 허용되지 않는 태그/기호가 포함됨: {mut}")
    return violations


def validate_assembled_question(
    indexed: IndexedPassage,
    spec: Any,
    rw: RewrittenQuestion,
    meta: AssemblyMetadata,
) -> list[str]:
    """Run all Gate 1 deterministic checks; empty list means PASS."""
    _ = spec
    violations: list[str] = []
    if not _is_valid_cached_english_question(rw):
        violations.append("표준 출력 구조 무결성 검사(_is_valid_cached_english_question)를 통과하지 못했습니다.")
    violations.extend(_verify_source_invariance(indexed, rw, meta))
    violations.extend(_verify_blank_paraphrase(rw, meta))
    violations.extend(_verify_choice_symmetry(rw, meta))
    violations.extend(_verify_mutations(meta))
    return violations
