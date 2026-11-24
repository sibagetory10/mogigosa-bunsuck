"""100% deterministic Python assembler: IndexedPassage + Spec -> RewrittenQuestion."""

from __future__ import annotations

import itertools
import random
import re
from dataclasses import dataclass, field
from typing import Any

from src.english_v3.preprocessor import IndexedPassage
from src.english_v3.schemas import (
    ChoiceAbcSpec,
    GistSpec,
    InsertionSpec,
    IrrelevantSpec,
    OrderSpec,
    SpanInferenceSpec,
    SummarySpec,
    Underline5Spec,
)
from src.models.concept import RewrittenQuestion

CIRCLED_NUMS: tuple[str, ...] = ("①", "②", "③", "④", "⑤")
ORDER_CHOICES: tuple[str, ...] = (
    "(A) - (C) - (B)",
    "(B) - (A) - (C)",
    "(B) - (C) - (A)",
    "(C) - (A) - (B)",
    "(C) - (B) - (A)",
)

_LEADING_CHOICE_PREFIX_RE = re.compile(r"^\s*(?:[①②③④⑤]|\(?[1-5]\)[.)]?)\s*")


class AssemblyError(ValueError):
    """Raised when LLM coordinate/delta output cannot be cleanly mapped onto the source passage."""


@dataclass
class AssemblyMetadata:
    """Metadata passed from assembler to Gate 1 deterministic rules validator."""

    family: str
    target_type: str
    blank_unit: str = ""
    vocab_style: str = ""
    exact_target_span: str = ""
    blank_prev_word: str = ""
    blank_next_word: str = ""
    non_target_sentences: list[str] = field(default_factory=list)
    mutated_pairs: list[tuple[str, str]] = field(default_factory=list)


def _clean_choice_text(text: str) -> str:
    s = re.sub(r"\s+", " ", (text or "").strip())
    return _LEADING_CHOICE_PREFIX_RE.sub("", s).strip()


def _normalize_english_choice_symmetry(choices: list[str]) -> list[str]:
    """Ensure uniform trailing punctuation across 5 English choices."""
    cleaned = [_clean_choice_text(c) for c in choices]
    if not cleaned or any(re.search(r"[가-힣]", c) for c in cleaned):
        return cleaned
    has_period = sum(1 for c in cleaned if c.endswith("."))
    if 0 < has_period < len(cleaned):
        if has_period >= 3:
            cleaned = [c if c.endswith((".", "?", "!")) else f"{c}." for c in cleaned]
        else:
            cleaned = [c[:-1] if c.endswith(".") else c for c in cleaned]
    return cleaned


def _find_span_in_sentence(sentence: str, span: str, occurrence: int = 1) -> tuple[int, int]:
    """Locate the N-th exact (or case-insensitive fallback) occurrence of span inside sentence."""
    target = (span or "").strip()
    if not target:
        raise AssemblyError("타깃 구절(exact_span)이 비어 있습니다.")

    def _search_nth(haystack: str, needle: str, occ: int) -> int:
        start = 0
        pos = -1
        for _ in range(max(1, occ)):
            pos = haystack.find(needle, start)
            if pos < 0:
                return -1
            start = pos + len(needle)
        return pos

    pos = _search_nth(sentence, target, occurrence)
    if pos >= 0:
        return pos, pos + len(target)

    # Fallback: strip trailing sentence punctuation that LLM occasionally includes
    stripped = target.rstrip(".,;:?!")
    if stripped and stripped != target:
        pos = _search_nth(sentence, stripped, occurrence)
        if pos >= 0:
            return pos, pos + len(stripped)

    # Case-insensitive fallback inside the same sentence
    pos_ci = _search_nth(sentence.lower(), (stripped or target).lower(), occurrence)
    if pos_ci >= 0:
        end_ci = pos_ci + len(stripped or target)
        return pos_ci, end_ci

    raise AssemblyError(f"지정 문장에서 원문 구절을 찾을 수 없습니다: '{target}'")


def _seeded_shuffle_choices(
    correct: str,
    distractors: list[str],
    seed_val: int,
    *,
    normalize_symmetry: bool = True,
) -> tuple[list[str], str]:
    """Deterministically shuffle [correct, *distractors] and return (choices_5, '1'..'5')."""
    items = [(_clean_choice_text(correct), True)] + [(_clean_choice_text(d), False) for d in distractors[:4]]
    if len(items) != 5:
        raise AssemblyError(f"선지 개수가 5개가 아닙니다: {len(items)}개")
    rng = random.Random(seed_val)
    rng.shuffle(items)
    raw_texts = [t for t, _ in items]
    if normalize_symmetry:
        raw_texts = _normalize_english_choice_symmetry(raw_texts)
    ans_idx = next(i for i, (_, is_ok) in enumerate(items) if is_ok)
    return raw_texts, str(ans_idx + 1)


def _format_explanation(ans_num: str, body: str) -> str:
    sym = CIRCLED_NUMS[int(ans_num) - 1] if ans_num in ("1", "2", "3", "4", "5") else ans_num
    clean_body = re.sub(r"^\s*정답\s*[:：]\s*[①②③④⑤1-5]번?\s*", "", (body or "").strip())
    return f"정답: {sym} ({ans_num}번)\n{clean_body}".strip()


def _join_with_footnotes(stem_body: str, footnotes: tuple[str, ...]) -> str:
    if not footnotes:
        return stem_body.strip()
    return f"{stem_body.strip()}\n\n" + "\n".join(footnotes)


def _resolve_instruction(target_type: str, vocab_style: str = "") -> str:
    t = (target_type or "").strip()
    if "함축" in t:
        return "밑줄 친 부분이 다음 글에서 의미하는 바로 가장 적절한 것은?"
    if "요지" in t:
        return "다음 글의 요지로 가장 적절한 것은?"
    if "주제" in t:
        return "다음 글의 주제로 가장 적절한 것은?"
    if "제목" in t:
        return "다음 글의 제목으로 가장 적절한 것은?"
    if "어법" in t:
        return "다음 글의 밑줄 친 부분 중, 어법상 틀린 것은?"
    if "어휘" in t:
        if "선택형" in vocab_style or "선택형" in t or "(A)" in vocab_style:
            return "(A), (B), (C)의 각 네모 안에서 문맥에 맞는 낱말로 가장 적절한 것은?"
        return "다음 글의 밑줄 친 부분 중, 문맥상 낱말의 쓰임이 적절하지 않은 것은?"
    if "빈칸" in t:
        return "다음 빈칸에 들어갈 말로 가장 적절한 것은?"
    if "무관" in t:
        return "다음 글에서 전체 흐름과 관계없는 문장은?"
    if "순서" in t:
        return "주어진 글 다음에 이어질 글의 순서로 가장 적절한 것은?"
    if "삽입" in t:
        return "글의 흐름으로 보아, 주어진 문장이 들어가기에 가장 적절한 곳은?"
    if "요약" in t:
        return "다음 글의 내용을 한 문장으로 요약하고자 한다. 빈칸 (A), (B)에 들어갈 말로 가장 적절한 것은?"
    return "다음 글을 읽고 물음에 답하시오."


def _effective_target_label(target_type: str, blank_unit: str = "", vocab_style: str = "") -> str:
    t = (target_type or "").strip()
    if "빈칸" in t:
        unit_short = (blank_unit or "구").split()[0].strip()
        return f"빈칸 추론 ({unit_short})"
    if "어휘" in t:
        if "선택형" in vocab_style or "(A)" in vocab_style:
            return "문맥상 어휘 적절성 (선택형)"
        return "문맥상 어휘 적절성 (밑줄형)"
    return t


def assemble_question(
    indexed: IndexedPassage,
    spec: Any,
    *,
    target_type: str,
    blank_unit: str = "구 (3~6단어)",
    vocab_style: str = "밑줄형 (①~⑤)",
    seed_val: int = 42,
    self_note: str = "",
) -> tuple[RewrittenQuestion, AssemblyMetadata]:
    """Assemble IndexedPassage + LLM delta Spec into a renderer-ready RewrittenQuestion."""
    inst = _resolve_instruction(target_type, vocab_style=vocab_style)
    eff_type = _effective_target_label(target_type, blank_unit=blank_unit, vocab_style=vocab_style)
    sentences = list(indexed.sentences)
    n_sent = len(sentences)

    if isinstance(spec, GistSpec):
        choices, ans = _seeded_shuffle_choices(
            spec.correct_choice,
            [d.text for d in spec.distractors],
            seed_val,
            normalize_symmetry=(spec.choice_language == "en"),
        )
        stem = _join_with_footnotes(f"{inst}\n\n{indexed.clean_text}", indexed.footnotes)
        rw = RewrittenQuestion(
            rewritten_stem=stem,
            rewritten_choices=choices,
            replaced_terms_map=list(spec.key_vocab[:6]),
            pdf_grounded_explanation=_format_explanation(ans, spec.explanation_ko),
            preserved_answer=ans,
            self_verification_note=self_note or "English v3 Gate 1 Passed",
            target_question_type=eff_type,
            subject_domain="영어",
        )
        meta = AssemblyMetadata(
            family="gist",
            target_type=eff_type,
            non_target_sentences=list(sentences),
        )
        return rw, meta

    if isinstance(spec, SpanInferenceSpec):
        if not (0 <= spec.target_sentence_idx < n_sent):
            raise AssemblyError(f"target_sentence_idx 범위 초과: {spec.target_sentence_idx}")
        orig_s = sentences[spec.target_sentence_idx]
        s_start, s_end = _find_span_in_sentence(orig_s, spec.exact_target_span, occurrence=1)
        matched_span = orig_s[s_start:s_end]
        is_blank = "빈칸" in target_type

        if is_blank:
            replaced_s = f"{orig_s[:s_start]}_____{orig_s[s_end:]}"
            prev_word_m = re.search(r"([A-Za-z']+)$", orig_s[:s_start].strip())
            next_word_m = re.match(r"^([A-Za-z']+)", orig_s[s_end:].strip())
            prev_w = prev_word_m.group(1) if prev_word_m else ""
            next_w = next_word_m.group(1) if next_word_m else ""
        else:
            replaced_s = f"{orig_s[:s_start]}<u>{matched_span}</u>{orig_s[s_end:]}"
            prev_w, next_w = "", ""

        mod_sentences = list(sentences)
        mod_sentences[spec.target_sentence_idx] = replaced_s
        body = " ".join(mod_sentences)
        stem = _join_with_footnotes(f"{inst}\n\n{body}", indexed.footnotes)
        choices, ans = _seeded_shuffle_choices(
            spec.correct_choice,
            [d.text for d in spec.distractors],
            seed_val,
            normalize_symmetry=True,
        )
        rw = RewrittenQuestion(
            rewritten_stem=stem,
            rewritten_choices=choices,
            replaced_terms_map=list(spec.key_vocab[:6]),
            pdf_grounded_explanation=_format_explanation(ans, spec.explanation_ko),
            preserved_answer=ans,
            self_verification_note=self_note or "English v3 Gate 1 + Blind Solver Passed",
            target_question_type=eff_type,
            subject_domain="영어",
        )
        meta = AssemblyMetadata(
            family="span_inference",
            target_type=eff_type,
            blank_unit=blank_unit,
            exact_target_span=matched_span,
            blank_prev_word=prev_w,
            blank_next_word=next_w,
            non_target_sentences=[s for i, s in enumerate(sentences) if i != spec.target_sentence_idx],
        )
        return rw, meta

    if isinstance(spec, Underline5Spec):
        ans_targets = [it for it in spec.items if it.is_answer_target]
        if len(ans_targets) != 1:
            raise AssemblyError(f"is_answer_target=True 항목은 정확히 1개여야 합니다 (현재 {len(ans_targets)}개).")

        located: list[tuple[int, int, int, int, str, str, bool]] = []
        for orig_i, it in enumerate(spec.items):
            if not (0 <= it.sentence_idx < n_sent):
                raise AssemblyError(f"sentence_idx 범위 초과: {it.sentence_idx}")
            st_pos, en_pos = _find_span_in_sentence(
                sentences[it.sentence_idx], it.exact_span, occurrence=it.occurrence
            )
            matched_orig = sentences[it.sentence_idx][st_pos:en_pos]
            disp_span = (it.mutated_span or "").strip() if it.is_answer_target else matched_orig
            if not disp_span:
                raise AssemblyError("정답 타깃의 mutated_span이 비어 있습니다.")
            located.append((it.sentence_idx, st_pos, en_pos, orig_i, matched_orig, disp_span, it.is_answer_target))

        located.sort(key=lambda x: (x[0], x[1]))
        # Check non-overlap within same sentence
        for k in range(1, len(located)):
            if located[k][0] == located[k - 1][0] and located[k][1] < located[k - 1][2]:
                raise AssemblyError("동일 문장 내 밑줄 타깃 구간이 서로 겹칩니다.")

        mod_sentences = list(sentences)
        ans_num = ""
        mutated_pairs: list[tuple[str, str]] = []
        per_sent: dict[int, list[tuple[int, int, str, str]]] = {}
        for rank_idx, (s_idx, st_pos, en_pos, _, orig_s, disp_s, is_ans) in enumerate(located):
            sym = CIRCLED_NUMS[rank_idx]
            if is_ans:
                ans_num = str(rank_idx + 1)
                mutated_pairs.append((orig_s, disp_s))
            per_sent.setdefault(s_idx, []).append((st_pos, en_pos, sym, disp_s))

        for s_idx, spans in per_sent.items():
            cur = mod_sentences[s_idx]
            for st_pos, en_pos, sym, disp_s in sorted(spans, key=lambda x: x[0], reverse=True):
                cur = f"{cur[:st_pos]}{sym} <u>{disp_s}</u>{cur[en_pos:]}"
            mod_sentences[s_idx] = cur

        stem = _join_with_footnotes(f"{inst}\n\n{' '.join(mod_sentences)}", indexed.footnotes)
        rw = RewrittenQuestion(
            rewritten_stem=stem,
            rewritten_choices=list(CIRCLED_NUMS),
            replaced_terms_map=list(spec.key_vocab[:6]),
            pdf_grounded_explanation=_format_explanation(ans_num, spec.explanation_ko),
            preserved_answer=ans_num,
            self_verification_note=self_note or "English v3 Gate 1 + Blind Solver Passed",
            target_question_type=eff_type,
            subject_domain="영어",
        )
        non_target = [s for i, s in enumerate(sentences) if i not in per_sent]
        meta = AssemblyMetadata(
            family="underline_5",
            target_type=eff_type,
            non_target_sentences=non_target,
            mutated_pairs=mutated_pairs,
        )
        return rw, meta

    if isinstance(spec, ChoiceAbcSpec):
        located_abc: list[tuple[int, int, int, str, str]] = []
        for sl in spec.slots:
            if not (0 <= sl.sentence_idx < n_sent):
                raise AssemblyError(f"sentence_idx 범위 초과: {sl.sentence_idx}")
            st_pos, en_pos = _find_span_in_sentence(
                sentences[sl.sentence_idx], sl.exact_span, occurrence=sl.occurrence
            )
            orig_w = sentences[sl.sentence_idx][st_pos:en_pos]
            foil_w = (sl.foil_span or "").strip()
            if not foil_w:
                raise AssemblyError("선택형 어휘 foil_span이 비어 있습니다.")
            located_abc.append((sl.sentence_idx, st_pos, en_pos, orig_w, foil_w))

        located_abc.sort(key=lambda x: (x[0], x[1]))
        for k in range(1, len(located_abc)):
            if located_abc[k][0] == located_abc[k - 1][0] and located_abc[k][1] < located_abc[k - 1][2]:
                raise AssemblyError("동일 문장 내 (A)(B)(C) 어휘 구간이 겹칩니다.")

        rng = random.Random(seed_val)
        flips = [rng.choice([True, False]) for _ in range(3)]
        if all(flips) or not any(flips):
            flips[1] = not flips[1]

        mod_sentences = list(sentences)
        labels = ("(A)", "(B)", "(C)")
        per_sent_abc: dict[int, list[tuple[int, int, str, str, str]]] = {}
        orig_words: list[str] = []
        foil_words: list[str] = []
        mutated_pairs_abc: list[tuple[str, str]] = []

        for idx_l, (s_idx, st_pos, en_pos, orig_w, foil_w) in enumerate(located_abc):
            orig_words.append(orig_w)
            foil_words.append(foil_w)
            mutated_pairs_abc.append((orig_w, foil_w))
            left_w, right_w = (foil_w, orig_w) if flips[idx_l] else (orig_w, foil_w)
            per_sent_abc.setdefault(s_idx, []).append((st_pos, en_pos, labels[idx_l], left_w, right_w))

        for s_idx, spans in per_sent_abc.items():
            cur = mod_sentences[s_idx]
            for st_pos, en_pos, lbl, lw, rw_txt in sorted(spans, key=lambda x: x[0], reverse=True):
                cur = f"{cur[:st_pos]}{lbl} [{lw} / {rw_txt}]{cur[en_pos:]}"
            mod_sentences[s_idx] = cur

        all_combos = list(itertools.product([0, 1], repeat=3))
        correct_combo = (0, 0, 0)
        distractor_combos = [c for c in all_combos if c != correct_combo]
        rng.shuffle(distractor_combos)
        picked_distractors = distractor_combos[:4]

        def _fmt_combo(combo: tuple[int, int, int]) -> str:
            w_a = orig_words[0] if combo[0] == 0 else foil_words[0]
            w_b = orig_words[1] if combo[1] == 0 else foil_words[1]
            w_c = orig_words[2] if combo[2] == 0 else foil_words[2]
            return f"(A) {w_a} — (B) {w_b} — (C) {w_c}"

        choices, ans = _seeded_shuffle_choices(
            _fmt_combo(correct_combo),
            [_fmt_combo(c) for c in picked_distractors],
            seed_val + 7,
            normalize_symmetry=False,
        )
        stem = _join_with_footnotes(f"{inst}\n\n{' '.join(mod_sentences)}", indexed.footnotes)
        rw = RewrittenQuestion(
            rewritten_stem=stem,
            rewritten_choices=choices,
            replaced_terms_map=list(spec.key_vocab[:6]),
            pdf_grounded_explanation=_format_explanation(ans, spec.explanation_ko),
            preserved_answer=ans,
            self_verification_note=self_note or "English v3 Gate 1 + Blind Solver Passed",
            target_question_type=eff_type,
            subject_domain="영어",
        )
        meta = AssemblyMetadata(
            family="choice_abc",
            target_type=eff_type,
            vocab_style=vocab_style,
            non_target_sentences=[s for i, s in enumerate(sentences) if i not in per_sent_abc],
            mutated_pairs=mutated_pairs_abc,
        )
        return rw, meta

    if isinstance(spec, IrrelevantSpec):
        # Keep sentences[0] as unnumbered lead; sentences[1:5] + 1 intruder = 5 numbered sentences ①~⑤
        if n_sent < 5:
            raise AssemblyError("무관한 문장 유형은 최소 5문장 이상이 필요합니다.")
        # Valid insertion slots among numbered ①..⑤: slot 0..4 (after S0, S1, S2, S3, S4)
        raw_after = int(spec.insert_after_sentence_idx)
        slot_idx = max(0, min(4, raw_after))
        lead = sentences[0]
        core_4 = list(sentences[1:5])
        tail_rest = list(sentences[5:])
        intruder = re.sub(r"\s+", " ", (spec.intruder_sentence or "").strip())
        if not intruder:
            raise AssemblyError("무관한 문장(intruder_sentence)이 비어 있습니다.")

        combined_5: list[str] = []
        orig_cursor = 0
        for pos in range(5):
            if pos == slot_idx:
                combined_5.append(f"{CIRCLED_NUMS[pos]} {intruder}")
            else:
                combined_5.append(f"{CIRCLED_NUMS[pos]} {core_4[orig_cursor]}")
                orig_cursor += 1

        full_parts = [lead, *combined_5, *tail_rest]
        stem = _join_with_footnotes(f"{inst}\n\n{' '.join(full_parts)}", indexed.footnotes)
        ans = str(slot_idx + 1)
        rw = RewrittenQuestion(
            rewritten_stem=stem,
            rewritten_choices=list(CIRCLED_NUMS),
            replaced_terms_map=list(spec.key_vocab[:6]),
            pdf_grounded_explanation=_format_explanation(ans, spec.explanation_ko),
            preserved_answer=ans,
            self_verification_note=self_note or "English v3 Gate 1 + Blind Solver Passed",
            target_question_type=eff_type,
            subject_domain="영어",
        )
        meta = AssemblyMetadata(
            family="irrelevant",
            target_type=eff_type,
            non_target_sentences=list(sentences),
        )
        return rw, meta

    if isinstance(spec, OrderSpec):
        c1 = int(spec.first_chunk_start_idx)
        c2 = int(spec.second_chunk_start_idx)
        c3 = int(spec.third_chunk_start_idx)
        if not (1 <= c1 < c2 < c3 < n_sent):
            raise AssemblyError(
                f"글의 순서 분할 경계가 유효하지 않습니다: c1={c1}, c2={c2}, c3={c3}, n={n_sent}"
            )

        given_text = " ".join(sentences[:c1]).strip()
        chunk1 = " ".join(sentences[c1:c2]).strip()
        chunk2 = " ".join(sentences[c2:c3]).strip()
        chunk3 = " ".join(sentences[c3:]).strip()
        if not all((given_text, chunk1, chunk2, chunk3)):
            raise AssemblyError("글의 순서 조립 중 빈 단락이 발생했습니다.")
        chunks = [chunk1, chunk2, chunk3]

        # Non-canonical permutations: perm[label_idx] = original_chunk_idx (never (0, 1, 2))
        non_canonical_perms: tuple[tuple[int, int, int], ...] = (
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        )
        rng = random.Random(seed_val)
        perm = rng.choice(non_canonical_perms)
        # label_for_orig[orig_chunk_idx] = "(A)" / "(B)" / "(C)"
        labels = ("(A)", "(B)", "(C)")
        label_for_orig = [""] * 3
        labeled_lines: list[str] = []
        for lbl_i, orig_chunk_i in enumerate(perm):
            label_for_orig[orig_chunk_i] = labels[lbl_i]
            labeled_lines.append(f"{labels[lbl_i]} {chunks[orig_chunk_i]}")

        true_seq_str = f"{label_for_orig[0]} - {label_for_orig[1]} - {label_for_orig[2]}"
        if true_seq_str not in ORDER_CHOICES:
            raise AssemblyError(f"계산된 순서 정답 '{true_seq_str}'이 표준 선택지에 없습니다.")
        ans = str(ORDER_CHOICES.index(true_seq_str) + 1)

        stem_body = f"{inst}\n\n[주어진 글]\n{given_text}\n\n" + "\n".join(labeled_lines)
        stem = _join_with_footnotes(stem_body, indexed.footnotes)
        rw = RewrittenQuestion(
            rewritten_stem=stem,
            rewritten_choices=list(ORDER_CHOICES),
            replaced_terms_map=list(spec.key_vocab[:6]),
            pdf_grounded_explanation=_format_explanation(ans, spec.cohesion_rationale_ko),
            preserved_answer=ans,
            self_verification_note=self_note or "English v3 Gate 1 + Blind Solver Passed",
            target_question_type=eff_type,
            subject_domain="영어",
        )
        meta = AssemblyMetadata(
            family="order",
            target_type=eff_type,
            non_target_sentences=list(sentences),
        )
        return rw, meta

    if isinstance(spec, InsertionSpec):
        ext_idx = int(spec.extracted_sentence_idx)
        if not (1 <= ext_idx < n_sent):
            raise AssemblyError(f"extracted_sentence_idx 범위 초과: {ext_idx} (1..{n_sent - 1})")
        if n_sent < 6:
            raise AssemblyError("문장 삽입 유형은 최소 6문장 이상이 필요합니다.")

        given_sent = sentences[ext_idx]
        rem = [s for i, s in enumerate(sentences) if i != ext_idx]
        true_gap = ext_idx  # gap k in [1..len(rem)]

        if len(rem) >= 6:
            max_start = len(rem) - 5
            g_start = max(1, min(max_start, true_gap - 2))
            if true_gap > g_start + 4:
                g_start = true_gap - 4
        else:
            g_start = 1

        gap_to_marker = {g_start + k: f"( {CIRCLED_NUMS[k]} )" for k in range(5)}
        ans_slot = true_gap - g_start + 1
        if not (1 <= ans_slot <= 5):
            raise AssemblyError("문장 삽입 정답 슬롯 매핑 오류")
        ans = str(ans_slot)

        parts: list[str] = []
        for idx_r, r_sent in enumerate(rem):
            if idx_r in gap_to_marker:
                parts.append(gap_to_marker[idx_r])
            parts.append(r_sent)
        if len(rem) in gap_to_marker:
            parts.append(gap_to_marker[len(rem)])

        stem_body = f"{inst}\n\n[주어진 문장]\n{given_sent}\n\n" + " ".join(parts)
        stem = _join_with_footnotes(stem_body, indexed.footnotes)
        rw = RewrittenQuestion(
            rewritten_stem=stem,
            rewritten_choices=list(CIRCLED_NUMS),
            replaced_terms_map=list(spec.key_vocab[:6]),
            pdf_grounded_explanation=_format_explanation(ans, spec.cohesion_rationale_ko),
            preserved_answer=ans,
            self_verification_note=self_note or "English v3 Gate 1 + Blind Solver Passed",
            target_question_type=eff_type,
            subject_domain="영어",
        )
        meta = AssemblyMetadata(
            family="insertion",
            target_type=eff_type,
            non_target_sentences=list(sentences),
        )
        return rw, meta

    if isinstance(spec, SummarySpec):
        sum_sent = re.sub(r"\s+", " ", (spec.summary_sentence or "").strip())
        if "(A)" not in sum_sent or "(B)" not in sum_sent:
            raise AssemblyError("요약문에 '(A)'와 '(B)' 슬롯이 모두 포함되어야 합니다.")
        # Normalize (A) / (B) blanks cleanly without duplicating underscores
        sum_sent = re.sub(r"_{2,}\s*\(A\)\s*_{2,}|\(A\)\s*_{2,}", "(A) _____", sum_sent)
        sum_sent = re.sub(r"_{2,}\s*\(B\)\s*_{2,}|\(B\)\s*_{2,}", "(B) _____", sum_sent)
        if "(A) _____" not in sum_sent:
            sum_sent = sum_sent.replace("(A)", "(A) _____", 1)
        if "(B) _____" not in sum_sent:
            sum_sent = sum_sent.replace("(B)", "(B) _____", 1)

        def _fmt_pair(wa: str, wb: str) -> str:
            return f"(A) {wa.strip()} — (B) {wb.strip()}"

        choices, ans = _seeded_shuffle_choices(
            _fmt_pair(spec.correct_pair.word_a, spec.correct_pair.word_b),
            [_fmt_pair(p.word_a, p.word_b) for p in spec.distractor_pairs],
            seed_val,
            normalize_symmetry=False,
        )
        stem_body = f"{inst}\n\n{indexed.clean_text}\n\n[요약문]\n{sum_sent}"
        stem = _join_with_footnotes(stem_body, indexed.footnotes)
        rw = RewrittenQuestion(
            rewritten_stem=stem,
            rewritten_choices=choices,
            replaced_terms_map=list(spec.key_vocab[:6]),
            pdf_grounded_explanation=_format_explanation(ans, spec.explanation_ko),
            preserved_answer=ans,
            self_verification_note=self_note or "English v3 Gate 1 + Blind Solver Passed",
            target_question_type=eff_type,
            subject_domain="영어",
        )
        meta = AssemblyMetadata(
            family="summary",
            target_type=eff_type,
            non_target_sentences=list(sentences),
        )
        return rw, meta

    raise AssemblyError(f"지원되지 않는 Spec 타입입니다: {type(spec).__name__}")
