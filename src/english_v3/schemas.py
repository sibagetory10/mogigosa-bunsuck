"""Compact coordinate/delta Pydantic v2 schemas for English Engine v3 (token-minimized)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GistSpec(BaseModel):
    correct_choice: str = Field(..., min_length=2)
    distractors: list[str] = Field(..., min_length=4, max_length=4)
    explanation_ko: str = Field(..., min_length=8, max_length=220)


class SpanInferenceSpec(BaseModel):
    target_sentence_idx: int = Field(..., ge=0)
    exact_target_span: str = Field(..., min_length=2)
    correct_choice: str = Field(..., min_length=1)
    distractors: list[str] = Field(..., min_length=4, max_length=4)
    explanation_ko: str = Field(..., min_length=8, max_length=220)


class UnderlineTargetItem(BaseModel):
    sentence_idx: int = Field(..., ge=0)
    exact_span: str = Field(..., min_length=1)
    occurrence: int = Field(default=1, ge=1)


class Underline5Spec(BaseModel):
    items: list[UnderlineTargetItem] = Field(..., min_length=5, max_length=5)
    wrong_item_index: Literal[0, 1, 2, 3, 4]
    mutated_span: str = Field(..., min_length=1)
    explanation_ko: str = Field(..., min_length=8, max_length=220)


class AbcVocabSlot(BaseModel):
    sentence_idx: int = Field(..., ge=0)
    exact_span: str = Field(..., min_length=1)
    occurrence: int = Field(default=1, ge=1)
    foil_span: str = Field(..., min_length=1)


class ChoiceAbcSpec(BaseModel):
    slots: list[AbcVocabSlot] = Field(..., min_length=3, max_length=3)
    explanation_ko: str = Field(..., min_length=8, max_length=220)


class IrrelevantSpec(BaseModel):
    insert_after_sentence_idx: int = Field(..., ge=0)
    intruder_sentence: str = Field(..., min_length=25)
    explanation_ko: str = Field(..., min_length=8, max_length=220)


class OrderSpec(BaseModel):
    first_chunk_start_idx: int = Field(..., ge=1)
    second_chunk_start_idx: int = Field(..., ge=2)
    third_chunk_start_idx: int = Field(..., ge=3)
    cohesion_rationale_ko: str = Field(
        default="단락 간 지시어와 연결어 단서에 따라 유일한 논리적 순서로 이어진다."
    )


class InsertionSpec(BaseModel):
    extracted_sentence_idx: int = Field(..., ge=1)
    cohesion_rationale_ko: str = Field(
        default="주어진 문장의 지시·연결 단서가 해당 위치 앞뒤 문맥의 단절을 필연적으로 연결한다."
    )


class SummaryPair(BaseModel):
    word_a: str = Field(..., min_length=1)
    word_b: str = Field(..., min_length=1)


class SummarySpec(BaseModel):
    summary_sentence: str = Field(..., min_length=20)
    correct_pair: SummaryPair
    distractor_pairs: list[SummaryPair] = Field(..., min_length=4, max_length=4)
    explanation_ko: str = Field(..., min_length=8, max_length=220)


class BlindSolverVerdict(BaseModel):
    selected_answer: Literal["1", "2", "3", "4", "5", "NONE"]
    ambiguous_choices: list[Literal["1", "2", "3", "4", "5"]] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    note_ko: str = Field(default="")
