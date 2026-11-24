"""Coordinate/delta-only Pydantic v2 schemas for English Engine v3 (zero full-passage output)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TrapType = Literal[
    "too_narrow",
    "too_broad",
    "opposite",
    "distorted_relation",
    "irrelevant_keyword",
]


class DistractorItem(BaseModel):
    """Single plausible CSAT distractor with explicit trap category."""

    text: str = Field(..., min_length=1, description="오답 선지 텍스트")
    trap_type: TrapType = Field(..., description="오답 함정 설계 유형")


class GistSpec(BaseModel):
    """Family A: 요지 파악 / 주제 파악 / 제목 추론 (no passage text output)."""

    choice_language: Literal["ko", "en"] = Field(
        ...,
        description="요지 파악은 'ko', 주제 파악·제목 추론은 'en'",
    )
    core_thesis: str = Field(..., min_length=5, description="지문 핵심 논지 1문장 요약")
    correct_choice: str = Field(..., min_length=2, description="정답 선지")
    distractors: list[DistractorItem] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="정확히 4개의 매력적 오답 선지",
    )
    explanation_ko: str = Field(..., min_length=10, description="한국어 핵심 정답 근거 1~2문장")
    key_vocab: list[str] = Field(
        default_factory=list,
        max_length=6,
        description="핵심 어휘 목록 ('word: 뜻' 형식, 최대 6개)",
    )


class SpanInferenceSpec(BaseModel):
    """Family B: 함축적 의미 추론 / 빈칸 추론 (coordinate + paraphrased choices only)."""

    target_sentence_idx: int = Field(..., ge=0, description="타깃 구절이 속한 문장 인덱스 (0-based)")
    exact_target_span: str = Field(
        ...,
        min_length=2,
        description="sentences[target_sentence_idx]에 실제로 존재하는 원문 그대로의 타깃 구절(문장 부호 제외)",
    )
    correct_choice: str = Field(
        ...,
        min_length=1,
        description="빈칸 추론의 경우 원문 타깃 구절을 동의어·구문 전환으로 반드시 재진술(Paraphrase)한 정답 선지",
    )
    distractors: list[DistractorItem] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="정답과 문법적 품사·길이·어조가 대칭인 4개의 오답 선지",
    )
    explanation_ko: str = Field(..., min_length=10, description="한국어 정답 근거 1~2문장")
    key_vocab: list[str] = Field(default_factory=list, max_length=6)


class UnderlineTargetItem(BaseModel):
    """Single underlined slot for 어법 or 밑줄형 어휘."""

    sentence_idx: int = Field(..., ge=0, description="문장 인덱스 (0-based)")
    exact_span: str = Field(..., min_length=1, description="원문 해당 문장 내 실제 연속 구절 (1~4단어)")
    occurrence: int = Field(default=1, ge=1, description="해당 문장 내 몇 번째 등장인지 (1-based)")
    is_answer_target: bool = Field(
        ...,
        description="5개 항목 중 오직 1개만 True(출제자가 변형할 틀린/부적절한 정답 항목)",
    )
    mutated_span: str = Field(
        default="",
        description="is_answer_target=True일 때 원문 exact_span을 대체할 비문법/반의어 변형 표현 (False면 빈 문자열)",
    )
    point_note_ko: str = Field(default="", description="어법/문맥 포인트 1줄 메모")


class Underline5Spec(BaseModel):
    """Family C-1: 어법상 틀린 것 (밑줄 ①~⑤) / 문맥상 어휘 적절성 (밑줄형 ①~⑤)."""

    items: list[UnderlineTargetItem] = Field(
        ...,
        min_length=5,
        max_length=5,
        description="정확히 5개의 밑줄 타깃 (is_answer_target=True는 정확히 1개)",
    )
    explanation_ko: str = Field(..., min_length=10, description="한국어 정답 근거 및 5개 포인트 요약")
    key_vocab: list[str] = Field(default_factory=list, max_length=6)


class AbcVocabSlot(BaseModel):
    """Single (A)/(B)/(C) lexical choice slot."""

    sentence_idx: int = Field(..., ge=0, description="문장 인덱스 (0-based)")
    exact_span: str = Field(..., min_length=1, description="원문 문맥에 맞는 원래 단어/구 (정답)")
    occurrence: int = Field(default=1, ge=1, description="해당 문장 내 등장 순서 (1-based)")
    foil_span: str = Field(
        ...,
        min_length=1,
        description="품사와 어형이 동일하지만 문맥상 반대/부적절한 대조 어휘",
    )


class ChoiceAbcSpec(BaseModel):
    """Family C-2: 문맥상 어휘 적절성 (선택형 (A)(B)(C))."""

    slots: list[AbcVocabSlot] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="지문 등장 순서대로 정확히 3개의 (A)/(B)/(C) 어휘 쌍",
    )
    explanation_ko: str = Field(..., min_length=10, description="한국어 (A)(B)(C) 문맥 근거 요약")
    key_vocab: list[str] = Field(default_factory=list, max_length=6)


class IrrelevantSpec(BaseModel):
    """Family D: 전체 흐름과 무관한 문장."""

    insert_after_sentence_idx: int = Field(
        ...,
        ge=0,
        description="도입부 직후~결론 직전 사이, 무관한 문장을 삽입할 기준 문장 인덱스",
    )
    intruder_sentence: str = Field(
        ...,
        min_length=25,
        description="지문의 핵심 소재 명사 1~2개는 공유하지만 글의 중심 논지와 어긋나는 자연스러운 영어 함정 문장 1개",
    )
    explanation_ko: str = Field(..., min_length=10, description="한국어 흐름 이탈 근거")
    key_vocab: list[str] = Field(default_factory=list, max_length=6)


class OrderSpec(BaseModel):
    """Family E-1: 글의 순서 ((A)-(B)-(C)) — strict 3-chunk start boundaries."""

    first_chunk_start_idx: int = Field(
        ...,
        ge=1,
        description="[주어진 글] 직후 이어질 첫 번째 단락의 시작 문장 인덱스 (1 이상, [주어진 글]은 S0 ~ S[first_chunk_start_idx-1])",
    )
    second_chunk_start_idx: int = Field(
        ...,
        ge=2,
        description="원래 논리 순서상 두 번째 단락의 시작 문장 인덱스 (반드시 first_chunk_start_idx < second_chunk_start_idx)",
    )
    third_chunk_start_idx: int = Field(
        ...,
        ge=3,
        description="원래 논리 순서상 세 번째(마지막) 단락의 시작 문장 인덱스 (반드시 second_chunk_start_idx < third_chunk_start_idx <= N-1)",
    )
    cohesion_rationale_ko: str = Field(
        ...,
        min_length=10,
        description="지시어·대명사·연결어 기반 단락 연결 논리 한국어 해설",
    )
    key_vocab: list[str] = Field(default_factory=list, max_length=6)


class InsertionSpec(BaseModel):
    """Family E-2: 주어진 문장 삽입 — extracted sentence index only."""

    extracted_sentence_idx: int = Field(
        ...,
        ge=1,
        description="[주어진 문장]으로 추출할 문장 인덱스 (지시어·연결어 단서가 명확한 문장)",
    )
    cohesion_rationale_ko: str = Field(
        ...,
        min_length=10,
        description="추출 문장 앞뒤 단절·연결 단서 한국어 해설",
    )
    key_vocab: list[str] = Field(default_factory=list, max_length=6)


class SummaryPair(BaseModel):
    """Pair of words/phrases for (A) and (B) summary blanks."""

    word_a: str = Field(..., min_length=1)
    word_b: str = Field(..., min_length=1)


class SummarySpec(BaseModel):
    """Family F: 요약문 완성 ((A), (B))."""

    summary_sentence: str = Field(
        ...,
        min_length=20,
        description="반드시 '(A)'와 '(B)' 표기를 각각 1회씩 포함한 1~2문장 영어 요약문",
    )
    correct_pair: SummaryPair = Field(..., description="정답 (A), (B) 어휘 쌍")
    distractor_pairs: list[SummaryPair] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="정확히 4개의 오답 (A), (B) 어휘 쌍",
    )
    explanation_ko: str = Field(..., min_length=10, description="한국어 요약문 근거 해설")
    key_vocab: list[str] = Field(default_factory=list, max_length=6)


class BlindSolverVerdict(BaseModel):
    """Gate 2: Answer-hidden Blind Solver output."""

    brief_analysis: str = Field(..., min_length=5, description="선지별 핵심 판정 요약")
    selected_answer: Literal["1", "2", "3", "4", "5", "NONE"] = Field(
        ...,
        description="지문과 선지만 보고 도출한 유일한 정답 번호 (불가능 시 'NONE')",
    )
    ambiguous_choices: list[Literal["1", "2", "3", "4", "5"]] = Field(
        default_factory=list,
        description="정답 외에 복수 정답으로 방어 가능해 보이는 경합 선지 번호 목록",
    )
    confidence: Literal["high", "medium", "low"] = Field(..., description="단일 정답 확신도")
    defect_reason: str = Field(
        default="",
        description="복수 정답 의심·오류·모호성 발견 시 구체적 결함 사유",
    )
