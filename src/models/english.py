from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.models.korean import KoreanVisualDiagram

class EnglishSingleExtractionOutput(BaseModel):
    """영어 기출 단일 문항 추출 스키마 (2022 성취기준·학년 제한 없이 지문·유형·어휘 추출)."""

    english_section: Literal["듣기", "독해"] = Field(
        ...,
        description="1~17번은 '듣기', 18~45번은 '독해'",
    )
    english_question_type: str = Field(
        ...,
        description="원래 문항 유형 (예: 목적, 심경, 함축 의미, 요지, 주제, 제목, 도표, 내용 일치, 실용문, 어법, 어휘, 빈칸 추론, 무관한 문장, 글의 순서, 문장 삽입, 요약문 완성, 장문 독해, 듣기)",
    )
    passage_group_id: str = Field(
        default="",
        description="장문 세트 문항인 경우 공통 지문 그룹 (예: '41~42', '43~45', 단일 문항은 빈 문자열)",
    )
    passage_text: str = Field(
        default="",
        description="독해(18~45번) 문항의 순수 영어 지문 원문 (발문과 선택지 ①~⑤를 제외한 본문 전체; 듣기 1~17번은 빈 문자열)",
    )
    passage_topic: str = Field(
        default="",
        description="영어 지문의 중심 주제·소재 한국어 1줄 요약",
    )
    key_vocabulary: list[str] = Field(
        default_factory=list,
        description="지문 핵심 어휘 3~6개 목록 (형식: 'word: 한국어 뜻')",
    )
    official_explanation: str = Field(
        ...,
        description="해설지 기준 핵심 풀이 및 정답 근거 요약",
    )
    answer: str = Field(
        ...,
        description="공식 정답 ('1'~'5' 단일 숫자 문자열)",
    )
    standard_code: str = Field(
        default="",
        description="2022 개정 교육과정 성취기준 대표 코드 (기준표가 제공된 경우에만 매칭, 없으면 빈 문자열)",
    )
    standard_codes: list[str] = Field(
        default_factory=list,
        description="2022 개정 교육과정 관련 성취기준 코드 목록",
    )
    needs_review: bool = Field(
        default=False,
        description="성취기준 코드 검증 필요 여부",
    )
    review_notes: list[str] = Field(
        default_factory=list,
        description="검증 시 발견된 주의사항",
    )


class EnglishQuestionExtractionItem(EnglishSingleExtractionOutput):
    """영어 모의고사 전체 PDF 일괄 추출 시 문항 번호가 포함된 항목."""

    q_num: int = Field(..., ge=1, le=45, description="문항 번호 (1~45)")


class BatchEnglishExamExtractionOutput(BaseModel):
    """영어 모의고사 PDF 전체 일괄 추출 출력 스키마."""

    questions: list[EnglishQuestionExtractionItem] = Field(default_factory=list)
CURRENT_ENGLISH_ANALYSIS_VERSION: int = 5
CURRENT_ENGLISH_TRANSFORM_VERSION: int = 2


class EnglishGistTransformSpec(BaseModel):
    """요지 파악 · 주제 파악 · 제목 추론 전용 구조화 출력 스키마."""

    choices: list[str] = Field(
        ...,
        description="번호(①~⑤)를 제외한 정확히 5개의 선택지 리스트 (요지는 한국어 완결 문장, 주제는 영어 명사구, 제목은 영어 타이틀 형식)",
    )
    correct_index: int = Field(
        ...,
        ge=1,
        le=5,
        description="유일한 정답 선지 번호 (1~5 정수)",
    )
    explanation: str = Field(
        ...,
        description="정답 도출 근거와 오답 핵심 요약 (한국어 2문장, 120자 이내)",
    )
    key_terms: list[str] = Field(
        default_factory=list,
        description="지문 핵심 어휘 2개 ('word → 한국어 뜻' 형식)",
    )


class EnglishSpanTargetTransformSpec(BaseModel):
    """빈칸 추론(단어/구/절) · 함축적 의미 추론 전용 구조화 출력 스키마."""

    target_sentence_index: int = Field(
        ...,
        ge=0,
        description="[S0]~[Sn-1] 중 빈칸으로 비우거나 밑줄을 그을 구절이 포함된 원문 문장 번호 (0부터 시작)",
    )
    target_span: str = Field(
        ...,
        description="해당 문장 안에 실제로 존재하는 연속된 영어 원문 구절 (원문 철자 100% 그대로)",
    )
    correct_choice: str = Field(
        ...,
        description="정답 영어 선지 1개 (빈칸 추론은 빈칸에 들어갈 정답 표현, 함축적 의미는 밑줄 구절의 문맥상 함축 의미를 나타낸 영어 표현)",
    )
    distractor_choices: list[str] = Field(
        ...,
        description="정답과 길이·품사·문법 구조가 동일하면서 문맥상 틀린 매력적인 영어 오답 선지 4개 (번호 제외)",
    )
    explanation: str = Field(
        ...,
        description="정답 도출 근거 및 문맥 단서 요약 (한국어 2문장, 120자 이내)",
    )
    key_terms: list[str] = Field(
        default_factory=list,
        description="지문 핵심 어휘 2개 ('word → 한국어 뜻' 형식)",
    )


class EnglishUnderlineFiveItem(BaseModel):
    """어법/어휘 밑줄형(①~⑤) 단일 밑줄 포인트 명세."""

    sentence_index: int = Field(
        ...,
        ge=0,
        description="해당 구절이 위치한 원문 문장 번호 ([S0] -> 0)",
    )
    original_span: str = Field(
        ...,
        description="해당 원문 문장에 실제로 존재하는 1~3단어 영어 구절 (원문 철자 그대로)",
    )
    displayed_span: str = Field(
        ...,
        description="시험지에 밑줄 그어 표시할 구절 (정상 항목 4개는 original_span과 동일, 정답인 1개만 틀린 어법 또는 문맥상 반의어로 변형)",
    )
    is_error_target: bool = Field(
        default=False,
        description="이 항목이 어법상 틀리거나 문맥상 부적절한 유일한 정답 항목이면 True, 나머지 4개는 False",
    )


class EnglishUnderlineFiveTransformSpec(BaseModel):
    """어법상 틀린 것(①~⑤) · 문맥상 어휘 적절성(밑줄형 ①~⑤) 전용 구조화 출력 스키마."""

    items: list[EnglishUnderlineFiveItem] = Field(
        ...,
        description="원문 등장 순서대로 선정한 정확히 5개의 밑줄 항목 (반드시 1개만 is_error_target=True 및 original_span != displayed_span)",
    )
    explanation: str = Field(
        ...,
        description="정답 밑줄이 왜 틀렸는지(원래 올바른 형태 명시 포함) 한국어 2문장 이내 해설",
    )
    key_terms: list[str] = Field(
        default_factory=list,
        description="지문 핵심 어휘 2개 ('word → 한국어 뜻' 형식)",
    )


class EnglishVocabChoiceTripleItem(BaseModel):
    """선택형 어휘((A)(B)(C)) 단일 네모 어휘 쌍 명세."""

    sentence_index: int = Field(
        ...,
        ge=0,
        description="해당 단어가 위치한 원문 문장 번호 ([S0] -> 0)",
    )
    original_word: str = Field(
        ...,
        description="해당 원문 문장에 실제로 존재하는 문맥상 올바른 핵심 영단어 1개",
    )
    antonym_word: str = Field(
        ...,
        description="original_word와 품사·어미 형태가 100% 동일한 문맥상 반대(부적절) 영단어 1개",
    )


class EnglishVocabChoiceTransformSpec(BaseModel):
    """문맥상 어휘 적절성(선택형 (A)(B)(C)) 전용 구조화 출력 스키마."""

    items: list[EnglishVocabChoiceTripleItem] = Field(
        ...,
        description="원문 등장 순서대로 선정한 정확히 3개의 (A), (B), (C) 어휘 쌍",
    )
    explanation: str = Field(
        ...,
        description="(A), (B), (C) 각 위치에서 왜 원문 어휘가 문맥상 적절한지 한국어 2문장 이내 해설",
    )
    key_terms: list[str] = Field(
        default_factory=list,
        description="지문 핵심 어휘 2개 ('word → 한국어 뜻' 형식)",
    )


class EnglishOrderTransformSpec(BaseModel):
    """글의 순서((A)-(B)-(C)) 전용 구조화 출력 스키마."""

    split_sentence_indices: list[int] = Field(
        ...,
        description="[주어진 글] 뒤의 3개 단락이 시작되는 문장 인덱스 3개 [s1, s2, s3] (반드시 0 < s1 < s2 < s3 < 전체 문장 수)",
    )
    explanation: str = Field(
        ...,
        description="단락 간 지시어·연결어·논리 흐름 근거 한국어 2문장 이내 해설",
    )
    key_terms: list[str] = Field(
        default_factory=list,
        description="지문 핵심 어휘 2개 ('word → 한국어 뜻' 형식)",
    )


class EnglishInsertionTransformSpec(BaseModel):
    """주어진 문장 삽입 전용 구조화 출력 스키마."""

    extracted_sentence_index: int = Field(
        ...,
        ge=1,
        description="[주어진 문장]으로 발췌할 원문 문장 인덱스 (첫 문장 S0을 제외한 1 이상 인덱스 중 연결어·지시어 단서가 명확한 문장 1개)",
    )
    explanation: str = Field(
        ...,
        description="주어진 문장의 연결어·지시어와 앞뒤 문장 단서를 바탕으로 한 한국어 2문장 이내 해설",
    )
    key_terms: list[str] = Field(
        default_factory=list,
        description="지문 핵심 어휘 2개 ('word → 한국어 뜻' 형식)",
    )


class EnglishIrrelevantTransformSpec(BaseModel):
    """전체 흐름과 무관한 문장 전용 구조화 출력 스키마."""

    insert_position: int = Field(
        ...,
        ge=2,
        le=4,
        description="①~⑤ 중 무관한 문장을 삽입할 정답 번호 (2, 3, 4 중 하나)",
    )
    unrelated_sentence: str = Field(
        ...,
        description="원문의 핵심 소재 단어만 표면적으로 공유하되 글의 중심 논지와 어긋나는 자연스러운 새 영어 문장 1개",
    )
    explanation: str = Field(
        ...,
        description="글의 중심 주제와 비교해 해당 번호 문장이 왜 흐름과 무관한지 한국어 2문장 이내 해설",
    )
    key_terms: list[str] = Field(
        default_factory=list,
        description="지문 핵심 어휘 2개 ('word → 한국어 뜻' 형식)",
    )


class EnglishSummaryPairItem(BaseModel):
    """요약문 완성 (A), (B) 오답 어휘 쌍."""

    word_a: str = Field(..., description="(A)에 들어갈 영단어 1개")
    word_b: str = Field(..., description="(B)에 들어갈 영단어 1개")


class EnglishSummaryTransformSpec(BaseModel):
    """요약문 완성((A), (B)) 전용 구조화 출력 스키마."""

    summary_template: str = Field(
        ...,
        description="지문 전체를 1문장으로 요약한 영어 문장 (반드시 '(A) ________'와 '(B) ________' 빈칸 표기 포함)",
    )
    correct_word_a: str = Field(
        ...,
        description="빈칸 (A)의 정답 영단어 1개 (원문 핵심어를 패러프레이징한 어휘)",
    )
    correct_word_b: str = Field(
        ...,
        description="빈칸 (B)의 정답 영단어 1개 (원문 핵심어를 패러프레이징한 어휘)",
    )
    distractor_pairs: list[EnglishSummaryPairItem] = Field(
        ...,
        description="정답 쌍과 구별되는 매력적인 오답 (A)-(B) 영단어 쌍 4개",
    )
    explanation: str = Field(
        ...,
        description="지문 핵심 내용을 바탕으로 (A), (B)에 해당 어휘가 들어가야 하는 근거 한국어 2문장 이내 해설",
    )
    key_terms: list[str] = Field(
        default_factory=list,
        description="지문 핵심 어휘 2개 ('word → 한국어 뜻' 형식)",
    )


class EnglishPassageSummaryDiagramOutput(BaseModel):
    """AI 전용 영어 지문 핵심 주제 · 논리 전개 흐름 · 구조 도식 출력 스키마 (문장별 분석 제외)."""

    title: str = Field(..., description="지문 핵심 제목 (한국어 중심 또는 영문+한국어 병기)")
    core_theme: str = Field(..., description="지문 전체의 핵심 주제 · 요지 요약 (1~2문장)")
    structure_flow: str = Field(
        ...,
        description="지문 전체의 논리 전개 흐름 요약 (예: '도입 : ...\\n전개 : ...\\n결론 : ...')",
    )
    visual_diagram: KoreanVisualDiagram | None = Field(
        default=None,
        description="지문의 핵심 논리·비교·인과·과정을 시각화하는 도식 그림 명세",
    )


class EnglishSentenceAnalysisItem(BaseModel):
    """영어 지문 문장별 원문 및 해설지 기반 한국어 해석 항목 (AI 미사용, 순수 PDF/DB 복붙)."""

    sentence_num: int = Field(..., ge=1, description="문장 번호 (1부터 시작)")
    is_topic_sentence: bool = Field(
        default=False,
        description="하위 호환용 필드 (v5부터 미사용)",
    )
    original_english: str = Field(
        default="",
        description="문제지/DB 원문에서 분리한 순수 영어 원문 문장 (AI 미사용)",
    )
    chunked_english: str = Field(
        default="",
        description="원문 영문 문장 (v5부터 original_english와 동일)",
    )
    korean_translation: str = Field(
        default="",
        description="모의고사 해설지 PDF [해석] 섹션에서 추출한 한국어 해석 문장",
    )
    syntax_notes: list[str] = Field(
        default_factory=list,
        description="하위 호환용 빈 배열 []",
    )
    original_blank_phrase: str = Field(
        default="",
        description="하위 호환용 필드 (v5부터 미사용)",
    )
    is_inserted_sentence: bool = Field(
        default=False,
        description="하위 호환용 필드 (v5부터 미사용)",
    )


class EnglishPassageAnalysisOutput(BaseModel):
    """영어 정밀 지문 분석지 구조화 출력 스키마 (주제 · 논리 흐름 · 시각 도식 · 문장별 원문/해설지 해석)."""

    title: str = Field(..., description="지문 핵심 제목 (한국어 중심 또는 영문+한국어 병기)")
    core_theme: str = Field(..., description="지문 전체의 핵심 주제 · 요지 요약 (1~2문장)")
    structure_flow: str = Field(
        ...,
        description="지문 전체의 논리 전개 흐름 요약 (예: '도입 : ...\\n전개 : ...\\n결론 : ...')",
    )
    visual_diagram: KoreanVisualDiagram | None = Field(
        default=None,
        description="지문의 핵심 논리·비교·인과·과정을 시각화하는 도식 그림 명세",
    )
    sentence_analyses: list[EnglishSentenceAnalysisItem] = Field(
        default_factory=list,
        description="지문 전체 문장에 대한 1번부터 끝번까지의 원문·해설지 해석 리스트",
    )
    exam_point_title: str = Field(
        default="",
        description="하위 호환용 필드 (v5부터 미사용)",
    )
    exam_point_note: str = Field(
        default="",
        description="하위 호환용 필드 (v5부터 미사용)",
    )
    footnotes: list[str] = Field(
        default_factory=list,
        description="문제지 공식 각주 어휘 목록 (예: ['delineate: ~의 경계를 정하다'])",
    )
    translation_source: str = Field(
        default="",
        description="해석 출처 ('answer_pdf' | 'direct_input' | 'none')",
    )
    is_source_restored: bool = Field(
        default=False,
        description="원문 추출 완료 여부",
    )
    analysis_version: int = Field(
        default=1,
        description="분석지 엔진 스키마 버전 (v5: 기출 포인트 제거, 3번 섹션 AI 미사용 원문+해설지 PDF 순수 복붙)",
    )
