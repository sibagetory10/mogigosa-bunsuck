from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator

SubjectCategory = Literal["수학", "사회", "과학", "영어"]

ENGLISH_DEFAULT_EXCLUDED_Q_NUMS: frozenset[int] = frozenset({18, 19, 25, 26, 27, 28, 43, 44, 45})

# 실제로 하나의 영어 지문을 공유하는 세트 문항만 그룹으로 인정한다.
# ([31~34], [36~37], [38~39] 등은 '발문 공통 헤더'일 뿐 지문은 문항마다 각각 다르다.)
ENGLISH_SHARED_PASSAGE_SETS: dict[int, str] = {
    41: "41~42",
    42: "41~42",
    43: "43~45",
    44: "43~45",
    45: "43~45",
}


def normalize_english_group_id(q_num: int, raw_group: str = "") -> str:
    """영어 세트 지문 그룹 ID를 문항 번호 기준으로 정규화한다 (41~42, 43~45 외에는 항상 빈 문자열)."""
    return ENGLISH_SHARED_PASSAGE_SETS.get(int(q_num), "")

ENGLISH_TRANSFORM_TYPES: tuple[str, ...] = (
    "함축적 의미 추론",
    "요지 파악",
    "주제 파악",
    "제목 추론",
    "어법상 틀린 것 (밑줄 ①~⑤)",
    "문맥상 어휘 적절성",
    "빈칸 추론",
    "전체 흐름과 무관한 문장",
    "글의 순서 ((A)-(B)-(C))",
    "주어진 문장 삽입",
    "요약문 완성 ((A), (B))",
)


def _coerce_grade_int(value: Any) -> int:
    if isinstance(value, bool):
        return 1
    if isinstance(value, int):
        return min(3, max(1, value))
    s = str(value or "").strip()
    for d in ("1", "2", "3"):
        if d in s:
            return int(d)
    return 1


class QuestionRecord(BaseModel):
    """저장소에 영구 보관되는 단일 기출 문항 스키마 (2022 개정 교육과정 기준 및 영어 전용 확장 포함)."""

    id: str = Field(..., description="고유 문항 ID (예: 2026_고3_6월_화학__Q05)")
    exam_name: str = Field(..., description="시험 회차명 (예: 2026_고3_6월_화학)")
    q_num: int = Field(..., ge=1, description="문항 번호")
    subject_category: Literal["수학", "사회", "과학", "영어"] = Field(..., description="대분류 과목")
    curriculum_subject: str = Field(..., description="2022 개정 교육과정 세부 과목명")
    unit_name: str = Field(..., description="단원명")
    min_grade: int = Field(default=1, ge=1, le=3, description="최저 학년 수준 (1=고1, 2=고2, 3=고3)")

    @field_validator("min_grade", mode="before")
    @classmethod
    def _validate_record_grade(cls, value: Any) -> int:
        return _coerce_grade_int(value)
    core_principles: list[str] = Field(
        default_factory=list,
        description="이 문제를 푸는 데 필요한 본질적 풀이 원리·법칙 리스트",
    )
    required_terms: list[str] = Field(
        default_factory=list,
        description="문제 지문과 보기에 등장하는 핵심 전문 용어·단위 리스트 (상위+기본 통합)",
    )
    advanced_terms: list[str] = Field(
        default_factory=list,
        description="상위 학년/선택과목에서 도입되는 낯선 전문 용어·기호 리스트",
    )
    basic_terms: list[str] = Field(
        default_factory=list,
        description="해당 최저 학년(min_grade) 수준에서 이미 학습해 당연한 기본 용어·기호 리스트",
    )
    standard_code: str = Field(
        default="",
        description="대표 2022 개정 성취기준 코드 또는 공용 개념서 기준 단원 식별자",
    )
    standard_codes: list[str] = Field(
        default_factory=list,
        description="복합 문항에 걸쳐 있는 모든 2022 개정 성취기준 코드 목록",
    )
    grade_evidence: str = Field(
        default="",
        description="최저 학년(min_grade) 판정 근거 요약",
    )
    needs_review: bool = Field(
        default=False,
        description="2022 기준표 대조 시 과목-학년 모순 또는 근거 부족으로 검토가 필요한지 여부",
    )
    review_notes: list[str] = Field(
        default_factory=list,
        description="할루시네이션·모순 검사에서 감지된 검토 사유 목록",
    )
    is_single_concept: bool = Field(
        default=True,
        description="단일 핵심 원리만으로 풀리는지 여부",
    )
    answer: str = Field(..., description="공식 정답")
    official_explanation: str = Field(..., description="공식 해설 핵심 풀이 요약")
    question_text: str = Field(
        default="",
        description="PyMuPDF로 추출한 문항 원문 텍스트 (AI 서칭·변형 시 토큰 절감용)",
    )
    english_section: str = Field(
        default="",
        description="영어 영역 구분 ('듣기' | '독해', 비영어 과목은 빈 문자열)",
    )
    english_question_type: str = Field(
        default="",
        description="영어 원래 문항 유형 (예: 목적, 심경, 함축 의미, 요지, 주제, 제목, 도표, 내용 일치, 어법, 어휘, 빈칸, 무관한 문장, 순서, 삽입, 요약문, 장문)",
    )
    passage_text: str = Field(
        default="",
        description="영어 독해 지문 원문 (41~42, 43~45 세트 문항은 공통 지문 연결)",
    )
    passage_group_id: str = Field(
        default="",
        description="세트 문항 공통 지문 그룹 식별자 (예: '41~42', '43~45')",
    )
    passage_topic: str = Field(
        default="",
        description="영어 지문 주제·소재 요약",
    )
    key_vocabulary: list[str] = Field(
        default_factory=list,
        description="영어 지문 핵심 어휘 목록 (예: ['resilience: 회복탄력성'])",
    )
    image_filename: str = Field(default="", description="크롭된 문항 PNG 파일명")
    source_pdf: str = Field(default="", description="원본 문제지 PDF 파일명")
    answer_pdf: str = Field(default="", description="원본 해설지 PDF 파일명")
    created_at: str = Field(default="", description="생성 시각 (ISO 8601)")


class QuestionExtractionOutput(BaseModel):
    """Gemini가 크롭 이미지 + 해설지 PDF로부터 추출하는 구조화 메타데이터."""

    curriculum_subject: str = Field(
        ...,
        description="2022 개정 교육과정 세부 과목명 (예: 공통수학1, 공통수학2, 대수, 미적분1, 통합사회1, 통합사회2, 경제, 통합과학1, 통합과학2, 화학, 물리학 등)",
    )
    unit_name: str = Field(..., description="단원명")
    min_grade: int = Field(
        ...,
        ge=1,
        le=3,
        description="이 문항의 본질적 풀이 원리를 처음 배우는 최저 학년 (반드시 1, 2, 3 중 하나의 정수: 1=고1, 2=고2, 3=고3)",
    )

    @field_validator("min_grade", mode="before")
    @classmethod
    def _validate_extracted_grade(cls, value: Any) -> int:
        return _coerce_grade_int(value)
    core_principles: list[str] = Field(
        ...,
        description="표면적인 상위 용어를 배제하고 문제를 푸는 데 필요한 본질적 풀이 원리·법칙 리스트 (예: 질량 보존 법칙, 비례식, 수요·공급 이동)",
    )
    required_terms: list[str] = Field(
        ...,
        description="실제 문제 지문, 그림, 보기에 등장하는 핵심 전문 용어·기호 목록 (일상어 및 단순 길이/질량 단위 제외)",
    )
    advanced_terms: list[str] = Field(
        default_factory=list,
        description="고2~3 선택과목 등 상위 과정에서 도입되어 하위 학년 학생에게는 낯선 전문 용어·기호 목록 (예: ['mol', '몰농도', '한계효용', '도함수'])",
    )
    basic_terms: list[str] = Field(
        default_factory=list,
        description="min_grade 수준에서 이미 학습해 당연한 기본 교과 용어·기호 목록 (예: ['속력', '이차방정식', '인권'])",
    )
    standard_code: str = Field(
        default="",
        description="대표 2022 성취기준 코드 (기준표에 있을 경우 공식 코드, 예: '10통과1-02-01')",
    )
    standard_codes: list[str] = Field(
        default_factory=list,
        description="복합 문항 등 여러 성취기준이 걸치는 경우 해당하는 모든 2022 공식 성취기준 코드 목록 (목록에 없으면 비움)",
    )
    grade_evidence: str = Field(
        default="",
        description="왜 이 최저 학년(min_grade)과 과목으로 판정했는지 2022 개정 교육과정 기준 1줄 근거",
    )
    needs_review: bool = Field(
        default=False,
        description="자동 검증기 전용 플래그 (기본 False)",
    )
    review_notes: list[str] = Field(
        default_factory=list,
        description="자동 검증기 전용 메모 (기본 빈 리스트)",
    )
    is_single_concept: bool = Field(
        ...,
        description="단일 핵심 원리만으로 풀리면 True, 여러 원리가 복합되면 False",
    )
    official_explanation: str = Field(
        ...,
        description="공식 해설 요약 또는 해설지 미첨부 시 단계별 직접 풀이·검산 과정 및 선지 판정 근거",
    )
    answer: str = Field(
        ...,
        description="최종 정답 (객관식은 반드시 '1'~'5' 중 단일 숫자 문자열, 주관식/단답형은 정확한 최종 값)",
    )


class BatchQuestionTagItem(QuestionExtractionOutput):
    """전체 모의고사 PDF 일괄 태깅 시 각 문항 번호가 포함된 구조화 메타데이터."""

    q_num: int = Field(..., ge=1, description="문항 번호 (1-based)")


class BatchExamExtractionOutput(BaseModel):
    """전체 모의고사 PDF를 한 번에 태깅해 반환하는 문항 목록 스키마."""

    questions: list[BatchQuestionTagItem] = Field(
        default_factory=list,
        description="요청된 전체 문항 번호별 원리·용어·정답·해설 태깅 결과 목록",
    )


class ConceptUnit(BaseModel):
    """단원별 개념 요약·허용 용어 및 실제 페이지 번호 스키마."""

    unit_name: str = Field(..., description="단원명 (예: Ⅲ. 변화와 다양성 - 1. 산화와 환원)")
    page_range: str = Field(..., description="실제 개념서 페이지 범위 (예: p.12 ~ p.25)")
    core_concepts: list[str] = Field(
        default_factory=list,
        description="핵심 원리 및 정의 요약 리스트",
    )
    allowed_terms_and_units: list[str] = Field(
        default_factory=list,
        description="해당 단원에서 허용되는 용어, 공식, 단위 리스트 (예: mL, 온도, 이온 수 등)",
    )


class ConceptBookExtractionOutput(BaseModel):
    """Gemini가 개념서 PDF 지정 페이지들로부터 추출하는 단원별 구조화 스키마."""

    book_title: str = Field(..., description="교재명 (예: 완자 통합과학)")
    subject: str = Field(..., description="과목명 (예: 통합과학)")
    grade: str = Field(..., description="학년 (예: 고1)")
    units: list[ConceptUnit] = Field(
        default_factory=list,
        description="단원(대단원/중단원)별 개념 및 허용 용어 구조화 목록",
    )


class AutoConceptRangeItem(BaseModel):
    """AI가 개념서·교과서 PDF 전체에서 자동 판별한 순수 개념 설명 페이지 구간."""

    unit_hint: str = Field(..., description="해당 구간이 다루는 대단원/중단원 또는 주제명")
    start_page: int = Field(..., ge=1, description="개념 설명 시작 PDF 페이지 번호 (1-based)")
    end_page: int = Field(..., ge=1, description="개념 설명 끝 PDF 페이지 번호 (1-based)")
    reason: str = Field(default="", description="개념 설명 구간으로 판별한 근거 요약")


class TocUnitItem(BaseModel):
    """개념서·교과서 목차(차례)에서 추출한 단원 항목."""

    unit_name: str = Field(..., description="목차에 기재된 대단원/중단원명")
    book_page_hint: str = Field(default="", description="목차에 적힌 책 쪽수 (예: '12', 'p.184~189')")


class TocExtractionOutput(BaseModel):
    """개념서·교과서 앞부분 목차 분석 결과."""

    last_toc_page: int = Field(
        default=0,
        ge=0,
        description="표지·교재 구성 안내·차례(목차)가 모두 끝나는 마지막 실제 PDF 페이지 번호 (1-based, 미확인 시 0)",
    )
    toc_units: list[TocUnitItem] = Field(
        default_factory=list,
        description="목차 순서대로 추출한 전체 대단원/중단원 목록",
    )


class AutoConceptPageDetectionOutput(BaseModel):
    """AI가 개념서 PDF 전체를 훑어 목차·정답·대단원 종합문제를 제외하고 개념 페이지만 추출한 결과."""

    toc_units: list[str] = Field(
        default_factory=list,
        description="목차(차례)에서 먼저 추출한 전체 단원명 기준 목록",
    )
    detected_ranges: list[AutoConceptRangeItem] = Field(
        default_factory=list,
        description="순수 개념/원리/이론/탐구/확인퀴즈가 수록된 PDF 페이지 구간 목록",
    )
    excluded_summary: str = Field(
        default="",
        description="제외된 구간 요약 (예: p.1~7 표지/목차, p.26~33 대단원 종합문제, p.180~210 정답·해설)",
    )
    failed_chunks: list[str] = Field(
        default_factory=list,
        description="재시도 후에도 판별에 실패한 페이지 구간 요약 목록 (예: ['p.21~p.40 (429 Rate Limit)'])",
    )


class ConceptBookRecord(BaseModel):
    """./concept_db 및 Google Drive 공용 폴더에 저장되는 개념서·교과서 DB 레코드."""

    id: str = Field(..., description="고유 개념서 ID (예: 완자_통합과학__고1_통합과학)")
    book_title: str = Field(..., description="교재명")
    subject: str = Field(..., description="과목 및 학년 표기")
    grade: str = Field(default="고1", description="학년 표기 (예: 고1)")
    page_ranges_spec: str = Field(default="", description="추출 시 입력한 개념 페이지 범위 (예: 12-25, 34-48)")
    units: list[ConceptUnit] = Field(default_factory=list, description="단원별 개념 목록")
    created_by: str = Field(default="", description="업로드한 사용자 아이디")
    created_at: str = Field(default="", description="생성 시각 (ISO 8601)")


class ConceptWhitelist(BaseModel):
    """업로드된 개념서·교과서 PDF 지정 페이지 또는 공용 개념서 DB에서 추출한 지식 화이트리스트."""

    subject_guess: str = Field(..., description="PDF 내용 기반 추정 과목명")
    pdf_summary: str = Field(..., description="지정 페이지 범위의 핵심 학습 내용 요약")
    pdf_concepts: list[str] = Field(
        ...,
        description="이 PDF에 실제로 설명되어 있는 개념·원리·법칙 목록",
    )
    pdf_vocabulary: list[str] = Field(
        ...,
        description="이 PDF에 실제로 등장하여 학생이 이해할 수 있는 허용 용어·기호·단위 목록",
    )
    book_title: str = Field(default="", description="참고 교재명 (예: 완자 통합과학)")
    grade: str = Field(default="", description="학년 정보 (예: 고1)")
    units: list[ConceptUnit] = Field(
        default_factory=list,
        description="단원별 개념 및 페이지 범위 메타데이터 목록",
    )


class QuestionMatchJudgment(BaseModel):
    """단일 기출 문항과 PDF 화이트리스트 간 엄격한 범위 일치 판정."""

    question_id: str = Field(..., description="평가 대상 문항 ID")
    is_solvable_by_pdf_only: bool = Field(
        ...,
        description="문항의 core_principles가 오직 pdf_concepts 범위만으로 완전히 해결 가능하면 True",
    )
    out_of_scope_terms: list[str] = Field(
        default_factory=list,
        description="required_terms 중 pdf_vocabulary에 없는 상위/범위 외 용어·단위 목록",
    )
    needs_rewriting: bool = Field(
        ...,
        description="원리는 PDF로 해결 가능(is_solvable_by_pdf_only=True)하지만 out_of_scope_terms가 존재하여 개조가 필요하면 True",
    )
    matching_reason: str = Field(..., description="매칭 또는 제외 사유 간략 설명")
    matched_unit_name: str = Field(
        default="",
        description="이 문항의 풀이 원리가 수록된 개념서 단원명 (예: Ⅲ. 변화와 다양성)",
    )
    matched_page_range: str = Field(
        default="",
        description="이 문항의 풀이 원리가 수록된 개념서 페이지 번호 (예: p.184~189)",
    )


class BatchMatchResult(BaseModel):
    """배치 문항 매칭 판정 결과."""

    judgments: list[QuestionMatchJudgment] = Field(default_factory=list)


class RewrittenQuestion(BaseModel):
    """업로드된 PDF 용어·개념만 사용해 개조한 변형 문제 카드 및 맞춤 해설."""

    rewritten_stem: str = Field(
        ...,
        description="업로드된 PDF의 개념과 허용 용어·일상어만 사용해 자연스럽게 개조한 문제 지문 및 <조건> (원본 논리·수치·정답 100% 유지)",
    )
    rewritten_choices: list[str] = Field(
        default_factory=list,
        description="객관식 5개 선지 텍스트 리스트 (선지 번호 ①~⑤ 제외하고 내용만, 주관식이면 빈 리스트)",
    )
    replaced_terms_map: list[str] = Field(
        default_factory=list,
        description="치환된 상위 용어 → PDF 기반 대체 표현 내역 (예: 'mol → 일정 개수 묶음(입자 수 비)')",
    )
    pdf_grounded_explanation: str = Field(
        ...,
        description="반드시 업로드된 PDF의 개념과 허용 용어만 사용해 작성한 단계별 맞춤 해설",
    )
    preserved_answer: str = Field(..., description="원본과 100% 동일하게 유지된 정답")
    self_verification_note: str = Field(
        ...,
        description="범위 외 용어 제거 및 정답/논리 일치 여부 자가 검증 결과",
    )
    visual_term_note: str = Field(
        default="",
        description="원본 그림·그래프·표 속에 범위 외 상위 용어/단위(예: mol, M)가 포함되어 있을 때 본문 허용 용어와 연결해 주는 안내 문구 (없으면 빈 문자열)",
    )
    target_question_type: str = Field(
        default="",
        description="영어 수능형 변형 목표 유형명 (예: '빈칸 추론 (구)', '어법상 틀린 것')",
    )
    subject_domain: str = Field(
        default="",
        description="개조/변형 도메인 구분 (''=개념서 기반 개조, '영어'=영어 수능형 지문 변형)",
    )


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


class EnglishQuestionExtractionItem(EnglishSingleExtractionOutput):
    """영어 모의고사 전체 PDF 일괄 추출 시 문항 번호가 포함된 항목."""

    q_num: int = Field(..., ge=1, le=45, description="문항 번호 (1~45)")


class BatchEnglishExamExtractionOutput(BaseModel):
    """영어 모의고사 PDF 전체 일괄 추출 출력 스키마."""

    questions: list[EnglishQuestionExtractionItem] = Field(default_factory=list)


class ExamHeaderMetadata(BaseModel):
    year: str = Field(..., description="시행 년도 (예: '2026')")
    month: str = Field(..., description="시행 월 (예: '3월', '6월', '9월', '11월')")
    grade: str = Field(..., description="학년 (예: '고1', '고2', '고3')")
    subject: str = Field(
        ...,
        description="시험지 헤더 기준 세부 과목명 (예: '국어', '수학', '영어', '한국사', '통합사회', '통합과학', '화학', '경제')",
    )
    exam_title: str = Field(
        ...,
        description="표준 회차명 (형식: '{year}년 {month} {grade} {subject}', 예: '2026년 9월 고1 통합사회')",
    )


class ConceptBookCoverMetadata(BaseModel):
    """개념서·교과서 PDF 표지/앞부분에서 자동 추출한 교재명·과목 및 학년."""

    book_title: str = Field(..., description="교재명 또는 교과서명 (예: '완자 통합과학', '개념원리 공통수학1', '비상 통합과학1')")
    subject_and_grade: str = Field(..., description="과목 및 학년 (예: '고1 통합과학', '고2 물리학Ⅰ', '고1 공통수학1')")


class ProblemAnswerPairItem(BaseModel):
    """문제지 PDF와 해설지 PDF 간 AI 자동 매칭 단일 항목."""

    problem_index: int = Field(..., ge=0, description="문제지 목록 인덱스 (P0 -> 0, P1 -> 1 ...)")
    answer_index: int | None = Field(default=None, description="대응하는 해설지 인덱스 (A0 -> 0, 없으면 None)")
    is_shared_combined_answer: bool = Field(
        default=False,
        description="여러 과목/시험지가 합본된 통합 해설지여서 복수 문제지에 공유되는 경우 True",
    )
    reason: str = Field(default="", description="매칭 또는 미매칭 근거 1줄 요약 (40자 이내)")


class BatchFilePairingOutput(BaseModel):
    """일괄 업로드된 문제지·해설지 PDF 전체의 1:1(또는 통합 해설지 공유) 자동 매칭 결과."""

    pairs: list[ProblemAnswerPairItem] = Field(default_factory=list)


class MixedPdfRoleItem(BaseModel):
    """혼합 업로드된 단일 PDF의 문제지/해설지 역할 판별 결과."""

    file_index: int = Field(..., ge=0, description="업로드된 PDF 인덱스 (F0 -> 0 ...)")
    role: Literal["problem", "answer"] = Field(..., description="문제지이면 'problem', 정답·해설지이면 'answer'")
    exam_summary: str = Field(default="", description="연도·월·학년·과목 간략 요약")


class MixedPdfClassificationOutput(BaseModel):
    """단일 업로더에 혼합 업로드된 PDF들의 역할 분류 결과."""

    items: list[MixedPdfRoleItem] = Field(default_factory=list)


def infer_subject_category(
    subject: str,
    exam_title: str = "",
    default: SubjectCategory = "과학",
) -> SubjectCategory:
    """세부 과목명 또는 시험지 제목을 대분류('수학' | '사회' | '과학' | '영어')로 매핑한다."""
    combined = " ".join(f"{subject or ''} {exam_title or ''}".split()).strip()
    if not combined:
        return default
    c_low = combined.lower()
    c_nospace = combined.replace(" ", "")

    # 1) 구체적인 과목명 최우선 매칭 (사회탐구 / 과학탐구 / 수학 / 영어)
    social_exact = (
        "세계사",
        "동아시아사",
        "한국지리",
        "세계지리",
        "생활과윤리",
        "윤리와사상",
        "경제",
        "정치와법",
        "사회·문화",
        "사회문화",
        "통합사회",
        "한국사",
    )
    if any(k in c_nospace for k in social_exact):
        return "사회"

    science_exact = ("물리학", "물리", "화학", "생명과학", "지구과학", "통합과학")
    if any(k in c_nospace for k in science_exact):
        return "과학"

    if any(k in c_nospace for k in ("수학", "미적분", "기하", "확률과통계", "확률", "통계", "대수")):
        return "수학"

    if "영어" in c_nospace or "english" in c_low:
        return "영어"

    # 2) 포괄 키워드 매칭
    if any(k in c_nospace for k in ("사회", "사탐", "역사", "지리", "윤리", "사상", "정치", "법", "문화")):
        return "사회"
    if any(k in c_nospace for k in ("과학", "과탐", "생명", "지구")):
        return "과학"

    return default


def normalize_exam_year(
    raw_year: str = "",
    *,
    text: str = "",
    filename: str = "",
    exam_title: str = "",
    month: str = "",
    grade: str = "",
    current_year: int | None = None,
) -> str:
    """PDF 첫 페이지 텍스트·파일명·AI 추출 회차명에서 실제 시행 연도를 표준화한다."""
    combined = " ".join(str(x).strip() for x in (text, exam_title, filename) if str(x or "").strip())
    cur_yr = int(current_year) if current_year is not None else datetime.now().year

    # 규칙 1: '(\d{4})\s*학년도' 표기가 있으면 무조건 year - 1 확정
    hak_m = re.search(r"(20\d{2})\s*학년도", combined)
    if hak_m:
        yr = int(hak_m.group(1)) - 1
        if yr > cur_yr:
            yr -= 1
        return str(yr)

    # 기본 연도 후보 추출
    base_year: int | None = None
    if raw_year and re.fullmatch(r"20\d{2}", str(raw_year).strip()):
        base_year = int(str(raw_year).strip())
    else:
        ym = re.search(r"(20\d{2})\s*년", combined) or re.search(r"(?:^|[\s_\-.(])(20\d{2})(?:$|[\s_\-.)])", combined)
        if ym:
            base_year = int(ym.group(1))

    if base_year is None:
        return ""

    # 월·학년 추출 (인자 우선, 없으면 combined에서 추출)
    eff_month = (month or "").strip()
    if not eff_month:
        mm = re.search(r"(?:^|[\s_\-.(])(1[0-2]|[1-9])\s*월", combined)
        if mm:
            eff_month = f"{int(mm.group(1))}월"
        elif "대학수학능력시험" in combined or "수능" in combined:
            eff_month = "11월"

    eff_grade = re.sub(r"\s+", "", (grade or "").strip())
    if not eff_grade:
        gm = re.search(r"(고\s*[123])", combined)
        if gm:
            eff_grade = gm.group(1).replace(" ", "")
        elif "대학수학능력시험" in combined or "모의평가" in combined or "수능" in combined:
            eff_grade = "고3"

    # 규칙 2: '학년도' 글자가 누락되었더라도 고3 + 6/9/11월 + 모의평가/수능 키워드면 year - 1
    is_hakpyeong = "전국연합학력평가" in combined or "학력평가" in combined
    is_kice = ("모의평가" in combined or "대학수학능력시험" in combined or "수능" in combined) and not is_hakpyeong
    if is_kice and eff_grade == "고3" and eff_month in ("6월", "9월", "11월"):
        yr = base_year - 1
    else:
        # 규칙 3: 'YYYY년' 또는 학력평가 표기인 경우 연도 그대로 사용
        yr = base_year

    # 규칙 4: 미래 연도 안전장치
    if yr > cur_yr:
        yr -= 1

    return str(yr)


def normalize_exam_title_year(
    exam_title: str,
    *,
    text: str = "",
    filename: str = "",
    current_year: int | None = None,
) -> str:
    """회차명 문자열 내 연도(예: 2027년 9월 고3...)를 실제 시행 연도로 보정한다."""
    title = re.sub(r"\s+", " ", (exam_title or "").strip())
    if not title:
        return ""
    ym = re.search(r"(20\d{2})\s*(학년도|년)", title)
    if not ym:
        return title
    raw_yr = ym.group(1)
    norm_yr = normalize_exam_year(
        raw_yr,
        text=text,
        filename=filename,
        exam_title=title,
        current_year=current_year,
    )
    if not norm_yr:
        return title
    if ym.group(2) == "학년도" or norm_yr != raw_yr:
        return f"{title[:ym.start()]}{norm_yr}년{title[ym.end():]}".strip()
    return title


TierType = Literal["FREE", "CUSTOM", "PRO", "ULTRA", "ADMIN"]
LEGACY_TIER_MAP: dict[str, str] = {
    "BYOK": "CUSTOM",
    "VIP": "PRO",
}


class UserRecord(BaseModel):
    """회원 계정 및 등급·개인 API 키·사용량 스키마 (users.json 저장용)."""

    username: str = Field(..., description="로그인 아이디 (학번 또는 영문·숫자 ID)")
    nickname: str = Field(..., description="표시 닉네임")
    password_hash: str = Field(..., description="salt$sha256 형식의 암호화 비밀번호")
    tier: TierType = Field(default="FREE", description="회원 등급 (FREE / CUSTOM / PRO / ULTRA / ADMIN)")
    custom_api_key: str = Field(default="", description="CUSTOM 회원의 개인 Gemini API 키")
    daily_count: int = Field(default=0, ge=0, description="금일 사용한 기출 추출 문항 수 (하위 호환)")
    last_used_date: str = Field(default="", description="마지막 추출 날짜 (YYYY-MM-DD, KST)")
    uploaded_exam_count: int = Field(default=0, ge=0, description="누적 업로드·태깅한 기출 회차 수")
    premium_unlock_credits: int = Field(
        default=0,
        ge=0,
        description="FREE 등급 기출 3회 업로드당 1회 해금되는 고급 모델(3.8/Hydrogen) 크레딧",
    )
    weekly_premium_count: int = Field(
        default=0,
        ge=0,
        description="PRO 등급 주간 고급 모델(3.8/Hydrogen) 사용 횟수",
    )
    last_used_week: str = Field(default="", description="마지막 주간 사용 주차 (YYYY-Www, KST)")
    created_at: str = Field(default="", description="가입 시각 (ISO 8601)")

    @field_validator("tier", mode="before")
    @classmethod
    def _normalize_legacy_tier(cls, value: Any) -> str:
        raw = str(value or "FREE").strip().upper()
        return LEGACY_TIER_MAP.get(raw, raw)


def build_whitelist_from_concept_units(
    book_title: str,
    subject: str,
    grade: str,
    selected_units: list[ConceptUnit],
) -> tuple[ConceptWhitelist, str]:
    """공용 개념서 DB에서 선택한 단원 목록을 기존 매칭·개조 엔진용 ConceptWhitelist와 보조 텍스트로 변환한다."""
    concepts: list[str] = []
    vocab: list[str] = []
    unit_summaries: list[str] = []
    context_blocks: list[str] = []

    for u in selected_units:
        for c in u.core_concepts:
            if c.strip() and c.strip() not in concepts:
                concepts.append(c.strip())
        for v in u.allowed_terms_and_units:
            if v.strip() and v.strip() not in vocab:
                vocab.append(v.strip())
        unit_summaries.append(f"{u.unit_name} ({u.page_range})")
        context_blocks.append(
            f"[{book_title} | {u.unit_name} ({u.page_range})]\n"
            f"- 핵심 개념: {', '.join(u.core_concepts)}\n"
            f"- 허용 용어·단위: {', '.join(u.allowed_terms_and_units)}"
        )

    summary_str = (
        f"{book_title} ({subject}) 선택 단원: " + ", ".join(unit_summaries)
        if unit_summaries
        else f"{book_title} ({subject})"
    )
    wl = ConceptWhitelist(
        subject_guess=subject,
        pdf_summary=summary_str,
        pdf_concepts=concepts,
        pdf_vocabulary=vocab,
        book_title=book_title,
        grade=grade,
        units=list(selected_units),
    )
    return wl, "\n\n".join(context_blocks)


def resolve_question_concept_reference(
    question: QuestionRecord,
    judgment: QuestionMatchJudgment | None,
    whitelist: ConceptWhitelist | None,
    fallback_source_label: str = "",
) -> str:
    """
    매칭·개조된 문항에 대해
    '📖 참고 개념: {교재명} - {단원명} ({페이지 범위})'
    (영어 과목인 경우 '📚 출처: {회차명} {문항번호}번') 형식의 출처 문자열을 결정한다.
    """
    if question.subject_category == "영어":
        if question.exam_name and "직접 입력" in question.exam_name:
            return f"📚 출처: {question.exam_name}"
        if question.exam_name and question.q_num:
            return f"📚 출처: {question.exam_name} {question.q_num}번"
        return f"📚 출처: {fallback_source_label or question.exam_name or '영어 기출 지문'}"

    book_title = (whitelist.book_title if whitelist and whitelist.book_title else "").strip()
    unit_name = (judgment.matched_unit_name if judgment and judgment.matched_unit_name else "").strip()
    page_range = (judgment.matched_page_range if judgment and judgment.matched_page_range else "").strip()

    units = whitelist.units if whitelist and whitelist.units else []

    if units:
        # 1) judgment의 matched_unit_name과 일치하거나 포함 관계인 단원 탐색
        matched_unit: ConceptUnit | None = None
        if unit_name:
            for u in units:
                if u.unit_name == unit_name or unit_name in u.unit_name or u.unit_name in unit_name:
                    matched_unit = u
                    break

        # 2) 없으면 문항의 core_principles / unit_name과 가장 많이 겹치는 단원 탐색
        if matched_unit is None:
            best_score = -1
            q_keywords = set(question.core_principles) | {question.unit_name}
            for u in units:
                score = 0
                u_blob = " ".join([u.unit_name] + u.core_concepts + u.allowed_terms_and_units)
                for kw in q_keywords:
                    for token in kw.split():
                        if len(token) >= 2 and token in u_blob:
                            score += 2
                if score > best_score:
                    best_score = score
                    matched_unit = u

        if matched_unit is None:
            matched_unit = units[0]

        unit_name = matched_unit.unit_name
        page_range = matched_unit.page_range

    if not book_title:
        book_title = fallback_source_label.strip() or (whitelist.subject_guess if whitelist else "개념서 PDF")

    if not unit_name:
        unit_name = (whitelist.subject_guess if whitelist and whitelist.subject_guess else question.unit_name) or "지정 개념 범위"

    if page_range:
        clean_p = page_range.strip()
        if not clean_p.lower().startswith("p"):
            clean_p = f"p.{clean_p}"
        return f"📖 참고 개념: {book_title} - {unit_name} ({clean_p})"

    return f"📖 참고 개념: {book_title} - {unit_name}"


