from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator

SubjectCategory = Literal["수학", "사회", "과학", "한국사", "국어", "영어"]

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


def normalize_korean_group_id(raw_group: str = "", q_num: int = 0) -> str:
    """국어 세트 지문 그룹 ID('[4~9]', '4-9', '4~9' 등)를 '4~9' 형식으로 정규화한다."""
    s = re.sub(r"[\[\]\s번]+", "", (raw_group or "").strip()).replace("-", "~")
    m = re.match(r"^(\d{1,2})~(\d{1,2})$", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a <= b:
            return f"{a}~{b}"
        return f"{b}~{a}"
    return str(int(q_num)) if q_num > 0 else ""

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
    """저장소에 영구 보관되는 단일 기출 문항 스키마 (2022 개정 교육과정 기준 및 영어·국어 전용 확장 포함)."""

    id: str = Field(..., description="고유 문항 ID (예: 2026_고3_6월_화학__Q05)")
    exam_name: str = Field(..., description="시험 회차명 (예: 2026_고3_6월_화학)")
    q_num: int = Field(..., ge=1, description="문항 번호")
    subject_category: Literal["수학", "사회", "과학", "한국사", "국어", "영어"] = Field(..., description="대분류 과목")
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
    passage_image_filename: str = Field(
        default="",
        description="세트 공통 지문 분리 크롭 PNG 파일명 (국어 등 지문·문제 분리 저장용)",
    )
    source_pdf: str = Field(default="", description="원본 문제지 PDF 파일명")
    answer_pdf: str = Field(default="", description="원본 해설지 PDF 파일명")
    created_at: str = Field(default="", description="생성 시각 (ISO 8601)")


class QuestionExtractionOutput(BaseModel):
    """Gemini가 크롭 이미지 + 해설지 PDF로부터 추출하는 구조화 메타데이터."""

    curriculum_subject: str = Field(
        ...,
        description="2022 개정 교육과정 세부 과목명 (예: 공통수학1, 공통수학2, 대수, 미적분1, 통합사회1, 통합사회2, 한국사1, 한국사2, 경제, 통합과학1, 통합과학2, 화학, 물리학 등)",
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
        description="개조/변형 도메인 구분 (''=개념서 기반 개조, '영어'=영어 수능형 지문 변형, '국어'=국어 지문 세트 재구성)",
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


class KoreanSingleExtractionOutput(BaseModel):
    """국어 기출 단일 문항 추출 스키마 (독서·문학 공통 영역 지문 세트 중심 추출)."""

    korean_section: Literal["독서", "문학", "화법과 작문", "언어와 매체"] = Field(
        ...,
        description="영역 구분 ('독서' | '문학' | '화법과 작문' | '언어와 매체')",
    )
    korean_subgenre: str = Field(
        default="",
        description="세부 갈래 (예: 독서론, 인문·예술, 사회·경제, 과학·기술, 주제 통합, 현대시, 고전시가, 현대소설, 고전소설, 극·수필, 갈래 복합)",
    )
    question_type: str = Field(
        default="",
        description="원래 문항 유형 (예: 세부 내용 일치, 핵심 논지 추론, <보기> 사례 적용, 서술상 특징, 표현상 특징, 문맥적 어휘 의미)",
    )
    passage_group_id: str = Field(
        default="",
        description="공통 지문 세트 범위 (예: '1~3', '4~9', '18~21', 단독 문항은 해당 번호 문자열)",
    )
    passage_text: str = Field(
        default="",
        description="발문과 선지를 제외한 순수 국어 지문 원문 ((가)/(나) 구분, [A]~[E], ㉠~㉤, ⓐ~ⓔ 표지 포함; 화작/언매는 빈 문자열)",
    )
    passage_topic: str = Field(
        default="",
        description="지문 핵심 주제 또는 작품명·작가 1줄 요약 (예: '박완서, 「나목」', '카르납의 논리실증주의')",
    )
    key_vocabulary: list[str] = Field(
        default_factory=list,
        description="지문 내 핵심 어휘·문학적 개념어 3~6개 목록",
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


class KoreanQuestionExtractionItem(KoreanSingleExtractionOutput):
    """국어 모의고사 전체 PDF 일괄 추출 시 문항 번호가 포함된 항목."""

    q_num: int = Field(..., ge=1, le=45, description="문항 번호 (1~45)")


class BatchKoreanExamExtractionOutput(BaseModel):
    """국어 모의고사 PDF 전체 일괄 추출 출력 스키마."""

    questions: list[KoreanQuestionExtractionItem] = Field(default_factory=list)


class KoreanSetQuestionOutput(BaseModel):
    """국어 지문 세트 전체 재구성 시 생성되는 개별 문항."""

    slot_index: int = Field(..., ge=1, description="세트 내 문항 순번 (1부터 시작)")
    question_type_label: str = Field(
        ...,
        description="새로 출제한 문항 유형 (예: 세부 내용 일치, 핵심 논지 추론, <보기> 적용, 표현·서술상 특징, 문맥 어휘)",
    )
    stem: str = Field(
        ...,
        description="발문 및 필요 시 <보기> 박스 포함 문항 본문 (공통 지문 본문은 포함하지 않음)",
    )
    choices: list[str] = Field(
        ...,
        description="①~⑤ 선지 번호를 제외한 5개 선지 텍스트 리스트",
    )
    answer: str = Field(..., description="정답 ('1'~'5' 단일 숫자 문자열)")
    explanation: str = Field(
        ...,
        description="지문 근거를 바탕으로 한 단계별 정답 근거 및 오답 선지 소거 해설",
    )


class KoreanSetReconstructionOutput(BaseModel):
    """국어 지문 1개 세트에 대한 전체 문항(3~6문항) 자유 유형 재구성 출력 스키마."""

    set_intro: str = Field(
        ...,
        description="세트 공통 발문 (예: '[1~4] 다음 글을 읽고 물음에 답하시오.')",
    )
    questions: list[KoreanSetQuestionOutput] = Field(default_factory=list)


class KoreanParagraphSummary(BaseModel):
    """국어 지문의 문단/연/장면별 핵심 요약."""

    label: str = Field(..., description="구분 라벨 (예: '1문단', '(가) 1연', '[A] 부분')")
    summary: str = Field(..., description="해당 문단/연의 핵심 내용 1~2문장 요약")
    key_point: str = Field(default="", description="핵심 논지·전개 기능 또는 표현 포인트 1줄")


class KoreanMarkerVocabItem(BaseModel):
    """지문 내 기호(㉠~㉤, ⓐ~ⓔ) 및 핵심 어휘/개념 해설."""

    term: str = Field(..., description="기호 또는 핵심 어휘 (예: '㉠', 'ⓐ 반증', '의제와 간주')")
    meaning: str = Field(..., description="문맥적 의미, 지시 대상 또는 개념 설명")


class KoreanPoetryLineNote(BaseModel):
    """시(현대시·고전시가 등) 원문에서 밑줄을 그어 해설할 핵심 시구·시어 항목."""

    quote: str = Field(..., description="원문에 실제로 등장하는 시구·시어 그대로 발췌 (1개 행 또는 구절)")
    explanation: str = Field(..., description="해당 밑줄 시구의 표현상 특징, 시적 의미 및 화자의 정서·태도 해설")


class KoreanPoetryMark(BaseModel):
    """시 원문 특정 행·시어에 직접 그리는 밑줄·물결선·네모·동그라미·세모 및 행간 주석."""

    line_index: int = Field(..., description="해당 시 작품 `lines` 리스트 내 1부터 시작하는 행 번호 (빈 줄 포함)")
    quote: str = Field(..., description="해당 행 원문에 실제로 존재하는 시어·시구 그대로 발췌")
    mark_type: Literal["underline", "wavy", "box", "circle", "triangle"] = Field(
        default="underline",
        description="표기 도형: underline(직선 밑줄), wavy(물결선), box(네모 박스), circle(동그라미), triangle(세모/시련·부정)",
    )
    color: Literal["red", "blue", "dark"] = Field(
        default="red",
        description="주석 색상: red(화자 정서·감각 이미지·부정/시련), blue(구조적 특징·긍정/희망/극복 의지), dark(기본 대상/지시어)",
    )
    position: Literal["above", "below"] = Field(
        default="below",
        description="행간 주석 배치 위치 (below=행 아래, above=행 위)",
    )
    note: str = Field(..., description="시어 위/아래 행간에 화살표로 연결할 짧은 핵심 해설 (25자 이내 권장)")


class KoreanPoetryMarginBracket(BaseModel):
    """시 좌·우 여백에 여러 행/연을 묶어 표시하는 구조 브라켓·수미상관·시상전환 주석."""

    start_line: int = Field(..., description="묶음 시작 행 번호 (1부터 시작)")
    end_line: int = Field(..., description="묶음 끝 행 번호 (1부터 시작)")
    side: Literal["left", "right"] = Field(
        default="right",
        description="여백 위치 (right=연/행 묶음·대구·시상전환 등, left=수미상관 큰 연결선 등)",
    )
    style: Literal["brace", "arc", "divider"] = Field(
        default="brace",
        description="선 스타일: brace(중괄호 } / { 묶음), arc(두 연을 잇는 큰 호 - 수미상관 등), divider(시상 전환 점선 구분선)",
    )
    color: Literal["red", "blue", "dark"] = Field(
        default="blue",
        description="여백 주석 색상 (blue/red/dark)",
    )
    note: str = Field(..., description="여백에 표시할 구조·시상 전개 핵심 주석 (예: '대구법 & 연쇄법', '수미상관 구조', '시상 전환')")


class KoreanPoetryWorkAnnotation(BaseModel):
    """(가)/(나) 등 개별 시 작품 1편의 행간·여백 주석형 원문 분석 데이터."""

    work_label: str = Field(default="", description="작품 구분 기호 (예: '(가)', '(나)', 단일 작품이면 '')")
    title_author: str = Field(..., description="작품명 및 작가 (예: '신동엽, 「산에 언덕에」')")
    lines: list[str] = Field(
        default_factory=list,
        description="원문 시의 각 행 리스트 (연과 연 사이는 빈 문자열 '' 포함). 반드시 원문 표기 그대로 수록.",
    )
    marks: list[KoreanPoetryMark] = Field(
        default_factory=list,
        description="시어·시구별 밑줄/물결/네모/동그라미/세모 및 행간 주석 목록 (8~16개 권장)",
    )
    brackets: list[KoreanPoetryMarginBracket] = Field(
        default_factory=list,
        description="좌·우 여백 묶음 괄호, 수미상관 연결선, 시상 전환 구분선 목록 (2~5개 권장)",
    )


class KoreanDiagramNode(BaseModel):
    """비문학 지문 이해를 돕는 도식 그림의 개별 개념/단계 노드."""

    title: str = Field(..., description="카드 제목 (예: '1단계: 압력 상승', '현명주의')")
    subtitle: str = Field(default="", description="부제 또는 역할 태그 (예: '원인', '핵심 입장')")
    details: list[str] = Field(
        default_factory=list,
        description="카드 안에 들어갈 1~3줄 핵심 설명 (각 줄 28자 이내 권장)",
    )
    accent_color: Literal["blue", "green", "amber", "purple", "slate"] = Field(
        default="blue",
        description="카드 강조 색상 (blue / green / amber / purple / slate)",
    )


class KoreanMatrixRow(BaseModel):
    """비교 매트릭스표(`matrix`)의 1개 비교 기준 행."""

    criterion: str = Field(..., description="비교 기준 항목명 (예: '통치 목적', '법의 적용', '핵심 수단')")
    values: list[str] = Field(
        default_factory=list,
        description="상단 비교 대상(nodes 순서)별 대응 내용 목록 (각 항목 28자 이내 권장)",
    )


class KoreanGraphCurve(BaseModel):
    """X-Y 함수/비례 그래프(`graph`)의 개별 곡선·직선 정보."""

    label: str = Field(..., description="곡선/변수 이름 (예: '수요의 가격탄력성', '평균비용')")
    trend: Literal["up", "down", "u_shape", "inv_u", "flat"] = Field(
        default="up",
        description="그래프 추세 (up=우상향 비례, down=우하향 반비례, u_shape=U자형, inv_u=역U자형, flat=일정)",
    )
    accent_color: Literal["blue", "green", "amber", "purple", "slate"] = Field(default="blue")
    annotation: str = Field(default="", description="곡선 특징 설명 (예: 'X 증가 시 Y 감소')")


class KoreanGraphSpec(BaseModel):
    """X-Y 좌표 그래프(`graph`) 명세."""

    x_label: str = Field(..., description="가로축(X축) 변수명 (예: '수요의 가격탄력성')")
    y_label: str = Field(..., description="세로축(Y축) 변수명 (예: '이용료 수준')")
    curves: list[KoreanGraphCurve] = Field(
        default_factory=list,
        description="좌표평면에 그릴 1~2개 추세선",
    )
    highlight_point: str = Field(
        default="",
        description="균형점·임계점·교차점 설명 (예: '임계점: 간접 네트워크 외부성 극대화')",
    )


class KoreanCharacterEdge(BaseModel):
    """소설 인물관계도(`character_map`)의 인물 간 관계선 정보."""

    source_idx: int = Field(default=0, ge=0, description="출발 인물 노드 인덱스 (0-based)")
    target_idx: int = Field(default=1, ge=0, description="도착 인물 노드 인덱스 (0-based)")
    relation: str = Field(default="", description="관계 설명 (예: '조력', '모함·대립', '부자 갈등')")
    polarity: Literal["positive", "negative", "neutral"] = Field(
        default="neutral",
        description="positive=조력/지지(+), negative=갈등/대립(-), neutral=일반 관계",
    )


class KoreanEmotionPoint(BaseModel):
    """시상 전개 정서 변화 그래프(`emotion_arc`)의 구간별 정서 좌표."""

    stage: str = Field(..., description="구간·시점 라벨 (예: '1연', '전반부', '과거')")
    emotion: str = Field(..., description="해당 구간 핵심 정서·태도 (예: '절망·상실', '극복 의지')")
    level: int = Field(
        default=0,
        ge=-2,
        le=2,
        description="정서 극성 수준 (-2=극심한 부정, -1=부정, 0=중립/전환, 1=긍정, 2=강한 긍정/극복)",
    )


class KoreanVisualDiagram(BaseModel):
    """비문학·문학 지문의 핵심 원리·관계·정서·구조를 시각화하는 16종 특화 도식 그림 명세."""

    diagram_type: Literal[
        "flow",
        "matrix",
        "graph",
        "anatomy",
        "cycle",
        "timeline",
        "venn",
        "scale",
        "fishbone",
        "quadrant",
        "decision_tree",
        "pyramid",
        "character_map",
        "emotion_arc",
        "poetic_contrast",
        "space_journey",
        "comparison",
        "hierarchy",
    ] = Field(
        default="flow",
        description=(
            "도식 유형: 비문학 12종(flow, matrix, graph, anatomy, cycle, timeline, venn, scale, "
            "fishbone, quadrant, decision_tree, pyramid) + 문학 4종(character_map=인물관계·갈등도, "
            "emotion_arc=정서변화 그래프, poetic_contrast=(+)/(-) 시어 대립도, space_journey=공간·시선 이동 경로도)"
        ),
    )
    title: str = Field(..., description="도식 상단 제목 (예: '양면시장 가격탄력성과 이용료 반비례 관계')")
    center_label: str = Field(
        default="",
        description="anatomy(중앙 장치명), cycle(순환 주제), venn(교집합), fishbone(최종 결과/현상), decision_tree(핵심 조건 질문) 라벨",
    )
    quadrant_x_label: str = Field(
        default="",
        description="diagram_type='quadrant'일 때 가로축(X축) 분류 기준 (예: '배제성 (좌: 불가 ↔ 우: 가능)')",
    )
    quadrant_y_label: str = Field(
        default="",
        description="diagram_type='quadrant'일 때 세로축(Y축) 분류 기준 (예: '경합성 (하: 낮음 ↔ 상: 높음)')",
    )
    nodes: list[KoreanDiagramNode] = Field(
        default_factory=list,
        description="도식을 구성하는 핵심 항목/대상/부위/단계 카드 2~4개",
    )
    relation_labels: list[str] = Field(
        default_factory=list,
        description="노드 사이 화살표/연결선에 표시할 짧은 관계 라벨 목록 (각 8자 이내)",
    )
    matrix_rows: list[KoreanMatrixRow] = Field(
        default_factory=list,
        description="diagram_type='matrix'일 때 사용하는 기준별 비교 행 리스트 (3~5행)",
    )
    graph_spec: KoreanGraphSpec | None = Field(
        default=None,
        description="diagram_type='graph'일 때 사용하는 X-Y 좌표 그래프 명세",
    )
    scale_tilt: Literal["balanced", "left", "right"] = Field(
        default="balanced",
        description="diagram_type='scale'일 때 저울 기울기 (balanced=균형/절충, left=좌측 우위, right=우측 우위)",
    )
    character_edges: list[KoreanCharacterEdge] = Field(
        default_factory=list,
        description="diagram_type='character_map'일 때 인물 간 조력/갈등 관계선 목록",
    )
    emotion_points: list[KoreanEmotionPoint] = Field(
        default_factory=list,
        description="diagram_type='emotion_arc'일 때 연·시간별 정서 변화 좌표 목록 (3~5개)",
    )
    takeaway: str = Field(
        default="",
        description="도식 하단에 표시할 1줄 핵심 정리 (시험 출제 포인트)",
    )


class KoreanPassageAnalysisOutput(BaseModel):
    """수능·학원형 국어 지문 정밀 분석지 구조화 출력 스키마."""

    title: str = Field(..., description="분석지 제목 (예: '법적 의제와 간주의 차이' 또는 '오세영, 「보석 1」 외')")
    core_theme: str = Field(..., description="지문 전체의 핵심 주제 · 중심 화제(또는 작품 주제 의식) 1~2문장")
    structure_flow: str = Field(
        ...,
        description="지문 전체 논리 전개 구조 또는 작품 시상 전개·서사 흐름 요약 (화살표 → 활용)",
    )
    paragraph_summaries: list[KoreanParagraphSummary] = Field(
        default_factory=list,
        description="문단별(독서) 또는 연·장면별(문학) 핵심 요약 리스트",
    )
    marker_and_vocab_notes: list[KoreanMarkerVocabItem] = Field(
        default_factory=list,
        description="지문 속에서 '~~는 ~~~~이다' 형태로 정의가 제시된 핵심 개념어 및 정의 정리 리스트 (기호 표지 제외)",
    )
    area_specific_notes: list[str] = Field(
        default_factory=list,
        description="영역별 핵심 정리 (독서: 개념 대조·인과관계 / 문학: 화자·정서·태도·표현상 특징 / 화작·언매: 말하기 전략·문법 원리)",
    )
    exam_checkpoints: list[str] = Field(
        default_factory=list,
        description="수능·내신 출제 예상 포인트 및 오답 함정 주의사항 3~5개",
    )
    poetry_underline_notes: list[KoreanPoetryLineNote] = Field(
        default_factory=list,
        description="시(현대시·고전시가 등) 갈래일 때 원문에서 밑줄을 그어 해설할 핵심 시구·시어 목록 (4~7개). 산문/비문학은 빈 리스트.",
    )
    poetry_works: list[KoreanPoetryWorkAnnotation] = Field(
        default_factory=list,
        description="시(현대시·고전시가 등) 갈래일 때 (가)/(나) 작품별로 분리한 행간·여백 주석형 원문 분석 데이터 리스트. 산문/비문학은 빈 리스트.",
    )
    visual_diagram: KoreanVisualDiagram | None = Field(
        default=None,
        description="비문학(독서) 등 지문 이해를 돕는 개념·원리·비교 도식 그림 데이터 (불필요한 경우 None)",
    )


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
    """세부 과목명 또는 시험지 제목을 대분류('수학' | '사회' | '과학' | '한국사' | '영어')로 매핑한다."""
    combined = " ".join(f"{subject or ''} {exam_title or ''}".split()).strip()
    if not combined:
        return default
    c_low = combined.lower()
    c_nospace = combined.replace(" ", "")

    # 1) 한국사 최우선 매칭 (독립 대분류)
    if any(k in c_nospace for k in ("한국사", "한국근현대사")):
        return "한국사"

    # 2) 국어 매칭 (독서·문학·화법과작문·언어와매체)
    if any(k in c_nospace for k in ("국어", "독서", "문학", "화법과작문", "언어와매체", "화작", "언매")):
        return "국어"

    # 2) 구체적인 과목명 매칭 (사회탐구 / 과학탐구 / 수학 / 영어)
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

    # 3) 포괄 키워드 매칭
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
) -> str:
    """PDF 첫 페이지 텍스트·파일명·AI 추출 회차명에서 실제 시행 연도를 표준화한다.

    규칙:
    1. 고3 모의고사 중 '대학수학능력시험' 글자가 검출되면 연도에 -1
    2. 혹은 고3이면서 6월 / 9월 / 수능일 경우 연도에 -1
    3. 나머지는 연도 그대로 유지
    """
    combined = " ".join(str(x).strip() for x in (text, exam_title, filename) if str(x or "").strip())
    c_nospace = combined.replace(" ", "")

    # 기본 연도 추출: 원문 text에 명시된 연도를 최우선 사용 (이미 -1 보정된 raw_year의 이중 감산 방지)
    base_year: int | None = None
    text_str = str(text or "").strip()
    if text_str:
        tm = re.search(r"(20\d{2})\s*(?:학년도|년)", text_str) or re.search(
            r"(?:^|[\s_\-.(])(20\d{2})(?:$|[\s_\-.)])", text_str
        )
        if tm:
            base_year = int(tm.group(1))
    if base_year is None and raw_year and re.fullmatch(r"20\d{2}", str(raw_year).strip()):
        base_year = int(str(raw_year).strip())
    if base_year is None:
        ym = re.search(r"(20\d{2})\s*(?:학년도|년)", combined) or re.search(
            r"(?:^|[\s_\-.(])(20\d{2})(?:$|[\s_\-.)])", combined
        )
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

    eff_grade = re.sub(r"\s+", "", (grade or "").strip())
    if not eff_grade:
        gm = re.search(r"(고\s*[123])", combined)
        if gm:
            eff_grade = gm.group(1).replace(" ", "")
        elif "대학수학능력시험" in c_nospace or "모의평가" in c_nospace or "수능" in c_nospace:
            eff_grade = "고3"

    has_csat_full = "대학수학능력시험" in c_nospace
    is_suneung = has_csat_full or "수능" in c_nospace
    is_g3_target = eff_grade == "고3" and (
        has_csat_full or eff_month in ("6월", "9월") or is_suneung
    )

    if is_g3_target:
        return str(base_year - 1)
    return str(base_year)


def normalize_exam_title_year(
    exam_title: str,
    *,
    text: str = "",
    filename: str = "",
    current_year: int | None = None,
) -> str:
    """회차명 문자열 내 연도(예: 2027학년도 / 미래 연도 2027년)를 실제 시행 연도로 보정한다."""
    title = re.sub(r"\s+", " ", (exam_title or "").strip())
    if not title:
        return ""
    ym = re.search(r"(20\d{2})\s*(학년도|년)", title)
    if not ym:
        return title
    raw_yr = ym.group(1)
    cur_yr = int(current_year) if current_year is not None else datetime.now().year
    # 이미 'YYYY년'으로 저장된 정상 과거/현재 연도는 재호출 시 중복 -1 되지 않도록 보존
    if ym.group(2) == "년" and int(raw_yr) <= cur_yr and not text:
        return title
    norm_yr = normalize_exam_year(
        raw_yr,
        text=text,
        filename=filename,
        exam_title=title,
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
    password_hash: str = Field(
        default="",
        description="PBKDF2 또는 레거시 salt$sha256 암호화 비밀번호 (구글 전용 가입자는 빈 문자열)",
    )
    google_sub: str = Field(default="", description="연동된 Google 계정 고유 식별자 (OIDC sub)")
    google_email: str = Field(default="", description="연동된 Google 이메일 주소")
    has_local_credentials: bool = Field(
        default=True,
        description="직접 등록한 아이디·비밀번호 보유 여부 (구글 전용 가입자는 False)",
    )
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
        description="하위 호환용 레거시 주간 고급 모델 사용 횟수",
    )
    last_used_week: str = Field(default="", description="마지막 주간 사용 주차 (YYYY-Www, KST)")

    # 실제 Gemini API 토큰 사용량 및 USD 예상 비용 (월간 KST YYYY-MM 자동 초기화 + 전체 누적)
    last_used_month: str = Field(default="", description="마지막 토큰 사용 월 (YYYY-MM, KST)")
    monthly_prompt_tokens: int = Field(default=0, ge=0, description="이번 달 일반 입력 토큰(캐시 제외)")
    monthly_cached_tokens: int = Field(default=0, ge=0, description="이번 달 캐시 입력 토큰")
    monthly_output_tokens: int = Field(default=0, ge=0, description="이번 달 출력+사고 토큰")
    monthly_total_tokens: int = Field(default=0, ge=0, description="이번 달 총 사용 토큰")
    monthly_api_calls: int = Field(default=0, ge=0, description="이번 달 API 호출 횟수")
    monthly_cost_usd: float = Field(default=0.0, ge=0.0, description="이번 달 전체 예상 API 비용 (USD)")
    monthly_server_cost_usd: float = Field(
        default=0.0, ge=0.0, description="이번 달 서버 API 키 부담 비용 (USD)"
    )

    total_prompt_tokens: int = Field(default=0, ge=0, description="누적 일반 입력 토큰(캐시 제외)")
    total_cached_tokens: int = Field(default=0, ge=0, description="누적 캐시 입력 토큰")
    total_output_tokens: int = Field(default=0, ge=0, description="누적 출력+사고 토큰")
    total_total_tokens: int = Field(default=0, ge=0, description="누적 총 사용 토큰")
    total_api_calls: int = Field(default=0, ge=0, description="누적 API 호출 횟수")
    total_cost_usd: float = Field(default=0.0, ge=0.0, description="누적 전체 예상 API 비용 (USD)")
    total_server_cost_usd: float = Field(
        default=0.0, ge=0.0, description="누적 서버 API 키 부담 비용 (USD)"
    )

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


# ---------------------------------------------------------------------------
# 탭 4: 수능·모의고사 영단어 사전 & 개인 단어장 스키마
# ---------------------------------------------------------------------------
def normalize_english_dict_key(raw: str) -> str:
    """영단어·숙어·한글 검색어를 사전 공용 캐시 키용으로 정규화한다."""
    s = (raw or "").strip()
    s = s.strip(" \t\r\n.,!?;:'\"()[]{}“”‘’")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


class EnglishDictSense(BaseModel):
    """영단어·숙어의 단일 품사·의미 항목."""

    pos: str = Field(..., description="품사 표기 (명사, 동사, 형용사, 부사, 전치사, 접속사, 숙어 등 한국어 표기)")
    meaning_ko: str = Field(..., description="해당 품사에서의 핵심 한국어 뜻 (쉼표 구분 2~4개 이내)")
    nuance_note: str = Field(default="", description="수능 독해에서의 문맥적 뉘앙스 또는 혼동 어휘 주의점")
    example_en: str = Field(default="", description="고교·수능 독해 수준의 자연스러운 영문 예문 1문장")
    example_ko: str = Field(default="", description="위 예문의 자연스러운 한국어 해석")


class EnglishDictEntry(BaseModel):
    """Gemini 기반 공용 영단어·숙어 사전 표제어 엔트리."""

    headword: str = Field(..., description="표제어 원형 (단어는 동사/명사 기본형, 숙어는 기본 형태)")
    query_normalized: str = Field(default="", description="정규화된 표제어 검색 키 (소문자·단일 공백)")
    phonetic: str = Field(default="", description="미국식 IPA 발음 기호 (예: [rɪˈzɪliəns])")
    cefr_level: str = Field(
        default="",
        description="수능 어휘 난이도 등급 ('필수 기초', '고1~2 핵심', '수능 1등급 필수', '고난도 심화' 중 택1)",
    )
    summary_ko: str = Field(..., description="표제어 핵심 의미 1줄 요약 (예: 회복탄력성, (충격 등에서) 빠르게 회복함)")
    senses: list[EnglishDictSense] = Field(
        default_factory=list,
        description="품사·의미별 상세 풀이 목록 (수능 빈출 의미 우선, 최대 4개)",
    )
    synonyms: list[str] = Field(
        default_factory=list,
        description="수능 빈출 패러프레이징 유의어 (형식: 'english_word (한국어 뜻)', 3~5개)",
    )
    antonyms: list[str] = Field(
        default_factory=list,
        description="대표 반의어 (형식: 'english_word (한국어 뜻)', 1~3개)",
    )
    derivatives: list[str] = Field(
        default_factory=list,
        description="파생어 목록 (형식: 'word (품사: 뜻)', 2~4개)",
    )
    collocations: list[str] = Field(
        default_factory=list,
        description="자주 쓰이는 연어·숙어 결합 표현 (형식: 'phrase (뜻)', 2~4개)",
    )
    etymology_tip: str = Field(default="", description="접두사·어근·어원 기반 암기 팁 1~2문장")
    csat_tip: str = Field(default="", description="수능·모의고사 빈칸/어휘 추론 함정 및 출제 포인트 1~2문장")
    updated_at: str = Field(default="", description="캐시 생성/갱신 시각 (ISO 8601)")


class UserVocabItem(BaseModel):
    """계정별 나만의 단어장에 저장되는 영단어·숙어 항목."""

    headword: str = Field(..., description="표제어")
    normalized_key: str = Field(..., description="중복 방지용 정규화 키")
    summary_ko: str = Field(..., description="핵심 뜻 요약")
    phonetic: str = Field(default="", description="IPA 발음 기호")
    cefr_level: str = Field(default="", description="수능 난이도 등급")
    pos_summary: str = Field(default="", description="주요 품사 요약 (예: 명사 · 형용사)")
    synonyms: list[str] = Field(default_factory=list, description="대표 유의어 목록")
    example_en: str = Field(default="", description="대표 영문 예문")
    example_ko: str = Field(default="", description="대표 예문 한국어 해석")
    source_exam: str = Field(default="", description="연동 출처 시험/메모 태그")
    memo: str = Field(default="", description="사용자 개인 암기 메모")
    mastered: bool = Field(default=False, description="암기 완료 체크 여부")
    wrong_count: int = Field(default=0, ge=0, description="퀴즈 오답 누적 횟수")
    added_at: str = Field(default="", description="추가 시각 (ISO 8601)")



