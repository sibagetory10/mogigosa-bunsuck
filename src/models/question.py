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
