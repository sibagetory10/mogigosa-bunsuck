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
CURRENT_ENGLISH_ANALYSIS_VERSION: int = 4


class EnglishSentenceAnalysisItem(BaseModel):
    """영어 지문 문장별 끊어읽기·괄호 구문독해 및 해석·어법 포인트."""

    sentence_num: int = Field(..., ge=1, description="문장 번호 (1부터 시작)")
    is_topic_sentence: bool = Field(
        default=False,
        description="이 문장이 지문의 핵심 주제문(Topic Sentence)이면 True (단순 전환·연결 단문에는 False)",
    )
    chunked_english: str = Field(
        ...,
        description="원문 영문 단어를 100% 그대로 보존한 상태에서 슬래시(/) 끊어읽기와 [절]·(수식어구) 괄호 기호를 삽입한 구문독해용 영문",
    )
    korean_translation: str = Field(
        ...,
        description="구문 기호(/, [], ())나 직독직해 끊어쓰기 없이 한국어 어순에 맞게 작성한 자연스러운 일반 한국어 해석 문장",
    )
    syntax_notes: list[str] = Field(
        default_factory=list,
        description="해당 문장의 핵심 구문·어법 구조 해설 (현재 빈 배열 [] 유지)",
    )
    original_blank_phrase: str = Field(
        default="",
        description="원래 빈칸 추론 문항에서 빈칸이었던 원문 구절 (이 문장에 빈칸이 있었을 때만 기재)",
    )
    is_inserted_sentence: bool = Field(
        default=False,
        description="원래 문장 삽입 문항에서 정답 위치에 삽입된 주어진 문장이면 True",
    )


class EnglishPassageAnalysisOutput(BaseModel):
    """내신·수능형 영어 지문 정밀 분석지 구조화 출력 스키마 (주제 · 논리 흐름 · 시각 도식 · 문장별 구문 분석)."""

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
        description="지문 전체 문장에 대한 1번부터 끝번까지의 문장별 정밀 구문독해 리스트",
    )
    exam_point_title: str = Field(
        default="",
        description="문항 유형에 맞는 기출 포인트 박스 제목 (예: '기출 포인트 (원문 빈칸)', '기출 포인트 (글의 순서)')",
    )
    exam_point_note: str = Field(
        default="",
        description="원래 기출 유형 핵심 포인트 (빈칸 추론의 원래 빈칸 위치·정답 구절, 무관한 문장의 제거된 정답 문장, 글의 순서 정답 배열, 문장 삽입 원문·정답 위치 등)",
    )
    footnotes: list[str] = Field(
        default_factory=list,
        description="문제지 공식 각주 어휘 목록 (예: ['delineate: ~의 경계를 정하다'])",
    )
    is_source_restored: bool = Field(
        default=False,
        description="빈칸·어법·어휘·삽입·순서 등 기출 변형 원문 복원 및 교정 완료 여부",
    )
    analysis_version: int = Field(
        default=1,
        description="분석지 엔진 스키마·검증 버전 (v4: 빈칸 덮어쓰기 차단·하이픈 보존·각주/번호/괄호 정밀 검증·PDF 누락 zero 가드)",
    )
