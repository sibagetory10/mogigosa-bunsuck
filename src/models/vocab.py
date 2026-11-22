from __future__ import annotations

import re

from pydantic import BaseModel, Field

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



