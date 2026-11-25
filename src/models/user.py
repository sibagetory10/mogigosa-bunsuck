from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

TierType = Literal["FREE", "PRO", "ULTRA", "ADMIN"]
LEGACY_TIER_MAP: dict[str, str] = {
    "BYOK": "FREE",
    "CUSTOM": "FREE",
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
    tier: TierType = Field(default="FREE", description="회원 등급 (FREE / PRO / ULTRA / ADMIN)")
    tier_expires_at: str = Field(
        default="",
        description="유료 결제 멤버십 만료 시각 ISO 8601 (KST, 빈 문자열이면 무기한 수동 승급 또는 비해당)",
    )
    custom_api_key: str = Field(
        default="",
        description="폐지된 레거시 필드 (마이그레이션 시 자동 공백 초기화)",
    )
    daily_count: int = Field(default=0, ge=0, description="금일 사용한 기출 추출 문항 수 (하위 호환)")
    last_used_date: str = Field(default="", description="마지막 사용 날짜 (YYYY-MM-DD, KST)")
    daily_free_ai_count: int = Field(
        default=0,
        ge=0,
        description="금일 사용한 0C 무료 AI 호출 횟수(해설+채팅+신규영단어, KST 자정 초기화)",
    )
    daily_shared_upload_count: int = Field(
        default=0,
        ge=0,
        description="금일 수행한 공용 DB 신규 업로드 횟수(탭1 기출+탭3 개념서, KST 자정 초기화)",
    )
    monthly_used_credits: int = Field(
        default=0,
        ge=0,
        description="이번 달 사용한 기본 크레딧 (매월 1일 KST 초기화)",
    )
    bonus_credits: int = Field(
        default=0,
        ge=0,
        description="월초 소멸되지 않는 관리자 지급 보너스 크레딧 잔액",
    )
    total_used_credits: int = Field(
        default=0,
        ge=0,
        description="누적 사용한 총 크레딧",
    )
    total_base_credits_used: int = Field(
        default=0,
        ge=0,
        description="보너스 크레딧과 분리된 순수 기본 크레딧 단조 증가 누적 사용량 (환불 역산 기준)",
    )
    refund_credit_debt: int = Field(
        default=0,
        ge=0,
        description="환불된 주문에서 초과 소비한 미정산 기본 크레딧 부채 (월초 비소멸, 재결제·익월 리셋 시 자동 상계)",
    )
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
