from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

PaymentPlanType = Literal["PRO", "ULTRA"]
PaymentOrderStatus = Literal["PENDING", "APPROVED", "REJECTED", "CANCELLED", "REFUNDED"]


class PaymentSettings(BaseModel):
    """관리자가 설정하는 무통장입금·간편송금 안내 및 Discord 실시간 알림 설정."""

    enabled: bool = Field(default=True, description="입금 신청 접수 활성화 여부")
    bank_name: str = Field(default="", description="입금 은행명")
    account_number: str = Field(default="", description="입금 계좌번호")
    account_holder: str = Field(default="", description="예금주 성명")
    kakaopay_link: str = Field(default="", description="카카오페이 송금 링크 (https://)")
    toss_link: str = Field(default="", description="토스 송금 링크 (https://)")
    guide_notice: str = Field(
        default="송금 완료 후 아래 [입금 확인 요청 보내기]를 눌러주시면 관리자 확인 즉시 멤버십이 활성화됩니다.",
        description="결제 모달 안내 문구",
    )
    discord_webhook_url: str = Field(default="", description="관리자 입금 신청 실시간 알림 Discord Webhook URL")
    updated_at: str = Field(default="", description="마지막 설정 변경 시각 (ISO 8601 KST)")

    @field_validator("kakaopay_link", "toss_link", mode="before")
    @classmethod
    def _validate_https_link(cls, value: object) -> str:
        s = str(value or "").strip()
        if s and not s.startswith("https://"):
            raise ValueError("송금 링크는 보안을 위해 'https://'로 시작해야 합니다.")
        return s

    @field_validator("discord_webhook_url", mode="before")
    @classmethod
    def _validate_discord_webhook(cls, value: object) -> str:
        s = str(value or "").strip()
        if s and not (
            s.startswith("https://discord.com/api/webhooks/")
            or s.startswith("https://discordapp.com/api/webhooks/")
        ):
            raise ValueError(
                "Discord 웹훅 URL은 'https://discord.com/api/webhooks/...' 형식이어야 합니다."
            )
        return s

    @property
    def has_transfer_method(self) -> bool:
        return bool(
            (self.bank_name.strip() and self.account_number.strip())
            or self.kakaopay_link.strip()
            or self.toss_link.strip()
        )


class PaymentOrder(BaseModel):
    """회원의 요금제 입금 확인 요청 및 관리자 승인·환불 스냅샷 스키마."""

    order_id: str = Field(..., description="고유 주문 번호 (예: PAY-20261125-A1B2C3)")
    username: str = Field(..., description="신청 회원 로그인 아이디 (소문자 정규화)")
    user_nickname: str = Field(default="", description="신청 당시 회원 닉네임")
    plan_tier: PaymentPlanType = Field(..., description="신청 요금제 (PRO / ULTRA)")
    duration_days: int = Field(default=30, ge=1, description="이용 기간 (일)")
    amount_krw: int = Field(..., ge=0, description="결제 금액 (원)")
    depositor_name: str = Field(..., description="실제 입금자명")
    user_note: str = Field(default="", description="회원 메모/연락처 (선택)")
    status: PaymentOrderStatus = Field(default="PENDING", description="주문 처리 상태")
    admin_memo: str = Field(default="", description="관리자 처리/반려/환불 메모")
    requested_at: str = Field(default="", description="입금 확인 신청 시각 (ISO 8601 KST)")
    processed_at: str = Field(default="", description="승인·반려·취소 처리 시각 (ISO 8601 KST)")
    processed_by: str = Field(default="", description="처리자 아이디")
    granted_expires_at: str = Field(default="", description="승인으로 부여된 멤버십 만료 시각 (ISO 8601 KST)")
    refunded_at: str = Field(default="", description="환불·승인취소 처리 시각 (ISO 8601 KST)")
    refunded_by: str = Field(default="", description="환불 처리 관리자 아이디")

    # 오입금·환불 시 무손실 원상 복구 및 크레딧 더블 수령 방지용 승인 시점 스냅샷
    prev_tier: str = Field(default="", description="승인 직전 회원 등급")
    prev_tier_expires_at: str = Field(default="", description="승인 직전 만료 시각")
    prev_monthly_used_credits: int = Field(default=0, ge=0, description="승인 직전 당월 기본 크레딧 사용량")
    prev_last_used_month: str = Field(default="", description="승인 직전 last_used_month (YYYY-MM)")
    base_used_at_approval: int = Field(default=0, ge=0, description="승인 시점 total_base_credits_used 스냅샷")
    applied_refund_debt: int = Field(default=0, ge=0, description="이 주문 승인 시 자동 상계된 기존 환불 부채")
