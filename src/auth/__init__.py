"""인증 · 세션 · Google OAuth · 월간 토큰 쿼터 관리 패키지 (PEP 562 lazy exports)."""

from __future__ import annotations

import importlib
from typing import Any

_SUBMODULE_EXPORTS: dict[str, tuple[str, ...]] = {
    "src.auth.tokens": (
        "CREDIT_COST_CONCEPT_MATCH_ORIG",
        "CREDIT_COST_CONCEPT_REWRITE",
        "CREDIT_COST_DIRECT_CONCEPT_PDF",
        "CREDIT_COST_ENGLISH_ANALYSIS",
        "CREDIT_COST_ENGLISH_VARIANT",
        "CREDIT_COST_KOREAN_ANALYSIS",
        "CREDIT_COST_KOREAN_SET_TRANSFORM",
        "DAILY_FREE_AI_LIMITS",
        "DAILY_SHARED_UPLOAD_LIMITS",
        "FREE_EXAMS_PER_UNLOCK",
        "FREE_MAX_BATCH_QUESTIONS",
        "FREE_MONTHLY_CREDIT_LIMIT",
        "FREE_MONTHLY_TOKEN_LIMIT",
        "KST",
        "MAX_CHAT_TURNS_PER_QUESTION",
        "MAX_SHARED_UPLOAD_PDF_BYTES",
        "MONTHLY_CREDIT_LIMITS",
        "MONTHLY_TOKEN_LIMITS",
        "PBKDF2_ITERATIONS",
        "PREMIUM_MODEL_IDS",
        "PRO_MONTHLY_CREDIT_LIMIT",
        "PRO_MONTHLY_PRICE_KRW",
        "PRO_MONTHLY_TOKEN_LIMIT",
        "PRO_WEEKLY_PREMIUM_LIMIT",
        "SESSION_TOKEN_TTL_SECONDS",
        "TIER_LABELS",
        "ULTRA_MONTHLY_CREDIT_LIMIT",
        "ULTRA_MONTHLY_PRICE_KRW",
        "ULTRA_MONTHLY_TOKEN_LIMIT",
        "ULTRA_WEEKLY_PREMIUM_LIMIT",
        "USERNAME_RE",
        "VALID_TIERS",
        "WEEKLY_PREMIUM_LIMITS",
        "_PBKDF2_PREFIX",
        "_PROCESS_FALLBACK_SIGNING_KEY",
        "_b64url_decode",
        "_b64url_encode",
        "_get_session_signing_key",
        "_storage_lock",
        "create_session_token",
        "format_credit_count",
        "format_krw",
        "format_token_count",
        "get_kst_month_str",
        "get_kst_today_str",
        "get_kst_week_str",
        "hash_password",
        "is_legacy_password_hash",
        "is_premium_model",
        "mask_api_key",
        "normalize_username",
        "verify_password",
        "verify_session_token",
    ),
    "src.auth.oauth": (
        "GOOGLE_AUTH_URL",
        "GOOGLE_TOKEN_URL",
        "GOOGLE_USERINFO_URL",
        "OAUTH_STATE_TTL_SECONDS",
        "build_google_authorize_url",
        "create_google_oauth_state",
        "exchange_google_code_for_userinfo",
        "verify_google_oauth_state",
    ),
    "src.auth.accounts": (
        "_flush_ledger_events",
        "_sync_user_state",
        "adjust_user_bonus_credits",
        "authenticate_google_user",
        "authenticate_user",
        "batch_update_user_tiers",
        "clear_user_custom_api_key",
        "delete_users_by_usernames",
        "get_effective_tier",
        "get_user_by_google_sub",
        "get_user_by_username",
        "is_paid_tier_expired",
        "link_google_account",
        "maybe_sweep_expired_users",
        "parse_kst_iso",
        "register_google_user",
        "register_local_credentials_for_user",
        "register_user",
        "save_user_custom_api_key",
        "sync_all_users_state",
        "unlink_google_account",
    ),
    "src.auth.payments": (
        "PAYMENT_MAX_DAILY_REJECTIONS",
        "PAYMENT_MAX_DAILY_SUBMISSIONS",
        "PAYMENT_PLAN_SPECS",
        "PAYMENT_SUBMIT_COOLDOWN_SECONDS",
        "approve_payment_order",
        "cancel_my_pending_order",
        "check_payment_submission_rate_limit",
        "format_tier_expiry_badge",
        "get_order_consumed_base_credits",
        "refund_payment_order",
        "reject_payment_order",
        "send_discord_payment_notification",
        "submit_payment_order",
        "summarize_payment_revenue",
        "test_discord_webhook",
    ),
    "src.auth.quota": (
        "CreditReservation",
        "acquire_credit_reservation",
        "can_access_shared_concept_db",
        "can_admin_shared_db",
        "can_contribute_shared_db",
        "can_use_premium_features",
        "check_credit_quota",
        "check_daily_shared_upload_quota",
        "check_token_quota",
        "consume_daily_free_ai_quota",
        "consume_daily_shared_upload_quota",
        "consume_extraction_quota",
        "consume_premium_model_usage",
        "get_bonus_credits",
        "get_daily_free_ai_limit",
        "get_daily_free_ai_used",
        "get_daily_shared_upload_limit",
        "get_daily_shared_upload_used",
        "get_max_batch_questions",
        "get_monthly_credit_limit",
        "get_monthly_token_limit",
        "get_remaining_free_quota",
        "get_remaining_monthly_base_credits",
        "get_remaining_monthly_tokens",
        "get_remaining_pro_weekly_quota",
        "get_total_available_credits",
        "get_used_monthly_credits",
        "get_used_monthly_tokens",
        "get_weekly_premium_limit",
        "is_server_api_key_user",
        "record_exam_upload_for_user",
        "record_token_usage",
        "refund_daily_free_ai_quota",
        "reset_user_monthly_credits",
        "reset_user_monthly_tokens",
        "resolve_active_gemini_api_key",
        "resolve_model_for_user",
        "settle_credit_reservation",
    ),
}

_SYMBOL_TO_MODULE: dict[str, str] = {
    sym: mod for mod, syms in _SUBMODULE_EXPORTS.items() for sym in syms
}

__all__ = sorted(_SYMBOL_TO_MODULE.keys())


def __getattr__(name: str) -> Any:
    mod_name = _SYMBOL_TO_MODULE.get(name)
    if mod_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(mod_name)
    val = getattr(mod, name)
    globals()[name] = val
    return val


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
