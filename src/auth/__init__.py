"""인증 · 세션 · Google OAuth · 월간 토큰 쿼터 관리 패키지 (PEP 562 lazy exports)."""

from __future__ import annotations

import importlib
from typing import Any

_SUBMODULE_EXPORTS: dict[str, tuple[str, ...]] = {
    "src.auth.tokens": (
        "FREE_EXAMS_PER_UNLOCK",
        "FREE_MAX_BATCH_QUESTIONS",
        "FREE_MONTHLY_TOKEN_LIMIT",
        "KST",
        "MONTHLY_TOKEN_LIMITS",
        "PBKDF2_ITERATIONS",
        "PREMIUM_MODEL_IDS",
        "PRO_MONTHLY_PRICE_KRW",
        "PRO_MONTHLY_TOKEN_LIMIT",
        "PRO_WEEKLY_PREMIUM_LIMIT",
        "SESSION_TOKEN_TTL_SECONDS",
        "TIER_LABELS",
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
        "_sync_user_state",
        "authenticate_google_user",
        "authenticate_user",
        "batch_update_user_tiers",
        "clear_user_custom_api_key",
        "delete_users_by_usernames",
        "get_user_by_google_sub",
        "get_user_by_username",
        "link_google_account",
        "register_google_user",
        "register_local_credentials_for_user",
        "register_user",
        "save_user_custom_api_key",
        "unlink_google_account",
    ),
    "src.auth.quota": (
        "can_access_shared_concept_db",
        "can_use_premium_features",
        "check_token_quota",
        "consume_extraction_quota",
        "consume_premium_model_usage",
        "get_max_batch_questions",
        "get_monthly_token_limit",
        "get_remaining_free_quota",
        "get_remaining_monthly_tokens",
        "get_remaining_pro_weekly_quota",
        "get_used_monthly_tokens",
        "get_weekly_premium_limit",
        "is_server_api_key_user",
        "record_exam_upload_for_user",
        "record_token_usage",
        "reset_user_monthly_tokens",
        "resolve_active_gemini_api_key",
        "resolve_model_for_user",
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
