from __future__ import annotations

from typing import TYPE_CHECKING

from src.auth.accounts import _sync_user_state, get_user_by_username
from src.auth.tokens import (
    FREE_EXAMS_PER_UNLOCK,
    FREE_MAX_BATCH_QUESTIONS,
    FREE_MONTHLY_TOKEN_LIMIT,
    MONTHLY_TOKEN_LIMITS,
    PREMIUM_MODEL_IDS,
    TIER_LABELS,
    WEEKLY_PREMIUM_LIMITS,
    _storage_lock,
    get_kst_month_str,
    get_kst_today_str,
    get_kst_week_str,
    is_premium_model,
    normalize_username,
)
from src.config import FLASH_LITE_MODEL, resolve_admin_config

if TYPE_CHECKING:
    from src.models import UserRecord
    from src.storage import StorageManager

def is_server_api_key_user(user: UserRecord | None) -> bool:
    """해당 사용자가 서버 Gemini API 키를 사용하는지(CUSTOM 개인 키가 아닌지) 반환한다."""
    if user is None:
        return True
    return user.tier != "CUSTOM"


def resolve_active_gemini_api_key(
    user: UserRecord,
    server_api_key: str,
    model_name: str | None = None,
    force_server_key: bool = False,
) -> tuple[str, str]:
    """
    등급에 따라 실제 Gemini 호출용 API 키를 라우팅한다 (전 기능 Gemini 3.5 Flash-Lite 단일 모델).
    - CUSTOM 등급: 본인이 등록한 개인 API 키를 모든 작업에 사용 (서버 원가 0원)
    - FREE / PRO / ULTRA / ADMIN: 서버 공용 Gemini API 키 사용 (월간 토큰 한도 적용)
    """
    _ = model_name
    server_key = (server_api_key or "").strip()
    if user.tier == "CUSTOM":
        custom_key = (user.custom_api_key or "").strip()
        if custom_key:
            return custom_key, "개인 키 모드 (CUSTOM · 본인 API 키)"
        if force_server_key:
            return server_key, "서버 키 (업로드·태깅 공용)"
        return "", "개인 키 미등록 (사이드바에서 Gemini API 키를 등록해 주세요)"
    if force_server_key:
        return server_key, "서버 키 (업로드·태깅 공용)"
    return server_key, f"서버 키 ({TIER_LABELS.get(user.tier, user.tier)})"


def get_remaining_free_quota(user: UserRecord) -> int | None:
    """하위 호환용; 월간 토큰 한도는 get_remaining_monthly_tokens()를 사용한다."""
    return get_remaining_monthly_tokens(user)


def get_max_batch_questions(user: UserRecord | None) -> int | None:
    """등급별 1회 최대 생성·변형 문항 수 제한을 반환한다 (FREE=5문항, 유료·개인키·관리자=제한 없음 None)."""
    if user is None or user.tier == "FREE":
        return FREE_MAX_BATCH_QUESTIONS
    return None


def get_monthly_token_limit(user: UserRecord | None) -> int | None:
    """등급별 월간 Gemini 3.5 Flash-Lite 토큰 한도를 반환한다 (FREE=50만, PRO=300만, ULTRA=1,200만, CUSTOM/ADMIN=None)."""
    if user is None:
        return FREE_MONTHLY_TOKEN_LIMIT
    return MONTHLY_TOKEN_LIMITS.get(user.tier)


def get_used_monthly_tokens(user: UserRecord | None) -> int:
    """이번 달(KST YYYY-MM) 누적 사용 토큰 수를 반환한다."""
    if user is None:
        return 0
    current_month = get_kst_month_str()
    if getattr(user, "last_used_month", "") != current_month:
        return 0
    return max(0, int(getattr(user, "monthly_total_tokens", 0) or 0))


def get_remaining_monthly_tokens(user: UserRecord | None) -> int | None:
    """이번 달(KST) 잔여 토큰 수를 반환한다. 무제한(CUSTOM/ADMIN)은 None."""
    limit = get_monthly_token_limit(user)
    if limit is None:
        return None
    used = get_used_monthly_tokens(user)
    return max(0, limit - used)


def check_token_quota(
    storage: StorageManager,
    username: str,
) -> tuple[bool, int | None, int | None, str]:
    """
    API 호출 직전 사용자의 이번 달 토큰 한도 초과 여부를 확인한다.
    반환: (허용 여부, 잔여 토큰 또는 None, 월간 한도 또는 None, 차단/안내 메시지)
    """
    user = get_user_by_username(storage, username)
    if user is None:
        return False, 0, FREE_MONTHLY_TOKEN_LIMIT, "로그인 계정 정보를 확인할 수 없습니다. 다시 로그인해 주세요."

    limit = get_monthly_token_limit(user)
    if limit is None:
        return True, None, None, ""

    used = get_used_monthly_tokens(user)
    rem = max(0, limit - used)
    if rem <= 0:
        tier_lbl = TIER_LABELS.get(user.tier, user.tier)
        if user.tier == "FREE":
            upgrade_hint = "사이드바에서 개인 API 키(CUSTOM)를 등록하거나 PRO(월 300만)/ULTRA(월 1,200만) 등급으로 업그레이드해 주세요."
        elif user.tier == "PRO":
            upgrade_hint = "매월 1일(KST) 초기화되며, 더 넉넉한 한도가 필요하시면 ULTRA(월 1,200만 토큰 · 4배)로 업그레이드해 주세요."
        else:
            upgrade_hint = "매월 1일(KST) 자동 초기화됩니다. 급한 경우 관리자에게 월간 한도 초기화를 요청해 주세요."
        return (
            False,
            0,
            limit,
            f"🚫 이번 달 토큰 한도를 모두 사용했습니다 ({tier_lbl} · {used:,} / {limit:,} 토큰). {upgrade_hint}",
        )
    return True, rem, limit, ""


def record_token_usage(
    storage: StorageManager,
    username: str,
    prompt_tokens: int,
    cached_tokens: int,
    output_tokens: int,
    total_tokens: int,
    cost_usd: float,
    api_calls: int = 1,
    used_server_key: bool = True,
    sync_drive: bool = True,
) -> UserRecord | None:
    """실제 Gemini API 응답에서 수집된 토큰 수와 USD 예상 비용을 사용자 레코드(월간+누적)에 기록한다."""
    p_tok = max(0, int(prompt_tokens or 0))
    c_tok = max(0, int(cached_tokens or 0))
    o_tok = max(0, int(output_tokens or 0))
    t_tok = max(0, int(total_tokens or 0))
    if t_tok <= 0:
        t_tok = p_tok + c_tok + o_tok
    calls = max(0, int(api_calls or 0))
    usd = max(0.0, float(cost_usd or 0.0))
    if t_tok == 0 and calls == 0 and usd == 0.0:
        return None

    norm_id = normalize_username(username)
    if not norm_id:
        return None

    with _storage_lock(storage):
        users = storage.load_users()
        today = get_kst_today_str()
        admin_names = resolve_admin_config().admin_usernames
        target: UserRecord | None = None

        for u in users:
            if u.username == norm_id:
                _sync_user_state(u, today, admin_names)
                u.monthly_prompt_tokens = int(getattr(u, "monthly_prompt_tokens", 0) or 0) + p_tok
                u.monthly_cached_tokens = int(getattr(u, "monthly_cached_tokens", 0) or 0) + c_tok
                u.monthly_output_tokens = int(getattr(u, "monthly_output_tokens", 0) or 0) + o_tok
                u.monthly_total_tokens = int(getattr(u, "monthly_total_tokens", 0) or 0) + t_tok
                u.monthly_api_calls = int(getattr(u, "monthly_api_calls", 0) or 0) + calls
                u.monthly_cost_usd = round(float(getattr(u, "monthly_cost_usd", 0.0) or 0.0) + usd, 8)
                if used_server_key:
                    u.monthly_server_cost_usd = round(
                        float(getattr(u, "monthly_server_cost_usd", 0.0) or 0.0) + usd, 8
                    )

                u.total_prompt_tokens = int(getattr(u, "total_prompt_tokens", 0) or 0) + p_tok
                u.total_cached_tokens = int(getattr(u, "total_cached_tokens", 0) or 0) + c_tok
                u.total_output_tokens = int(getattr(u, "total_output_tokens", 0) or 0) + o_tok
                u.total_total_tokens = int(getattr(u, "total_total_tokens", 0) or 0) + t_tok
                u.total_api_calls = int(getattr(u, "total_api_calls", 0) or 0) + calls
                u.total_cost_usd = round(float(getattr(u, "total_cost_usd", 0.0) or 0.0) + usd, 8)
                if used_server_key:
                    u.total_server_cost_usd = round(
                        float(getattr(u, "total_server_cost_usd", 0.0) or 0.0) + usd, 8
                    )
                target = u
                break

        if target is not None:
            _ = sync_drive
            storage.save_users(users)
        return target


def reset_user_monthly_tokens(
    storage: StorageManager,
    target_usernames: list[str],
) -> tuple[int, str | None]:
    """관리자가 지정한 회원의 이번 달 토큰 사용량을 0으로 초기화한다 (누적 통계는 보존)."""
    targets = {normalize_username(u) for u in target_usernames if normalize_username(u)}
    if not targets:
        return 0, None

    current_month = get_kst_month_str()
    with _storage_lock(storage):
        users = storage.load_users()
        reset_cnt = 0
        for u in users:
            if u.username in targets:
                u.monthly_prompt_tokens = 0
                u.monthly_cached_tokens = 0
                u.monthly_output_tokens = 0
                u.monthly_total_tokens = 0
                u.monthly_api_calls = 0
                u.monthly_cost_usd = 0.0
                u.monthly_server_cost_usd = 0.0
                u.last_used_month = current_month
                reset_cnt += 1
        if reset_cnt > 0:
            err = storage.save_users(users)
            return reset_cnt, err
        return 0, None


def get_weekly_premium_limit(user: UserRecord | None) -> int | None:
    """하위 호환용 (고급 모델 폐지 완료)."""
    _ = user
    return None


def get_remaining_pro_weekly_quota(user: UserRecord) -> int | None:
    """하위 호환용 (고급 모델 폐지 완료)."""
    _ = user
    return None


def can_use_premium_features(user: UserRecord) -> bool:
    """문제 자동 개조 및 맞춤형 시험지 PDF 병합 허용 여부 (전 등급 허용, 월간 토큰 한도 적용)."""
    return user.tier in ("FREE", "CUSTOM", "PRO", "ULTRA", "ADMIN")


def can_access_shared_concept_db(user: UserRecord | None) -> bool:
    """공용 개념서·교과서 DB 구축(업로드) 및 단원별 불러오기 권한 여부 (PRO / ULTRA / ADMIN 전용)."""
    if user is None:
        return False
    return user.tier in ("PRO", "ULTRA", "ADMIN")


def resolve_model_for_user(user: UserRecord, requested_model: str) -> tuple[str, str]:
    """전 등급·전 기능을 Gemini 3.5 Flash-Lite 단일 모델로 라우팅한다."""
    _ = (user, requested_model)
    return FLASH_LITE_MODEL, ""


def record_exam_upload_for_user(
    storage: StorageManager,
    username: str,
) -> tuple[UserRecord | None, bool]:
    """기출 PDF 업로드·태깅 완료 시 회원의 누적 업로드 횟수를 +1 한다."""
    norm_id = normalize_username(username)
    with _storage_lock(storage):
        users = storage.load_users()
        today = get_kst_today_str()
        admin_names = resolve_admin_config().admin_usernames
        target: UserRecord | None = None

        for u in users:
            if u.username == norm_id:
                _sync_user_state(u, today, admin_names)
                cur_uploads = int(getattr(u, "uploaded_exam_count", 0) or 0) + 1
                if hasattr(u, "uploaded_exam_count"):
                    u.uploaded_exam_count = cur_uploads
                target = u
                break

        if target is not None:
            storage.save_users(users)
        return target, False


def consume_premium_model_usage(
    storage: StorageManager,
    username: str,
    used_model: str,
    count: int = 1,
) -> UserRecord | None:
    """하위 호환용 no-op (실제 사용량은 record_token_usage에서 토큰 단위로 자동 기록됨)."""
    _ = (storage, username, used_model, count)
    return None


def consume_extraction_quota(
    storage: StorageManager,
    username: str,
    extracted_count: int,
) -> UserRecord | None:
    """
    추출한 문항 수만큼 서버 사용량(daily_count)을 기록한다 (하위 호환용; 개편 후 FREE 한도 제한 없음).
    """
    norm_id = normalize_username(username)
    with _storage_lock(storage):
        users = storage.load_users()
        today = get_kst_today_str()
        admin_names = resolve_admin_config().admin_usernames
        target: UserRecord | None = None

        for u in users:
            if u.username == norm_id:
                _sync_user_state(u, today, admin_names)
                if u.tier != "CUSTOM":
                    u.daily_count += max(0, extracted_count)
                    u.last_used_date = today
                target = u
                break

        if target is not None:
            storage.save_users(users)
        return target
