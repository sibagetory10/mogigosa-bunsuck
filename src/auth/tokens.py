from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from src.config import (
    DEFAULT_USD_TO_KRW,
    FLASH_LITE_MODEL,
    _safe_get_secret,
)

if TYPE_CHECKING:
    from src.models import TierType
    from src.storage import StorageManager

def _storage_lock(storage: StorageManager | None):
    return getattr(storage, "_lock", None) or nullcontext()

KST = timezone(timedelta(hours=9))
FREE_MAX_BATCH_QUESTIONS = 5

# 월간 기본 크레딧 한도 (KST 매월 1일 초기화 · PRO 대비 ULTRA 정확히 4배)
FREE_MONTHLY_CREDIT_LIMIT = 30
PRO_MONTHLY_CREDIT_LIMIT = 600
ULTRA_MONTHLY_CREDIT_LIMIT = 2_400
MONTHLY_CREDIT_LIMITS: dict[str, int] = {
    "FREE": FREE_MONTHLY_CREDIT_LIMIT,
    "PRO": PRO_MONTHLY_CREDIT_LIMIT,
    "ULTRA": ULTRA_MONTHLY_CREDIT_LIMIT,
}

# 기능별 정액 크레딧 단가 (신규 생성 · 공용 캐시 풀 히트 동일 차감)
CREDIT_COST_ENGLISH_VARIANT = 2          # 영어 변형 1문항
CREDIT_COST_ENGLISH_ANALYSIS = 4         # 영어 정밀 지문 분석지 1지문
CREDIT_COST_KOREAN_SET_TRANSFORM = 10    # 국어 세트 변형 1세트
CREDIT_COST_KOREAN_ANALYSIS = 10         # 국어 심층 분석지 1세트
CREDIT_COST_CONCEPT_MATCH_ORIG = 1       # 수·사·과·한 무개조 원문 추출 1문항
CREDIT_COST_CONCEPT_REWRITE = 3          # 수·사·과·한 상위용어 개조 1문항
CREDIT_COST_DIRECT_CONCEPT_PDF = 3       # 탭 2 개인 개념서 PDF 직접 업로드 추출 1회

# 0C 무료 기능 타이트 어뷰징 방어 한도
MAX_CHAT_TURNS_PER_QUESTION = 5
MAX_SHARED_UPLOAD_PDF_BYTES = 50 * 1024 * 1024  # 50 MB
DAILY_FREE_AI_LIMITS: dict[str, int] = {
    "FREE": 10,
    "PRO": 40,
    "ULTRA": 100,
}
DAILY_SHARED_UPLOAD_LIMITS: dict[str, int] = {
    "FREE": 0,
    "PRO": 3,
    "ULTRA": 8,
}

# 하위 호환용 월간 토큰 상수 (관리자 원가 통계용)
FREE_MONTHLY_TOKEN_LIMIT = 500_000
PRO_MONTHLY_TOKEN_LIMIT = 3_000_000
ULTRA_MONTHLY_TOKEN_LIMIT = 12_000_000
MONTHLY_TOKEN_LIMITS: dict[str, int] = {
    "FREE": FREE_MONTHLY_TOKEN_LIMIT,
    "PRO": PRO_MONTHLY_TOKEN_LIMIT,
    "ULTRA": ULTRA_MONTHLY_TOKEN_LIMIT,
}

# 월 구독료 (관리자 대시보드 예상 매출 대비 API 원가율 계산용)
PRO_MONTHLY_PRICE_KRW = 9_900
ULTRA_MONTHLY_PRICE_KRW = 19_900

# 하위 호환용 상수 (고급 모델 폐지 완료)
PRO_WEEKLY_PREMIUM_LIMIT = 50
ULTRA_WEEKLY_PREMIUM_LIMIT = 200
WEEKLY_PREMIUM_LIMITS: dict[str, int] = {}
FREE_EXAMS_PER_UNLOCK = 3
PREMIUM_MODEL_IDS: tuple[str, ...] = ()

VALID_TIERS: tuple[TierType, ...] = ("FREE", "PRO", "ULTRA", "ADMIN")
TIER_LABELS: dict[TierType, str] = {
    "FREE": "무료 (0원)",
    "PRO": "PRO (9,900원/월)",
    "ULTRA": "ULTRA (19,900원/월)",
    "ADMIN": "최고 관리자",
}
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{3,30}$")


def get_kst_today_str() -> str:
    """한국 표준시(KST) 기준 오늘 날짜 YYYY-MM-DD를 반환한다."""
    return datetime.now(KST).strftime("%Y-%m-%d")


def get_kst_week_str() -> str:
    """한국 표준시(KST) 기준 ISO 주차 YYYY-Www를 반환한다 (매주 월요일 초기화)."""
    return datetime.now(KST).strftime("%G-W%V")


def get_kst_month_str() -> str:
    """한국 표준시(KST) 기준 현재 월 YYYY-MM을 반환한다 (매월 1일 초기화)."""
    return datetime.now(KST).strftime("%Y-%m")


def format_credit_count(credits: int | float) -> str:
    """크레딧 수를 읽기 쉬운 표기(예: 600C, 2,400C)로 변환한다."""
    n = max(0, int(round(credits or 0)))
    return f"{n:,}C"


def format_token_count(tokens: int | float) -> str:
    """토큰 수를 읽기 쉬운 한국어 축약 표기로 변환한다 (예: 3,000,000 -> 300만, 12,500 -> 12,500)."""
    n = max(0, int(round(tokens or 0)))
    if n >= 100_000 and n % 10_000 == 0:
        return f"{n // 10_000:,}만"
    return f"{n:,}"


def format_krw(cost_usd: float, usd_to_krw: float = DEFAULT_USD_TO_KRW) -> str:
    """USD 비용을 원화(₩) 문자열로 변환한다."""
    krw = max(0.0, float(cost_usd or 0.0)) * float(usd_to_krw or DEFAULT_USD_TO_KRW)
    if 0 < krw < 1.0:
        return f"₩{krw:.2f}"
    if krw < 100.0 and krw != int(krw):
        return f"₩{krw:,.1f}"
    return f"₩{round(krw):,}"


def is_premium_model(model_name: str | None) -> bool:
    """전 기능 Gemini 3.5 Flash-Lite 단일 모델 통일로 인해 고급 모델 구분을 사용하지 않는다."""
    _ = model_name
    return False


PBKDF2_ITERATIONS = 240_000
_PBKDF2_PREFIX = "pbkdf2_sha256"


def hash_password(password: str, salt: str | None = None) -> str:
    """PBKDF2-HMAC-SHA256(24만 회) + random salt로 비밀번호를 해싱한다. 형식: pbkdf2_sha256$iters$salt$hex"""
    actual_salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), actual_salt.encode("utf-8"), PBKDF2_ITERATIONS
    ).hex()
    return f"{_PBKDF2_PREFIX}${PBKDF2_ITERATIONS}${actual_salt}${digest}"


def is_legacy_password_hash(stored_hash: str) -> bool:
    return bool(stored_hash) and not stored_hash.startswith(f"{_PBKDF2_PREFIX}$")


def verify_password(password: str, stored_hash: str) -> bool:
    """PBKDF2 해시(및 기존 salt$sha256 레거시 해시)를 상수 시간 비교로 검증한다."""
    if not stored_hash or "$" not in stored_hash:
        return False
    if stored_hash.startswith(f"{_PBKDF2_PREFIX}$"):
        try:
            _, iters_s, salt, expected = stored_hash.split("$", 3)
            candidate = hashlib.pbkdf2_hmac(
                "sha256", password.encode("utf-8"), salt.encode("utf-8"), int(iters_s)
            ).hex()
        except (ValueError, TypeError):
            return False
        return hmac.compare_digest(candidate, expected)
    salt, expected_digest = stored_hash.split("$", 1)
    candidate_digest = hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()
    return hmac.compare_digest(candidate_digest, expected_digest)


def normalize_username(username: str) -> str:
    return (username or "").strip().lower()


def mask_api_key(api_key: str) -> str:
    """API 키 전체를 노출하지 않고 앞 6자리 + 마스킹 기호로 반환한다."""
    key = (api_key or "").strip()
    if not key:
        return ""
    if len(key) <= 6:
        return key[:2] + "••••••••"
    return f"{key[:6]}••••••••"


SESSION_TOKEN_TTL_SECONDS: int = 7 * 24 * 3600
_PROCESS_FALLBACK_SIGNING_KEY: bytes | None = None


def _get_session_signing_key(storage: StorageManager | None = None) -> bytes:
    """새로고침 이후에도 유지되는 서명용 비밀키를 결정론적으로 로드한다."""
    parts: list[str] = []
    for key_name in ("SESSION_SECRET", "ADMIN_PASSWORD", "GEMINI_API_KEY"):
        sec_val = _safe_get_secret([key_name, key_name.lower()])
        if isinstance(sec_val, str) and sec_val.strip():
            parts.append(f"{key_name}:{sec_val.strip()}")
        elif os.environ.get(key_name, "").strip():
            parts.append(f"{key_name}:{os.environ[key_name].strip()}")

    if parts:
        seed = "|".join(parts)
        return hashlib.sha256(f"mogigosa-session-v1:{seed}".encode("utf-8")).digest()

    if storage is not None:
        secret_file = storage.local_root / ".session_secret"
        try:
            if secret_file.exists():
                existing = secret_file.read_text(encoding="utf-8").strip()
                if existing:
                    return hashlib.sha256(existing.encode("utf-8")).digest()
            generated = secrets.token_hex(32)
            secret_file.write_text(generated, encoding="utf-8")
            return hashlib.sha256(generated.encode("utf-8")).digest()
        except Exception:
            pass

    # 비밀값이 전혀 설정되지 않은 환경: 코드에 박힌 고정 키 대신 프로세스별 무작위 키 사용 (재시작 시 세션만 만료)
    global _PROCESS_FALLBACK_SIGNING_KEY
    if _PROCESS_FALLBACK_SIGNING_KEY is None:
        _PROCESS_FALLBACK_SIGNING_KEY = secrets.token_bytes(32)
    return _PROCESS_FALLBACK_SIGNING_KEY


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def create_session_token(
    username: str,
    storage: StorageManager | None = None,
    ttl_seconds: int = SESSION_TOKEN_TTL_SECONDS,
) -> str:
    """브라우저 새로고침 시 로그인 상태를 안전하게 복원하기 위한 HMAC 서명 세션 토큰을 생성한다."""
    norm = normalize_username(username)
    payload = {"u": norm, "exp": int(time.time()) + max(60, int(ttl_seconds))}
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
    sig = hmac.new(
        _get_session_signing_key(storage),
        payload_b64.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return f"{payload_b64}.{_b64url_encode(sig)}"


def verify_session_token(token: str, storage: StorageManager | None = None) -> str | None:
    """HMAC 서명 세션 토큰을 검증하고 유효한 경우 정규화된 username을 반환한다."""
    raw = (token or "").strip()
    if "." not in raw:
        return None
    payload_b64, sig_b64 = raw.split(".", 1)
    if not payload_b64 or not sig_b64:
        return None

    expected_sig = hmac.new(
        _get_session_signing_key(storage),
        payload_b64.encode("ascii"),
        hashlib.sha256,
    ).digest()
    try:
        actual_sig = _b64url_decode(sig_b64)
        if not hmac.compare_digest(actual_sig, expected_sig):
            return None
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    username = normalize_username(str(payload.get("u", "")))
    exp = payload.get("exp")
    if not username or not isinstance(exp, (int, float)):
        return None
    if time.time() > float(exp):
        return None
    return username
