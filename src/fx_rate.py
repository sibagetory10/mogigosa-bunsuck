from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from src.config import DEFAULT_USD_TO_KRW, resolve_usd_to_krw_rate

KST = timezone(timedelta(hours=9))
_FX_LOCK = threading.RLock()
_SCHEDULER_STARTED = False
_SCHEDULER_LOCK = threading.Lock()

# Compute Engine 공용 서비스 ID (Cloud Billing Catalog API에서 KRW currencyConversionRate 조회용)
_GCP_CATALOG_SERVICE_ID = "6F81-5844-456A"


@dataclass
class FxRateInfo:
    """Gemini API (Google Cloud Billing) 원화 환산에 적용되는 일일 자동 조회 환율 정보."""

    rate: float
    source: str
    checked_date: str  # YYYY-MM-DD (KST) — 매일 00:00 KST 갱신 판정 기준
    checked_at: str  # YYYY-MM-DD HH:MM:SS KST
    billing_month: str  # YYYY-MM (KST)
    reference_date: str = ""  # 환율 고시 기준일
    is_google_official: bool = False


def _now_kst() -> datetime:
    return datetime.now(KST)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _is_valid_krw_rate(val: Any) -> bool:
    try:
        f = float(val)
        return 500.0 <= f <= 5000.0
    except (ValueError, TypeError):
        return False


def _fetch_google_cloud_billing_krw_rate(api_keys: list[str], timeout_sec: int = 8) -> tuple[float, str] | None:
    """
    1순위: Google Cloud Billing Catalog API에서 실제 Google Cloud/Gemini API KRW 고시 환율
    (pricingInfo[].currencyConversionRate)을 직접 조회한다.
    """
    seen: set[str] = set()
    for raw_key in api_keys:
        key = (raw_key or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        try:
            resp = requests.get(
                f"https://cloudbilling.googleapis.com/v1/services/{_GCP_CATALOG_SERVICE_ID}/skus",
                params={
                    "currencyCode": "KRW",
                    "pageSize": 10,
                    "key": key,
                },
                timeout=timeout_sec,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            for sku in data.get("skus", []) or []:
                for pinfo in sku.get("pricingInfo", []) or []:
                    conv = pinfo.get("currencyConversionRate")
                    if _is_valid_krw_rate(conv):
                        eff_time = str(pinfo.get("effectiveTime") or "")[:10]
                        return round(float(conv), 4), eff_time
        except Exception:
            continue
    return None


def _fetch_frankfurter_monthly_krw_rate(billing_month: str, timeout_sec: int = 8) -> tuple[float, str, str] | None:
    """
    2순위: Google Cloud의 '매월 초 주요 금융기관 고시 환율 고정' 정책에 맞춰
    Frankfurter API(중앙은행 고시 환율)에서 해당 월 1일 기준 USD->KRW 환율을 조회한다.
    """
    month_start = f"{billing_month}-01"
    urls = [
        (f"https://api.frankfurter.dev/v1/{month_start}", f"금융기관 월초 고시 환율 (Frankfurter · {month_start})"),
        ("https://api.frankfurter.dev/v1/latest", "금융기관 최신 고시 환율 (Frankfurter)"),
    ]
    for url, label in urls:
        try:
            resp = requests.get(
                url,
                params={"base": "USD", "symbols": "KRW"},
                timeout=timeout_sec,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            rate = (data.get("rates") or {}).get("KRW")
            if _is_valid_krw_rate(rate):
                ref_date = str(data.get("date") or month_start)
                return round(float(rate), 2), ref_date, label
        except Exception:
            continue
    return None


def _fetch_open_er_api_krw_rate(timeout_sec: int = 8) -> tuple[float, str] | None:
    """3순위: ExchangeRate-API(open.er-api.com) 무료 공개 엔드포인트에서 USD->KRW 환율을 조회한다."""
    try:
        resp = requests.get("https://open.er-api.com/v6/latest/USD", timeout=timeout_sec)
        if resp.status_code != 200:
            return None
        data = resp.json()
        rate = (data.get("rates") or {}).get("KRW")
        if _is_valid_krw_rate(rate):
            ref_date = str(data.get("time_last_update_utc") or "")[:16]
            return round(float(rate), 2), ref_date
    except Exception:
        return None
    return None


def _load_cached_fx(cache_path: Path) -> FxRateInfo | None:
    if not cache_path.exists():
        return None
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not _is_valid_krw_rate(raw.get("rate")):
            return None
        return FxRateInfo(
            rate=float(raw["rate"]),
            source=str(raw.get("source") or "저장된 캐시"),
            checked_date=str(raw.get("checked_date") or ""),
            checked_at=str(raw.get("checked_at") or ""),
            billing_month=str(raw.get("billing_month") or ""),
            reference_date=str(raw.get("reference_date") or ""),
            is_google_official=bool(raw.get("is_google_official", False)),
        )
    except Exception:
        return None


def fetch_gemini_api_krw_rate(
    api_keys: list[str] | None = None,
    previous_cache: FxRateInfo | None = None,
) -> FxRateInfo:
    """
    Gemini API(Google Cloud Billing) 원화 환율을 우선순위에 따라 실시간 조회한다.
    1) Google Cloud Billing Catalog API (`currencyConversionRate`)
    2) Frankfurter API 매월 1일 기준 금융기관 고시 환율 (Google 월초 고정 정책 재현)
    3) ExchangeRate-API (`open.er-api.com`)
    4) 직전 캐시 / Secrets / 기본값
    """
    now = _now_kst()
    today_str = now.strftime("%Y-%m-%d")
    checked_at_str = now.strftime("%Y-%m-%d %H:%M:%S KST")
    billing_month = now.strftime("%Y-%m")

    # 1순위: Google Cloud Billing 공식 고시 환율
    gcp_res = _fetch_google_cloud_billing_krw_rate(api_keys or [])
    if gcp_res is not None:
        rate_val, eff_date = gcp_res
        return FxRateInfo(
            rate=rate_val,
            source="Google Cloud Billing 공식 고시 환율",
            checked_date=today_str,
            checked_at=checked_at_str,
            billing_month=billing_month,
            reference_date=eff_date or f"{billing_month}-01",
            is_google_official=True,
        )

    # 2순위: Frankfurter API (매월 초 고시 환율)
    ff_res = _fetch_frankfurter_monthly_krw_rate(billing_month)
    if ff_res is not None:
        rate_val, ref_date, label = ff_res
        return FxRateInfo(
            rate=rate_val,
            source=label,
            checked_date=today_str,
            checked_at=checked_at_str,
            billing_month=billing_month,
            reference_date=ref_date,
            is_google_official=False,
        )

    # 3순위: ExchangeRate-API
    er_res = _fetch_open_er_api_krw_rate()
    if er_res is not None:
        rate_val, ref_date = er_res
        return FxRateInfo(
            rate=rate_val,
            source="글로벌 고시 환율 (ExchangeRate-API)",
            checked_date=today_str,
            checked_at=checked_at_str,
            billing_month=billing_month,
            reference_date=ref_date,
            is_google_official=False,
        )

    # 4순위: 직전 캐시 유지 또는 Secrets/기본값 폴백
    if previous_cache is not None and _is_valid_krw_rate(previous_cache.rate):
        return FxRateInfo(
            rate=previous_cache.rate,
            source=f"{previous_cache.source} (오프라인 캐시 유지)",
            checked_date=today_str,
            checked_at=checked_at_str,
            billing_month=billing_month,
            reference_date=previous_cache.reference_date,
            is_google_official=previous_cache.is_google_official,
        )

    fallback_rate = resolve_usd_to_krw_rate() or DEFAULT_USD_TO_KRW
    return FxRateInfo(
        rate=float(fallback_rate),
        source="기본 설정 환율 (외부 조회 실패 폴백)",
        checked_date=today_str,
        checked_at=checked_at_str,
        billing_month=billing_month,
        reference_date=today_str,
        is_google_official=False,
    )


def get_daily_gemini_fx_rate(
    cache_dir: str | Path = "./exam_db",
    api_keys: list[str] | None = None,
    force_refresh: bool = False,
) -> FxRateInfo:
    """
    매일 00:00 KST 기준으로 하루 1회 자동 갱신되는 Gemini API USD->KRW 환율을 반환한다.
    오늘(KST YYYY-MM-DD) 이미 확인된 캐시가 있으면 네트워크 호출 없이 즉시 반환한다.
    """
    cache_path = Path(cache_dir) / "fx_rate_cache.json"
    today_str = _now_kst().strftime("%Y-%m-%d")

    with _FX_LOCK:
        cached = _load_cached_fx(cache_path)
        if not force_refresh and cached is not None and cached.checked_date == today_str:
            return cached

        fresh = fetch_gemini_api_krw_rate(api_keys=api_keys, previous_cache=cached)
        try:
            _atomic_write_json(cache_path, asdict(fresh))
        except Exception:
            pass
        return fresh


def seconds_until_next_kst_midnight(now_dt: datetime | None = None) -> float:
    """다음 한국 표준시(KST) 00:00:05까지 남은 초를 반환한다."""
    cur = now_dt.astimezone(KST) if now_dt is not None else _now_kst()
    next_midnight = (cur + timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
    return max(5.0, (next_midnight - cur).total_seconds())


def start_daily_midnight_fx_scheduler(
    cache_dir: str | Path = "./exam_db",
    api_keys: list[str] | None = None,
) -> bool:
    """
    매일 00:00 KST에 Gemini API 환율을 자동 갱신하는 단일 백그라운드 데몬 스레드를 시작한다.
    프로세스당 1회만 기동된다.
    """
    global _SCHEDULER_STARTED
    with _SCHEDULER_LOCK:
        if _SCHEDULER_STARTED:
            return False
        _SCHEDULER_STARTED = True

    keys_snapshot = list(api_keys or [])
    target_dir = Path(cache_dir)

    def _loop() -> None:
        # 기동 직후 오늘(KST) 00시 이후 갱신 이력이 없으면 즉시 1회 확인
        try:
            get_daily_gemini_fx_rate(cache_dir=target_dir, api_keys=keys_snapshot, force_refresh=False)
        except Exception:
            pass

        while True:
            wait_sec = seconds_until_next_kst_midnight()
            # 장시간 sleep 중 스레드 종료 반응성을 위해 최대 3600초 단위로 분할 대기
            remaining = wait_sec
            while remaining > 0:
                chunk = min(3600.0, remaining)
                threading.Event().wait(chunk)
                remaining -= chunk
            try:
                get_daily_gemini_fx_rate(cache_dir=target_dir, api_keys=keys_snapshot, force_refresh=True)
            except Exception:
                pass

    t = threading.Thread(target=_loop, name="gemini-fx-midnight-kst", daemon=True)
    t.start()
    return True
