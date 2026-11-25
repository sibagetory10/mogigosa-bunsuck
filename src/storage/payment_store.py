from __future__ import annotations

import json
from typing import TYPE_CHECKING

from src.storage.base import (
    _GLOBAL_CACHE_LOCK,
    _GLOBAL_JSON_CACHE,
    _atomic_write_text,
    _file_stat_sig,
)

if TYPE_CHECKING:
    from src.models.payment import PaymentOrder, PaymentSettings


class PaymentStoreMixin:
    """입금 계좌 설정(payment_settings.json) 및 결제 신청 내역(payment_orders.json) 영구 저장소 믹스인."""

    def load_payment_settings(self) -> PaymentSettings:
        from src.models.payment import PaymentSettings

        with self._lock:
            if not self.payment_settings_path.exists():
                return PaymentSettings()
            sig = _file_stat_sig(self.payment_settings_path)
            if sig is None or sig[1] == 0:
                return PaymentSettings()
            cache_key = f"payment_settings::{self.payment_settings_path.resolve()}"
            with _GLOBAL_CACHE_LOCK:
                cached = _GLOBAL_JSON_CACHE.get(cache_key)
                if cached is not None and cached[0] == sig:
                    return cached[1].model_copy(deep=True)
            try:
                raw = json.loads(self.payment_settings_path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    return PaymentSettings()
                settings = PaymentSettings.model_validate(raw)
                with _GLOBAL_CACHE_LOCK:
                    _GLOBAL_JSON_CACHE[cache_key] = (sig, settings)
                return settings.model_copy(deep=True)
            except Exception:
                return PaymentSettings()

    def save_payment_settings(self, settings: PaymentSettings) -> None:
        with self._lock:
            raw_text = json.dumps(settings.model_dump(), ensure_ascii=False, indent=2)
            _atomic_write_text(self.payment_settings_path, raw_text, encoding="utf-8")
            new_sig = _file_stat_sig(self.payment_settings_path)
            if new_sig is not None:
                with _GLOBAL_CACHE_LOCK:
                    _GLOBAL_JSON_CACHE[f"payment_settings::{self.payment_settings_path.resolve()}"] = (
                        new_sig,
                        settings.model_copy(deep=True),
                    )

    def load_payment_orders(self) -> list[PaymentOrder]:
        from src.models.payment import PaymentOrder

        with self._lock:
            if not self.payment_orders_path.exists():
                return []
            sig = _file_stat_sig(self.payment_orders_path)
            if sig is None or sig[1] == 0:
                return []
            cache_key = f"payment_orders::{self.payment_orders_path.resolve()}"
            with _GLOBAL_CACHE_LOCK:
                cached = _GLOBAL_JSON_CACHE.get(cache_key)
                if cached is not None and cached[0] == sig:
                    return [o.model_copy(deep=True) for o in cached[1]]
            try:
                raw = json.loads(self.payment_orders_path.read_text(encoding="utf-8"))
                if not isinstance(raw, list):
                    return []
                orders = [PaymentOrder.model_validate(item) for item in raw if isinstance(item, dict)]
                with _GLOBAL_CACHE_LOCK:
                    _GLOBAL_JSON_CACHE[cache_key] = (sig, orders)
                return [o.model_copy(deep=True) for o in orders]
            except Exception:
                return []

    def save_payment_orders(self, orders: list[PaymentOrder]) -> None:
        with self._lock:
            payload = [o.model_dump() for o in orders]
            raw_text = json.dumps(payload, ensure_ascii=False, indent=2)
            _atomic_write_text(self.payment_orders_path, raw_text, encoding="utf-8")
            new_sig = _file_stat_sig(self.payment_orders_path)
            if new_sig is not None:
                with _GLOBAL_CACHE_LOCK:
                    _GLOBAL_JSON_CACHE[f"payment_orders::{self.payment_orders_path.resolve()}"] = (
                        new_sig,
                        [o.model_copy(deep=True) for o in orders],
                    )

    def count_pending_payment_orders(self) -> int:
        orders = self.load_payment_orders()
        return sum(1 for o in orders if o.status == "PENDING")
