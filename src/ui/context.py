"""Shared application context passed across UI tabs and panels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.models import UserRecord
from src.storage import StorageManager


@dataclass
class AppContext:
    storage: StorageManager
    creds: Any
    google_login_cfg: Any
    current_user: UserRecord | None
    is_admin: bool
    selected_model: str
    active_gemini_key: str
    active_key_label: str
    upload_gemini_key: str
    upload_key_label: str
