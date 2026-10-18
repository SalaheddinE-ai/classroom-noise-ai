"""Shared FastAPI dependencies."""
from __future__ import annotations

from functools import lru_cache

from engine.runner import MonitorService


@lru_cache(maxsize=1)
def get_service() -> MonitorService:
    """Process-wide MonitorService singleton."""
    return MonitorService()
