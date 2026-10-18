"""Dependency-free event enum (importable without SQLAlchemy)."""
from __future__ import annotations

import enum


class EventType(str, enum.Enum):
    """Kind of intervention issued by the policy engine."""

    SINGLE_STUDENT = "SINGLE_STUDENT"
    GROUP = "GROUP"
    GENERAL = "GENERAL"
