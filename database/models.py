"""SQLAlchemy ORM models: students, voice profiles and intervention logs."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

import numpy as np
from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from database.event_types import EventType  # noqa: F401  (re-exported for backward compatibility)


class Base(DeclarativeBase):
    """Declarative base class."""


def _uuid() -> str:
    return str(uuid.uuid4())


class Student(Base):
    """A registered student."""

    __tablename__ = "students"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    age: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    student_code: Mapped[Optional[str]] = mapped_column(
        String(50), unique=True, nullable=True, index=True
    )
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    group_name: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    voice_profiles: Mapped[List["VoiceProfile"]] = relationship(
        back_populates="student", cascade="all, delete-orphan", passive_deletes=True
    )


class VoiceProfile(Base):
    """A voice embedding (192-D ECAPA-TDNN) belonging to a student."""

    __tablename__ = "voice_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("students.id", ondelete="CASCADE"), nullable=False, index=True
    )
    embedding_blob: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    audio_sample_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    quality_weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 0..1, see core/quality.py
    quality_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # enrollment metadata (levels, clipping)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    student: Mapped[Student] = relationship(back_populates="voice_profiles")

    @staticmethod
    def pack_embedding(embedding: np.ndarray) -> bytes:
        """Serialise an embedding as raw float32 bytes (no pickle)."""
        return np.asarray(embedding, dtype=np.float32).tobytes()

    @property
    def embedding(self) -> np.ndarray:
        """Deserialised float32 embedding."""
        return np.frombuffer(self.embedding_blob, dtype=np.float32).copy()


class InterventionLog(Base):
    """One detected noise event (spoken or suppressed by the cooldown)."""

    __tablename__ = "intervention_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)
    event_type: Mapped[EventType] = mapped_column(
        Enum(EventType, native_enum=False, length=20), nullable=False, index=True
    )
    student_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("students.id", ondelete="SET NULL"), nullable=True
    )
    dB_level: Mapped[float] = mapped_column("db_level", Float, nullable=False)
    message_spoken: Mapped[str] = mapped_column(Text, default="")
    match_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    was_spoken: Mapped[bool] = mapped_column(Boolean, default=False)

    # --- structured multi-speaker event (all nullable: rows written by older versions stay valid)
    event_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    category: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    start_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    overlap_duration: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    speaker_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_overlap: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    detection_method: Mapped[Optional[str]] = mapped_column(String(60), nullable=True)
    details_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # speakers, notes (no raw audio)
