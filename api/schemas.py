"""Pydantic schemas for the REST API."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class StudentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    age: Optional[int] = None
    student_code: Optional[str] = None
    created_at: datetime
    voice_samples: int = 0
    # --- added with the voice-profile feature (optional: older clients simply ignore them)
    notes: Optional[str] = None
    group_name: Optional[str] = None
    # no_samples | needs_samples (fewer than CNAI_MIN_VOICE_SAMPLES) | ready
    recognition_status: str = "no_samples"


class VoiceSampleOut(BaseModel):
    id: int
    created_at: datetime
    duration_seconds: Optional[float] = None
    has_audio: bool = False
    quality_weight: Optional[float] = None  # 0..1 weight of this sample in the profile centroid


class StudentProfileOut(StudentOut):
    """A student with the list of registered voice samples (no raw embeddings are exposed)."""

    samples: List[VoiceSampleOut] = Field(default_factory=list)
    min_samples_required: int = 1


class StudentUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=120)
    age: Optional[int] = Field(None, ge=3, le=99)
    student_code: Optional[str] = Field(None, max_length=50)
    notes: Optional[str] = Field(None, max_length=2000)
    group_name: Optional[str] = Field(None, max_length=80)


class LogOut(BaseModel):
    id: int
    timestamp: datetime
    event_type: str
    student_id: Optional[str] = None
    student_name: Optional[str] = None
    dB_level: float
    message_spoken: str
    match_score: Optional[float] = None
    was_spoken: bool
    # --- structured multi-speaker fields (null for rows written by older versions)
    event_id: Optional[str] = None
    category: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    overlap_duration: Optional[float] = None
    speaker_count: Optional[int] = None
    is_overlap: Optional[bool] = None
    detection_method: Optional[str] = None
    speakers: Optional[List[Dict[str, Any]]] = None
    notes: Optional[List[str]] = None


class SettingsUpdate(BaseModel):
    threshold_db: Optional[float] = Field(None, ge=0, le=90)
    cooldown_seconds: Optional[float] = Field(None, ge=15, le=30)
    speaker_match_threshold: Optional[float] = Field(None, ge=0, le=1)
    tts_volume: Optional[float] = Field(None, ge=0, le=2)
    language: Optional[str] = Field(None, pattern="^(en|fr)$")
    speaker_cooldown_seconds: Optional[float] = Field(None, ge=0, le=600)
    min_identify_seconds: Optional[float] = Field(None, ge=0.3, le=10)
    max_simultaneous_speakers: Optional[int] = Field(None, ge=1, le=6)
    overlap_sensitivity: Optional[float] = Field(None, ge=0, le=1)
    unknown_speaker_handling: Optional[str] = Field(None, pattern="^(report|log_only|ignore)$")
