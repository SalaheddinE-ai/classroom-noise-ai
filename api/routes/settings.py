"""Runtime settings endpoints."""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends

from api.deps import get_service
from api.schemas import SettingsUpdate
from engine.runner import MonitorService

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("")
def read_settings(service: MonitorService = Depends(get_service)) -> Dict[str, Any]:
    """Current tunable settings."""
    s = service.settings
    return {
        "threshold_db": s.threshold_db,
        "cooldown_seconds": s.cooldown_seconds,
        "speaker_match_threshold": s.speaker_match_threshold,
        "tts_volume": s.tts_volume,
        "language": s.language,
        "calibration_offset_db": s.calibration_offset_db,
        "audio_backend": s.audio_backend,
        "tts_backend": s.tts_backend,
        "speaker_cooldown_seconds": s.speaker_cooldown_seconds,
        "min_identify_seconds": s.min_identify_seconds,
        "max_simultaneous_speakers": s.max_simultaneous_speakers,
        "overlap_sensitivity": s.overlap_sensitivity,
        "unknown_speaker_handling": s.unknown_speaker_handling,
        "samples_per_student": s.samples_per_student,
        "min_voice_samples": s.min_voice_samples,
        "separation_enabled": s.separation_enabled,
        "diarization_backend": s.diarization_backend,
        "viz_history_seconds": s.viz_history_seconds,
        "raw_audio_retention_seconds": s.raw_audio_retention_seconds,
    }


@router.put("")
def update_settings(body: SettingsUpdate, service: MonitorService = Depends(get_service)) -> Dict[str, Any]:
    """Update thresholds / cooldown / language live (not persisted across restarts)."""
    service.update_settings(**body.model_dump(exclude_none=True))
    return read_settings(service)
