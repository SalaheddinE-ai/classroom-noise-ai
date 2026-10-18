"""Structured alert events (grouped detections with per-speaker details)."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.deps import get_service
from api.schemas import EventOut
from database.db_session import get_db
from database.models import InterventionLog
from engine.events import iso
from engine.runner import MonitorService

router = APIRouter(prefix="/events", tags=["events"])


def _row_to_event(row: InterventionLog) -> Dict[str, Any]:
    """Rebuild the structured event from a log row (legacy rows get a single-speaker view)."""
    speakers = json.loads(row.speakers_json) if row.speakers_json else []
    if not speakers and row.student_id:
        speakers = [
            {
                "speaker_id": "spk_1", "student_id": row.student_id, "student_name": "Registered student",
                "start_time": iso(row.timestamp.timestamp()), "end_time": iso(row.timestamp.timestamp()),
                "similarity_score": row.match_score, "is_identified": True,
            }
        ]
    start = row.start_time or row.timestamp
    end = row.end_time or row.timestamp
    return {
        "event_id": row.event_uid or f"legacy-{row.id}",
        "event_type": row.event_type.value,
        "labels": ["Overlapping Speech"] if row.is_overlap else [],
        "timestamp": iso(row.timestamp.timestamp()),
        "start_time": iso(start.timestamp()),
        "end_time": iso(end.timestamp()),
        "duration": round((end - start).total_seconds(), 3),
        "overlap_duration": row.overlap_duration,
        "overlap_estimated": bool(row.overlap_estimated),
        "speakers": speakers,
        "speaker_count": row.speaker_count or len(speakers) or 1,
        "is_overlap": bool(row.is_overlap),
        "detection_method": row.detection_method,
        "warning_triggered": bool(row.was_spoken),
        "warning_status": "spoken" if row.was_spoken else "suppressed",
        "message": row.message_spoken or "",
        "db_level": row.dB_level,
        "notes": json.loads(row.notes_json) if row.notes_json else [],
    }


@router.get("", response_model=List[EventOut])
def list_events(
    limit: int = Query(50, ge=1, le=500),
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    db: Session = Depends(get_db),
    service: MonitorService = Depends(get_service),
) -> List[Dict[str, Any]]:
    """Recent events (newest first). Live events carry waveform previews; history comes from SQLite."""
    stmt = select(InterventionLog).order_by(InterventionLog.timestamp.desc(), InterventionLog.id.desc()).limit(limit)
    if start:
        stmt = stmt.where(InterventionLog.timestamp >= start)
    if end:
        stmt = stmt.where(InterventionLog.timestamp <= end)
    events = [_row_to_event(r) for r in db.execute(stmt).scalars().all()]
    if service.engine is not None:
        live = {e.event_id: e.to_dict() for e in list(service.engine.events)}
        events = [live.get(e["event_id"], e) for e in events]
    return events
