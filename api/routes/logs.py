"""Intervention log endpoints."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.schemas import LogOut
from database.db_session import get_db
from database.models import EventType, InterventionLog, Student

router = APIRouter(prefix="/logs", tags=["logs"])


@router.get("", response_model=List[LogOut])
def list_logs(
    start: Optional[datetime] = Query(None, description="Inclusive lower bound (ISO 8601)"),
    end: Optional[datetime] = Query(None, description="Inclusive upper bound (ISO 8601)"),
    event_type: Optional[List[EventType]] = Query(None),
    limit: int = Query(500, ge=1, le=5000),
    db: Session = Depends(get_db),
) -> List[LogOut]:
    """Past interventions, newest first, filterable by time and event type."""
    stmt = (
        select(InterventionLog, Student.name)
        .outerjoin(Student, Student.id == InterventionLog.student_id)
        .order_by(InterventionLog.timestamp.desc())
        .limit(limit)
    )
    if start:
        stmt = stmt.where(InterventionLog.timestamp >= start)
    if end:
        stmt = stmt.where(InterventionLog.timestamp <= end)
    if event_type:
        stmt = stmt.where(InterventionLog.event_type.in_(event_type))
    rows = db.execute(stmt).all()
    return [_to_out(log, name) for log, name in rows]


def _details(raw: Optional[str]) -> Dict[str, Any]:
    try:
        return json.loads(raw) if raw else {}
    except ValueError:
        return {}


def _to_out(log: InterventionLog, student_name: Optional[str]) -> LogOut:
    details = _details(log.details_json)
    return LogOut(
        id=log.id,
        timestamp=log.timestamp,
        event_type=log.event_type.value,
        student_id=log.student_id,
        student_name=student_name,
        dB_level=log.dB_level,
        message_spoken=log.message_spoken,
        match_score=log.match_score,
        was_spoken=log.was_spoken,
        event_id=log.event_id,
        category=log.category,
        start_time=log.start_time,
        end_time=log.end_time,
        overlap_duration=log.overlap_duration,
        speaker_count=log.speaker_count,
        is_overlap=log.is_overlap,
        detection_method=log.detection_method,
        speakers=details.get("speakers"),
        notes=details.get("notes"),
    )
