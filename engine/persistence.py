"""Persistence of grouped detection events (one ``intervention_logs`` row per event, updated in place)."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from database.db_session import session_scope
from database.event_types import EventType
from database.models import InterventionLog
from engine.events import DetectionEvent

logger = logging.getLogger(__name__)


def event_details(event: DetectionEvent) -> Dict[str, Any]:
    """JSON stored in ``details_json``: speakers (without raw audio), notes and overlap spans."""
    return {
        "labels": event.labels,
        "notes": event.notes,
        "overlap_segments": [s.model_dump(mode="json") for s in event.overlap_segments],
        "schema_version": event.schema_version,
        "decision_status": event.decision_status,
        "identification_method": event.identification_method,
        "metadata": event.metadata,
        "speakers": [s.model_dump(mode="json") for s in event.speakers],
    }


def _write(event: DetectionEvent, student_id: Optional[str]) -> None:
    message = event.message if event.warning_triggered else f"[suppressed by cooldown] {event.message}"
    with session_scope() as session:
        row = session.execute(
            select(InterventionLog).where(InterventionLog.event_id == event.event_id)
        ).scalar_one_or_none()
        if row is None:
            row = InterventionLog(event_id=event.event_id, dB_level=event.db,
                                  event_type=EventType(event.event_type))
            session.add(row)
        row.timestamp = event.timestamp.replace(tzinfo=None)
        row.event_type = EventType(event.event_type)
        row.student_id = student_id
        row.dB_level = event.db
        row.message_spoken = message
        row.match_score = event.match_score
        row.was_spoken = event.warning_triggered
        row.category = event.category
        row.start_time = event.start_time.replace(tzinfo=None)
        row.end_time = event.end_time.replace(tzinfo=None)
        row.overlap_duration = None if event.overlap_duration_ms is None else event.overlap_duration_ms / 1000.0
        row.speaker_count = event.speaker_count
        row.is_overlap = event.is_overlap
        row.detection_method = f"{event.detection_method}|{event.identification_method}"[:60]
        row.details_json = json.dumps(event_details(event))


def persist_event(event: DetectionEvent) -> None:
    """Insert the event, or update the row written for the same ``event_id`` earlier.

    ``student_id`` is only stored for a single *confirmed* student. If that student was deleted
    meanwhile (foreign-key error) the event is stored without the link instead of being lost.
    """
    confirmed = [s for s in event.speakers if s.is_identified and s.student_id]
    student_id = confirmed[0].student_id if len(confirmed) == 1 and event.speaker_count == 1 else None
    try:
        try:
            _write(event, student_id)
        except IntegrityError:
            if student_id is None:
                raise
            _write(event, None)
    except Exception:
        logger.exception("Could not persist the detection event")


def anonymize_student_in_logs(student_id: str) -> int:
    """Remove a deleted student's identity from stored event details (names inside ``details_json``).

    ``intervention_logs.student_id`` is already set to NULL by the foreign key; this also scrubs the
    speaker list of grouped events. Returns the number of rows rewritten.
    """
    rewritten = 0
    try:
        with session_scope() as session:
            rows = session.execute(
                select(InterventionLog).where(InterventionLog.details_json.like(f"%{student_id}%"))
            ).scalars().all()
            for row in rows:
                try:
                    details = json.loads(row.details_json or "{}")
                except ValueError:
                    continue
                for speaker in details.get("speakers", []):
                    if speaker.get("student_id") == student_id:
                        speaker.update(student_id=None, student_name="Unknown Speaker",
                                       is_identified=False, identification_status="deleted_profile")
                row.details_json = json.dumps(details)
                rewritten += 1
    except Exception:
        logger.exception("Could not anonymise stored events")
    return rewritten
