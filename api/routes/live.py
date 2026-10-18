"""Structured events and live visualisation data for the dashboard."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import get_service
from engine.runner import MonitorService

router = APIRouter(tags=["live"])


@router.get("/events")
def list_events(
    limit: int = Query(50, ge=1, le=100),
    full: bool = Query(True, description="Include the per-speaker waveform series"),
    service: MonitorService = Depends(get_service),
) -> List[Dict[str, Any]]:
    """Grouped detection events of the current session, newest first (in memory, bounded)."""
    return service.engine.events_snapshot(limit, full) if service.engine else []


@router.get("/events/{event_id}")
def get_event(event_id: str, service: MonitorService = Depends(get_service)) -> Dict[str, Any]:
    """One grouped event with every speaker, segment and waveform series."""
    for event in service.engine.events_snapshot(100, True) if service.engine else []:
        if event.get("event_id") == event_id:
            return event
    raise HTTPException(404, "Event not found (only the most recent events are kept in memory; see /logs)")


@router.get("/live/visualization")
def live_visualization(
    seconds: float = Query(None, ge=5, le=900),
    service: MonitorService = Depends(get_service),
) -> Dict[str, Any]:
    """Mixed-microphone waveform + speaker events of the last ``seconds`` seconds.

    ``waveform`` is the *combined* microphone signal. Per-speaker tracks live in each event's
    ``speakers[].waveform`` and are labelled by ``signal_type`` (separated / masked_mixture /
    activity_only) so the UI never presents a mixture as an isolated voice.
    """
    window = seconds or service.settings.viz_history_seconds
    if service.engine is None:
        return {"now": None, "history_seconds": window, "waveform": {"t0_epoch": None, "dt_seconds": 0.05, "peaks": []},
                "events": [], "capabilities": {}}
    return service.engine.visualization_snapshot(window)
