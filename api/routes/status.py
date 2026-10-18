"""Live status, monitor control and the WebSocket stream."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect

from api.deps import get_service
from core.audio_stream import AudioStream
from engine.runner import MonitorService

router = APIRouter(tags=["status"])


@router.get("/status")
def get_status(service: MonitorService = Depends(get_service)) -> Dict[str, Any]:
    """Current monitor status (live dB, state, recent events)."""
    return service.status()


@router.post("/monitor/start")
def start_monitor(service: MonitorService = Depends(get_service)) -> Dict[str, Any]:
    """Load models if needed and start listening to the microphone."""
    try:
        service.start()
    except Exception as exc:
        raise HTTPException(500, f"Could not start monitoring: {exc}") from exc
    return service.status()


@router.post("/monitor/stop")
def stop_monitor(service: MonitorService = Depends(get_service)) -> Dict[str, Any]:
    """Stop listening."""
    service.stop()
    return service.status()


@router.get("/devices")
def list_devices() -> List[Dict[str, Any]]:
    """Audio devices (use the index with CNAI_INPUT_DEVICE / CNAI_OUTPUT_DEVICE)."""
    return AudioStream.list_devices()


@router.websocket("/ws/stream")
async def stream_status(websocket: WebSocket) -> None:
    """Push the status JSON twice per second."""
    service = get_service()
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(service.status())
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, RuntimeError):
        return
