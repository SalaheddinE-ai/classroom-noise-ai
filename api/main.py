"""FastAPI application: REST endpoints + WebSocket status stream."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.deps import get_service
from api.routes import live, logs, settings, status, students
from database.db_session import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Create the schema and warm the models in the background."""
    init_db()
    service = get_service()
    service.preload_async()
    yield
    service.shutdown()


app = FastAPI(title="classroom-noise-ai", version="2.1.1", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(students.router)
app.include_router(status.router)
app.include_router(live.router)
app.include_router(logs.router)
app.include_router(settings.router)


@app.get("/health", tags=["status"])
def health() -> dict:
    return {"status": "ok"}
