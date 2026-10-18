"""Retention of enrolment audio (embeddings are kept; raw audio is limited and configurable)."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import select

from config import Settings
from database.db_session import session_scope
from database.models import VoiceProfile

logger = logging.getLogger(__name__)


def purge_expired_samples(settings: Settings) -> int:
    """Delete enrolment audio older than ``voice_sample_retention_days`` (0 = keep). Returns the count."""
    days = settings.voice_sample_retention_days
    if days <= 0:
        return 0
    cutoff = datetime.now() - timedelta(days=days)
    removed = 0
    with session_scope() as session:
        rows = session.execute(
            select(VoiceProfile).where(VoiceProfile.audio_sample_path.is_not(None), VoiceProfile.created_at < cutoff)
        ).scalars().all()
        for row in rows:
            Path(row.audio_sample_path).unlink(missing_ok=True)
            row.audio_sample_path = None
            removed += 1
    if removed:
        logger.info("Retention: deleted %d expired voice sample file(s); embeddings kept", removed)
    return removed
