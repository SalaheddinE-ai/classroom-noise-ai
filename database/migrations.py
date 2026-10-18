"""Lightweight, idempotent schema upgrade for SQLite (adds missing nullable/defaulted columns).

``Base.metadata.create_all`` creates missing tables but never alters existing ones; this module
adds the columns introduced by newer versions so existing ``classroom.db`` files keep working.
"""
from __future__ import annotations

import logging
from typing import List

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from database.models import Base

logger = logging.getLogger(__name__)


def upgrade_schema(engine: Engine) -> List[str]:
    """Add every column that exists in the models but not in the database. Returns what was added."""
    added: List[str] = []
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue
            existing = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing:
                    continue
                default = ""
                if column.server_default is not None:
                    arg = column.server_default.arg
                    default = f" DEFAULT {getattr(arg, 'text', arg)}"
                if not column.nullable and not default:
                    raise RuntimeError(f"Cannot auto-add NOT NULL column {table.name}.{column.name} without default")
                ddl_type = column.type.compile(dialect=engine.dialect)
                conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {ddl_type}{default}'))
                added.append(f"{table.name}.{column.name}")
    if added:
        logger.info("Schema upgraded, added columns: %s", ", ".join(added))
    return added
