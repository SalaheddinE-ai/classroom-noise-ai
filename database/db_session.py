"""SQLite engine / session factory.

Run ``python -m database.db_session`` to create the schema ("migration").
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from config import get_settings
from database.models import Base

_settings = get_settings()

engine = create_engine(
    _settings.sqlite_url,
    connect_args={"check_same_thread": False},
    future=True,
)


@event.listens_for(engine, "connect")
def _enable_sqlite_fk(dbapi_connection, _record) -> None:  # pragma: no cover
    """Enforce foreign keys (ON DELETE CASCADE) on every SQLite connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def _add_missing_columns(bind=None) -> list[str]:
    """Lightweight forward-only migration: ``ALTER TABLE ... ADD COLUMN`` for new model columns.

    ``create_all`` never alters an existing table, so databases created by an older version of the
    application would miss the multi-speaker columns. Every added column is nullable, existing rows
    stay valid. Returns the list of ``table.column`` names that were added.
    """
    bind = bind or engine
    added: list[str] = []
    inspector = inspect(bind)
    existing_tables = set(inspector.get_table_names())
    with bind.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            present = {col["name"] for col in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in present:
                    continue
                ddl_type = column.type.compile(dialect=bind.dialect)
                conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {ddl_type}'))
                added.append(f"{table.name}.{column.name}")
    return added


def init_db() -> None:
    """Create all tables if they do not exist and add any column introduced by a newer version."""
    Base.metadata.create_all(bind=engine)
    added = _add_missing_columns()
    if added:
        logging.getLogger(__name__).info("Database migrated, columns added: %s", ", ".join(added))


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, rollback on error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


if __name__ == "__main__":
    init_db()
    print(f"Database initialised at {_settings.sqlite_url}")
