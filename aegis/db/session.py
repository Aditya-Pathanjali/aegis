"""SQLAlchemy 2 database engine and session management for Aegis."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from aegis.config import get_settings

logger = logging.getLogger(__name__)

_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker[Session]] = None


def get_engine(url: Optional[str] = None) -> Engine:
    """Returns the singleton SQLAlchemy Engine, failing closed if no database URL is set."""
    global _engine
    if _engine is not None and url is None:
        return _engine

    db_url = url or get_settings().database_url
    if not db_url:
        raise RuntimeError("DATABASE_URL is not configured. Aegis requires PostgreSQL persistence.")

    connect_args = {}
    if "postgres" in db_url:
        connect_args["connect_timeout"] = 3

    engine = create_engine(
        db_url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        connect_args=connect_args,
    )
    if url is None:
        _engine = engine
    return engine


def get_session_factory(engine: Optional[Engine] = None) -> sessionmaker[Session]:
    """Returns the session factory for creating database sessions."""
    global _session_factory
    if _session_factory is not None and engine is None:
        return _session_factory

    eng = engine or get_engine()
    factory = sessionmaker(bind=eng, autoflush=False, autocommit=False, expire_on_commit=False)
    if engine is None:
        _session_factory = factory
    return factory


@contextmanager
def get_db_session(factory: Optional[sessionmaker[Session]] = None) -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""
    sf = factory or get_session_factory()
    session = sf()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_db_health(url: Optional[str] = None, timeout: float = 3.0) -> bool:
    """Verifies that the database is reachable and accepting queries."""
    try:
        db_url = url or get_settings().database_url
        if not db_url:
            return False
        eng = get_engine(db_url)
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("Database health check failed: %s", exc)
        return False


def reset_db_connection() -> None:
    """Disposes existing engine and resets cached instances (useful for testing)."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None

