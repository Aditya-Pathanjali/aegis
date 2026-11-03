"""LangGraph checkpointer factory supporting PostgreSQL and test environments."""

from __future__ import annotations

import logging
from typing import Any, Optional, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from aegis.config import get_settings
from aegis.orchestration.serialization import get_checkpointer_serializer

logger = logging.getLogger(__name__)

_global_pool: Optional[ConnectionPool] = None
_global_checkpointer: Optional[BaseCheckpointSaver] = None


def get_checkpointer(connection_url: Optional[str] = None) -> BaseCheckpointSaver:
    """Returns a checkpointer configured with Aegis serializer.

    Uses PostgreSQL PostgresSaver when DATABASE_URL is configured,
    and falls back to MemorySaver only when no database URL is set (for offline unit tests).
    """
    global _global_pool, _global_checkpointer
    db_url = connection_url or get_settings().database_url
    if db_url:
        if _global_checkpointer is not None and connection_url is None:
            return _global_checkpointer
        try:
            pool = ConnectionPool(
                db_url,
                min_size=1,
                max_size=10,
                open=True,
                kwargs={"autocommit": True, "row_factory": dict_row},
            )
            saver = PostgresSaver(conn=cast(Any, pool), serde=get_checkpointer_serializer())
            saver.setup()
            if connection_url is None:
                _global_pool = pool
                _global_checkpointer = saver
            return saver
        except Exception as exc:
            logger.exception("Failed to initialize PostgresSaver with URL: %s", exc)
            raise

    # For offline unit tests when no database is configured
    return MemorySaver(serde=get_checkpointer_serializer())


def reset_checkpointer() -> None:
    """Closes connections and clears cached checkpointer."""
    global _global_pool, _global_checkpointer
    if _global_pool is not None:
        try:
            _global_pool.close()
        except Exception:
            pass
    _global_pool = None
    _global_checkpointer = None

