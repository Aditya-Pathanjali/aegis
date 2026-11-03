"""Shared pytest fixtures for Aegis testing suite."""

from __future__ import annotations

import os
from typing import Generator

import pytest
from alembic.config import Config
from testcontainers.postgres import PostgresContainer

from aegis.config import Settings, reset_settings
from aegis.db.session import reset_db_connection
from aegis.orchestration.checkpointer import reset_checkpointer
from aegis.rag.vector_store import reset_vector_store
from aegis.services.orchestrator_service import reset_orchestrator_service
from alembic import command


@pytest.fixture(scope="session")
def postgres_container() -> Generator[str, None, None]:
    """Spins up a real PostgreSQL container with pgvector:pg16 and applies migrations."""
    container = PostgresContainer("pgvector/pgvector:pg16")
    container.start()
    raw_url = container.get_connection_url()
    url = raw_url.replace("+psycopg2", "")
    os.environ["DATABASE_URL"] = url
    reset_settings(Settings(database_url=url))

    # Run Alembic migrations
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")

    yield url

    reset_db_connection()
    reset_checkpointer()
    reset_vector_store()
    reset_orchestrator_service()
    os.environ.pop("DATABASE_URL", None)
    reset_settings(Settings(database_url=None))
    container.stop()

