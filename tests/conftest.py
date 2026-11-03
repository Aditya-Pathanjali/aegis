"""Shared pytest fixtures for Aegis testing suite."""

from __future__ import annotations

import os
from typing import Generator

import pytest
from alembic.config import Config
from testcontainers.postgres import PostgresContainer

from aegis.config import reset_settings
from aegis.db.session import reset_db_connection
from aegis.orchestration.checkpointer import reset_checkpointer
from aegis.rag.vector_store import reset_vector_store
from aegis.services.orchestrator_service import reset_orchestrator_service
from alembic import command


@pytest.fixture(scope="session")
def postgres_container(redis_container) -> Generator[str, None, None]:
    """Spins up a real PostgreSQL container with pgvector:pg16 and applies migrations."""
    container = PostgresContainer("pgvector/pgvector:pg16")
    container.start()
    raw_url = container.get_connection_url()
    url = raw_url.replace("+psycopg2", "")
    os.environ["DATABASE_URL"] = url
    reset_settings(None)

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
    reset_settings(None)
    container.stop()


@pytest.fixture(scope="session")
def redis_container() -> Generator[str, None, None]:
    """Spins up a real Redis container."""
    from testcontainers.redis import RedisContainer

    from aegis.queue import reset_redis_pools

    container = RedisContainer("redis:7-alpine")
    container.start()
    host = container.get_container_host_ip()
    port = container.get_exposed_port(container.port)
    url = f"redis://{host}:{port}/0"
    os.environ["REDIS_URL"] = url
    reset_settings(None)

    yield url

    reset_redis_pools()
    os.environ.pop("REDIS_URL", None)
    reset_settings(None)
    container.stop()


