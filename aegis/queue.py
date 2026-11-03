"""Queue and Redis connection manager for Aegis async workers."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from typing import Any, Coroutine, Optional, TypeVar

import redis
import redis.asyncio as aioredis
from arq.connections import ArqRedis, RedisSettings, create_pool

from aegis.config import get_settings
from aegis.models.schemas import TaskSubmission, TraceEvent

logger = logging.getLogger(__name__)

T = TypeVar("T")

_arq_pool: Optional[ArqRedis] = None
_arq_pool_loop: Optional[asyncio.AbstractEventLoop] = None
_async_redis_client: Optional[aioredis.Redis] = None
_async_redis_loop: Optional[asyncio.AbstractEventLoop] = None
_sync_redis_client: Optional[redis.Redis] = None


def get_redis_settings() -> RedisSettings:
    """Returns arq RedisSettings parsed from configured REDIS_URL."""
    settings = get_settings()
    return RedisSettings.from_dsn(settings.redis_url)


async def get_arq_pool() -> ArqRedis:
    """Returns singleton ArqRedis pool connection bound to the current event loop."""
    global _arq_pool, _arq_pool_loop
    current_loop = asyncio.get_running_loop()
    if _arq_pool is None or _arq_pool_loop is not current_loop or current_loop.is_closed():
        _arq_pool = await create_pool(get_redis_settings())
        _arq_pool_loop = current_loop
    return _arq_pool


def get_sync_redis() -> redis.Redis:
    """Returns synchronous redis client for quick status and cancel checks."""
    global _sync_redis_client
    if _sync_redis_client is None:
        settings = get_settings()
        _sync_redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    return _sync_redis_client


def get_async_redis() -> aioredis.Redis:
    """Returns async redis client for SSE streaming and pub/sub."""
    global _async_redis_client, _async_redis_loop
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    if _async_redis_client is None or (_async_redis_loop and _async_redis_loop is not current_loop):
        settings = get_settings()
        _async_redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)
        _async_redis_loop = current_loop
    return _async_redis_client


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Runs an async coroutine synchronously from sync callers."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


async def publish_trace_event_async(event: TraceEvent) -> None:
    """Publishes a TraceEvent to the task's Redis pub/sub channel."""
    try:
        client = get_async_redis()
        channel = f"aegis:tasks:{event.task_id}:events"
        payload = json.dumps(event.model_dump(mode="json"))
        await client.publish(channel, payload)
    except Exception as exc:
        logger.debug("Failed to publish trace event over Redis pub/sub: %s", exc)


def publish_trace_event_sync(event: TraceEvent) -> None:
    """Synchronously publishes a TraceEvent to the task's Redis pub/sub channel."""
    try:
        client = get_sync_redis()
        channel = f"aegis:tasks:{event.task_id}:events"
        payload = json.dumps(event.model_dump(mode="json"))
        client.publish(channel, payload)
    except Exception as exc:
        logger.debug("Failed to publish trace event over Redis pub/sub: %s", exc)


async def enqueue_task_job(task_id: str, submission: TaskSubmission) -> str:
    """Enqueues task execution onto arq worker queue."""
    pool = await get_arq_pool()
    job = await pool.enqueue_job(
        "run_task_job",
        task_id=task_id,
        submission_dict=submission.model_dump(mode="json"),
        _job_id=f"task:{task_id}",
    )
    return job.job_id if job else f"task:{task_id}"


async def enqueue_resume_job(task_id: str, action: str, feedback: Optional[str] = None) -> str:
    """Enqueues human gate resumption onto arq worker queue."""
    pool = await get_arq_pool()
    job = await pool.enqueue_job(
        "resume_task_job",
        task_id=task_id,
        action=action,
        feedback=feedback,
        _job_id=f"resume:{task_id}:{uuid_short()}",
    )
    return job.job_id if job else f"resume:{task_id}"


def uuid_short() -> str:
    import uuid
    return uuid.uuid4().hex[:8]


def set_task_cancelled(task_id: str) -> None:
    """Sets the cancellation flag in Redis. Raises if Redis is unreachable (fail closed)."""
    get_sync_redis().set(f"aegis:task:{task_id}:cancel", "1", ex=3600)


def is_task_cancelled(task_id: str) -> bool:
    """Checks the cancellation flag. Raises if Redis is unreachable, which fails the task
    rather than letting it keep spending with no way to stop it."""
    return bool(get_sync_redis().exists(f"aegis:task:{task_id}:cancel"))


# The worker owns Docker access; the API learns sandbox health from this heartbeat instead of
# mounting the Docker socket. A missing (expired) key means no healthy worker has reported.
SANDBOX_HEARTBEAT_KEY = "aegis:worker:sandbox_ok"
SANDBOX_HEARTBEAT_TTL_S = 30


def write_sandbox_heartbeat(ok: bool) -> None:
    get_sync_redis().set(SANDBOX_HEARTBEAT_KEY, "1" if ok else "0", ex=SANDBOX_HEARTBEAT_TTL_S)


def read_sandbox_heartbeat() -> bool:
    return get_sync_redis().get(SANDBOX_HEARTBEAT_KEY) == "1"


async def reset_queue_connections() -> None:
    """Closes all active Redis and arq connections."""
    global _arq_pool, _async_redis_client, _sync_redis_client
    if _arq_pool is not None:
        try:
            await _arq_pool.close()
        except Exception:
            pass
        _arq_pool = None

    if _async_redis_client is not None:
        try:
            await _async_redis_client.aclose()
        except Exception:
            pass
        _async_redis_client = None

    if _sync_redis_client is not None:
        try:
            _sync_redis_client.close()
        except Exception:
            pass
        _sync_redis_client = None


def reset_redis_pools() -> None:
    """Synchronously resets cached redis clients and pools."""
    global _arq_pool, _arq_pool_loop, _async_redis_client, _async_redis_loop, _sync_redis_client
    if _sync_redis_client is not None:
        try:
            _sync_redis_client.close()
        except Exception:
            pass
        _sync_redis_client = None
    _async_redis_client = None
    _async_redis_loop = None
    _arq_pool = None
    _arq_pool_loop = None


