"""Run the real arq worker in burst mode against the test Redis, with a scripted LLM."""

from __future__ import annotations

import asyncio
from typing import Any

from arq.worker import create_worker

from aegis.worker import get_worker_settings


def run_worker_burst(llm: Any) -> None:
    """Process every queued job once, then return (what a worker container does continuously)."""

    async def _run() -> None:
        worker = create_worker(get_worker_settings(), burst=True, ctx={"llm": llm})
        await worker.async_run()
        await worker.close()

    asyncio.run(_run())
