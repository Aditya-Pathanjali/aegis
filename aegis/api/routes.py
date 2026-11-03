"""FastAPI API routes for Aegis."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse

from aegis.api.schemas import (
    ApprovalActionRequest,
    DocumentUploadRequest,
    RAGSearchRequest,
)
from aegis.config import Settings, get_settings
from aegis.db.session import check_db_health
from aegis.models.schemas import (
    ApprovalAction,
    TaskResponse,
    TaskStatus,
    TaskSubmission,
    TraceEvent,
)
from aegis.rag.retriever import get_rag_knowledge_base
from aegis.services.orchestrator_service import get_orchestrator_service

router = APIRouter(prefix="/api")


def check_redis_health(redis_url: str) -> bool:
    try:
        import redis
        client = redis.from_url(redis_url, socket_timeout=2.0)
        return bool(client.ping())
    except Exception:
        return False


def check_sandbox_health() -> bool:
    try:
        import docker
        client = docker.from_env(timeout=2)  # type: ignore[attr-defined]
        return bool(client.ping())
    except Exception:
        return False


def compute_health_status() -> tuple[int, Dict[str, Any]]:
    settings = get_settings()
    db_ok = check_db_health(settings.database_url)
    redis_ok = check_redis_health(settings.redis_url)
    sandbox_ok = check_sandbox_health()

    all_healthy = db_ok and redis_ok and sandbox_ok
    status_code = status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE

    data = {
        "status": "healthy" if all_healthy else "unhealthy",
        "service": "Aegis Multi-Agent Orchestrator",
        "version": "1.0.0",
        "database": "healthy" if db_ok else "down",
        "redis": "healthy" if redis_ok else "down",
        "sandbox": "healthy" if sandbox_ok else "down",
        "agents": [
            "PlannerAgent",
            "ResearchAgent",
            "DataAnalystAgent",
            "VerificationAgent",
            "CriticAgent",
            "SynthesizerAgent",
        ],
    }
    return status_code, data


@router.get("/health")
def health_check():
    status_code, data = compute_health_status()
    return JSONResponse(status_code=status_code, content=data)


def _sanitize_columns(columns: List[Any]) -> List[str]:
    seen: Dict[str, int] = {}
    new_cols: List[str] = []
    for c in columns:
        clean = re.sub(r"[^a-zA-Z0-9_]+", "_", str(c).strip().lower()).strip("_")
        if not clean:
            clean = "col"
        if clean in seen:
            seen[clean] += 1
            new_cols.append(f"{clean}_{seen[clean]}")
        else:
            seen[clean] = 0
            new_cols.append(clean)
    return new_cols


def _process_dataset_bytes(raw_bytes: bytes, settings: Settings) -> str:
    if len(raw_bytes) > settings.dataset_max_bytes:
        raise HTTPException(
            status_code=422,
            detail=f"Dataset size ({len(raw_bytes)} bytes) exceeds limit of {settings.dataset_max_bytes} bytes.",
        )
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse CSV dataset: {e}")

    if len(df) > settings.dataset_max_rows:
        raise HTTPException(
            status_code=422,
            detail=f"Dataset row count ({len(df)}) exceeds limit of {settings.dataset_max_rows}.",
        )
    if len(df.columns) > settings.dataset_max_cols:
        raise HTTPException(
            status_code=422,
            detail=f"Dataset column count ({len(df.columns)}) exceeds limit of {settings.dataset_max_cols}.",
        )

    df.columns = _sanitize_columns(list(df.columns))
    return df.to_csv(index=False)


@router.post("/tasks", response_model=Dict[str, str], status_code=status.HTTP_202_ACCEPTED)
async def submit_task(request: Request):
    """Submits a new task for autonomous multi-agent decomposition and execution."""
    import os

    settings = get_settings()
    if not settings.provider or not settings.model:
        raise HTTPException(
            status_code=503,
            detail="No LLM provider configured. Set AEGIS_PROVIDER and AEGIS_MODEL in environment.",
        )

    if settings.provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY is not configured in environment.",
        )

    if settings.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured in environment.",
        )

    content_type = request.headers.get("content-type", "")
    query = ""
    dataset_name: Optional[str] = None
    dataset_content: Optional[str] = None
    require_human_approval = False
    max_retries = 2

    if "multipart/form-data" in content_type:
        form = await request.form()
        query = str(form.get("query", ""))
        raw_ds_name = form.get("dataset_name")
        if isinstance(raw_ds_name, str):
            dataset_name = raw_ds_name
        req_app = form.get("require_human_approval", False)
        if isinstance(req_app, str):
            require_human_approval = req_app.lower() in ("true", "1")
        else:
            require_human_approval = bool(req_app)
        m_retries = form.get("max_retries")
        if isinstance(m_retries, str):
            try:
                max_retries = int(m_retries)
            except ValueError:
                pass

        file = form.get("file")
        if file is not None and hasattr(file, "read"):
            file_bytes = await file.read()
            if not dataset_name and hasattr(file, "filename") and isinstance(file.filename, str):
                dataset_name = file.filename
            if file_bytes:
                dataset_content = _process_dataset_bytes(file_bytes, settings)
        elif "dataset_content" in form and form.get("dataset_content"):
            raw_str = str(form.get("dataset_content"))
            dataset_content = _process_dataset_bytes(raw_str.encode("utf-8"), settings)
    else:
        try:
            body = await request.json()
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid JSON request body: {e}")
        query = body.get("query", "")
        raw_ds_name = body.get("dataset_name")
        if isinstance(raw_ds_name, str):
            dataset_name = raw_ds_name
        raw_content = body.get("dataset_content")
        require_human_approval = body.get("require_human_approval", False)
        max_retries = body.get("max_retries", 2)
        if raw_content:
            dataset_content = _process_dataset_bytes(raw_content.encode("utf-8"), settings)

    if not query or not query.strip():
        raise HTTPException(status_code=422, detail="Query cannot be empty.")

    service = get_orchestrator_service()
    sub_obj = TaskSubmission(
        query=query,
        dataset_name=dataset_name,
        dataset_content=dataset_content,
        require_human_approval=require_human_approval,
        max_retries=max_retries,
    )
    try:
        task_id = service.submit_task(sub_obj, enqueue=True)
    except TypeError:
        task_id = service.submit_task(sub_obj)
    return {"task_id": task_id, "status": "queued"}


@router.get("/tasks", response_model=List[TaskResponse])
def list_tasks():
    service = get_orchestrator_service()
    return service.task_store.list_tasks()


@router.get("/tasks/{task_id}", response_model=TaskResponse)
def get_task(task_id: str):
    service = get_orchestrator_service()
    resp = service.task_store.get_response(task_id)
    if not resp:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    return resp


@router.get("/tasks/{task_id}/events")
async def stream_task_events(
    task_id: str,
    request: Request,
    last_event_id: Optional[str] = Header(None, alias="Last-Event-ID"),
):
    """Server-Sent Events (SSE) endpoint streaming real-time agent execution traces over Redis pub/sub."""
    service = get_orchestrator_service()
    state = service.task_store.get_state(task_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")

    channel = f"aegis:tasks:{task_id}:events"
    seen_event_ids: set[str] = set()

    # Load historical events from Postgres
    history = service.task_store.get_trace_events(task_id)

    # Determine replay slice based on Last-Event-ID
    replay_events = []
    if last_event_id:
        found_idx = -1
        for idx, ev in enumerate(history):
            if ev.event_id == last_event_id:
                found_idx = idx
                break
        if found_idx >= 0:
            replay_events = history[found_idx + 1 :]
        else:
            replay_events = history
    else:
        replay_events = history

    async def event_generator():
        # 1. Replay historical events
        for ev in replay_events:
            seen_event_ids.add(ev.event_id)
            ev_json = json.dumps(ev.model_dump(mode="json"))
            yield f"id: {ev.event_id}\nevent: message\ndata: {ev_json}\n\n"

        # Check if task is already in terminal state
        curr = service.task_store.get_state(task_id)
        curr_status = curr.get("status") if curr else None
        if curr_status in (TaskStatus.COMPLETED, TaskStatus.REJECTED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            status_val = curr_status.value if hasattr(curr_status, "value") else str(curr_status)
            yield f"data: {json.dumps({'event_type': 'stream_end', 'status': status_val})}\n\n"
            return

        # 2. Subscribe to Redis pub/sub for real-time events
        from aegis.queue import get_async_redis
        redis_client = get_async_redis()
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(channel)

        try:
            while True:
                if await request.is_disconnected():
                    break

                try:
                    msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.5)
                    if msg and msg.get("type") == "message":
                        data_str = msg.get("data")
                        if isinstance(data_str, bytes):
                            data_str = data_str.decode("utf-8")
                        try:
                            ev_payload = json.loads(data_str)
                            ev_id = ev_payload.get("event_id")
                            if ev_id and ev_id not in seen_event_ids:
                                seen_event_ids.add(ev_id)
                                yield f"id: {ev_id}\nevent: message\ndata: {data_str}\n\n"
                        except Exception:
                            yield f"data: {data_str}\n\n"

                    # Check DB status periodically
                    st = service.task_store.get_state(task_id)
                    term_status = st.get("status") if st else None
                    if term_status in (TaskStatus.COMPLETED, TaskStatus.REJECTED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                        # Drain any remaining pubsub messages
                        while True:
                            rem = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1)
                            if not rem or rem.get("type") != "message":
                                break
                            d_str = rem.get("data")
                            if isinstance(d_str, bytes):
                                d_str = d_str.decode("utf-8")
                            try:
                                p = json.loads(d_str)
                                eid = p.get("event_id")
                                if eid and eid not in seen_event_ids:
                                    seen_event_ids.add(eid)
                                    yield f"id: {eid}\nevent: message\ndata: {d_str}\n\n"
                            except Exception:
                                pass
                        status_str = term_status.value if hasattr(term_status, "value") else str(term_status)
                        yield f"data: {json.dumps({'event_type': 'stream_end', 'status': status_str})}\n\n"
                        break

                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logging.getLogger(__name__).debug("Error in pubsub event streaming: %s", exc)
                    await asyncio.sleep(0.1)

        finally:
            try:
                await pubsub.unsubscribe(channel)
                await pubsub.close()
            except Exception as e:
                logging.getLogger(__name__).debug("Error unsubscribing from Redis channel: %s", e)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/tasks/{task_id}/cancel", response_model=TaskResponse)
async def cancel_task(task_id: str):
    """Cancels a queued or executing task."""
    service = get_orchestrator_service()
    state = service.task_store.get_state(task_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")

    curr_status = state.get("status")
    if curr_status in (TaskStatus.COMPLETED, TaskStatus.REJECTED, TaskStatus.FAILED, TaskStatus.CANCELLED):
        resp = service.task_store.get_response(task_id)
        if resp:
            return resp
        return TaskResponse(task_id=task_id, status=curr_status)

    from aegis.queue import set_task_cancelled
    set_task_cancelled(task_id)

    state["status"] = TaskStatus.CANCELLED
    service.task_store.save_state(task_id, state)

    cancel_ev = TraceEvent(
        event_id=uuid.uuid4().hex[:8],
        task_id=task_id,
        event_type="task_cancelled",
        message=f"Task {task_id} cancelled by operator request.",
        details={"status": "cancelled"},
    )
    service.task_store.save_trace_event(cancel_ev)
    from aegis.queue import publish_trace_event_sync
    publish_trace_event_sync(cancel_ev)

    resp = service.task_store.get_response(task_id)
    if not resp:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' response not found.")
    return resp


@router.post("/tasks/{task_id}/approve", response_model=TaskResponse)
@router.post("/tasks/{task_id}/approval", response_model=TaskResponse)
async def handle_human_approval(task_id: str, req: ApprovalActionRequest, response: Response):
    """Handles Human-in-the-Loop approval decision (Approve, Reject, Request Re-analysis)."""
    service = get_orchestrator_service()
    state = service.task_store.get_state(task_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")

    if state.get("status") != TaskStatus.WAITING_APPROVAL:
        raise HTTPException(
            status_code=409,
            detail=f"Task '{task_id}' is not in WAITING_APPROVAL status (current: {state.get('status')}).",
        )

    try:
        if req.action == ApprovalAction.REQUEST_REANALYSIS:
            response.status_code = status.HTTP_202_ACCEPTED
        updated_task = service.resolve_human_approval(
            task_id=task_id,
            action=req.action,
            feedback=req.feedback,
        )
        return updated_task
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")


# RAG Knowledge Base Endpoints
@router.post("/rag/documents")
def upload_document(req: DocumentUploadRequest):
    kb = get_rag_knowledge_base()
    chunks_count = kb.index_document(
        doc_id=req.doc_id,
        title=req.title,
        text=req.text,
        source=req.source,
        metadata=req.metadata,
    )
    return {"status": "indexed", "doc_id": req.doc_id, "chunks_created": chunks_count}


@router.get("/rag/documents")
def list_documents():
    kb = get_rag_knowledge_base()
    return kb.vector_store.list_documents()


@router.post("/rag/search")
def search_rag(req: RAGSearchRequest):
    kb = get_rag_knowledge_base()
    results = kb.retrieve(query=req.query, top_k=req.top_k)
    return [r.model_dump() for r in results]


# Evaluation Benchmark Endpoints (Phase 3)
@router.post("/evaluation/run")
def run_evaluation():
    raise HTTPException(status_code=501, detail="Phase 3")


@router.get("/evaluation/results")
def get_evaluation_results():
    raise HTTPException(status_code=501, detail="Phase 3")
