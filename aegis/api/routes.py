"""FastAPI API routes for Aegis."""

from __future__ import annotations

import asyncio
import io
import json
import queue
import re
from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse

from aegis.api.schemas import (
    ApprovalActionRequest,
    DocumentUploadRequest,
    RAGSearchRequest,
)
from aegis.config import Settings, get_settings
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


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Aegis Multi-Agent Orchestrator",
        "version": "1.0.0",
        "agents": [
            "PlannerAgent",
            "ResearchAgent",
            "DataAnalystAgent",
            "VerificationAgent",
            "CriticAgent",
            "SynthesizerAgent",
        ],
    }


@router.post("/tasks", response_model=Dict[str, str])
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
    dataset_name = None
    dataset_content = None
    require_human_approval = False
    max_retries = 2

    if "multipart/form-data" in content_type:
        form = await request.form()
        query = str(form.get("query", ""))
        dataset_name = form.get("dataset_name")
        req_app = form.get("require_human_approval", False)
        if isinstance(req_app, str):
            require_human_approval = req_app.lower() in ("true", "1")
        else:
            require_human_approval = bool(req_app)
        m_retries = form.get("max_retries")
        if m_retries is not None:
            try:
                max_retries = int(m_retries)
            except ValueError:
                pass

        file = form.get("file")
        if file is not None and hasattr(file, "read"):
            file_bytes = await file.read()
            if not dataset_name and hasattr(file, "filename"):
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
        dataset_name = body.get("dataset_name")
        raw_content = body.get("dataset_content")
        require_human_approval = body.get("require_human_approval", False)
        max_retries = body.get("max_retries", 2)
        if raw_content:
            dataset_content = _process_dataset_bytes(raw_content.encode("utf-8"), settings)

    if not query or not query.strip():
        raise HTTPException(status_code=422, detail="Query cannot be empty.")

    service = get_orchestrator_service()
    task_id = service.submit_task(
        TaskSubmission(
            query=query,
            dataset_name=dataset_name,
            dataset_content=dataset_content,
            require_human_approval=require_human_approval,
            max_retries=max_retries,
        )
    )
    return {"task_id": task_id, "status": "submitted"}


@router.get("/tasks", response_model=List[TaskResponse])
def list_tasks():
    service = get_orchestrator_service()
    return service.state_store.list_tasks()


@router.get("/tasks/{task_id}", response_model=TaskResponse)
def get_task(task_id: str):
    service = get_orchestrator_service()
    resp = service.state_store.get_response(task_id)
    if not resp:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    return resp


@router.get("/tasks/{task_id}/events")
async def stream_task_events(task_id: str, request: Request):
    """Server-Sent Events (SSE) endpoint streaming real-time agent execution traces."""
    service = get_orchestrator_service()
    event_queue: queue.Queue = queue.Queue()
    seen_event_ids: set[str] = set()

    # Callback to push into queue
    def on_event(event: TraceEvent):
        event_queue.put(event)

    service.subscribe(task_id, on_event)

    # First send existing historical events
    state = service.state_store.get_state(task_id)
    existing_events = state.get("trace_events", []) if state else []

    async def event_generator():
        try:
            # Send historical events with de-duplication
            for ev in existing_events:
                if ev.event_id not in seen_event_ids:
                    seen_event_ids.add(ev.event_id)
                    yield f"data: {ev.model_dump_json()}\n\n"

            # Stream new events
            while True:
                if await request.is_disconnected():
                    break
                try:
                    # Check for new event non-blocking
                    ev = event_queue.get_nowait()
                    if ev.event_id not in seen_event_ids:
                        seen_event_ids.add(ev.event_id)
                        yield f"data: {ev.model_dump_json()}\n\n"
                except queue.Empty:
                    # Check if task is finished
                    current_state = service.state_store.get_state(task_id)
                    status = current_state.get("status") if current_state else None
                    if status in ["completed", "failed", "rejected", "waiting_approval"]:
                        while not event_queue.empty():
                            ev = event_queue.get_nowait()
                            if ev.event_id not in seen_event_ids:
                                seen_event_ids.add(ev.event_id)
                                yield f"data: {ev.model_dump_json()}\n\n"
                        yield f"data: {json.dumps({'event_type': 'stream_end', 'status': status})}\n\n"
                        break
                    await asyncio.sleep(0.1)
        finally:
            service.unsubscribe(task_id, on_event)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/tasks/{task_id}/approve", response_model=TaskResponse)
@router.post("/tasks/{task_id}/approval", response_model=TaskResponse)
def handle_human_approval(task_id: str, req: ApprovalActionRequest, response: Response):
    """Handles Human-in-the-Loop approval decision (Approve, Reject, Request Re-analysis)."""
    service = get_orchestrator_service()
    state = service.state_store.get_state(task_id)
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
