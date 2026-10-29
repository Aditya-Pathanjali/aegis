"""FastAPI API routes for Aegis."""

from __future__ import annotations

import asyncio
import json
import queue
from typing import Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from aegis.api.schemas import (
    ApprovalActionRequest,
    DocumentUploadRequest,
    RAGSearchRequest,
    TaskCreateRequest,
)
from aegis.config import get_settings
from aegis.models.schemas import TaskResponse, TaskSubmission, TraceEvent
from aegis.rag.retriever import get_rag_knowledge_base
from aegis.services.orchestrator_service import get_orchestrator_service

router = APIRouter(prefix="/api")


# Sample sensor dataset pre-loaded for immediate demonstration
SAMPLE_SENSOR_CSV = """timestamp,vibration_rms_x,vibration_rms_y,bearing_temperature_c,operating_pressure_bar,shaft_speed_rpm,status_flag
2026-03-01 00:00:00,1.20,1.15,58.2,12.1,1495,NORMAL
2026-03-03 00:00:00,1.35,1.28,59.5,12.1,1494,NORMAL
2026-03-05 00:00:00,1.80,1.65,62.4,12.0,1492,NORMAL
2026-03-08 00:00:00,2.30,2.10,65.8,11.9,1491,NORMAL
2026-03-10 00:00:00,2.90,2.75,69.1,11.8,1490,WARNING
2026-03-12 00:00:00,3.30,3.12,72.4,11.6,1487,WARNING
2026-03-15 00:00:00,3.85,3.50,76.5,11.5,1482,ALERT
2026-03-18 00:00:00,4.40,4.10,80.1,11.3,1480,ALERT
2026-03-20 00:00:00,5.20,4.90,84.3,11.2,1475,CRITICAL
"""


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


@router.get("/sample-telemetry")
def get_sample_telemetry():
    """Returns sample industrial equipment telemetry for instant testing."""
    return {
        "filename": "turbine_journal_bearing_telemetry.csv",
        "csv_content": SAMPLE_SENSOR_CSV,
        "description": "500kW Slurry Pump Journal Bearing Telemetry showing accelerating RMS vibration and thermal surge.",
    }


@router.post("/tasks", response_model=Dict[str, str])
def submit_task(req: TaskCreateRequest):
    """Submits a new complex task for autonomous multi-agent decomposition and execution."""
    settings = get_settings()
    if not settings.provider:
        raise HTTPException(
            status_code=503,
            detail="No LLM provider configured. Set AEGIS_PROVIDER to 'anthropic' or 'openai'.",
        )
    service = get_orchestrator_service()
    task_id = service.submit_task(
        TaskSubmission(
            query=req.query,
            dataset_name=req.dataset_name,
            dataset_content=req.dataset_content,
            require_human_approval=req.require_human_approval,
            max_retries=req.max_retries,
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
async def stream_task_events(task_id: str):
    """Server-Sent Events (SSE) endpoint streaming real-time agent execution traces."""
    service = get_orchestrator_service()
    event_queue: queue.Queue = queue.Queue()

    # Callback to push into queue
    def on_event(event: TraceEvent):
        event_queue.put(event)

    service.subscribe(task_id, on_event)

    # First send existing historical events
    state = service.state_store.get_state(task_id)
    existing_events = state.get("trace_events", []) if state else []

    async def event_generator():
        # Send historical
        for ev in existing_events:
            yield f"data: {ev.model_dump_json()}\n\n"

        # Stream new events
        while True:
            try:
                # Check for new event non-blocking
                ev = event_queue.get_nowait()
                yield f"data: {ev.model_dump_json()}\n\n"
            except queue.Empty:
                # Check if task is finished
                current_state = service.state_store.get_state(task_id)
                status = current_state.get("status") if current_state else None
                if status in ["completed", "failed", "rejected", "waiting_approval"]:
                    yield f"data: {json.dumps({'event_type': 'stream_end', 'status': status})}\n\n"
                    break
                await asyncio.sleep(0.3)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/tasks/{task_id}/approval", response_model=TaskResponse)
def handle_human_approval(task_id: str, req: ApprovalActionRequest):
    """Handles Human-in-the-Loop approval decision (Approve, Reject, Request Re-analysis)."""
    service = get_orchestrator_service()
    try:
        updated_task = service.resolve_human_approval(
            task_id=task_id,
            action=req.action,
            feedback=req.feedback,
        )
        return updated_task
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
