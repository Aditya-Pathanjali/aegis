"""Docker integration tests proving PR 9 persistence guarantees."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from alembic.config import Config
from fastapi import FastAPI
from fastapi.testclient import TestClient
from psycopg import Connection
from psycopg.rows import dict_row

from aegis.admin import ingest_directory
from aegis.agents.critic import CriticRubricOutput
from aegis.agents.verification import JudgeVerdict, VerificationAgent
from aegis.api.main import lifespan
from aegis.config import Settings, reset_settings
from aegis.models.llm_outputs import AgentRole as LLMAgentRole
from aegis.models.llm_outputs import PlannedSubtask, PlanOutput
from aegis.models.schemas import (
    AgentExecutionResult,
    AgentRole,
    ApprovalAction,
    EvidenceRef,
    ExecutionPlan,
    Finding,
    SpecialistOutput,
    SubTask,
    SynthesizerOutput,
    TaskStatus,
    TaskSubmission,
    TraceEvent,
)
from aegis.orchestration.checkpointer import get_checkpointer, reset_checkpointer
from aegis.orchestration.serialization import get_checkpointer_serializer
from aegis.rag.chunker import DocumentChunker
from aegis.rag.embeddings import SentenceTransformerEmbeddings
from aegis.rag.retriever import RAGKnowledgeBase
from aegis.rag.vector_store import PostgresVectorStore, reset_vector_store
from aegis.services.orchestrator_service import (
    OrchestratorService,
    reset_orchestrator_service,
)
from aegis.tools.rag_tool import RAGSearchTool
from aegis.tools.registry import ToolRegistry
from alembic import command
from tests.fakes import FakeLLM, fake_text
from tests.worker_utils import run_worker_burst


@pytest.mark.docker
def test_migrations_apply_and_downgrade_cleanly(postgres_container):
    """Proves: migrations apply on an empty database, downgrade to base, and upgrade again cleanly."""
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", postgres_container)

    # 1. Downgrade to base
    command.downgrade(cfg, "base")

    # Verify tables dropped
    with Connection.connect(postgres_container, autocommit=True, row_factory=dict_row) as conn:
        tables = [
            r["table_name"]
            for r in conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
            ).fetchall()
        ]
        assert "tasks" not in tables
        assert "aegis_documents" not in tables
        assert "aegis_document_chunks" not in tables

    # 2. Upgrade to head
    command.upgrade(cfg, "head")

    # Verify tables created
    with Connection.connect(postgres_container, autocommit=True, row_factory=dict_row) as conn:
        tables = [
            r["table_name"]
            for r in conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
            ).fetchall()
        ]
        assert "tasks" in tables
        assert "trace_events" in tables
        assert "human_approval_requests" in tables
        assert "aegis_documents" in tables
        assert "aegis_document_chunks" in tables


@pytest.mark.docker
def test_human_gate_interrupted_task_approved_after_app_recreated(postgres_container):
    """Proves: a task waiting at the human gate is approved after the app is recreated, and reaches the report."""
    reset_orchestrator_service()
    reset_checkpointer()

    plan_output = PlanOutput(
        objective="Inspect bearing and pause for gate",
        rationale="Check safety thresholds",
        subtasks=[
            PlannedSubtask(
                id="st-1",
                description="Search literature guidelines",
                role=LLMAgentRole.RESEARCH,
                tools=["rag_search"],
            )
        ],
    )
    spec_out = SpecialistOutput(
        summary="Research complete.",
        findings=[],
        limitations=[],
    )
    critic_rubric = CriticRubricOutput(
        answers_question=9.0, uses_task_data=9.0, claims_verified=9.0, limitations_stated=9.0
    )
    synth_draft = SynthesizerOutput(
        title="Overhaul Audit",
        final_finding="Approved maintenance schedule confirmed.",
        key_conclusions=["All subsystems normal."],
        actionable_recommendations=["Sign off on operation."],
    )

    fake_llm = FakeLLM(
        responses=[plan_output, fake_text("Done."), spec_out, critic_rubric, synth_draft]
    )

    # 1. Run initial service instance
    service1 = OrchestratorService()
    task_id = service1.submit_task(
        TaskSubmission(
            query="Safety overhaul audit",
            require_human_approval=True,
            max_retries=0,
        )
    )

    # A worker runs the task until it pauses at WAITING_APPROVAL
    run_worker_burst(fake_llm)

    initial_resp = service1.task_store.get_response(task_id)
    assert initial_resp is not None
    assert initial_resp.status == TaskStatus.WAITING_APPROVAL

    # 2. Simulate complete app recreation / process restart:
    reset_orchestrator_service()
    reset_checkpointer()

    service2 = OrchestratorService()

    # Operator approves via recreated app; a fresh worker resumes from the Postgres checkpoint
    queued = service2.resolve_human_approval(
        task_id=task_id,
        action=ApprovalAction.APPROVE,
        feedback="Chief Engineer signed off after reboot.",
    )
    assert queued.status == TaskStatus.QUEUED
    run_worker_burst(fake_llm)

    updated = service2.task_store.get_response(task_id)
    assert updated is not None
    assert updated.status == TaskStatus.COMPLETED
    assert updated.report is not None
    assert "Approved maintenance schedule confirmed." in updated.report.executive_summary.final_finding
    assert updated.approval_request is not None
    assert updated.approval_request.status == "approved"
    assert updated.approval_request.feedback == "Chief Engineer signed off after reboot."


@pytest.mark.docker
def test_app_refuses_to_start_without_database_url_or_unreachable():
    """Proves: the app refuses to start without DATABASE_URL, or with the database unreachable."""
    try:
        # Case 1: DATABASE_URL is missing
        reset_settings(Settings(database_url=None))
        test_app = FastAPI(lifespan=lifespan)
        with pytest.raises(RuntimeError, match="DATABASE_URL is not configured"):
            with TestClient(test_app):
                pass

        # Case 2: DATABASE_URL is unreachable
        reset_settings(Settings(database_url="postgresql://aegis:secret@127.0.0.1:54329/nonexistent_db"))
        with pytest.raises(RuntimeError, match="is unreachable"):
            with TestClient(test_app):
                pass
    finally:
        reset_settings(Settings(database_url=None))


@pytest.mark.docker
def test_ingest_idempotency_and_rag_search_verification(tmp_path: Path, postgres_container: str):
    """Proves: running ingest twice gives the same chunk count, and rag_search returns a chunk ID that verification resolves."""
    reset_vector_store()

    # Create mixed .md and .txt corpus
    doc1 = tmp_path / "bearing_standard.md"
    doc1.write_text(
        "---\ntitle: ISO 10816 Bearing Standards\nsource_url: https://standards.iso.org/sample\n---\n\n"
        "ISO 10816 defines acceptable machinery vibration levels. Zone A vibration threshold is below 2.8 mm/s.",
        encoding="utf-8",
    )
    doc2 = tmp_path / "sensor_notes.txt"
    doc2.write_text(
        "Sensor 4 telemetry measured peak vibration amplitude of 2.1 mm/s under standard thermal load.",
        encoding="utf-8",
    )

    store = PostgresVectorStore(connection_url=postgres_container)
    store.clear()
    kb = RAGKnowledgeBase(vector_store=store, embeddings=SentenceTransformerEmbeddings(), chunker=DocumentChunker())

    # 1. Run ingest first time
    count1 = ingest_directory(tmp_path, kb=kb)
    assert count1 >= 2
    docs1 = store.list_documents()
    assert len(docs1) == 2

    with Connection.connect(postgres_container, autocommit=True, row_factory=dict_row) as conn:
        row1 = conn.execute("SELECT count(*) as total FROM aegis_document_chunks;").fetchone()
        assert row1 is not None
        initial_chunk_count = row1["total"]

    # 2. Run ingest second time: must give identical count without duplicates
    count2 = ingest_directory(tmp_path, kb=kb)
    assert count2 == count1

    with Connection.connect(postgres_container, autocommit=True, row_factory=dict_row) as conn:
        row2 = conn.execute("SELECT count(*) as total FROM aegis_document_chunks;").fetchone()
        assert row2 is not None
        assert row2["total"] == initial_chunk_count

    # 3. rag_search returns a chunk ID that verification resolves
    query = "Zone A vibration threshold"
    results = kb.retrieve(query=query, top_k=1)
    assert len(results) > 0
    top_chunk = results[0]
    assert top_chunk.chunk_id is not None
    assert "2.8" in top_chunk.content

    # Verification agent resolves evidence chunk ID
    spec_result = AgentExecutionResult(
        subtask_id="st-rag",
        agent_role=AgentRole.RESEARCH,
        output_summary="Found ISO standard.",
        findings=[
            Finding(
                statement="Zone A vibration threshold is below 2.8 mm/s.",
                evidence=[EvidenceRef(kind="chunk", ref_id=top_chunk.chunk_id)],
            )
        ],
        limitations=[],
    )

    judge = JudgeVerdict(verdict="supported", quote_span="2.8", rationale="Matches ISO chunk text")
    tool_reg = ToolRegistry()
    tool_reg.register(RAGSearchTool(retriever=kb))

    v_agent = VerificationAgent(
        llm_provider=FakeLLM(responses=[judge]),
        tool_registry=tool_reg,
    )

    audit = v_agent.execute([spec_result], original_query=query)
    assert audit.status == "passed"
    assert len(audit.claims) == 1
    assert audit.claims[0].status == "supported"
    assert top_chunk.chunk_id in audit.claims[0].evidence_sources


@pytest.mark.docker
def test_no_checkpoint_deserialization_warnings(postgres_container: str):
    """Proves: there is no checkpoint deserialization warning in the test output."""
    reset_checkpointer()
    serde = get_checkpointer_serializer()
    get_checkpointer(connection_url=postgres_container)

    # Capture warnings during round-trip workflow serialization
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")

        plan = ExecutionPlan(
            task_id="t-serde-test",
            original_query="Check pump",
            objective="Verify serde clean",
            subtasks=[SubTask(id="st-1", title="Sub", description="Desc", assigned_agent=AgentRole.RESEARCH)],
        )
        typed_tuple = serde.dumps_typed({
            "plan": plan,
            "status": TaskStatus.EXECUTING,
            "trace_events": [TraceEvent(event_id="ev-1", task_id="t-serde-test", event_type="test", message="msg")],
        })
        loaded = serde.loads_typed(typed_tuple)
        assert isinstance(loaded["plan"], ExecutionPlan)

        # Check for any deserialization warning
        unregistered_warnings = [
            w for w in recorded_warnings
            if "Deserializing unregistered type" in str(w.message)
        ]
        assert len(unregistered_warnings) == 0
