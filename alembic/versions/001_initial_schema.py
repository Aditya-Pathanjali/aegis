"""001_initial_schema: Initial database schema for tasks, traces, approvals, and documents.

Revision ID: 001_initial
Revises: None
Create Date: 2026-11-03 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Ensure pgvector extension exists
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 1. Tasks table
    op.create_table(
        "tasks",
        sa.Column("task_id", sa.String(length=64), primary_key=True, nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("dataset_name", sa.String(length=255), nullable=True),
        sa.Column("dataset_content", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("state_data", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # 2. Trace events table
    op.create_table(
        "trace_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("event_id", sa.String(length=64), nullable=False),
        sa.Column(
            "task_id",
            sa.String(length=64),
            sa.ForeignKey("tasks.task_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("agent_role", sa.String(length=64), nullable=True),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("details", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_trace_events_event_id", "trace_events", ["event_id"])
    op.create_index("ix_trace_events_task_id", "trace_events", ["task_id"])
    op.create_index("ix_trace_events_timestamp", "trace_events", ["timestamp"])

    # 3. Human approval requests table
    op.create_table(
        "human_approval_requests",
        sa.Column(
            "task_id",
            sa.String(length=64),
            sa.ForeignKey("tasks.task_id", ondelete="CASCADE"),
            primary_key=True,
            nullable=False,
        ),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("required", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("proposed_actions", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
        sa.Column("requested_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("feedback", sa.Text(), nullable=True),
    )

    # 4. Documents table
    op.create_table(
        "aegis_documents",
        sa.Column("doc_id", sa.String(length=255), primary_key=True, nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("source", sa.Text(), nullable=True),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # 5. Document chunks table
    op.create_table(
        "aegis_document_chunks",
        sa.Column("chunk_id", sa.String(length=255), primary_key=True, nullable=False),
        sa.Column(
            "doc_id",
            sa.String(length=255),
            sa.ForeignKey("aegis_documents.doc_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
    )
    op.create_index("ix_aegis_document_chunks_doc_id", "aegis_document_chunks", ["doc_id"])

    # Vector column & HNSW index
    op.execute("ALTER TABLE aegis_document_chunks ADD COLUMN IF NOT EXISTS embedding vector(384);")
    op.execute(
        "CREATE INDEX IF NOT EXISTS aegis_chunks_hnsw_idx "
        "ON aegis_document_chunks USING hnsw (embedding vector_cosine_ops);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS aegis_chunks_hnsw_idx;")
    op.drop_index("ix_aegis_document_chunks_doc_id", table_name="aegis_document_chunks")
    op.drop_table("aegis_document_chunks")
    op.drop_table("aegis_documents")
    op.drop_table("human_approval_requests")
    op.drop_index("ix_trace_events_timestamp", table_name="trace_events")
    op.drop_index("ix_trace_events_task_id", table_name="trace_events")
    op.drop_index("ix_trace_events_event_id", table_name="trace_events")
    op.drop_table("trace_events")
    op.drop_table("tasks")

