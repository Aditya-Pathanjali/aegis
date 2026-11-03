"""SQLAlchemy 2 database models for Aegis persistence."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy 2 ORM entities in Aegis."""
    pass


class TaskModel(Base):
    __tablename__ = "tasks"

    task_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    dataset_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    dataset_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    state_data: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    trace_events: Mapped[List[TraceEventModel]] = relationship(
        "TraceEventModel",
        back_populates="task",
        cascade="all, delete-orphan",
        order_by="TraceEventModel.timestamp",
    )
    approval_request: Mapped[Optional[HumanApprovalModel]] = relationship(
        "HumanApprovalModel",
        back_populates="task",
        uselist=False,
        cascade="all, delete-orphan",
    )


class TraceEventModel(Base):
    __tablename__ = "trace_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("tasks.task_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    agent_role: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        index=True,
    )

    task: Mapped[TaskModel] = relationship("TaskModel", back_populates="trace_events")


class HumanApprovalModel(Base):
    __tablename__ = "human_approval_requests"

    task_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("tasks.task_id", ondelete="CASCADE"),
        primary_key=True,
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    required: Mapped[bool] = mapped_column(Boolean, default=True)
    proposed_actions: Mapped[List[str]] = mapped_column(JSON, default=list)
    requested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )
    decided_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    feedback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    task: Mapped[TaskModel] = relationship("TaskModel", back_populates="approval_request")


class DocumentModel(Base):
    __tablename__ = "aegis_documents"

    doc_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )

    chunks: Mapped[List[DocumentChunkModel]] = relationship(
        "DocumentChunkModel",
        back_populates="document",
        cascade="all, delete-orphan",
    )


class DocumentChunkModel(Base):
    __tablename__ = "aegis_document_chunks"

    chunk_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    doc_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("aegis_documents.doc_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column("metadata", JSON, nullable=False, default=dict)

    document: Mapped[DocumentModel] = relationship("DocumentModel", back_populates="chunks")

