"""API request and response schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from aegis.models.schemas import ApprovalAction, TaskStatus


class TaskCreateRequest(BaseModel):
    query: str
    dataset_name: Optional[str] = None
    dataset_content: Optional[str] = None
    require_human_approval: bool = False
    max_retries: int = 2
    llm_provider: str = "auto"
    llm_model: Optional[str] = None
    api_key: Optional[str] = None


class ApprovalActionRequest(BaseModel):
    action: ApprovalAction
    feedback: Optional[str] = None


class DocumentUploadRequest(BaseModel):
    doc_id: str
    title: str
    text: str
    source: str = "manual_upload"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RAGSearchRequest(BaseModel):
    query: str
    top_k: int = 3
