"""RAG retrieval tool for internal technical knowledge base."""

from __future__ import annotations

from typing import Any, Dict

from aegis.rag.retriever import get_rag_knowledge_base
from aegis.tools.base import BaseTool


class RAGRetrieverTool(BaseTool):
    """Retrieves relevant evidence from the internal RAG knowledge base."""

    name = "rag_retriever"
    description = (
        "Queries the internal engineering knowledge base using dense vector embeddings and semantic search. "
        "Returns chunk citations, author sources, and relevance scores."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The technical inquiry or hypothesis to verify"},
            "top_k": {"type": "integer", "description": "Number of evidence chunks to retrieve", "default": 3},
        },
        "required": ["query"],
    }

    def _run(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        kb = get_rag_knowledge_base()
        evidence_list = kb.retrieve(query=query, top_k=top_k)

        formatted_evidence = []
        for e in evidence_list:
            formatted_evidence.append({
                "chunk_id": e.chunk_id,
                "title": e.title,
                "source": e.source,
                "content": e.content,
                "relevance_score": e.relevance_score,
            })

        return {
            "query": query,
            "evidence_count": len(formatted_evidence),
            "evidence": formatted_evidence,
        }
