"""RAG Knowledge Base Retriever with source attribution."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from aegis.rag.chunker import DocumentChunker
from aegis.rag.embeddings import BaseEmbeddings, FastDenseEmbeddings
from aegis.rag.vector_store import BaseVectorStore, VectorSearchResult, get_vector_store


class RetrievedEvidence(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    source: str
    content: str
    relevance_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RAGKnowledgeBase:
    """Orchestrates indexing, embedding, and evidence retrieval."""

    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        embeddings: Optional[BaseEmbeddings] = None,
        chunker: Optional[DocumentChunker] = None,
    ):
        self.vector_store = vector_store or get_vector_store()
        self.embeddings = embeddings or FastDenseEmbeddings()
        self.chunker = chunker or DocumentChunker()

    def index_document(
        self,
        doc_id: str,
        title: str,
        text: str,
        source: str = "internal_kb",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        meta = metadata or {}
        meta["title"] = title
        meta["source"] = source

        chunks = self.chunker.chunk_text(text=text, doc_id=doc_id, metadata=meta)
        if not chunks:
            return 0

        chunk_texts = [c.text for c in chunks]
        embeddings = self.embeddings.embed_batch(chunk_texts)
        self.vector_store.add_chunks(chunks, embeddings)
        return len(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        min_relevance: float = 0.25,
    ) -> List[RetrievedEvidence]:
        q_emb = self.embeddings.embed_text(query)
        raw_results: List[VectorSearchResult] = self.vector_store.search(
            query_embedding=q_emb,
            top_k=top_k * 2,
            min_score=min_relevance,
        )

        query_tokens = set(query.lower().split())
        evidence_list: List[RetrievedEvidence] = []

        for r in raw_results:
            # Hybrid lexical boost
            chunk_tokens = set(r.text.lower().split())
            overlap = len(query_tokens.intersection(chunk_tokens))
            hybrid_score = min(1.0, r.score + (0.05 * overlap))

            evidence_list.append(
                RetrievedEvidence(
                    chunk_id=r.chunk_id,
                    doc_id=r.doc_id,
                    title=r.metadata.get("title", r.doc_id),
                    source=r.metadata.get("source", "Internal Repository"),
                    content=r.text,
                    relevance_score=round(hybrid_score, 4),
                    metadata=r.metadata,
                )
            )

        evidence_list.sort(key=lambda x: x.relevance_score, reverse=True)
        return evidence_list[:top_k]


_global_kb: Optional[RAGKnowledgeBase] = None


def get_rag_knowledge_base() -> RAGKnowledgeBase:
    global _global_kb
    if _global_kb is None:
        _global_kb = RAGKnowledgeBase()
    return _global_kb
