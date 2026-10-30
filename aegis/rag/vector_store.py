"""Vector database abstractions supporting PostgreSQL + pgvector and high-speed in-memory store."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

from aegis.rag.chunker import DocumentChunk


class VectorSearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseVectorStore(ABC):
    """Abstract Vector Store interface."""

    @abstractmethod
    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        pass

    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieves a single chunk by its ID."""
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5, min_score: float = 0.0) -> List[VectorSearchResult]:
        pass

    @abstractmethod
    def list_documents(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class InMemoryVectorStore(BaseVectorStore):
    """High-speed in-memory vector store with cosine similarity via NumPy."""

    def __init__(self):
        self.chunks: Dict[str, DocumentChunk] = {}
        self.vectors: Dict[str, np.ndarray] = {}
        self.doc_registry: Dict[str, Dict[str, Any]] = {}

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        for chunk, emb in zip(chunks, embeddings):
            is_new = chunk.chunk_id not in self.chunks
            self.chunks[chunk.chunk_id] = chunk
            vec = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self.vectors[chunk.chunk_id] = vec

            # Track documents
            if chunk.doc_id not in self.doc_registry:
                self.doc_registry[chunk.doc_id] = {
                    "doc_id": chunk.doc_id,
                    "title": chunk.metadata.get("title", chunk.doc_id),
                    "source": chunk.metadata.get("source", "manual_upload"),
                    "chunk_count": 0,
                }
            if is_new:
                self.doc_registry[chunk.doc_id]["chunk_count"] += 1

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        return self.chunks.get(chunk_id)

    def search(self, query_embedding: List[float], top_k: int = 5, min_score: float = 0.0) -> List[VectorSearchResult]:
        if not self.vectors:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        chunk_ids = list(self.vectors.keys())
        matrix = np.stack([self.vectors[cid] for cid in chunk_ids])
        scores = np.dot(matrix, q_vec)

        # Sort descending
        top_indices = np.argsort(scores)[::-1][:top_k]
        results: List[VectorSearchResult] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= min_score:
                cid = chunk_ids[idx]
                chunk = self.chunks[cid]
                results.append(
                    VectorSearchResult(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        text=chunk.text,
                        score=round(score, 4),
                        metadata=chunk.metadata,
                    )
                )
        return results

    def list_documents(self) -> List[Dict[str, Any]]:
        return list(self.doc_registry.values())

    def clear(self) -> None:
        self.chunks.clear()
        self.vectors.clear()
        self.doc_registry.clear()


class PostgresVectorStore(BaseVectorStore):
    """PostgreSQL + pgvector implementation using psycopg_pool and HNSW cosine index."""

    def __init__(self, connection_url: Optional[str] = None, timeout: float = 3.0):
        from psycopg_pool import ConnectionPool

        self.timeout = timeout
        conn_url = connection_url or os.getenv("DATABASE_URL") or "postgresql://aegis:aegis_secret@localhost:5432/aegis_db"
        self.connection_url: str = conn_url
        self.pool = ConnectionPool(self.connection_url, min_size=1, max_size=10, open=True)
        self._init_db()

    def _init_db(self) -> None:
        with self.pool.connection(timeout=self.timeout) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS aegis_documents (
                        doc_id TEXT PRIMARY KEY,
                        title TEXT,
                        source TEXT,
                        chunk_count INT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS aegis_document_chunks (
                        chunk_id TEXT PRIMARY KEY,
                        doc_id TEXT REFERENCES aegis_documents(doc_id) ON DELETE CASCADE,
                        text TEXT,
                        metadata JSONB,
                        embedding vector(384)
                    );
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS aegis_chunks_hnsw_idx
                    ON aegis_document_chunks USING hnsw (embedding vector_cosine_ops);
                """)
            conn.commit()

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                for chunk, emb in zip(chunks, embeddings):
                    cur.execute(
                        """
                        INSERT INTO aegis_documents (doc_id, title, source, chunk_count)
                        VALUES (%s, %s, %s, 1)
                        ON CONFLICT (doc_id) DO UPDATE SET chunk_count = aegis_documents.chunk_count + 1;
                        """,
                        (chunk.doc_id, chunk.metadata.get("title", chunk.doc_id), chunk.metadata.get("source", "upload")),
                    )
                    cur.execute(
                        """
                        INSERT INTO aegis_document_chunks (chunk_id, doc_id, text, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (chunk_id) DO UPDATE SET text = EXCLUDED.text, metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding;
                        """,
                        (chunk.chunk_id, chunk.doc_id, chunk.text, json.dumps(chunk.metadata), str(emb)),
                    )
            conn.commit()

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT chunk_id, doc_id, text, metadata FROM aegis_document_chunks WHERE chunk_id = %s;",
                    (chunk_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                meta = row[3] if isinstance(row[3], dict) else {}
                return DocumentChunk(
                    chunk_id=row[0],
                    doc_id=row[1],
                    text=row[2],
                    metadata=meta,
                )

    def search(self, query_embedding: List[float], top_k: int = 5, min_score: float = 0.0) -> List[VectorSearchResult]:
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chunk_id, doc_id, text, metadata, 1 - (embedding <=> %s::vector) AS score
                    FROM aegis_document_chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (str(query_embedding), str(query_embedding), top_k),
                )
                rows = cur.fetchall()
                results: List[VectorSearchResult] = []
                for row in rows:
                    score = float(row[4])
                    if score >= min_score:
                        meta = row[3] if isinstance(row[3], dict) else {}
                        results.append(
                            VectorSearchResult(
                                chunk_id=row[0],
                                doc_id=row[1],
                                text=row[2],
                                score=round(score, 4),
                                metadata=meta,
                            )
                        )
                return results

    def list_documents(self) -> List[Dict[str, Any]]:
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT doc_id, title, source, chunk_count FROM aegis_documents;")
                docs = []
                for row in cur.fetchall():
                    docs.append({
                        "doc_id": row[0],
                        "title": row[1],
                        "source": row[2],
                        "chunk_count": row[3],
                    })
                return docs

    def clear(self) -> None:
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE aegis_document_chunks, aegis_documents CASCADE;")
            conn.commit()

    def close(self) -> None:
        if hasattr(self, "pool") and self.pool is not None:
            self.pool.close()


# Global Singleton Vector Store instance
_global_vector_store: Optional[BaseVectorStore] = None


def get_vector_store(prefer_postgres: Optional[bool] = None) -> BaseVectorStore:
    global _global_vector_store
    if _global_vector_store is None:
        use_pg = (
            prefer_postgres
            if prefer_postgres is not None
            else (os.getenv("USE_POSTGRES", "false").lower() == "true")
        )
        if use_pg:
            _global_vector_store = PostgresVectorStore()
        else:
            _global_vector_store = InMemoryVectorStore()
    return _global_vector_store
