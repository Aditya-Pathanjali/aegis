"""Aegis RAG package."""

from aegis.rag.chunker import DocumentChunk, DocumentChunker
from aegis.rag.embeddings import BaseEmbeddings, FastDenseEmbeddings, OpenAIEmbeddings
from aegis.rag.retriever import RAGKnowledgeBase, RetrievedEvidence, get_rag_knowledge_base
from aegis.rag.vector_store import (
    BaseVectorStore,
    InMemoryVectorStore,
    PostgresVectorStore,
    VectorSearchResult,
    get_vector_store,
)

__all__ = [
    "DocumentChunk",
    "DocumentChunker",
    "BaseEmbeddings",
    "FastDenseEmbeddings",
    "OpenAIEmbeddings",
    "BaseVectorStore",
    "InMemoryVectorStore",
    "PostgresVectorStore",
    "VectorSearchResult",
    "get_vector_store",
    "RAGKnowledgeBase",
    "RetrievedEvidence",
    "get_rag_knowledge_base",
]
