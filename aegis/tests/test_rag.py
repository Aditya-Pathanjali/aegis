"""Unit tests for RAG pipeline components."""

from aegis.rag.chunker import DocumentChunker
from aegis.rag.embeddings import FastDenseEmbeddings
from aegis.rag.retriever import RAGKnowledgeBase
from aegis.rag.vector_store import InMemoryVectorStore


def test_chunker_basic():
    chunker = DocumentChunker(chunk_size=150, chunk_overlap=30)
    text = (
        "Paragraph 1 discusses industrial predictive maintenance and telemetry.\n\n"
        "Paragraph 2 discusses vibration spectrum analysis using Fast Fourier Transforms.\n\n"
        "Paragraph 3 discusses bearing lubrication degradation under extreme thermal stress."
    )
    chunks = chunker.chunk_text(text=text, doc_id="doc-test-1")
    assert len(chunks) >= 2
    assert all(c.doc_id == "doc-test-1" for c in chunks)


def test_embeddings_fast_dense():
    embedder = FastDenseEmbeddings(dimension=128)
    v1 = embedder.embed_text("vibration analysis and sensor failure")
    v2 = embedder.embed_text("vibration sensors in machinery")
    v3 = embedder.embed_text("culinary baking techniques and chocolate cake")

    assert len(v1) == 128
    # Semantic proximity between v1 and v2 should exceed v1 and v3
    import numpy as np
    dot12 = np.dot(v1, v2)
    dot13 = np.dot(v1, v3)
    assert dot12 > dot13


def test_vector_store_and_retrieval():
    kb = RAGKnowledgeBase(vector_store=InMemoryVectorStore())
    kb.index_document(
        doc_id="pump-specs",
        title="Centrifugal Slurry Pump Engineering Manual",
        text="Continuous operation max temperature is 80 degrees Celsius. Alarm set at 75C.",
    )
    results = kb.retrieve(query="What is the maximum operating temperature for slurry pump?", top_k=2)
    assert len(results) > 0
    assert "80 degrees" in results[0].content or "temperature" in results[0].content
