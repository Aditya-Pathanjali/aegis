"""Unit tests for RAG pipeline components, embedding models, ingestion, and stores."""

import pytest

from aegis.rag.chunker import DocumentChunk, DocumentChunker
from aegis.rag.embeddings import HashingEmbeddings, SentenceTransformerEmbeddings
from aegis.rag.ingest import ingest_corpus
from aegis.rag.retriever import RAGKnowledgeBase
from aegis.rag.vector_store import InMemoryVectorStore, PostgresVectorStore


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


def test_embeddings_hashing():
    embedder = HashingEmbeddings(dimension=384)
    v1 = embedder.embed_text("vibration analysis and sensor failure")
    v2 = embedder.embed_text("vibration sensors in machinery")
    v3 = embedder.embed_text("culinary baking techniques and chocolate cake")

    assert len(v1) == 384
    import numpy as np

    dot12 = np.dot(v1, v2)
    dot13 = np.dot(v1, v3)
    assert dot12 > dot13


def test_retrieval_ranks_relevant_above_irrelevant():
    store = InMemoryVectorStore()
    embedder = SentenceTransformerEmbeddings()
    kb = RAGKnowledgeBase(vector_store=store, embeddings=embedder)

    kb.index_document(
        doc_id="pump-specs",
        title="Pump Manual",
        text="Continuous operation maximum temperature is 80 degrees Celsius. Vibration threshold is 4.5 mm/s.",
    )
    kb.index_document(
        doc_id="cooking-recipe",
        title="Culinary Guide",
        text="Bake sourdough bread at 220 degrees Celsius with steam. Knead the dough thoroughly.",
    )

    results = kb.retrieve(query="What is the pump vibration threshold and temperature limit?", top_k=2, min_relevance=0.0)
    assert len(results) == 2
    # Relevant pump document must be ranked first
    assert results[0].doc_id == "pump-specs"
    assert results[0].relevance_score > results[1].relevance_score


def test_get_chunk_works():
    store = InMemoryVectorStore()
    chunk = DocumentChunk(
        chunk_id="chunk-abc-123",
        doc_id="doc-1",
        text="Thermal sensor reading elevated above baseline.",
        metadata={"title": "Sensor Report", "source": "test"},
    )
    store.add_chunks([chunk], [[0.1] * 384])

    retrieved = store.get_chunk("chunk-abc-123")
    assert retrieved is not None
    assert retrieved.chunk_id == "chunk-abc-123"
    assert retrieved.doc_id == "doc-1"
    assert "Thermal sensor" in retrieved.text

    assert store.get_chunk("nonexistent-chunk") is None


def test_ingest_is_idempotent(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    f1 = corpus_dir / "doc1.md"
    f1.write_text(
        "---\ntitle: Machine Guidelines\nsource_url: https://example.com/guide\n---\n\n"
        "Guidelines for industrial pump telemetry and vibration monitoring.",
        encoding="utf-8",
    )

    f2 = corpus_dir / "doc2.md"
    f2.write_text(
        "---\ntitle: Bearing Standards\nsource_url: https://example.com/bearing\n---\n\n"
        "Standards for bearing wear detection and operational life bounds.",
        encoding="utf-8",
    )

    store = InMemoryVectorStore()
    embedder = HashingEmbeddings(dimension=384)
    kb = RAGKnowledgeBase(vector_store=store, embeddings=embedder)

    count1 = ingest_corpus(corpus_dir, kb=kb)
    assert count1 >= 2
    assert len(store.chunks) == count1
    assert len(store.list_documents()) == 2

    # Re-running ingest must not duplicate chunks
    count2 = ingest_corpus(corpus_dir, kb=kb)
    assert count2 == count1
    assert len(store.chunks) == count1
    assert len(store.list_documents()) == 2


def test_postgres_failure_raises():
    import psycopg

    # Connecting to an invalid port/host must raise without silent fallback
    with pytest.raises((psycopg.OperationalError, Exception)):
        PostgresVectorStore(connection_url="postgresql://invalid:invalid@127.0.0.1:54329/nonexistent_db")


def test_ingest_malformed_frontmatter_raises_with_filename(tmp_path):
    corpus_dir = tmp_path / "corpus_bad"
    corpus_dir.mkdir()

    bad_file = corpus_dir / "bad_syntax.md"
    bad_file.write_text(
        "---\ntitle: [Unclosed list\nkey: value\n---\n\nBody text here",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Malformed front-matter in '.*bad_syntax\.md'"):
        ingest_corpus(corpus_dir)

    # Test non-dict front-matter (e.g. YAML scalar or list)
    scalar_file = corpus_dir / "scalar_meta.md"
    scalar_file.write_text(
        "---\njust a string not a mapping\n---\n\nBody text",
        encoding="utf-8",
    )
    bad_file.unlink()  # remove first bad file to test second

    with pytest.raises(ValueError, match=r"Malformed front-matter in '.*scalar_meta\.md'"):
        ingest_corpus(corpus_dir)
