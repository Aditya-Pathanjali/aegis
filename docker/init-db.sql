-- Aegis Database Schema with pgvector
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS aegis_documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    chunk_count INT DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS aegis_document_chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT REFERENCES aegis_documents(doc_id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding vector(128)
);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding_cosine 
ON aegis_document_chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE TABLE IF NOT EXISTS aegis_tasks (
    task_id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    status TEXT NOT NULL,
    plan JSONB,
    report JSONB,
    human_approval JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
