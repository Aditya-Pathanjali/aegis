"""Admin CLI for administrative operations, corpus ingestion, and maintenance.

Usage:
    python -m aegis.admin ingest <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Optional

from aegis.rag.chunker import DocumentChunk, DocumentChunker
from aegis.rag.embeddings import SentenceTransformerEmbeddings
from aegis.rag.ingest import parse_markdown_with_frontmatter
from aegis.rag.retriever import RAGKnowledgeBase
from aegis.rag.vector_store import get_vector_store


def ingest_directory(
    directory: str | Path,
    kb: Optional[RAGKnowledgeBase] = None,
) -> int:
    """Ingests .md and .txt files from directory into PostgreSQL + pgvector with stable chunk IDs.

    Idempotent: safe to run multiple times without duplicating chunks.
    """
    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Target directory does not exist: {path}")

    knowledge_base = kb or RAGKnowledgeBase(
        vector_store=get_vector_store(),
        embeddings=SentenceTransformerEmbeddings(),
        chunker=DocumentChunker(),
    )
    chunker = knowledge_base.chunker or DocumentChunker()

    files = sorted(
        [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in (".md", ".txt")]
    )
    total_chunks = 0

    for file_path in files:
        content = file_path.read_text(encoding="utf-8")
        doc_id = file_path.stem

        if file_path.suffix.lower() == ".md":
            meta, body = parse_markdown_with_frontmatter(content, file_path=file_path)
            title = meta.get("title", doc_id)
            source = meta.get("source_url") or meta.get("source") or file_path.name
        else:
            meta = {}
            body = content.strip()
            title = doc_id
            source = file_path.name

        metadata = {
            "title": title,
            "source": str(source),
            "file_path": str(file_path),
            **meta,
        }

        raw_chunks = chunker.chunk_text(text=body, doc_id=doc_id, metadata=metadata)
        if not raw_chunks:
            continue

        stable_chunks: list[DocumentChunk] = []
        chunk_texts: list[str] = []

        for idx, rc in enumerate(raw_chunks):
            # Compute stable, deterministic chunk ID from doc_id, index, and text content
            chunk_hash = hashlib.sha256(f"{doc_id}:{idx}:{rc.text}".encode("utf-8")).hexdigest()[:16]
            stable_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_hash,
                    doc_id=doc_id,
                    text=rc.text,
                    char_start=rc.char_start,
                    char_end=rc.char_end,
                    metadata=rc.metadata,
                )
            )
            chunk_texts.append(rc.text)

        embeddings = knowledge_base.embeddings.embed_batch(chunk_texts)
        knowledge_base.vector_store.add_chunks(stable_chunks, embeddings)
        total_chunks += len(stable_chunks)

    return total_chunks


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Aegis Admin CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest .md and .txt documents into pgvector")
    ingest_parser.add_argument("directory", type=str, help="Directory path to scan and ingest")

    args = parser.parse_args(argv)

    if args.subcommand == "ingest":
        try:
            count = ingest_directory(args.directory)
            print(f"Successfully ingested {count} chunks into pgvector.")
            return 0
        except Exception as exc:
            print(f"Error during ingestion: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

