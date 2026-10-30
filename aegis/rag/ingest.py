"""CLI and ingestion pipeline for loading Markdown technical corpus into RAG."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from aegis.rag.chunker import DocumentChunk, DocumentChunker
from aegis.rag.retriever import RAGKnowledgeBase, get_rag_knowledge_base


def parse_markdown_with_frontmatter(content: str, file_path: str | Path = "<unknown>") -> Tuple[Dict[str, Any], str]:
    """Extracts front-matter metadata and body from a markdown document.

    Raises ValueError with the file name if front-matter is malformed.
    """
    frontmatter: Dict[str, Any] = {}
    body = content

    pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
    match = re.match(pattern, content, re.DOTALL)
    if match:
        raw_meta = match.group(1)
        body = match.group(2).strip()
        try:
            parsed = yaml.safe_load(raw_meta)
        except Exception as e:
            raise ValueError(f"Malformed front-matter in '{file_path}': {e}") from e
        if parsed is not None and not isinstance(parsed, dict):
            raise ValueError(
                f"Malformed front-matter in '{file_path}': expected YAML mapping, got {type(parsed).__name__}"
            )
        if isinstance(parsed, dict):
            frontmatter = parsed

    return frontmatter, body


def ingest_corpus(
    corpus_dir: str | Path,
    kb: Optional[RAGKnowledgeBase] = None,
) -> int:
    """
    Ingests all Markdown files in corpus_dir into the RAG vector store.
    Uses content hashes as chunk IDs for idempotent re-runs.
    """
    path = Path(corpus_dir)
    if not path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {path}")

    kb = kb or get_rag_knowledge_base()
    chunker = kb.chunker or DocumentChunker()
    total_chunks_indexed = 0

    md_files = sorted(list(path.rglob("*.md")))
    for file_path in md_files:
        content = file_path.read_text(encoding="utf-8")
        meta, body = parse_markdown_with_frontmatter(content, file_path=file_path)

        doc_id = file_path.stem
        title = meta.get("title", doc_id)
        source_url = meta.get("source_url", str(file_path.name))

        metadata = {
            "title": title,
            "source": source_url,
            "source_url": source_url,
            "file_path": str(file_path),
        }

        raw_chunks = chunker.chunk_text(text=body, doc_id=doc_id, metadata=metadata)
        if not raw_chunks:
            continue

        # Re-assign deterministic content-hash IDs for idempotence
        idempotent_chunks = []
        chunk_texts = []
        for c in raw_chunks:
            chunk_hash = hashlib.sha256(f"{doc_id}:{c.text}".encode("utf-8")).hexdigest()[:16]
            idempotent_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_hash,
                    doc_id=doc_id,
                    text=c.text,
                    char_start=c.char_start,
                    char_end=c.char_end,
                    metadata=c.metadata,
                )
            )
            chunk_texts.append(c.text)

        embeddings = kb.embeddings.embed_batch(chunk_texts)
        kb.vector_store.add_chunks(idempotent_chunks, embeddings)
        total_chunks_indexed += len(idempotent_chunks)

    return total_chunks_indexed


def main():
    parser = argparse.ArgumentParser(description="Ingest technical corpus Markdown files into Aegis RAG.")
    parser.add_argument(
        "corpus_path",
        nargs="?",
        default="corpus",
        help="Path to directory containing markdown files (default: corpus)",
    )
    args = parser.parse_args()

    count = ingest_corpus(args.corpus_path)
    print(f"Successfully ingested {count} chunks into Aegis RAG knowledge base.")


if __name__ == "__main__":
    main()
