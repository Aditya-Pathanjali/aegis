"""Document chunking with metadata preservation for RAG."""

from __future__ import annotations

import re
from typing import Any, Dict, List
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    char_start: int = 0
    char_end: int = 0


class DocumentChunker:
    """Chunks documents into semantic segments with overlap and metadata tracking."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        metadata: Dict[str, Any] | None = None,
    ) -> List[DocumentChunk]:
        clean_text = text.strip()
        if not clean_text:
            return []

        base_meta = metadata or {}
        # Split primarily by double newline (paragraphs), then single newlines, then sentences
        paragraphs = re.split(r"\n\s*\n", clean_text)
        chunks: List[DocumentChunk] = []

        current_text = ""
        current_start = 0
        char_pointer = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_text) + len(para) + 2 <= self.chunk_size:
                if current_text:
                    current_text += "\n\n" + para
                else:
                    current_text = para
                    current_start = char_pointer
            else:
                if current_text:
                    chunk_id = f"{doc_id}_chunk_{len(chunks)}"
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            text=current_text,
                            metadata=dict(base_meta),
                            char_start=current_start,
                            char_end=current_start + len(current_text),
                        )
                    )
                    # Prepare overlap
                    overlap_chars = current_text[-self.chunk_overlap :] if len(current_text) > self.chunk_overlap else ""
                    current_text = (overlap_chars + "\n\n" + para).strip() if overlap_chars else para
                    current_start = max(0, char_pointer - len(overlap_chars))
                else:
                    # Single paragraph is larger than chunk_size, split by words
                    words = para.split()
                    word_buf: List[str] = []
                    for w in words:
                        word_buf.append(w)
                        joined = " ".join(word_buf)
                        if len(joined) >= self.chunk_size:
                            chunk_id = f"{doc_id}_chunk_{len(chunks)}"
                            chunks.append(
                                DocumentChunk(
                                    chunk_id=chunk_id,
                                    doc_id=doc_id,
                                    text=joined,
                                    metadata=dict(base_meta),
                                    char_start=char_pointer,
                                    char_end=char_pointer + len(joined),
                                )
                            )
                            word_buf = word_buf[-max(1, self.chunk_overlap // 6) :]
                    if word_buf:
                        current_text = " ".join(word_buf)
                        current_start = char_pointer

            char_pointer += len(para) + 2

        if current_text:
            chunk_id = f"{doc_id}_chunk_{len(chunks)}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=current_text,
                    metadata=dict(base_meta),
                    char_start=current_start,
                    char_end=current_start + len(current_text),
                )
            )

        return chunks
