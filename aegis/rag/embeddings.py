"""Embedding generation for RAG vector search."""

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class BaseEmbeddings(ABC):
    """Base interface for embedding providers."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        pass


class FastDenseEmbeddings(BaseEmbeddings):
    """
    High-performance semantic-hash dense embedding generator (128-dim).
    Guarantees deterministic, normalized unit vectors with zero network dependencies.
    """

    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    def _hash_token(self, token: str) -> int:
        h = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(h[:4], "little")

    def embed_text(self, text: str) -> List[float]:
        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = text.lower().split()
        if not tokens:
            vec[0] = 1.0
            return vec.tolist()

        for i, token in enumerate(tokens):
            h = self._hash_token(token)
            idx = h % self.dimension
            sign = 1.0 if ((h >> 8) & 1) else -1.0
            weight = 1.0 + 1.0 / (i + 1)
            vec[idx] += sign * weight

            # Also hash character bigrams for subword robustness
            for j in range(len(token) - 1):
                bg = token[j : j + 2]
                bg_h = self._hash_token(bg)
                bg_idx = bg_h % self.dimension
                vec[bg_idx] += 0.3 * (1.0 if ((bg_h >> 4) & 1) else -1.0)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            vec[0] = 1.0

        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


class OpenAIEmbeddings(BaseEmbeddings):
    """OpenAI text embeddings client."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def embed_text(self, text: str) -> List[float]:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        resp = client.embeddings.create(input=text, model=self.model)
        return resp.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        resp = client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in resp.data]
