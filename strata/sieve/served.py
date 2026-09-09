"""Served embedding drone — calls llama-server's /v1/embeddings endpoint.

Drop-in replacement for :class:`UltraSmallDrone` when an embedding model is
loaded in a managed llama-server instance (e.g. nomic-embed, bge-m3). The
strata's retrieval pipeline (dedup, drift, assembly) uses the same ``embed``
and ``score`` interface, so switching is a one-line change in the sidecar:

    drone = ServedEmbeddingDrone(base_url="http://127.0.0.1:1236/v1")

Advantages over in-process sentence-transformers:
- No torch/CPU dependency on the serving path
- VRAM-shared with the LLM (same llama-server process or a separate one)
- Model is swappable from the hub panel without restarting the sidecar

The API is intentionally identical: ``embed(text) -> np.ndarray`` and
``score(query, chunks) -> list[ChunkScore]``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import requests

from sieve.scores import ChunkScore
from sieve.ultra_small import cosine_similarity_rows


class ServedEmbeddingDrone:
    """Embedding + scoring via a llama-server /v1/embeddings endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str = "default",
        vocab=None,
        vocab_boost: float = 0.15,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.vocab = vocab
        self.vocab_boost = vocab_boost
        self.timeout = timeout

    def embed(self, text: str) -> np.ndarray:
        """Single-text embedding via POST /v1/embeddings."""
        resp = requests.post(
            f"{self.base_url}/v1/embeddings",
            json={"model": self.model, "input": [text]},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return np.array(data[0]["embedding"], dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Batch embedding — one HTTP round-trip for the whole list."""
        if not texts:
            return np.array([])
        resp = requests.post(
            f"{self.base_url}/v1/embeddings",
            json={"model": self.model, "input": texts},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        items = sorted(resp.json()["data"], key=lambda d: d["index"])
        return np.array([d["embedding"] for d in items], dtype=np.float32)

    def score(self, query: str, chunks: list[str]) -> list[ChunkScore]:
        """Cosine-similarity scoring, same contract as UltraSmallDrone."""
        if not chunks:
            return []
        q = self.embed(query)
        embs = self.embed_batch(chunks)
        if embs.size == 0:
            return []
        sims = cosine_similarity_rows(q, embs)
        results = []
        for i, s in enumerate(sims):
            relevance = float(np.clip(s + self._boost(chunks[i]), 0.0, 1.0))
            results.append(ChunkScore(index=i, relevance_score=relevance,
                                      confidence=1.0))
        return results

    def _boost(self, text: str) -> float:
        """Domain vocabulary relevance boost (same as UltraSmallDrone)."""
        if self.vocab is None or self.vocab_boost <= 0:
            return 0.0
        return self.vocab.score(text) * self.vocab_boost
