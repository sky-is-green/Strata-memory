"""Ultra-small drone: paraphrase-MiniLM-L3-v2 (~60MB, CPU-capable).

Swapped from all-MiniLM-L6-v2 (2026-08-23, B-avenue footprint decision):
the encoder probe showed L3-v2 matches-or-beats L6 on the hard live 264-pair
curve (top-1/3/5) at ~2.4x the scoring speed, and the retrieval-ceiling
research (B1-B3: bge-m3, contrastive tuning) proved encoder choice does not
move the precision ceiling — so the smallest encoder that holds retrieval
quality is the right default. L6 remains available via HiveConfig.ultra_model.

Fast semantic similarity scoring with a confidence estimate. Confidence is
computed as ``1 - normalized_std_dev`` across 3 forward passes with dropout
(MC-dropout). Note: at inference SentenceTransformer disables dropout, so a
stock model yields confidence ~1.0; the injected-encoder path (used in tests and
in the escalation design) is where the variance signal is meaningful.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from sieve.embedding_cache import EmbeddingCache
from sieve.scores import ChunkScore
from sieve.vocabulary import Vocabulary

COSINE_EPS = 1e-9


def _guard_pyarrow_parquet() -> None:
    """Workaround for hosts whose AppControl/WDAC policy blocks pyarrow's
    parquet DLL (``_parquet.pyd``): sentence-transformers 6.x imports
    ``datasets`` unconditionally at package import time, and ``datasets``
    imports ``pyarrow.parquet`` at module level — even though inference never
    touches parquet. When the real module cannot load, a stub keeps the import
    chain alive. Machines without the policy block are unaffected (the real
    module wins)."""
    import sys
    import types

    try:
        import pyarrow.parquet  # noqa: F401
    except ImportError:
        if "pyarrow.parquet" not in sys.modules:
            sys.modules["pyarrow.parquet"] = types.ModuleType("pyarrow.parquet")


def cosine_similarity_rows(query: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query vector and each row of ``rows``."""
    dots = rows @ query
    q_norm = float(np.linalg.norm(query))
    row_norms = np.linalg.norm(rows, axis=1)
    denom = q_norm * row_norms + COSINE_EPS
    return dots / denom


class UltraSmallDrone:
    """Wraps paraphrase-MiniLM-L3-v2, produces relevance + confidence per chunk."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-MiniLM-L3-v2",
        device: Optional[str] = None,
        encode_fn: Optional[Callable] = None,
        vocab: Optional[Vocabulary] = None,
        vocab_boost: float = 0.15,
        cache: Optional[EmbeddingCache] = None,
        confidence_mode: str = "mcdropout",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._encode = encode_fn
        self._model = None
        self.vocab = vocab if vocab is not None else Vocabulary.load("code", "general")
        self.vocab_boost = vocab_boost
        self.cache = cache if cache is not None else EmbeddingCache()
        # Tag the cache with the model so persisted embeddings are never reused
        # across a model swap with a different dimension.
        self.cache.model = self._model_name
        if confidence_mode not in ("mcdropout", "single", "off"):
            raise ValueError(f"unknown confidence_mode: {confidence_mode!r}")
        self.confidence_mode = confidence_mode

    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._encode is not None:
            return
        _guard_pyarrow_parquet()
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._model_name, device=self._device)
        self._model.eval()
        self._encode = lambda texts: self._model.encode(texts, convert_to_numpy=True)

    def _embed(self, text: str) -> np.ndarray:
        return self.cache.get_or_compute(text, lambda t: self._encode([t])[0])

    def embed(self, text: str) -> np.ndarray:
        """Public embedding accessor (used by drift detection, dedup, etc.)."""
        self._ensure_loaded()
        return self._embed(text)

    def score(self, query: str, chunks: list[str]) -> list[ChunkScore]:
        """Score each chunk against the query, returning relevance + confidence."""
        self._ensure_loaded()
        if not chunks:
            return []

        q = self._embed(query)
        embs = np.stack([self._embed(c) for c in chunks])
        sims = cosine_similarity_rows(q, embs)

        confidences = [self._confidence(c) for c in chunks]

        results = []
        for i, (s, conf) in enumerate(zip(sims, confidences)):
            boosted = float(s) + self._boost(chunks[i])
            results.append(ChunkScore(i, boosted, conf))
        return results

    def _confidence(self, chunk: str) -> float:
        """Confidence from prediction variance.

        ``mcdropout``: std across 3 passes (most accurate, most expensive).
        ``single``: 1 pass (std 0 -> confidence 1.0).
        ``off``: skip the extra passes entirely (fastest; used under
        degradation / when confidence is not needed).
        """
        if self.confidence_mode == "off":
            return 1.0
        n = 1 if self.confidence_mode == "single" else 3
        passes = np.stack([self._encode([chunk])[0] for _ in range(n)])
        std = float(np.std(passes, axis=0).mean())
        conf = max(0.0, 1.0 - std / 0.1)
        return min(1.0, conf)

    def _boost(self, text: str) -> float:
        if self.vocab is not None and self.vocab.matches(text):
            return float(self.vocab_boost)
        return 0.0
