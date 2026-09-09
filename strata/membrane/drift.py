"""Topic drift detection (membrane layer).

Detects when the conversation topic changes significantly and flags a context
reset. Drift score = 1 - cosine(recent_embedding, historical_embedding), where
"recent" is the last few turns and "historical" is the average-ish embedding of
the older half of the store. When drift exceeds threshold, the assembler applies
drift penalties to old-topic chunks (Membrane-before-Retention).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class DriftResult:
    drift_score: float
    should_reset: bool


class TopicDriftDetector:
    DRIFT_THRESHOLD = 0.6

    def __init__(self, embed_fn: Optional[Callable[[str], object]] = None,
                 threshold: float = 0.6) -> None:
        # embed_fn(text) -> vector; if absent, falls back to a drone's .embed().
        self.embed_fn = embed_fn
        self.threshold = threshold

    def check(self, recent_chunks: list, all_chunks: list, drone=None) -> DriftResult:
        if len(all_chunks) < 5 or len(recent_chunks) < 1:
            return DriftResult(drift_score=0.0, should_reset=False)

        recent_text = " ".join(c.content for c in recent_chunks)
        historical_text = " ".join(c.content for c in all_chunks[: len(all_chunks) // 2])

        embed = self._resolve_embed(drone)
        recent_emb = embed(recent_text)
        hist_emb = embed(historical_text)

        sim = self._cosine(recent_emb, hist_emb)
        drift_score = 1.0 - sim
        return DriftResult(
            drift_score=float(drift_score),
            should_reset=drift_score > self.threshold,
        )

    def _resolve_embed(self, drone):
        if self.embed_fn is not None:
            return self.embed_fn
        if drone is None or not hasattr(drone, "embed"):
            raise ValueError("drift detector needs embed_fn or a drone with .embed()")
        return drone.embed

    @staticmethod
    def _cosine(a, b) -> float:
        import numpy as np

        a = np.asarray(a, dtype=float).reshape(-1)
        b = np.asarray(b, dtype=float).reshape(-1)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(a @ b / denom)
