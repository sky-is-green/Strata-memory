"""Context deduplication (membrane layer).

Detects semantic duplicates (cosine > 0.92) and keeps the most information-dense
version. Returns the surviving chunks plus a refresh_map so the kept chunk's
decay state can be refreshed (Membrane-before-Retention rule).
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

from sieve.vocabulary import Vocabulary


def _cosine_matrix(rows: list) -> np.ndarray:
    m = np.asarray(rows, dtype=float)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    sim = m @ m.T
    denom = norms @ norms.T
    denom = np.where(denom == 0, 1e-9, denom)
    return sim / denom


class ContextDeduplicator:
    DUPLICATE_THRESHOLD = 0.92

    def __init__(self, vocab: Optional[Vocabulary] = None, threshold: float = 0.92) -> None:
        self.vocab = vocab if vocab is not None else Vocabulary.load("code", "general")
        self.threshold = threshold

    def deduplicate(
        self, chunks: list, embeddings: dict
    ) -> tuple[list, dict[str, int]]:
        if len(chunks) <= 1:
            return chunks, {}

        emb_list = [embeddings[c.id] for c in chunks]
        sim = _cosine_matrix(emb_list)

        keep = set(range(len(chunks)))
        refresh_map: dict[str, int] = {}
        for i in range(len(chunks)):
            if i not in keep:
                continue
            for j in range(i + 1, len(chunks)):
                if j not in keep:
                    continue
                if sim[i][j] > self.threshold:
                    if self._info_density(chunks[i].content) >= self._info_density(
                        chunks[j].content
                    ):
                        keep_idx, discard = i, j
                    else:
                        keep_idx, discard = j, i
                    keep.discard(discard)
                    freshest = max(chunks[i].turn, chunks[j].turn)
                    refresh_map[chunks[keep_idx].id] = max(
                        refresh_map.get(chunks[keep_idx].id, 0), freshest
                    )

        return [chunks[i] for i in sorted(keep)], refresh_map

    def _info_density(self, content: str) -> float:
        words = re.findall(r"\w+", content)
        total_words = len(words) or 1
        term_count = len(self.vocab.matched_terms(content))
        sentences = re.split(r"(?<=[.!?])\s+", content.strip())
        avg_sent_len = sum(len(re.findall(r"\w+", s)) for s in sentences) / (
            len(sentences) or 1
        )
        return (term_count / total_words) * avg_sent_len
