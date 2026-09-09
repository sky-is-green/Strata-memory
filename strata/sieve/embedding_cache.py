"""Content-hash embedding cache.

Reuses embeddings for repeated chunks (e.g. a system prompt re-injected every
turn) instead of recomputing them. The hash includes the full content, so any
edit to a chunk produces a new hash and an automatic cache miss (Pitfall 6).

The cache is model-aware: persisted caches are tagged with the model that wrote
them, so swapping to a different-dimension model never reuses stale embeddings
(``load`` returns an empty cache on mismatch).
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional

import numpy as np


class EmbeddingCache:
    """LRU cache mapping content-hash -> numpy embedding vector."""

    def __init__(self, max_size: int = 10000, model: Optional[str] = None) -> None:
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._max_size = max_size
        self.model = model
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    @staticmethod
    def namespace(model: str, directory: str | Path) -> Path:
        """A per-model cache file path (avoids mixing different-dim vectors)."""
        safe = re.sub(r"[^A-Za-z0-9_.\-]", "_", model or "default")
        return Path(directory) / f"emb_cache_{safe}.npz"

    def get_or_compute(
        self, text: str, compute_fn: Callable[[str], np.ndarray]
    ) -> np.ndarray:
        h = self.key(text)
        if h in self._cache:
            self._cache.move_to_end(h)
            self.hits += 1
            return self._cache[h]
        emb = compute_fn(text)
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        self._cache[h] = emb
        self.misses += 1
        return emb

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------
    # Optional disk persistence (reuse embeddings across processes).
    # ------------------------------------------------------------------
    def persist(self, path: str | Path) -> None:
        """Write the cache to disk (NPZ). Requires equal-dimension vectors."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self._cache)
        if not keys:
            np.savez(path, keys=np.array([], dtype=object),
                     arr=np.zeros((0, 0)), model=np.array([self.model or ""]))
            return
        arr = np.stack([self._cache[k] for k in keys])
        np.savez(path, keys=np.array(keys), arr=arr, model=np.array([self.model or ""]))

    @classmethod
    def load(cls, path: str | Path, model: Optional[str] = None) -> "EmbeddingCache":
        """Restore a cache written by ``persist``.

        If a ``model`` is supplied and differs from the stored model, an empty
        cache is returned (dimensions may differ -> embeddings are incompatible).
        """
        import numpy as np

        data = np.load(path, allow_pickle=True)
        stored = str(data["model"][0]) if "model" in data.files else ""
        cache = cls(model=model or stored or None)
        if model and stored and model != stored:
            return cache  # model mismatch -> never reuse stale embeddings
        keys = data["keys"].tolist()
        arr = data["arr"]
        for k, vec in zip(keys, arr):
            cache._cache[str(k)] = np.asarray(vec)
        return cache
