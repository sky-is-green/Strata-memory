"""KV-cache management.

Two modes depending on the backend:

- **vLLM (surgical):** PagedAttention enables page-level KV-cache edits (pin
  persistent pages, evict on drift). The exact vLLM cache API is backend-version
  specific, so this is planned/stubbed for instrumentation.
- **LM Studio / llama.cpp (prefix caching):** no surgical KV-cache API is
  exposed over HTTP. Instead the manager exploits llama.cpp's *automatic prefix
  caching*: keep the pinned system context byte-identical and at the front of
  the prompt, so llama.cpp reuses its KV for that prefix every turn. The manager
  ensures the backend sends that prefix verbatim first, and tracks whether the
  prefix changed (which invalidates the cache).
"""

from __future__ import annotations


class KVCacheManager:
    def __init__(self, backend) -> None:
        self.backend = backend
        self.supports_surgical_edits = bool(
            getattr(backend, "supports_surgical_edits", False)
        )
        self._last_pinned_prefix = None

    def update_cache(self, assembled, persistent_prefix: str = "") -> dict:
        """Update the KV-cache for a new assembled context.

        Also configures the backend (``pinned_prefix``) so the next ``generate``
        call sends the stable prefix first. Returns a dict describing the action.
        """
        if self.supports_surgical_edits:
            return {
                "mode": "surgical",
                "backend": type(self.backend).__name__,
                "plan": self._plan_pages(assembled, persistent_prefix),
            }

        # LM Studio: automatic prefix caching.
        prefix_changed = (
            self._last_pinned_prefix is not None
            and persistent_prefix != self._last_pinned_prefix
        )
        self._last_pinned_prefix = persistent_prefix
        self.backend.pinned_prefix = persistent_prefix
        return {
            "mode": "prefix_caching",
            "backend": type(self.backend).__name__,
            "prefix_stable": not prefix_changed,
            "cache_invalidated": prefix_changed,
            "plan": {
                "pinned": persistent_prefix or None,
                "dynamic": assembled,
            },
        }

    def _plan_pages(self, assembled, persistent_prefix: str) -> dict:
        """vLLM surgical page layout (stub): pinned/cached/dynamic."""
        return {
            "pinned": [persistent_prefix] if persistent_prefix else [],
            "cached": [],
            "dynamic": [assembled],
            "invalidated": [],
        }
