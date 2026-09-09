"""Predictive pre-loading.

Based on conversation patterns, predict which chunks will be needed next so their
embeddings / contents can be pre-computed or pre-fetched, reducing latency.

Patterns:
  - "debugging": error-related queries -> pre-load error/traceback chunks
  - "function_exploration": code-analysis queries -> pre-load function chunks
  - "alternating_topics": rapid topic switches -> pre-load chunks overlapping
    recent queries
"""

from __future__ import annotations

from typing import Optional

ERROR_TERMS = ("error", "fail", "exception", "traceback", "crash", "bug", "debug")
FUNCTION_TERMS = ("function", "method", "implement", "how does", "caller", "callee", "api")


class PredictivePreloader:
    def predict_next_context(self, recent_queries: list[str], store) -> list[str]:
        """Return chunk IDs likely to be needed next, in a stable order."""
        patterns = self._detect_patterns(recent_queries)
        predicted: list[str] = []

        for cid, chunk in store.chunks.items():
            content = chunk.content.lower()
            if "debugging" in patterns and any(t in content for t in ERROR_TERMS):
                predicted.append(cid)
            elif "function_exploration" in patterns and any(t in content for t in FUNCTION_TERMS):
                predicted.append(cid)

        if "alternating_topics" in patterns and recent_queries:
            words = [w for w in recent_queries[-1].lower().split() if len(w) > 4]
            for cid, chunk in store.chunks.items():
                if any(w in chunk.content.lower() for w in words):
                    predicted.append(cid)

        # Dedupe, preserving order.
        seen: set = set()
        out: list[str] = []
        for cid in predicted:
            if cid not in seen:
                seen.add(cid)
                out.append(cid)
        return out

    def _detect_patterns(self, queries: list[str]) -> set:
        patterns: set = set()
        if not queries:
            return patterns
        joined = " ".join(queries).lower()
        if any(t in joined for t in ERROR_TERMS):
            patterns.add("debugging")
        if any(t in joined for t in FUNCTION_TERMS):
            patterns.add("function_exploration")
        if len(queries) >= 3:
            stopwords = {
                "the", "a", "an", "and", "of", "to", "in", "on", "for",
                "with", "is", "it", "this", "that", "what", "how", "why",
            }

            def _words(q):
                return {w for w in q.lower().split() if w not in stopwords}

            overlaps = []
            for i in range(len(queries) - 1):
                a, b = _words(queries[i]), _words(queries[i + 1])
                union = len(a | b)
                overlaps.append(len(a & b) / union if union else 1.0)
            if max(overlaps) < 0.1:
                patterns.add("alternating_topics")
        return patterns
