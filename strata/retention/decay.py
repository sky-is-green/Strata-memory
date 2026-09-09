"""Sharp decay matrix (retention layer).

Applies decay to the surviving (post-dedup) chunks on every assembly pass.

    effective_score = raw / (decay_multiplier ** age_factor)

where ``age_factor = min(turns_since_last_reference / 10, 3.0)``. Chunks saved by
the remembrance pass carry a higher decay multiplier and decay faster. Optional
drift penalties (from a Membrane topic reset) multiply the result. Stale chunks
(unreferenced for > STALE_THRESHOLD turns) get an extra decay acceleration so
"zombie context" doesn't linger.
"""

from __future__ import annotations

from typing import Optional


class DecayMatrix:
    STALE_THRESHOLD = 20
    STALE_FACTOR = 0.5

    def apply(
        self,
        chunks: list,
        current_turn: int,
        raw_scores: dict[str, float],
        drift_penalties: Optional[dict[str, float]] = None,
        exempt_ids: Optional[set[str]] = None,
    ) -> dict[str, float]:
        drift_penalties = drift_penalties or {}
        exempt_ids = exempt_ids or set()
        effective: dict[str, float] = {}
        for chunk in chunks:
            raw = raw_scores.get(chunk.id, 0.0)
            if chunk.id in exempt_ids:
                # Comb resurrections are explicit recalls, not zombies: they
                # compete on raw relevance, exempt from the stale factor and
                # drift penalties (the P4 measurement showed the stale factor
                # walls off every old fact at every multiplier).
                effective[chunk.id] = max(0.0, raw)
                continue
            age = current_turn - (chunk.last_referenced_turn or chunk.turn)
            age_factor = min(age / 10.0, 3.0)
            decayed = raw / (chunk.decay_multiplier ** age_factor)
            decayed *= drift_penalties.get(chunk.id, 1.0)
            if age > self.STALE_THRESHOLD:
                decayed *= self.STALE_FACTOR
            effective[chunk.id] = max(0.0, decayed)
        return effective
