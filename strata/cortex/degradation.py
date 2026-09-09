"""Graceful degradation.

When the pipeline is under stress, degrade gracefully instead of blocking or
crashing. Degradation levels escalate with congestion severity and recover one
level at a time (Appendix C.3 cooldown is handled by the caller cadence).

Level 0 (Normal):   full pipeline
Level 1 (Warning):  skip medium drone; skip dedup
Level 2 (Critical): skip remembrance; cached embeddings only; budget 2k
Level 3 (Emergency): fall back to naive FIFO truncation
"""

from __future__ import annotations

from cortex.congestion import CongestionReport

SEVERITY_TO_LEVEL = {"normal": 0, "warning": 1, "critical": 2, "emergency": 3}


class GracefulDegradation:
    def __init__(self) -> None:
        self.current_level = 0

    def update(self, congestion: CongestionReport) -> int:
        new_level = SEVERITY_TO_LEVEL.get(congestion.severity, 0)
        if new_level > self.current_level:
            self.current_level = new_level
        elif new_level < self.current_level:
            # Recover slowly: one level at a time (cooldown enforced by caller).
            self.current_level = max(self.current_level - 1, new_level)
        return self.current_level

    def reset(self) -> None:
        self.current_level = 0

    def should_skip_medium(self) -> bool:
        return self.current_level >= 1

    def should_skip_dedup(self) -> bool:
        return self.current_level >= 1

    def should_skip_remembrance(self) -> bool:
        return self.current_level >= 2

    def should_use_cached_only(self) -> bool:
        return self.current_level >= 2

    def should_fallback_fifo(self) -> bool:
        return self.current_level >= 3
