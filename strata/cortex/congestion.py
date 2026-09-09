"""Congestion detection for the strata pipeline.

See HIVE-HANDOFF.md §4.2 (S0.3) and Appendix C for the full specification.

Three signals, each with normal/warning/critical thresholds:

1. Queue depth        (context chunks waiting for drone processing)
2. Drone latency      (rolling average of the last 10 queries, ms)
3. Processing backlog (pending context assemblies since last user input)

Severity escalation: any critical signal -> "critical"; two or more critical
signals -> "emergency"; any warning signal -> "warning"; otherwise "normal".
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CongestionReport:
    """Result of a congestion evaluation, with escalating recommended actions."""

    severity: str  # normal | warning | critical | emergency
    queue_depth: int
    avg_drone_latency_ms: float
    pending_assemblies: int
    recommended_actions: list = field(default_factory=list)
    signal_breakdown: dict = field(default_factory=dict)


class RollingAverage:
    """Rolling mean over the last ``window_size`` values (dropping old ones)."""

    def __init__(self, window_size: int = 10) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self._window: deque = deque(maxlen=window_size)

    def push(self, value: float) -> None:
        self._window.append(float(value))

    @property
    def value(self) -> Optional[float]:
        if not self._window:
            return None
        return sum(self._window) / len(self._window)

    def __len__(self) -> int:
        return len(self._window)


class CongestionDetector:
    """Evaluates congestion signals and recommends escalating actions."""

    # Signal thresholds (S0.3 / Appendix C).
    QUEUE_WARNING = 6       # 6-15 chunks -> batching optimization
    QUEUE_CRITICAL = 15     # >15 chunks -> skip medium drone
    LATENCY_WARNING = 20.0  # 20-100 ms avg -> investigate GPU contention
    LATENCY_CRITICAL = 100.0  # >100 ms avg -> fall back to cached embeddings
    BACKLOG_WARNING = 1     # 1 pending assembly -> queue the message
    BACKLOG_CRITICAL = 1    # level-2 when value > 1, i.e. 2+ pending assemblies

    def evaluate(
        self,
        queue_depth: int = 0,
        avg_drone_latency_ms: Optional[float] = None,
        pending_assemblies: int = 0,
    ) -> CongestionReport:
        """Evaluate the three signals and return a report with actions."""
        queue_level = self._level(queue_depth, self.QUEUE_WARNING, self.QUEUE_CRITICAL)
        lat = 0.0 if avg_drone_latency_ms is None else float(avg_drone_latency_ms)
        latency_level = self._level(lat, self.LATENCY_WARNING, self.LATENCY_CRITICAL)
        backlog_level = self._level(
            pending_assemblies, self.BACKLOG_WARNING, self.BACKLOG_CRITICAL
        )

        actions = self._actions(queue_level, latency_level, backlog_level)

        criticals = sum(1 for lvl in (queue_level, latency_level, backlog_level)
                        if lvl == 2)
        if criticals >= 2:
            severity = "emergency"
        elif criticals == 1:
            severity = "critical"
        elif any(lvl == 1 for lvl in (queue_level, latency_level, backlog_level)):
            severity = "warning"
        else:
            severity = "normal"

        if severity == "critical" and "fallback_to_truncation" in actions:
            severity = "emergency"

        return CongestionReport(
            severity=severity,
            queue_depth=queue_depth,
            avg_drone_latency_ms=round(lat, 2),
            pending_assemblies=pending_assemblies,
            recommended_actions=actions,
            signal_breakdown={
                "queue": queue_level,
                "latency": latency_level,
                "backlog": backlog_level,
            },
        )

    @staticmethod
    def _level(value: float, warning: float, critical: float) -> int:
        """0 = normal, 1 = warning, 2 = critical."""
        if value > critical:
            return 2
        if value >= warning:
            return 1
        return 0

    @staticmethod
    def _actions(queue_level: int, latency_level: int, backlog_level: int) -> list:
        actions = []
        if queue_level >= 1:
            actions.append("batch_similar_chunks")
        if queue_level >= 2:
            actions.append("skip_medium_drone")
        if latency_level >= 1:
            actions.append("investigate_gpu_contention")
            actions.append("skip_medium_drone_low_confidence")
        if latency_level >= 2:
            actions.append("use_cached_embeddings")
        if backlog_level >= 1:
            actions.append("queue_messages")
        if backlog_level >= 2:
            actions.append("aggressive_compression")
        if queue_level >= 2 and backlog_level >= 2:
            actions.append("fallback_to_truncation")
        if queue_level == 2 and latency_level == 2:
            actions.append("fallback_to_truncation")
        if latency_level == 2 and backlog_level == 2:
            actions.append("fallback_to_truncation")
        return actions