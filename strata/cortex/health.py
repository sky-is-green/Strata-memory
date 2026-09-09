"""Pipeline health monitor.

Continuously tracks queue depth, rolling drone latency, pending assemblies and a
rolling window of metrics, and evaluates congestion against thresholds (reusing
cortex.congestion.CongestionDetector). Runs on a background cadence in production;
here it exposes synchronous checks for tests and orchestration.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from cortex.congestion import CongestionDetector, CongestionReport


class RollingMetrics:
    """Rolling window + rolling drone-latency average."""

    def __init__(self, window_size: int = 100, latency_window: int = 10) -> None:
        self.window = deque(maxlen=window_size)
        self._latencies = deque(maxlen=latency_window)
        self.queue_depth = 0
        self.pending_assemblies = 0

    def record_drone_latency(self, ms: float) -> None:
        self._latencies.append(float(ms))

    @property
    def avg_drone_latency_ms(self) -> Optional[float]:
        if not self._latencies:
            return None
        return sum(self._latencies) / len(self._latencies)

    def record(self, key: str, value: float) -> None:
        self.window.append((key, value))


class PipelineHealthMonitor:
    """Evaluates pipeline health and triggers congestion reports."""

    def __init__(self, logger=None, detector: Optional[CongestionDetector] = None) -> None:
        self.logger = logger
        self.detector = detector or CongestionDetector()
        self.metrics = RollingMetrics(window_size=100)

    def check_congestion(self) -> CongestionReport:
        """Evaluate the three congestion signals and log if abnormal."""
        report = self.detector.evaluate(
            queue_depth=self.metrics.queue_depth,
            avg_drone_latency_ms=self.metrics.avg_drone_latency_ms,
            pending_assemblies=self.metrics.pending_assemblies,
        )
        if self.logger is not None and report.severity != "normal":
            self.logger.log("congestion", "congestion_detected", report.__dict__)
        return report
