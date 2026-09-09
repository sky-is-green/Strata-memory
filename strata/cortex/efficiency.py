"""Pipeline Efficiency Score (PES) — composite 0-100 pipeline health metric.

See HIVE-HANDOFF.md Appendix B for the full specification.

PES = 0.30*RetrievalPrecision
    + 0.20*RoutingAccuracy
    + 0.20*LatencyHealth
    + 0.15*ThroughputHealth
    + 0.15*ContextUtilization

Missing components are dropped and the remaining weights renormalized, so the
scorer can produce meaningful partial scores before all sources exist (e.g. the
S0 baseline where only latency and throughput are measurable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

WEIGHTS = {
    "retrieval_precision": 0.30,
    "routing_accuracy": 0.20,
    "latency_health": 0.20,
    "throughput_health": 0.15,
    "context_utilization": 0.15,
}

# Appendix B.3 — interpretation bands and their actions.
BANDS = [
    (80.0, "GREEN", "Normal operation. Log periodic snapshot."),
    (60.0, "YELLOW", "Log warning. Trigger shadow-mode A/B test of optimized config."),
    (40.0, "RED", "Trigger automated rollback to last known good config."),
    (0.0, "CRITICAL", "Emergency fallback to FIFO truncation. Alert operator."),
]


@dataclass
class ScoreResult:
    """Composite PES plus per-component breakdown."""

    composite: float
    breakdown: dict  # component -> weighted contribution
    band: str
    action: str
    active_components: list = field(default_factory=list)


def latency_health(avg_latency_ms: float) -> float:
    """max(0, 100 - (avg - 50) * 0.67), capped at 100. 50ms = 100, 200ms = 0."""
    if avg_latency_ms is None:
        return 0.0
    score = 100.0 - (float(avg_latency_ms) - 50.0) * 0.67
    return max(0.0, min(100.0, score))


def throughput_health(actual_tps: float, baseline_tps: float) -> float:
    """actual_tps / baseline_tps * 100, capped at 100."""
    if actual_tps is None or baseline_tps in (None, 0):
        return 0.0
    return max(0.0, min(100.0, float(actual_tps) / float(baseline_tps) * 100.0))


def context_utilization(budget_used: float, budget_total: float) -> float:
    """Appendix B.2: 100 in the 60-95% sweet spot, penalized outside."""
    if budget_used is None or budget_total in (None, 0):
        return 0.0
    utilization = float(budget_used) / float(budget_total)
    if utilization < 0.60:
        return utilization * 100.0
    if utilization > 0.95:
        return 100.0 - (utilization - 0.95) * 1000.0
    return 100.0


def interpret(score: float) -> tuple[str, str]:
    """Map a composite score to a (band, action) pair per Appendix B.3."""
    for threshold, band, action in BANDS:
        if score >= threshold:
            return band, action
    return BANDS[-1][1], BANDS[-1][2]


class EfficiencyScorer:
    """Computes the composite PES with correct weight application."""

    WEIGHTS = WEIGHTS

    def compute(
        self,
        retrieval_precision: Optional[float] = None,
        routing_accuracy: Optional[float] = None,
        avg_latency_ms: Optional[float] = None,
        actual_tps: Optional[float] = None,
        baseline_tps: Optional[float] = None,
        budget_used: Optional[float] = None,
        budget_total: Optional[float] = None,
    ) -> ScoreResult:
        """Compute the composite score from the components that are present.

        Components passed as ``None`` are dropped and the remaining weights are
        renormalized so they still sum to 1.0. If no component is present the
        composite is 0.0.
        """
        active = {}
        if retrieval_precision is not None:
            active["retrieval_precision"] = max(0.0, min(100.0, float(retrieval_precision)))
        if routing_accuracy is not None:
            active["routing_accuracy"] = max(0.0, min(100.0, float(routing_accuracy)))
        if avg_latency_ms is not None:
            active["latency_health"] = latency_health(avg_latency_ms)
        if actual_tps is not None and baseline_tps is not None:
            active["throughput_health"] = throughput_health(actual_tps, baseline_tps)
        if budget_used is not None and budget_total is not None:
            active["context_utilization"] = context_utilization(budget_used, budget_total)

        if not active:
            result = ScoreResult(0.0, {}, *interpret(0.0))
            return result

        weight_sum = sum(WEIGHTS[name] for name in active)
        composite = sum(WEIGHTS[name] * active[name] for name in active) / weight_sum

        breakdown = {name: round(WEIGHTS[name] * active[name], 3) for name in active}
        band, action = interpret(composite)
        return ScoreResult(
            composite=round(composite, 2),
            breakdown=breakdown,
            band=band,
            action=action,
            active_components=list(active),
        )