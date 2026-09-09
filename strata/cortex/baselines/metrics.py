"""Baseline metric recording for S0.5.

Records per-conversation and aggregate metrics for the LM Studio (no-strata) and
naive-FIFO baselines into ``logs/baseline_*.json`` files. The BaselineMetrics
dataclass is the "before" measurement the strata is later compared against.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from cortex.efficiency import EfficiencyScorer

DEFAULT_MAX_CONTEXT = 32768  # assumed context window for Qwen3.6-family models


def estimate_tokens(text: str) -> int:
    """Approximate token count (no tokenizer dependency).

    Roughly one token per 4 characters for mixed prose/code. Accurate enough for
    baseline window sizing; the real tokenizer is applied by the LLM backend.
    """
    text = text or ""
    return max(1, len(text) // 4)


@dataclass
class BaselineMetrics:
    """Metrics recorded for a single baseline conversation."""

    conversation_id: str
    total_turns: int
    mode: str  # "lm_studio" (rolling) | "fifo" (4k truncation)
    avg_tokens_per_sec: float
    last_prompt_tokens: int
    max_prompt_tokens: int
    context_utilization: Optional[float]  # max_prompt_tokens / max_context
    task_completed: Optional[bool]  # human-judged after the run
    oom_events: int
    errors: int
    avg_latency_ms: float
    pes: float
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def record_baseline(
    metrics: list[BaselineMetrics],
    output_path: str | Path,
    max_context: int = DEFAULT_MAX_CONTEXT,
) -> dict:
    """Write metrics to ``output_path`` (NDJSON-friendly JSON) and return them.

    Includes the aggregate summary plus the per-conversation records so both the
    per-conversation and rollup numbers are queryable.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    aggregate = _aggregate(metrics)
    document = {
        "type": "baseline",
        "mode": metrics[0].mode if metrics else "unknown",
        "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "max_context": max_context,
        "conversation_count": len(metrics),
        "aggregate": aggregate,
        "conversations": [m.to_dict() for m in metrics],
    }
    output_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return document


def _aggregate(metrics: list[BaselineMetrics]) -> dict:
    if not metrics:
        return {}
    avg = lambda key: round(
        sum(getattr(m, key) or 0.0 for m in metrics) / len(metrics), 2
    )
    return {
        "avg_tokens_per_sec": avg("avg_tokens_per_sec"),
        "avg_latency_ms": avg("avg_latency_ms"),
        "avg_pes": avg("pes"),
        "avg_context_utilization": avg("context_utilization"),
        "total_oom_events": sum(m.oom_events for m in metrics),
        "total_errors": sum(m.errors for m in metrics),
        "completed_count": sum(1 for m in metrics if m.task_completed),
        "completed_with_oom": sum(1 for m in metrics if m.oom_events > 0),
    }


def compute_baseline_pes(
    avg_latency_ms: float,
    actual_tps: float,
    baseline_tps: float,
    budget_used: Optional[float] = None,
    budget_total: Optional[float] = None,
) -> float:
    """PES for baseline runs — only latency and throughput are measurable, so
    retrieval/routing/context components are absent and renormalized away."""
    return EfficiencyScorer().compute(
        avg_latency_ms=avg_latency_ms,
        actual_tps=actual_tps,
        baseline_tps=baseline_tps,
        budget_used=budget_used,
        budget_total=budget_total,
    ).composite


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def monotonic_ms() -> float:
    return time.monotonic() * 1000.0