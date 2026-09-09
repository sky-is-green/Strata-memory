"""Shared baseline runner for S0.5.

Replays synthetic conversations through a local LLM backend and records the
"before" metrics (throughput, latency, context utilization, OOM events) for the
no-strata (LM Studio rolling) and naive-FIFO baselines. Both baselines are the
comparison points the strata must beat in S3/S5.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

from cortex.baselines.metrics import (
    BaselineMetrics,
    compute_baseline_pes,
    estimate_tokens,
)

BuildMessagesFn = Callable[[list[dict]], list[dict]]

FIFO_WINDOW_TOKENS = 4000  # naive FIFO truncation cap (S0.5 baseline 2)


def load_conversations(directory: str | Path) -> list[dict]:
    """Load all conversation JSON files from *directory*, sorted by name."""
    directory = Path(directory)
    conversations = []
    for path in sorted(directory.glob("*.json")):
        conversations.append(json.loads(path.read_text(encoding="utf-8")))
    return conversations


def build_lmstudio_messages(history: list[dict]) -> list[dict]:
    """Full rolling context: send the entire history (backend handles the window)."""
    return history


def build_fifo_messages(history: list[dict]) -> list[dict]:
    """Naive FIFO truncation to FIFO_WINDOW_TOKENS: keep the most recent messages
    that fit within the window (the current user turn is the newest, so it is
    always retained), dropping the oldest."""
    total = 0
    kept: list[dict] = []
    for msg in reversed(history):
        cost = estimate_tokens(msg.get("content", ""))
        if kept and total + cost > FIFO_WINDOW_TOKENS:
            continue
        kept.append(msg)
        total += cost
    return list(reversed(kept))


class MockClient:
    """Deterministic, network-free stand-in for offline harness verification."""

    def __init__(self, latency_ms: float = 25.0, tps: float = 35.0, seed: int = 1) -> None:
        import random

        self._rng = random.Random(seed)
        self._latency_ms = latency_ms
        self._tps = tps
        self._n = 0

    def generate(self, messages: list[dict], sampling_params: Optional[dict] = None):
        from cortex.baselines.lm_studio_client import GenerationResult

        self._n += 1
        prompt_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)
        completion = self._rng.randint(120, 220)
        latency_ms = self._latency_ms + self._rng.uniform(-3, 3)
        return GenerationResult(
            text=f"[mock response #{self._n}]",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion,
            total_tokens=prompt_tokens + completion,
            latency_ms=max(1.0, latency_ms),
            model="mock",
        )


def run_baseline(
    conversations: list[dict],
    client,
    mode: str,
    build_messages: BuildMessagesFn,
    baseline_tps: float = 30.0,
    max_context: int = 32768,
    max_tokens: Optional[int] = None,
) -> list[BaselineMetrics]:
    """Run every conversation and return per-conversation BaselineMetrics."""
    results: list[BaselineMetrics] = []
    sampling = {"max_tokens": max_tokens} if max_tokens else None

    for conv in conversations:
        conv_id = conv.get("conversation_id", "unknown")
        history: list[dict] = []
        tps_list: list[float] = []
        latency_list: list[float] = []
        prompt_tokens: list[int] = []
        oom_events = 0
        errors = 0

        for turn in conv.get("turns", []):
            history.append(turn)
            if turn.get("role") != "user":
                continue
            try:
                messages = build_messages(history)
                result = client.generate(messages, sampling)
                tps = (
                    result.completion_tokens / (result.latency_ms / 1000.0)
                    if result.latency_ms > 0
                    else 0.0
                )
                tps_list.append(tps)
                latency_list.append(result.latency_ms)
                prompt_tokens.append(result.prompt_tokens)
            except Exception as exc:  # noqa: BLE001
                if client.is_oom(exc):
                    oom_events += 1
                else:
                    errors += 1

        if not latency_list:
            latency_list = [0.0]
            tps_list = [0.0]
        if not prompt_tokens:
            prompt_tokens = [0]

        avg_latency = sum(latency_list) / len(latency_list)
        avg_tps = sum(tps_list) / len(tps_list)
        max_prompt = max(prompt_tokens)
        utilization = min(1.0, max_prompt / max_context)
        pes = compute_baseline_pes(
            avg_latency_ms=avg_latency,
            actual_tps=avg_tps,
            baseline_tps=baseline_tps,
            budget_used=max_prompt,
            budget_total=max_context,
        )

        results.append(
            BaselineMetrics(
                conversation_id=conv_id,
                total_turns=len([t for t in conv.get("turns", []) if t.get("role") == "user"]),
                mode=mode,
                avg_tokens_per_sec=round(avg_tps, 2),
                last_prompt_tokens=prompt_tokens[-1],
                max_prompt_tokens=max_prompt,
                context_utilization=round(utilization, 3),
                task_completed=None,
                oom_events=oom_events,
                errors=errors,
                avg_latency_ms=round(avg_latency, 2),
                pes=round(pes, 2),
                config={"window": "rolling" if mode == "lm_studio" else "fifo"},
            )
        )

    return results


def elapsed_s() -> float:
    return time.monotonic()