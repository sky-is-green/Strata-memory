"""End-to-end runner: drive a real conversation through the full strata pipeline
against an LLM backend (default LM Studio on localhost:1234).

For each user turn it: adds the turn to the context store, assembles a bounded
context via the full strata pipeline, sends it to the backend (with a stable pinned
prefix for llama.cpp automatic prefix caching), and records per-turn metrics
(latency, tokens, utilization, PES) plus an NDJSON event log.

Usage::

    # Live run against LM Studio (must be running with a model loaded):
    python -m cortex.e2e --conversation hivebench/tests/fixtures/generated/short_001.json

    # Offline verification of the harness (no backend / no model needed):
    python -m cortex.e2e --conversation hivebench/tests/fixtures/generated/short_001.json --mock
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from backend.cache_manager import KVCacheManager
from backend.lmstudio import LMStudioBackend
from cortex.efficiency import EfficiencyScorer
from cortex.routing import DroneRouter, EscalationHandler
from focal.assembly import ContextAssembler
from focal.budget import AdaptiveBudget
from logs.event_logger import EventLogger
from membrane.dedup import ContextDeduplicator
from membrane.drift import TopicDriftDetector
from retention.store import ContextStore
from sieve.medium import MediumDrone
from sieve.ultra_small import UltraSmallDrone
from sieve.scores import ChunkScore

DEFAULT_PINNED_PREFIX = (
    "You are an assistant operating in the Hive Memory system. "
    "Answer using only the provided context and conversation history."
)


class FakeUltraSmall:
    """Fast deterministic drone for offline harness verification."""

    def score(self, query, chunks):
        return [ChunkScore(i, 0.9 if "JWT" in c else 0.2, 1.0) for i, c in enumerate(chunks)]

    def embed(self, text):
        return np.array([1.0, 0.0, 0.0])


class MockTransport:
    """Offline OpenAI-compatible transport returning a deterministic reply."""

    def __init__(self, latency_ms: float = 8.0) -> None:
        self.latency_ms = latency_ms

    def post(self, url, json=None, headers=None, timeout=None):
        self.last = (url, json)
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000.0)
        query = json["messages"][-1]["content"]
        return _Resp({"choices": [{"message": {"content": f"[mock] re: {query[:40]}"}}]})

    def get(self, url, headers=None, timeout=None):
        return _Resp({"data": []})


class _Resp:
    ok = True

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


class EndToEndRunner:
    def __init__(
        self,
        backend,
        ultra_small=None,
        medium=None,
        logger=None,
        pinned_prefix: str = DEFAULT_PINNED_PREFIX,
        max_context: int = 8192,
    ) -> None:
        self.backend = backend
        self.ultra = ultra_small if ultra_small is not None else UltraSmallDrone()
        self.medium = medium if medium is not None else MediumDrone(
            score_pair_fn=lambda q, c: 0.5
        )
        self.logger = logger
        self.pinned_prefix = pinned_prefix
        self.max_context = max_context
        self.manager = KVCacheManager(backend)
        self.assembler = ContextAssembler()
        self.scorer = EfficiencyScorer()
        self.dedup = ContextDeduplicator()
        self.drift = TopicDriftDetector(embed_fn=self.ultra.embed)

    def run(self, conversation: dict, max_turns: Optional[int] = None) -> dict:
        store = ContextStore(embed_fn=self.ultra.embed)
        turn = 0
        pes_history = []
        turns_report = []

        for turn_data in conversation.get("turns", []):
            if turn_data.get("role") != "user":
                continue
            turn += 1
            if max_turns and turn > max_turns:
                break
            query = turn_data["content"]

            # --- assemble context (full strata pipeline) ---
            t0 = time.perf_counter()
            assembled = self.assembler.assemble(
                query=query, current_turn=turn, store=store,
                router=DroneRouter(), ultra_small=self.ultra, medium=self.medium,
                escalation=EscalationHandler(), dedup=self.dedup,
                drift_detector=self.drift, budget=AdaptiveBudget(),
                max_context=self.max_context,
            )
            assembly_ms = (time.perf_counter() - t0) * 1000.0

            # --- prefix-cache + generate ---
            cache_state = self.manager.update_cache(
                assembled.content, persistent_prefix=self.pinned_prefix
            )
            t1 = time.perf_counter()
            reply = self.backend.generate(assembled.content, query)
            gen_ms = (time.perf_counter() - t1) * 1000.0

            # --- metrics ---
            util = assembled.token_count / max(assembled.budget, 1)
            pes = self.scorer.compute(
                avg_latency_ms=assembly_ms + gen_ms,
                budget_used=assembled.token_count,
                budget_total=assembled.budget,
            ).composite
            pes_history.append(pes)

            # --- grow context with the exchange ---
            store.add_chunk(turn, query)
            store.add_chunk(turn, reply)

            turns_report.append({
                "turn": turn,
                "query": query,
                "reply": reply,
                "assembly_ms": round(assembly_ms, 2),
                "generation_ms": round(gen_ms, 2),
                "total_ms": round(assembly_ms + gen_ms, 2),
                "token_count": assembled.token_count,
                "budget": assembled.budget,
                "utilization": round(util, 3),
                "routed_to": assembled.routing_decision.route_to,
                "drift_detected": assembled.drift_detected,
                "cache_mode": cache_state["mode"],
                "pes": round(pes, 2),
            })

            if self.logger is not None:
                self._log_turn(assembled, query, assembly_ms, gen_ms)

        return {
            "conversation_id": conversation.get("conversation_id", "unknown"),
            "backend": type(self.backend).__name__,
            "pinned_prefix": self.pinned_prefix,
            "turns": turns_report,
            "aggregate": {
                "user_turns": len(turns_report),
                "avg_total_ms": round(sum(t["total_ms"] for t in turns_report) / max(len(turns_report), 1), 2),
                "avg_assembly_ms": round(sum(t["assembly_ms"] for t in turns_report) / max(len(turns_report), 1), 2),
                "avg_generation_ms": round(sum(t["generation_ms"] for t in turns_report) / max(len(turns_report), 1), 2),
                "avg_utilization": round(sum(t["utilization"] for t in turns_report) / max(len(turns_report), 1), 3),
                "pes": {
                    "min": round(min(pes_history), 2) if pes_history else None,
                    "mean": round(sum(pes_history) / max(len(pes_history), 1), 2) if pes_history else None,
                },
                "drift_events": sum(1 for t in turns_report if t["drift_detected"]),
            },
        }

    def _log_turn(self, assembled, query, assembly_ms, gen_ms) -> None:
        self.logger.log("router", "task_classified", {
            "query_hash": hashlib.md5(query.encode("utf-8")).hexdigest()[:8],
            "routed_to": assembled.routing_decision.route_to,
            "latency_ms": 0,
        })
        self.logger.log("assembly", "context_assembled", {
            "total_tokens": assembled.token_count,
            "chunk_count": assembled.chunks_used,
            "chunk_ids": assembled.selected_chunk_ids,
            "budget_used": assembled.token_count,
            "budget_total": assembled.budget,
        }, latency_ms=assembly_ms + gen_ms)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hive end-to-end runner")
    parser.add_argument("--conversation", default="hivebench/tests/fixtures/generated/short_001.json")
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--base-url", default="http://localhost:1234")
    parser.add_argument("--model", default="")
    parser.add_argument("--pinned-prefix", default=DEFAULT_PINNED_PREFIX)
    parser.add_argument("--mock", action="store_true", help="run offline (no backend/model)")
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    conv_path = Path(args.conversation)
    if not conv_path.exists():
        print(f"conversation not found: {conv_path}")
        return 2
    conversation = json.loads(conv_path.read_text(encoding="utf-8"))

    if args.mock:
        ultra = FakeUltraSmall()
        backend = LMStudioBackend(base_url=args.base_url, model=args.model,
                                  transport=MockTransport())
    else:
        ultra = UltraSmallDrone()
        ultra._ensure_loaded()
        backend = LMStudioBackend(base_url=args.base_url, model=args.model)
        if not backend.health():
            print(
                f"LM Studio not reachable at {args.base_url}. "
                "Start LM Studio with a model loaded, or pass --mock to verify the harness."
            )
            return 3

    logger = EventLogger(log_dir="logs")
    runner = EndToEndRunner(
        backend, ultra_small=ultra, logger=logger, pinned_prefix=args.pinned_prefix
    )
    try:
        report = runner.run(conversation, max_turns=args.max_turns)
    finally:
        logger.flush()
        logger.close()

    output = Path(args.output) if args.output else Path("logs") / f"e2e_{int(time.time())}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    agg = report["aggregate"]
    print(f"E2E run: {report['conversation_id']} ({report['backend']})")
    print(f"  user turns      : {agg['user_turns']}")
    print(f"  avg assembly ms : {agg['avg_assembly_ms']}")
    print(f"  avg generation  : {agg['avg_generation_ms']} ms")
    print(f"  avg total       : {agg['avg_total_ms']} ms")
    print(f"  utilization     : {agg['avg_utilization']}")
    print(f"  PES min/mean    : {agg['pes']}")
    print(f"  drift events    : {agg['drift_events']}")
    print(f"Wrote {output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
