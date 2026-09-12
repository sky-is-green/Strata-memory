"""S0.5 baseline 2: naive FIFO truncation to 4k tokens, no strata.

Usage::

    python -m cortex.baselines.fifo_baseline [--conversations DIR] [--mock]

Records metrics to ``logs/baseline_fifo.json``. Pass ``--mock`` to verify the
harness end-to-end without a live LM Studio server.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cortex.baselines import metrics as m
from cortex.baselines import runner
from locations import generated_fixtures_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Naive FIFO 4k-truncation baseline")
    parser.add_argument(
        "--conversations",
        default=str(generated_fixtures_dir()),
        help="directory of conversation JSON files",
    )
    parser.add_argument(
        "--output", default="logs/baseline_fifo.json", help="output JSON path"
    )
    parser.add_argument(
        "--base-url", default="http://localhost:1234", help="LM Studio API base URL"
    )
    parser.add_argument("--model", default="", help="model id ('' = server default)")
    parser.add_argument("--baseline-tps", type=float, default=30.0)
    parser.add_argument(
        "--max-context", type=int, default=m.DEFAULT_MAX_CONTEXT, help="assumed window"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None, help="cap reply length (iteration)"
    )
    parser.add_argument("--mock", action="store_true", help="run with a mock client")
    args = parser.parse_args(argv)

    conversations = runner.load_conversations(args.conversations)
    if not conversations:
        print(f"No conversation files found in {args.conversations}")
        return 2

    if args.mock:
        client = runner.MockClient()
    else:
        from cortex.baselines.lm_studio_client import LMStudioClient

        client = LMStudioClient(base_url=args.base_url, model=args.model)
        if client.health() is None:
            print(
                f"LM Studio not reachable at {args.base_url}. "
                "Start it, or pass --mock to verify the harness."
            )
            return 3

    results = runner.run_baseline(
        conversations,
        client,
        mode="fifo",
        build_messages=runner.build_fifo_messages,
        baseline_tps=args.baseline_tps,
        max_context=args.max_context,
        max_tokens=args.max_tokens,
    )
    doc = m.record_baseline(results, args.output, max_context=args.max_context)

    agg = doc["aggregate"]
    print(f"Conversations: {doc['conversation_count']}")
    print(f"  avg tokens/sec : {agg['avg_tokens_per_sec']}")
    print(f"  avg latency ms : {agg['avg_latency_ms']}")
    print(f"  avg PES        : {agg['avg_pes']}")
    print(f"  utilization    : {agg['avg_context_utilization']}")
    print(f"  OOM events     : {agg['total_oom_events']}")
    print(f"  errors         : {agg['total_errors']}")
    print(f"Wrote {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())