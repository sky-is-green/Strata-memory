"""Log query / report tool.

Reads NDJSON event logs and prints a summary (per-component counts, event types,
latency stats, route distribution, PES). Use ``--include-archive`` to fold in
gzipped archives.

Usage::

    python -m logs.query --dir logs
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np


def load_entries(log_dir: str | Path, include_archive: bool = False) -> list[dict]:
    log_dir = Path(log_dir)
    entries: list[dict] = []
    for path in sorted(log_dir.glob("events-*.ndjson")):
        with open(path, encoding="utf-8") as f:
            entries.extend(_parse_lines(f))
    if include_archive:
        for path in sorted((log_dir / "archive").glob("*.ndjson.gz")):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                entries.extend(_parse_lines(f))
    return entries


def _parse_lines(stream) -> list[dict]:
    out = []
    for line in stream:
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _pct(values, p) -> Optional[float]:
    return round(float(np.percentile(values, p)), 3) if values else None


def summarize(entries: list[dict]) -> dict:
    by_component = Counter(e.get("component") for e in entries)
    by_event = Counter(e.get("event_type") for e in entries)
    latency = [e["latency_ms"] for e in entries
               if isinstance(e.get("latency_ms"), (int, float))]
    routes = Counter(
        e["payload"].get("routed_to") for e in entries
        if e.get("event_type") == "task_classified" and e.get("payload")
    )
    pes = [e["payload"]["composite_score"] for e in entries
           if e.get("event_type") == "score_computed"
           and e.get("payload", {}).get("composite_score") is not None]

    return {
        "entries": len(entries),
        "by_component": dict(by_component),
        "by_event": dict(by_event),
        "latency_ms": {
            "avg": round(statistics.mean(latency), 3) if latency else None,
            "p50": _pct(latency, 50),
            "p95": _pct(latency, 95),
            "p99": _pct(latency, 99),
        },
        "routes": dict(routes),
        "pes": {
            "min": round(min(pes), 2) if pes else None,
            "mean": round(statistics.mean(pes), 2) if pes else None,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strata log query/report tool")
    parser.add_argument("--dir", default="logs")
    parser.add_argument("--include-archive", action="store_true")
    args = parser.parse_args(argv)

    entries = load_entries(args.dir, include_archive=args.include_archive)
    if not entries:
        print(f"no log entries found in {args.dir}")
        return 1
    print(json.dumps(summarize(entries), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
