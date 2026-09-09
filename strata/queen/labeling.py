"""Ground-truth labeling workflow.

Produces the labeled datasets the white paper's experiments need:

  - query-chunk relevance pairs (for P2/P6 retrieval precision/recall)
  - routing decision labels (for P8 routing accuracy)
  - eviction decision labels (for false-eviction rate)

Labels are *auto-derived* from the synthetic corpus structure (topic membership
and cross-references), which is deterministic and reproducible. The generated
JSON/CSV files are also the review surface for human or LLM-queen annotation:
override any ``*_auto`` field with a reviewed label.

Usage::

    python -m queen.labeling --output hivebench/tests/fixtures/labels
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from queen.topics import TOPICS

DEFAULT_LABEL_DIR = Path(__file__).resolve().parents[2] / "hivebench" / "tests" / "fixtures" / "labels"


# ---------------------------------------------------------------------------
# Topic inference (shared by several labelers)
# ---------------------------------------------------------------------------
def topic_of(text: str, topics: dict = TOPICS) -> str | None:
    """Best topic for a text, by domain-vocabulary substring overlap."""
    text = text.lower()
    best, best_score = None, 0
    for name, data in topics.items():
        terms = (
            data["feature"].split()
            + data["aspects"]
            + [str(k) for k in data["decisions"]]
            + [str(v) for v in data["decisions"].values()]
        )
        score = sum(1 for term in terms if term.lower() in text)
        if score > best_score:
            best, best_score = name, score
    return best if best_score > 0 else None


def _user_queries(conversation) -> list[str]:
    return [t["content"] for t in conversation["turns"] if t["role"] == "user"]


def _assistant_chunks(conversation) -> list[str]:
    return [t["content"] for t in conversation["turns"] if t["role"] == "assistant"]


# ---------------------------------------------------------------------------
# Labelers
# ---------------------------------------------------------------------------
def generate_query_chunk_pairs(conversations, n: int = 200, seed: int = 0) -> list[dict]:
    """n (query, chunk, relevant) pairs, balanced relevant/irrelevant."""
    rng = random.Random(seed)
    pairs = []
    for conv in conversations:
        queries = _user_queries(conv)
        chunks = _assistant_chunks(conv)
        for q in queries:
            qt = topic_of(q)
            if qt is None:
                continue
            same = [c for c in chunks if topic_of(c) == qt]
            diff = [c for c in chunks if topic_of(c) is not None and topic_of(c) != qt]
            for c in same:
                pairs.append({"query": q, "chunk": c, "relevant": True})
                if len(pairs) >= n:
                    break
            for c in diff:
                pairs.append({"query": q, "chunk": c, "relevant": False})
                if len(pairs) >= n:
                    break
            if len(pairs) >= n:
                break
        if len(pairs) >= n:
            break
    rng.shuffle(pairs)
    return pairs[:n]


def generate_routing_decision_labels(conversations, n: int = 200) -> list[dict]:
    """Label the 'optimal' route per query. Uses the heuristic router as the
    proxy queen until a real queen labels these (see note in the record)."""
    from cortex.routing import DroneRouter

    router = DroneRouter()
    labels = []
    for conv in conversations:
        for q in _user_queries(conv):
            decision = router.route(q)
            labels.append({
                "query": q,
                "optimal_route_auto": decision.route_to,
                "reviewed": None,
            })
            if len(labels) >= n:
                return labels
    return labels


def generate_eviction_labels(conversations, n: int = 100) -> list[dict]:
    """Label whether an assistant chunk is needed by a later turn."""
    labels = []
    for conv in conversations:
        turns = conv["turns"]
        for i, t in enumerate(turns):
            if t["role"] != "assistant":
                continue
            chunk = t["content"]
            ct = topic_of(chunk)
            later_queries = [x["content"] for x in turns[i + 1:] if x["role"] == "user"]
            needed = any(ct is not None and ct == topic_of(lq) for lq in later_queries)
            labels.append({"chunk": chunk, "needed_later_auto": bool(needed), "reviewed": None})
            if len(labels) >= n:
                return labels
    return labels


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def write_labels(labels: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, indent=2), encoding="utf-8")


def generate_all(conversations, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    qcp = generate_query_chunk_pairs(conversations, n=200)
    rd = generate_routing_decision_labels(conversations, n=200)
    ev = generate_eviction_labels(conversations, n=100)

    write_labels(qcp, output_dir / "query_chunk_pairs.json")
    write_labels(rd, output_dir / "routing_decisions.json")
    write_labels(ev, output_dir / "eviction_decisions.json")
    return {"query_chunk_pairs": len(qcp), "routing_decisions": len(rd), "evictions": len(ev)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ground-truth labeling workflow")
    parser.add_argument("--conversations", default="hivebench/tests/fixtures/generated")
    parser.add_argument("--output", default=str(DEFAULT_LABEL_DIR))
    args = parser.parse_args(argv)

    from cortex.baselines.runner import load_conversations

    conversations = load_conversations(args.conversations)
    counts = generate_all(conversations, Path(args.output))
    for name, count in counts.items():
        print(f"  {name}: {count}")
    print(f"Wrote labels to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
