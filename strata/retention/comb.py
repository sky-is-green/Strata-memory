"""Comb (honeycomb) — the strata's surplus store (SSD tier).

Chunks the active store evicts are frozen here instead of dropped: content
the strata once selected as relevant may become relevant again after a topic
change returns (long-horizon topic return). The comb is JSONL-backed
(append-only puts, crash-safe: the file is the source of truth and the
in-memory index rebuilds from it), per-conversation, and its candidates
compete for the *same* token budget as the active store — it never enlarges
the context window.

On resurrection, comb records are exempt from the decay matrix's stale
factor and from drift penalties (explicit recalls, not zombies): the P4
measurement showed the stale factor (×0.5 at age > 20) makes old facts
unretrievable at every multiplier, so a returned topic can only win the
budget by competing on raw relevance.

Archiving happens on two triggers: (1) LRU eviction (store overflow) and
(2) *stale-out* — a once-curated chunk aging past the stale wall while not
selected for the current query moves from the active store into the comb
(so production runs with large ``max_chunks`` still archive surplus).

Conflict rule (v1): an archived fact can be outdated when its topic returns.
Dedup merges near-identical versions (>0.92 cosine); for the dissimilar
but conflicting range both versions surface and the newer store chunk is
ranked by recency-of-reference — the model sees both. A recency-aware
conflict rule is a documented future refinement, not v1 behavior.

The name: bees store surplus honey in the comb so it outlives the season;
the strata stores surplus context here so it outlives the topic.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from cortex.baselines.metrics import now_iso

_WORD = re.compile(r"[a-z0-9]{4,}")


@dataclass
class CombRecord:
    """A frozen chunk (chunk-shaped so assembly/dedup/selection treat it like one)."""

    id: str
    content: str
    turn: int
    fingerprint: str
    timestamp: str = ""
    decay_multiplier: float = 1.0
    times_saved: int = 0
    last_referenced_turn: Optional[int] = None
    relevance_history: list = field(default_factory=list)
    embedding: Optional[list] = None
    from_comb: bool = True

    @classmethod
    def from_chunk(cls, chunk, embedding: Optional[list] = None) -> "CombRecord":
        return cls(
            id=chunk.id,
            content=chunk.content,
            turn=chunk.turn,
            fingerprint=getattr(chunk, "fingerprint", ""),
            timestamp=getattr(chunk, "timestamp", "") or now_iso(),
            decay_multiplier=chunk.decay_multiplier,
            times_saved=chunk.times_saved,
            last_referenced_turn=chunk.last_referenced_turn or chunk.turn,
            relevance_history=[list(x) for x in getattr(chunk, "relevance_history", [])],
            embedding=(
                [float(x) for x in embedding] if embedding is not None else None
            ),
        )

    def to_json(self) -> dict:
        d = {
            "id": self.id,
            "content": self.content,
            "turn": self.turn,
            "fingerprint": self.fingerprint,
            "timestamp": self.timestamp,
            "decay_multiplier": self.decay_multiplier,
            "times_saved": self.times_saved,
            "last_referenced_turn": self.last_referenced_turn,
            "relevance_history": self.relevance_history,
        }
        if self.embedding is not None:
            d["embedding"] = self.embedding
        return d

    @classmethod
    def from_json(cls, d: dict) -> "CombRecord":
        return cls(
            id=d["id"],
            content=d["content"],
            turn=int(d["turn"]),
            fingerprint=d.get("fingerprint", ""),
            timestamp=d.get("timestamp", ""),
            decay_multiplier=float(d.get("decay_multiplier", 1.0)),
            times_saved=int(d.get("times_saved", 0)),
            last_referenced_turn=d.get("last_referenced_turn"),
            relevance_history=list(d.get("relevance_history", [])),
            embedding=d.get("embedding"),
        )


class CombStore:
    """Append-only JSONL surplus store with a rebuildable in-memory index."""

    def __init__(
        self,
        path: str | Path,
        max_records: int = 2000,
        embed_fn=None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_records = max_records
        self.embed_fn = embed_fn
        self._records: dict[str, CombRecord] = {}
        self.put_count = 0
        self._load()

    # ------------------------------------------------------------------
    def put(self, chunk, embedding: Optional[list] = None) -> bool:
        """Freeze a chunk into the comb (append-only write)."""
        record = CombRecord.from_chunk(chunk, embedding)
        self._records[record.id] = record
        self._append(record)
        self.put_count += 1
        if len(self._records) > self.max_records:
            self.prune()
        return True

    def retrieve(self, query: str, k: int = 3, drone=None) -> list[CombRecord]:
        """Top-k comb records for *query*, ranked by lexical word-overlap.

        Measured on the fixture's topic-return turns (``comb_probe``,
        2026-08-24): lexical ranking (recall@1 42% / @3 59% / @5 64% / @8 68%)
        beats the real L3-v2 drone's cosine ranking (22% / 31% / 35% / 39%) at
        every k — return queries lexically name their old topic, and the
        drone's semantic smoothing pulls in same-topic-but-wrong chunks.
        Lexical ranking is also ~1 ms instead of ~820 ms per gated turn (no
        drone pass). Ties break by recency (most recently referenced first).
        The *drone* parameter is kept for API compatibility; ranking is
        lexical by default.
        """
        qwords = {w for w in _WORD.findall(query.lower())}
        if not qwords:
            qwords = {query.lower()}
        ranked = sorted(
            (r for r in self._records.values() if self._overlap_count(r, qwords) > 0),
            key=lambda r: (self._overlap_count(r, qwords), r.last_referenced_turn or 0),
            reverse=True,
        )
        return ranked[:k]

    def touch(self, ids, turn: int) -> int:
        """Refresh last_referenced_turn for records actually selected this turn."""
        changed = 0
        for cid in ids:
            record = self._records.get(cid)
            if record is None:
                continue
            record.last_referenced_turn = max(record.last_referenced_turn or 0, turn)
            changed += 1
        if changed:
            self.flush()
        return changed

    def prune(self, max_records: Optional[int] = None,
              max_age_turns: Optional[int] = None, current_turn: Optional[int] = None) -> int:
        """Drop surplus records: least-recently-referenced over the count cap,
        and (when *max_age_turns* + *current_turn* are given) records unreferenced
        for more than *max_age_turns* — the comb forgets too (Postulate 2),
        just much slower than the store."""
        dropped = 0
        if current_turn is not None and max_age_turns is not None:
            cutoff = current_turn - max_age_turns
            for cid in [c for c, r in self._records.items()
                        if (r.last_referenced_turn or 0) < cutoff]:
                self._records.pop(cid, None)
                dropped += 1
        cap = max_records or self.max_records
        while len(self._records) > cap:
            oldest = min(self._records.values(), key=lambda r: r.last_referenced_turn or 0)
            self._records.pop(oldest.id, None)
            dropped += 1
        if dropped:
            self.flush()
        return dropped

    def flush(self) -> None:
        """Rewrite the file from the in-memory index (after mutations)."""
        with open(self.path, "w", encoding="utf-8") as f:
            for record in sorted(self._records.values(), key=lambda r: r.id):
                f.write(json.dumps(record.to_json()) + "\n")

    def close(self) -> None:
        self.flush()

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, cid: str) -> bool:
        return cid in self._records

    def all_records(self) -> list:
        """All frozen records (used by the P11 protocol's retrievability
        bookkeeping; the strata itself only reads via retrieve())."""
        return list(self._records.values())

    # ------------------------------------------------------------------
    def _append(self, record: CombRecord) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_json()) + "\n")

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = CombRecord.from_json(json.loads(line))
                except (ValueError, KeyError, TypeError):
                    continue
                self._records[record.id] = record

    @staticmethod
    def _overlaps(record: CombRecord, qwords: set[str]) -> bool:
        return CombStore._overlap_count(record, qwords) > 0

    @staticmethod
    def _overlap_count(record: CombRecord, qwords: set[str]) -> int:
        cwords = {w for w in _WORD.findall(record.content.lower())}
        return len(cwords & qwords)