"""Context store (retention layer) — the central chunk repository.

Stores conversation context as discrete chunks with metadata: decay state,
times saved, last-referenced turn, and a content fingerprint for dedup. An
optional ``embed_fn`` provides embeddings lazily (used by dedup/assembly).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from cortex.baselines.metrics import now_iso
from retention.hygiene import (
    DEFAULT_INGEST_BLOCK_PREFIXES,
    DEFAULT_MAX_CHUNK_CHARS,
    content_fingerprint,
    prepare_for_storage,
)

@dataclass
class ContextChunk:
    id: str
    content: str
    turn: int
    fingerprint: str
    timestamp: str = ""
    decay_multiplier: float = 1.0
    times_saved: int = 0
    last_referenced_turn: Optional[int] = None
    relevance_history: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "turn": self.turn,
            "fingerprint": self.fingerprint,
            "timestamp": self.timestamp,
            "decay_multiplier": self.decay_multiplier,
            "times_saved": self.times_saved,
            "last_referenced_turn": self.last_referenced_turn,
            "relevance_history": [list(x) for x in self.relevance_history],
        }


class ContextStore:
    """Chunk-level conversation store with turn indexing and decay metadata."""

    def __init__(
        self,
        embed_fn: Optional[Callable[[str], object]] = None,
        max_chunks: Optional[int] = None,
        comb=None,
        comb_relevant_only: bool = True,
        sanitize: bool = True,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
        ingest_block_prefixes: Optional[list] = None,
    ) -> None:
        self.chunks: dict[str, ContextChunk] = {}
        self.turn_index: dict[int, list[str]] = {}
        self.embed_fn = embed_fn
        self._embeddings: dict = {}
        self.max_chunks = max_chunks
        self.comb = comb
        self.comb_relevant_only = comb_relevant_only
        self.sanitize = sanitize
        self.max_chunk_chars = max_chunk_chars
        self.ingest_block_prefixes: list = (
            list(DEFAULT_INGEST_BLOCK_PREFIXES)
            if ingest_block_prefixes is None
            else list(ingest_block_prefixes)
        )

    def add_chunk(self, turn: int, content: str, chunk_id: Optional[str] = None) -> Optional[str]:
        # Write-side hygiene (retention.hygiene): boilerplate dropped and
        # secrets/blobs redacted BEFORE fingerprinting, so dedup groups the
        # sanitized form and every downstream tier inherits clean data. A
        # fully-boilerplate turn stores zero chunks; a mixed turn keeps only
        # the remainder.
        content = prepare_for_storage(
            content, self.max_chunk_chars, self.ingest_block_prefixes,
            sanitize=self.sanitize,
        )
        if content is None:
            return None
        fingerprint = content_fingerprint(content)
        # RC1 fingerprint guard: a chunk with the same (sanitized) fingerprint
        # already in the store is a verbatim duplicate — do NOT append a
        # parallel copy that would split decay bookkeeping across clones.
        # Ingest repeats are a harness artifact, not remembrance events, so
        # times_saved (the remembrance ladder counter) is left untouched;
        # only recency is refreshed and the existing id returned.
        for existing in self.chunks.values():
            if existing.fingerprint == fingerprint:
                existing.last_referenced_turn = turn
                # The dedup turn still belongs to the conversation: register
                # the existing id so the recency lookback (get_recent_chunks /
                # drift) sees it. Without this, turns that only re-ingest old
                # content vanish from the lookback window.
                self.turn_index.setdefault(turn, []).append(existing.id)
                return existing.id
        cid = chunk_id or hashlib.md5(f"{turn}:{content}".encode("utf-8")).hexdigest()[:12]
        self.chunks[cid] = ContextChunk(
            id=cid,
            content=content,
            turn=turn,
            fingerprint=fingerprint,
            timestamp=now_iso(),
            last_referenced_turn=turn,
        )
        self.turn_index.setdefault(turn, []).append(cid)
        if self.max_chunks and len(self.chunks) > self.max_chunks:
            self._evict_overflow()
        return cid

    def _evict_overflow(self) -> None:
        """Enforce ``max_chunks`` by evicting the least-recently-referenced
        chunks (the same ordering ``get_deletion_candidates`` exposes).

        Without this the store grows without bound within a conversation —
        every chunk (plus its cached embedding and turn-index entry) is
        retained forever, so long chats accumulate memory linearly and the
        per-turn dedup/scoring cost grows with the whole history.
        """
        excess = len(self.chunks) - self.max_chunks
        if excess <= 0:
            return
        ordered = sorted(
            self.chunks.values(), key=lambda c: c.last_referenced_turn or 0
        )
        for chunk in ordered[:excess]:
            self._remove_chunk(chunk.id)

    def _remove_chunk(self, cid: str) -> None:
        chunk = self.chunks.pop(cid, None)
        if chunk is None:
            return
        self._embeddings.pop(cid, None)
        # A deduped chunk can be registered on several turns (RC1 guard), so
        # purge it from every turn list, not just its creation turn; leaving
        # ghosts behind makes get_recent_chunks raise KeyError after eviction.
        for turn, ids in list(self.turn_index.items()):
            if cid not in ids:
                continue
            remaining = [i for i in ids if i != cid]
            if remaining:
                self.turn_index[turn] = remaining
            else:
                self.turn_index.pop(turn, None)
        if self.comb is not None:
            # Surplus curation: freeze into the comb what the strata once
            # engaged with (relevance history or remembrance-saved decay
            # multiplier), instead of dropping it entirely.
            once_curated = chunk.relevance_history or (chunk.decay_multiplier or 1.0) > 1.0
            if not self.comb_relevant_only or once_curated:
                embedding = self._embeddings.get(cid)
                if embedding is None and self.embed_fn is not None:
                    embedding = self.embed_fn(chunk.content)
                self.comb.put(chunk, embedding)

    def evict_stale(self, current_turn: int, keep_ids: set[str],
                    stale_threshold: int,
                    raw_scores: Optional[dict] = None,
                    relevance_floor: float = 0.0) -> int:
        """Stale-out archiving (comb): move once-curated chunks past the stale
        wall that are surplus — unselected this turn OR scored below the
        relevance floor (budget selection is a greedy fill, so low-score
        chunks still enter the context when leftover budget remains) — out of
        the active store into the comb (surplus tier). No-op without a comb.
        Returns the number of chunks moved."""
        if self.comb is None:
            return 0
        moved = 0
        for chunk in list(self.chunks.values()):
            if chunk.id in keep_ids and raw_scores.get(chunk.id, 1.0) >= relevance_floor:
                continue
            age = current_turn - (chunk.last_referenced_turn or chunk.turn)
            if age <= stale_threshold:
                continue
            once_curated = chunk.relevance_history or (chunk.decay_multiplier or 1.0) > 1.0
            if self.comb_relevant_only and not once_curated:
                continue
            self._remove_chunk(chunk.id)  # comb hook fires inside
            moved += 1
        return moved

    def apply_refresh(self, refresh_map: dict[str, int]) -> None:
        """Membrane-before-Retention fix: refresh kept chunks' decay state so a
        chunk merged with a fresher duplicate isn't penalized for stale age."""
        for cid, freshest_turn in refresh_map.items():
            chunk = self.chunks.get(cid)
            if chunk:
                chunk.last_referenced_turn = max(
                    chunk.last_referenced_turn or 0, freshest_turn
                )

    def all_chunks(self) -> list[ContextChunk]:
        return list(self.chunks.values())

    def all_contents(self) -> list[str]:
        return [c.content for c in self.all_chunks()]

    def all_embeddings(self) -> dict:
        for c in self.all_chunks():
            if c.id not in self._embeddings and self.embed_fn is not None:
                self._embeddings[c.id] = self.embed_fn(c.content)
        return self._embeddings

    def get_deletion_candidates(self) -> list[ContextChunk]:
        """Chunks that would be evicted on overflow (least-recently-referenced)."""
        if not self.max_chunks or len(self.chunks) <= self.max_chunks:
            return []
        ordered = sorted(self.all_chunks(), key=lambda c: c.last_referenced_turn or 0)
        return ordered[: len(self.chunks) - self.max_chunks]

    def get_recent_chunks(self, n: int) -> list[ContextChunk]:
        """Chunks from the last ``n`` turns, in chronological order."""
        ids: list[str] = []
        for t in sorted(self.turn_index)[-n:]:
            ids.extend(self.turn_index[t])
        return [self.chunks[i] for i in ids]

    def get_turns(self) -> list[int]:
        return sorted(self.turn_index)

    def count(self) -> int:
        return len(self.chunks)

    # ------------------------------------------------------------------
    # Checkpoint serialization (resume support)
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "max_chunks": self.max_chunks,
            "ingest_block_prefixes": list(self.ingest_block_prefixes),
            "chunks": [c.to_dict() for c in self.all_chunks()],
            "turn_index": {str(t): ids for t, ids in self.turn_index.items()},
        }

    @classmethod
    def from_dict(cls, data: dict, embed_fn: Optional[Callable[[str], object]] = None) -> "ContextStore":
        store = cls(
            embed_fn=embed_fn,
            max_chunks=data.get("max_chunks"),
            ingest_block_prefixes=data.get("ingest_block_prefixes", None),
        )
        for raw in data.get("chunks", []):
            chunk = ContextChunk(
                id=raw["id"],
                content=raw["content"],
                turn=raw["turn"],
                fingerprint=raw.get("fingerprint", ""),
                timestamp=raw.get("timestamp", ""),
                decay_multiplier=raw.get("decay_multiplier", 1.0),
                times_saved=raw.get("times_saved", 0),
                last_referenced_turn=raw.get("last_referenced_turn"),
                relevance_history=[tuple(x) for x in raw.get("relevance_history", [])],
            )
            store.chunks[chunk.id] = chunk
        store.turn_index = {
            int(t): list(ids) for t, ids in data.get("turn_index", {}).items()
        }
        return store
