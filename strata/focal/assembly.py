"""Context assembly (focal layer) — the full strata pipeline per user turn.

Order (Membrane runs before Retention):
  1. Remembrance pass on chunks approaching deletion.
  2. Route the query and score all chunks via the drone fleet.
  3. Deduplicate semantically similar chunks (Membrane) + refresh decay state.
  4. Detect topic drift; if a reset is flagged, build drift penalties (Membrane).
  5. Apply the decay matrix to the surviving chunks (Retention).
  6. Compute the adaptive budget.
  7. Sort by effective score and select within budget.
  8. Return the assembled context string.
"""

from __future__ import annotations

import copy
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cortex.baselines.metrics import estimate_tokens
from cortex.hedges import HEDGE_CONTRACTIONS
from cortex.routing import RoutingDecision
from retention.decay import DecayMatrix
from retention.filter import strip_boilerplate
from retention.remembrance import RemembrancePass


# Contraction expansions applied case-insensitively so "don't" and "do not"
# normalize identically on both sides of the similarity compare.
_CONTRACTION_RES = [
    (re.compile(re.escape(short), re.IGNORECASE), long)
    for short, long in HEDGE_CONTRACTIONS.items()
]


def normalize_for_scoring(text: str, prefixes=None) -> str:
    """Shared normalize+strip pass applied to BOTH the query side and every
    candidate side before similarity scoring (RC2).

    Meta/tool control lines (harness system-reminders) are removed with the
    same ingest blocklist the store enforces (``strip_boilerplate``), and
    contractions are expanded with the shared hedge map so "don't" and
    "do not" score identically. Returns ``""`` when nothing scorable
    remains; such candidates are dropped before the drones ever see them.
    """
    if not text or not text.strip():
        return ""
    stripped = strip_boilerplate(text, prefixes)
    if not stripped or not stripped.strip():
        return ""
    for pattern, long in _CONTRACTION_RES:
        stripped = pattern.sub(long, stripped)
    return stripped.strip()


@dataclass
class AssembledContext:
    content: str
    token_count: int
    budget: int
    chunks_used: int
    routing_decision: RoutingDecision
    drift_detected: bool
    selected_chunk_ids: list = field(default_factory=list)
    # Best raw drone score over the candidate pool (pre-decay); the comb gate
    # (Strata) uses it to decide whether the active store answered well enough.
    top_raw_score: float = 0.0
    # Id of the highest-raw-score chunk — the gate also fires when this is a
    # *query echo* (a near-duplicate of the query itself, which carries no
    # facts): measured in the P11 replay (2026-08-24) that template-repeated
    # query chunks accumulate in the store and keep the gate closed on every
    # subsequent return turn.
    top_chunk_id: Optional[str] = None
    # Raw (pre-decay) drone scores per candidate id; the stale-out archiver
    # uses them because budget selection is a greedy fill — small low-score
    # chunks still enter the context when leftover budget remains, so
    # "not selected" alone is not a reliable surplus signal.
    raw_scores: dict = field(default_factory=dict)
    # Stored chunks skipped because their fingerprint matched content already
    # present in the incoming request messages (recency echo); surfaced via
    # /v1/strata/inspect.
    payload_dedup_skipped: int = 0


class ContextAssembler:
    def __init__(self, collect_timings: bool = False) -> None:
        self.collect_timings = collect_timings
        self.last_timings: dict = {}

    def _tick(self, name: str, t0: float) -> None:
        if self.collect_timings:
            self.last_timings[name] = round((time.perf_counter() - t0) * 1000.0, 3)

    def assemble(
        self,
        query: str,
        current_turn: int,
        store,
        router,
        ultra_small,
        medium,
        escalation,
        dedup,
        drift_detector,
        budget,
        max_context: int = 8192,
        skip_remembrance: bool = False,
        skip_dedup: bool = False,
        comb_candidates: Optional[list] = None,
        payload_fingerprints: Optional[set] = None,
        dedup_against_payload: bool = True,
        relevance_floor: float = 0.25,
        max_chunk_share: float = 0.5,
    ) -> AssembledContext:
        comb_candidates = comb_candidates or []
        comb_by_id = {c.id: c for c in comb_candidates}
        # 1. Remembrance pass
        if not skip_remembrance:
            _t = time.perf_counter()
            deletion_candidates = store.get_deletion_candidates()
            current_topic = query
            remembrance_results = RemembrancePass().process(
                deletion_candidates, current_topic, ultra_small
            )
            self._tick("remembrance_ms", _t)

        # 2. Route + score all chunks (comb records join the candidate pool)
        _t = time.perf_counter()
        routing = router.route(query, store.get_turns())
        all_chunks = store.all_chunks() + comb_candidates
        # RC2: score the normalized forms — the query and every candidate go
        # through the same strip+expand pass, so meta/tool control text can
        # neither skew the query embedding nor win as a candidate. Candidates
        # with nothing scorable left are dropped before the drones run.
        prefixes = getattr(store, "ingest_block_prefixes", None)
        scoring_query = normalize_for_scoring(query, prefixes)
        scored_pairs = [
            (chunk, normalize_for_scoring(chunk.content, prefixes))
            for chunk in all_chunks
        ]
        scored_pairs = [(c, n) for c, n in scored_pairs if n]
        scoring_texts = [normalized for _, normalized in scored_pairs]
        if routing.route_to == "escalation":
            scores = escalation.process(scoring_query, scoring_texts, ultra_small, medium)
        elif routing.route_to == "medium":
            scores = medium.score(scoring_query, scoring_texts)
        else:
            scores = ultra_small.score(scoring_query, scoring_texts)
        self._tick("scoring_ms", _t)

        raw_scores = {}
        for k, s in enumerate(scores):
            if k < len(scored_pairs):
                raw_scores[scored_pairs[k][0].id] = s.relevance_score

        # 3. Deduplicate (Membrane first) + refresh decay state
        _t = time.perf_counter()
        embeddings = store.all_embeddings()
        for c in comb_candidates:
            if c.embedding is not None:
                embeddings[c.id] = np.asarray(c.embedding)
            elif store.embed_fn is not None:
                embeddings[c.id] = np.asarray(store.embed_fn(c.content))
        if not skip_dedup:
            surviving, refresh_map = dedup.deduplicate(
                [c for c, _ in scored_pairs], embeddings
            )
            store.apply_refresh(refresh_map)
        else:
            surviving, refresh_map = [c for c, _ in scored_pairs], {}
        self._tick("dedup_ms", _t)

        # 4. Topic drift -> drift penalties
        _t = time.perf_counter()
        recent = store.get_recent_chunks(3)
        drift = drift_detector.check(recent, all_chunks, ultra_small)
        drift_penalties = {}
        if drift.should_reset:
            drift_penalties = self._apply_drift_reset(surviving, recent)
        self._tick("drift_ms", _t)

        # 5. Decay on surviving chunks (Retention); comb resurrections are
        # exempt (raw relevance) so a returned topic can win the budget.
        _t = time.perf_counter()
        effective = DecayMatrix().apply(
            surviving, current_turn, raw_scores, drift_penalties,
            exempt_ids=set(comb_by_id),
        )
        self._tick("decay_ms", _t)

        # 5b. Payload dedup: skip stored chunks that are verbatim copies of
        # content already present in the incoming request messages (recency
        # echo). Matched on the same 12-hex fingerprint the store assigns at
        # write time. Runs before budget selection so echoes consume nothing.
        payload_dedup_skipped = 0
        if dedup_against_payload and payload_fingerprints:
            pool_by_id = {c.id: c for c in surviving}
            kept_ids = {
                cid for cid in effective
                if getattr(pool_by_id.get(cid), "fingerprint", None)
                not in payload_fingerprints
            }
            payload_dedup_skipped = len(effective) - len(kept_ids)
            effective = {cid: v for cid, v in effective.items() if cid in kept_ids}

        # 6. Budget
        high_relevance = sum(1 for v in effective.values() if v > 0.6)
        token_budget = budget.compute(routing.route_to, high_relevance, max_context)

        # 7. Sort + select within budget (comb records resolve from the pool)
        _t = time.perf_counter()
        scored = sorted(effective.items(), key=lambda kv: kv[1], reverse=True)
        pool = {**store.chunks, **comb_by_id}
        selected = self._select_within_budget(
            scored,
            pool,
            token_budget,
            raw_scores=raw_scores,
            relevance_floor=relevance_floor,
            max_chunk_share=max_chunk_share,
        )
        self._tick("select_ms", _t)

        # Selection is curation: mark selected store chunks so the surplus
        # tier (comb) can identify what the strata once judged relevant. The
        # remembrance pass only fires on overflow candidates, so an escalated
        # decay multiplier alone never marks chunks in practice — without this
        # record, comb_relevant_only would archive ~nothing (measured in the
        # P11 replay, 2026-08-24).
        for c in selected:
            if c.id in comb_by_id:
                continue
            target = pool.get(c.id, c)
            hist = list(getattr(target, "relevance_history", []) or [])
            hist.append((current_turn, round(raw_scores.get(c.id, 0.0), 3)))
            target.relevance_history = hist[-10:]

        result = AssembledContext(
            content=self._format_context(selected),
            token_count=self._count_tokens(selected),
            budget=token_budget,
            chunks_used=len(selected),
            routing_decision=routing,
            drift_detected=drift.should_reset,
            selected_chunk_ids=[c.id for c in selected],
            top_raw_score=max(raw_scores.values(), default=0.0),
            top_chunk_id=(
                max(raw_scores.items(), key=lambda kv: kv[1])[0] if raw_scores else None
            ),
            raw_scores=raw_scores,
            payload_dedup_skipped=payload_dedup_skipped,
        )
        return result

    def _select_within_budget(
        self,
        scored,
        pool: dict,
        token_budget: int,
        raw_scores=None,
        relevance_floor: float = 0.25,
        max_chunk_share: float = 0.5,
    ) -> list:
        # raw_scores=None keeps the original pure-greedy fill for legacy
        # callers: no floor and no per-chunk cap are applied.
        legacy = raw_scores is None
        share_cap = (
            int(token_budget * max_chunk_share)
            if not legacy and max_chunk_share
            else 0
        )
        selected = []
        used = 0
        for cid, _score in scored:
            chunk = pool.get(cid)
            if chunk is None:
                continue
            if not legacy and raw_scores.get(cid, 0.0) < relevance_floor:
                continue
            content = chunk.content
            cost = estimate_tokens(content)
            if share_cap and cost > share_cap:
                content = self._truncate_to_token_cap(content, share_cap)
                if len(content) < 32:
                    continue
                chunk = copy.copy(chunk)
                chunk.content = content
                cost = estimate_tokens(content)
            if used + cost > token_budget:
                continue
            selected.append(chunk)
            used += cost
        return selected

    @staticmethod
    def _truncate_to_token_cap(text: str, token_cap: int) -> str:
        if token_cap <= 0:
            return ""
        if estimate_tokens(text) <= token_cap:
            return text
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if estimate_tokens(text[:mid]) <= token_cap:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo]

    @staticmethod
    def _format_context(selected: list) -> str:
        return "\n\n".join(c.content for c in selected)

    @staticmethod
    def _count_tokens(selected: list) -> int:
        return sum(estimate_tokens(c.content) for c in selected)

    @staticmethod
    def _apply_drift_reset(surviving: list, recent: list) -> dict[str, float]:
        recent_ids = {c.id for c in recent}
        return {c.id: (1.0 if c.id in recent_ids else 0.1) for c in surviving}
