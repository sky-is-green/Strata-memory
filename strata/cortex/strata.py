"""Unified Hive orchestrator.

Wires all components (store, drones, router, assembler, health monitor, graceful
degradation, KV-cache, logger) into a single ``process_turn()`` entry point that
mirrors the plan's ``Hive`` used by shadow mode / A-B / rollback. It:

  - checks congestion and applies graceful degradation each turn,
  - assembles context (or falls back to FIFO at emergency level),
  - optionally generates via a backend with a stable pinned prefix,
  - records per-turn latency breakdown, PES, and NDJSON events,
  - grows the context store with each exchange.
"""

from __future__ import annotations

import sys
import time
import uuid
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from backend.cache_manager import KVCacheManager
from cortex.baselines.metrics import estimate_tokens
from cortex.baselines.runner import build_fifo_messages
from cortex.config import HiveConfig
from cortex.degradation import GracefulDegradation
from cortex.efficiency import EfficiencyScorer
from cortex.health import PipelineHealthMonitor
from cortex.hedges import is_hedge_reply
from cortex.routing import DroneRouter, EscalationHandler
from focal.assembly import ContextAssembler
from focal.budget import AdaptiveBudget
from membrane.dedup import ContextDeduplicator
from membrane.drift import TopicDriftDetector
from retention.comb import CombStore
from retention.store import ContextStore
from sieve.medium import MediumDrone
from sieve.ultra_small import UltraSmallDrone


@dataclass
class TurnResult:
    turn: int
    query: str
    reply: str
    assembled: object  # AssembledContext or None (FIFO fallback / error)
    timings: dict = field(default_factory=dict)
    congestion: object = None
    degradation_level: int = 0
    mode: str = "strata"  # strata | fifo_fallback | no_backend | error
    pes: float = 0.0
    token_count: int = 0
    budget: int = 0
    error: Optional[str] = None


class Hive:
    _WORD_RE = re.compile(r"[a-z0-9]{4,}")

    def __init__(
        self,
        config: Optional[HiveConfig] = None,
        ultra=None,
        medium=None,
        backend=None,
        logger=None,
        store: Optional[ContextStore] = None,
        pinned_prefix: str = "",
        router=None,
        confirmation_imprint=None,
    ) -> None:
        self.config = config or HiveConfig()
        self.ultra = ultra or UltraSmallDrone(
            model_name=self.config.ultra_model,
            vocab_boost=self.config.vocab_boost,
            confidence_mode=self.config.confidence_mode,
        )
        if medium is not None:
            self.medium = medium
        elif self.config.enable_medium:
            self.medium = MediumDrone(model_name=self.config.medium_model)
        else:
            # Medium drone is heavy and VRAM-contending; keep a lightweight
            # placeholder until enabled via HiveConfig.enable_medium.
            self.medium = MediumDrone(score_pair_fn=lambda q, c: 0.5)
        self.backend = backend
        self.logger = logger
        self.pinned_prefix = pinned_prefix
        self._comb_counter = 0
        self.comb = None
        self.comb_stats = {"archived": 0, "resurrected": 0, "comb_hits": 0, "gate_fired": 0}
        self.comb_stats_history: list[dict] = []
        self.store = store or self._build_store()
        self.router = router or DroneRouter()
        self.escalation = EscalationHandler()
        self.dedup = ContextDeduplicator(threshold=self.config.dedup_threshold)
        self.drift = TopicDriftDetector(
            embed_fn=self.ultra.embed, threshold=self.config.drift_threshold
        )
        self.budget = AdaptiveBudget()
        self.assembler = ContextAssembler(collect_timings=True)
        self.monitor = PipelineHealthMonitor(logger=logger)
        self.degradation = GracefulDegradation()
        self.cache = KVCacheManager(backend) if backend is not None else None
        self.scorer = EfficiencyScorer()
        self.run_id = uuid.uuid4().hex[:8]
        self.turn = 0
        self._warned_reasoning_starve = False
        # S6 confirmation gate (opt-in; disabled == rule-based hedge filter,
        # the mechanism-attribution condition). The imprint provider is
        # injected by the runner (fixture answers offline, digest live).
        self.gate = None
        self.gate_imprint = confirmation_imprint
        self.gate_stats: dict = {"decisions": []}
        if self.config.gate_enabled:
            from cortex.confirmation_gate import ConfirmationGate, DigestImprint

            self.gate = ConfirmationGate(
                accept_threshold=self.config.gate_accept_threshold,
                flag_threshold=self.config.gate_flag_threshold,
                substantive_floor=self.config.gate_substantive_floor,
            )
            self.gate_imprint = confirmation_imprint or DigestImprint()

    # ------------------------------------------------------------------
    def _build_store(self) -> ContextStore:
        """Fresh store for a conversation; wires the comb (surplus SSD tier)
        when enabled. Each conversation gets its own comb file so archived
        chunks never leak across conversations (same isolation rule as the
        store itself)."""
        if self.config.comb_enabled and self.config.comb_dir:
            self._comb_counter += 1
            self.comb = CombStore(
                Path(self.config.comb_dir) / f"conv_{self._comb_counter:03d}.jsonl",
                max_records=self.config.comb_max_records,
                embed_fn=self.ultra.embed,
            )
            return ContextStore(
                embed_fn=self.ultra.embed,
                max_chunks=self.config.max_chunks,
                comb=self.comb,
                comb_relevant_only=self.config.comb_relevant_only,
                sanitize=self.config.strip_secrets,
                max_chunk_chars=self.config.max_chunk_chars,
            )
        return ContextStore(
            embed_fn=self.ultra.embed, max_chunks=self.config.max_chunks,
            sanitize=self.config.strip_secrets,
            max_chunk_chars=self.config.max_chunk_chars,
        )

    # ------------------------------------------------------------------
    def reset_conversation(self) -> None:
        """Start a fresh conversation context.

        Clears the store and resets the turn counter so one conversation's
        chunks never leak into the next. Without this, a benchmark run over
        many conversations shares one store, so late conversations retrieve
        mostly *other* conversations' chunks (cross-conversation
        contamination) and P2 precision collapses. Comb stats are finalized
        per conversation.
        """
        if self.comb is not None:
            self.comb_stats_history.append(dict(self.comb_stats))
            self.comb_stats = {
                "archived": 0, "resurrected": 0, "comb_hits": 0, "gate_fired": 0,
            }
        self.store = self._build_store()
        self.turn = 0

    # ------------------------------------------------------------------
    def process_turn(
        self, query: str, conversation_id: Optional[str] = None, record_exchange: bool = True
    ) -> TurnResult:
        self.turn += 1

        report = self.monitor.check_congestion()
        self.degradation.update(report)
        level = self.degradation.current_level

        timings: dict = {}
        assembled = None
        reply = ""
        error = None
        token_count = 0
        budget = 0
        try:
            if self.degradation.should_fallback_fifo():
                content = self._fifo_context(query)
                reply = self._generate(content, query, timings)
                token_count = estimate_tokens(content)
                budget = 4096
                mode = "fifo_fallback"
            else:
                t0 = time.perf_counter()
                assembled = self.assembler.assemble(
                    query=query, current_turn=self.turn, store=self.store,
                    router=self.router, ultra_small=self.ultra, medium=self.medium,
                    escalation=self.escalation, dedup=self.dedup,
                    drift_detector=self.drift, budget=self.budget,
                    max_context=self.config.max_context,
                    skip_remembrance=self.degradation.should_skip_remembrance(),
                    skip_dedup=self.degradation.should_skip_dedup(),
                )
                # Comb gate: consult the surplus tier only when the active
                # store's best raw match is weak — normal turns pay zero comb
                # cost, and resurrected candidates cannot crowd out a store
                # that already answers (P11 falsification clause 2). The gate
                # also fires when the best match is a *query echo* (a
                # near-duplicate of the query that carries no facts): measured
                # in the P11 replay (2026-08-24) that template-repeated query
                # chunks accumulate in the store and otherwise keep the gate
                # closed on every subsequent return turn.
                comb_candidates = []
                if (
                    self.comb is not None
                    and len(self.comb) > 0
                    and self._comb_gate_fires(query, assembled)
                ):
                    self.comb_stats["gate_fired"] += 1
                    comb_candidates = self.comb.retrieve(
                        query, k=self.config.comb_top_k
                    )
                    if comb_candidates:
                        assembled = self.assembler.assemble(
                            query=query, current_turn=self.turn, store=self.store,
                            router=self.router, ultra_small=self.ultra,
                            medium=self.medium, escalation=self.escalation,
                            dedup=self.dedup, drift_detector=self.drift,
                            budget=self.budget, max_context=self.config.max_context,
                            skip_remembrance=self.degradation.should_skip_remembrance(),
                            skip_dedup=self.degradation.should_skip_dedup(),
                            comb_candidates=comb_candidates,
                        )
                timings.update(self.assembler.last_timings)
                timings["assembly_total_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
                content = assembled.content
                reply = self._generate(content, query, timings)
                if self.comb is not None and comb_candidates:
                    comb_ids = {c.id for c in comb_candidates}
                    selected_comb = [
                        cid for cid in assembled.selected_chunk_ids if cid in comb_ids
                    ]
                    self.comb.touch(selected_comb, self.turn)
                    self.comb_stats["resurrected"] += len(selected_comb)
                    if selected_comb:
                        self.comb_stats["comb_hits"] += 1
                token_count = assembled.token_count
                budget = assembled.budget
                mode = "no_backend" if self.backend is None else "strata"
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            mode = "error"
            if self.logger is not None:
                self.logger.log(
                    "error", "turn_failed",
                    {"query_hash": query[:40], "error": error, "degradation_level": level},
                    run_id=self.run_id, conversation_id=conversation_id, turn_id=self.turn,
                )

        if record_exchange:
            self.store.add_chunk(self.turn, query)
            if reply:
                if self.gate is not None:
                    decision = self.gate.decide(
                        conversation_id or "unknown", query, reply,
                        self.gate_imprint,
                    )
                    if decision.decision == "accept":
                        self.gate_imprint.confirm(
                            conversation_id or "unknown", decision.reply_facts
                        )
                    self.gate_stats["decisions"].append({
                        "turn": self.turn, "decision": decision.decision,
                        "ingestion_ratio": decision.ingestion_ratio,
                        "imprint_facts": decision.imprint_facts,
                        "reason": decision.reason, "rule_hedge": decision.rule_hedge,
                    })
                    if self.logger is not None:
                        self.logger.log(
                            "confirmation_gate", "graded",
                            {"decision": decision.decision,
                             "reason": decision.reason,
                             "ingestion_ratio": decision.ingestion_ratio,
                             "imprint_facts": decision.imprint_facts,
                             "rule_hedge": decision.rule_hedge},
                            run_id=self.run_id, conversation_id=conversation_id,
                            turn_id=self.turn,
                        )
                    if decision.decision in ("accept", "flag"):
                        self.store.add_chunk(self.turn, reply)
                elif not (self.config.filter_hedge_replies and self._is_hedge_reply(reply)):
                    self.store.add_chunk(self.turn, reply)

        # Comb maintenance: stale-out archiving (surplus tier) + archive-side
        # decay. Runs every turn; both are no-ops when the comb is disabled.
        if self.comb is not None and assembled is not None:
            self.comb_stats["archived"] += self.store.evict_stale(
                self.turn,
                set(assembled.selected_chunk_ids),
                self.config.stale_threshold,
                raw_scores=assembled.raw_scores,
                relevance_floor=self.config.comb_gate_threshold,
            )
            self.comb.prune(
                max_age_turns=self.config.comb_max_age_turns, current_turn=self.turn
            )

        util = token_count / max(budget, 1)
        pes = self.scorer.compute(
            avg_latency_ms=timings.get("total_ms", 0.0),
            budget_used=token_count, budget_total=budget,
        ).composite

        self._log(query, assembled, report, level, timings, conversation_id)
        result = TurnResult(
            turn=self.turn, query=query, reply=reply, assembled=assembled,
            timings=timings, congestion=report, degradation_level=level,
            mode=mode, pes=round(pes, 2), token_count=token_count,
            budget=budget, error=error,
        )
        self._last_turn_result = result  # for the prompt inspector
        return result

    # ------------------------------------------------------------------
    # Refusal/hedge detection. Implementation lives in cortex.hedges
    # (shared with the S6 confirmation gate); this staticmethod is kept as
    # the public API. Markers must fire in the *opening* of the reply
    # (lead-anchored): refusals announce themselves up front, a mid-reply
    # caveat in an otherwise factual answer must NOT be treated as a hedge.

    @staticmethod
    def _is_hedge_reply(reply: str) -> bool:
        return is_hedge_reply(reply)

    def _comb_gate_fires(self, query: str, assembled) -> bool:
        """True when the comb should be consulted for this turn: the store's
        best raw match is weak (below comb_gate_threshold), OR that best match
        is a *query echo* — a template-sibling chunk that shares >= 80% of its
        content words with the query (template-repeated question chunks score
        ~1.0 but carry no facts, keeping the gate closed; measured in the P11
        replay, 2026-08-24)."""
        if assembled.top_raw_score < self.config.comb_gate_threshold:
            return True
        cid = getattr(assembled, "top_chunk_id", None)
        if cid is None:
            return False
        chunk = self.store.chunks.get(cid)
        if chunk is None or not chunk.content or chunk.content == query:
            return False
        cwords = {w for w in Hive._WORD_RE.findall(chunk.content.lower())}
        if not cwords:
            return False
        qwords = {w for w in Hive._WORD_RE.findall(query.lower())}
        return len(cwords & qwords) / len(cwords) >= 0.8

    def inspect_turn(self, result: TurnResult) -> dict:
        """Full per-turn curation detail for the prompt inspector.

        Surfaces what the model actually received (and what it didn't) so
        the curation pipeline is debuggable without reading NDJSON logs.
        """
        a = result.assembled
        if a is None:
            return {"mode": result.mode, "error": result.error or "no assembly"}
        all_chunks = {c.id: c for c in self.store.all_chunks()}
        selected_ids = set(a.selected_chunk_ids or [])
        selected, dropped = [], []
        for cid, score in sorted((a.raw_scores or {}).items(),
                                 key=lambda kv: kv[1], reverse=True):
            chunk = all_chunks.get(cid)
            preview = (chunk.content[:200] + "…" if chunk and len(chunk.content) > 200
                       else chunk.content if chunk else "(evicted)")
            entry = {"id": cid, "raw_score": round(score, 3),
                     "selected": cid in selected_ids, "preview": preview}
            if cid in selected_ids:
                selected.append(entry)
            else:
                dropped.append(entry)
        route = a.routing_decision
        return {
            "mode": result.mode,
            "turn": result.turn,
            "query": result.query,
            "routing": {"route_to": route.route_to if route else "unknown"},
            "drift_detected": a.drift_detected,
            "budget": {"total": a.budget, "used": a.token_count,
                       "utilization": round(a.token_count / max(a.budget, 1), 3)},
            "top_raw_score": round(a.top_raw_score, 3),
            "selected_chunks": selected,
            "dropped_chunks": dropped[:20],
            "dropped_count": len(dropped),
            "store_chunks": len(all_chunks),
            "assembled_preview": (a.content[:500] + "…"
                                  if len(a.content) > 500 else a.content),
            "timings": {k: round(v, 1) for k, v in (result.timings or {}).items()
                        if isinstance(v, (int, float))},
        }

    def _fifo_context(self, query: str) -> str:
        history = [{"role": "user", "content": c.content} for c in self.store.all_chunks()]
        msgs = build_fifo_messages(history + [{"role": "user", "content": query}])
        return "\n\n".join(m["content"] for m in msgs)

    def _generate(self, content: str, query: str, timings: dict) -> str:
        if self.backend is None:
            timings["generation_ms"] = 0.0
            timings["total_ms"] = round(
                timings.get("assembly_total_ms", 0.0) + timings["generation_ms"], 3
            )
            return ""
        if self.config.sanitize_context:
            from cortex.sanitize import sanitize_context

            content = sanitize_context(content)
        if self.cache is not None:
            self.cache.update_cache(content, persistent_prefix=self.pinned_prefix)
        t1 = time.perf_counter()
        sampling = dict(self.config.sampling or {})
        if self.config.max_tokens:
            sampling["max_tokens"] = self.config.max_tokens
        reply = self.backend.generate(content, query, sampling or None)
        if not (reply or "").strip():
            usage = getattr(self.backend, "last_usage", {}) or {}
            reason_toks = usage.get("completion_tokens_details", {}).get("reasoning_tokens")
            if reason_toks and not self._warned_reasoning_starve:
                # A reasoning model burned its whole output budget on chain-of-thought,
                # so the visible reply is empty. Surface this loudly once per run.
                self._warned_reasoning_starve = True
                if self.logger is not None:
                    self.logger.log(
                        "backend", "empty_reply_reasoning_starved",
                        {"max_tokens": self.config.max_tokens, "reasoning_tokens": reason_toks},
                        run_id=self.run_id,
                    )
                print(
                    "\nWARNING: replies are empty — the loaded model spent its whole "
                    f"output budget on reasoning tokens ({reason_toks}). A small "
                    "--max-tokens cap produces no visible answer on a reasoning model. "
                    "Drop --max-tokens (the default 4096 ceiling lets reasoning "
                    "models reason and answer), or load a non-reasoning model.",
                    file=sys.stderr,
                )
        timings["generation_ms"] = round((time.perf_counter() - t1) * 1000.0, 3)
        timings["total_ms"] = round(
            timings.get("assembly_total_ms", 0.0) + timings["generation_ms"], 3
        )
        return reply

    # ------------------------------------------------------------------
    def _log(self, query, assembled, report, level, timings, conversation_id) -> None:
        if self.logger is None:
            return
        routed_to = assembled.routing_decision.route_to if assembled else "fifo"
        self.logger.log(
            "router", "task_classified",
            {"routed_to": routed_to, "latency_ms": 0},
            run_id=self.run_id, conversation_id=conversation_id, turn_id=self.turn,
        )
        self.logger.log(
            "assembly", "context_assembled",
            {
                "total_tokens": assembled.token_count if assembled else 0,
                "chunk_count": assembled.chunks_used if assembled else 0,
                "budget_used": assembled.token_count if assembled else 0,
                "budget_total": assembled.budget if assembled else 0,
                "degradation_level": level,
                "timings": timings,
            },
            latency_ms=timings.get("total_ms", 0.0),
            run_id=self.run_id, conversation_id=conversation_id, turn_id=self.turn,
        )
        if report.severity != "normal":
            self.logger.log(
                "congestion", "congestion_detected", report.__dict__,
                run_id=self.run_id, conversation_id=conversation_id, turn_id=self.turn,
            )
