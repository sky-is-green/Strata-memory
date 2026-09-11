"""StrataConfig — all tunable parameters in one place.

Lets A/B testing and automated rollback swap whole configurations cleanly, and
loads/saves via the Gatekeeper merge contract (cortex.interop).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional

from cortex.interop import GatekeeperSeam
from retention.filter import DEFAULT_INGEST_BLOCK_PREFIXES


@dataclass
class StrataConfig:
    # --- retention ---
    decay_multiplier_init: float = 1.8
    remembrance_threshold: float = 0.65
    stale_threshold: int = 20
    # --- membrane ---
    dedup_threshold: float = 0.92
    drift_threshold: float = 0.6
    # --- routing ---
    routing_threshold: int = 2
    # --- sieve ---
    ultra_model: str = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    medium_model: str = "microsoft/graphcodebert-base"
    enable_medium: bool = False  # medium drone is heavy + VRAM-contending; opt-in
    vocab_boost: float = 0.15
    confidence_mode: str = "mcdropout"  # mcdropout | single | off
    # --- security ---
    sanitize_context: bool = True
    # store-time hygiene (retention.store.sanitize_for_storage): redact
    # credential-shaped strings / base64 blobs and cap chunk length before
    # anything reaches the store, checkpoints, or comb archives
    strip_secrets: bool = True
    max_chunk_chars: int = 4000
    # Harness boilerplate blocklist (retention.filter.strip_boilerplate):
    # system-reminder control text the upstream harness injects around tool
    # calls must never become persistent chunks. Matched by normalized
    # prefix (first 80 chars, case-sensitive); [] disables the filter.
    # Overridable per conversation via the `config` dict, like
    # sanitize_context / strip_secrets.
    ingest_block_prefixes: list = field(
        default_factory=lambda: list(DEFAULT_INGEST_BLOCK_PREFIXES)
    )
    # --- focal ---
    generation_headroom: int = 2048
    max_context: int = 8192
    max_chunks: int = 1000
    max_tokens: Optional[int] = None  # reply cap for iteration/stability runs
    # Payload dedup: skip stored chunks whose fingerprint matches content
    # already present in the incoming request messages (recency echo adds
    # zero new information while consuming budget). Enabled by default.
    dedup_against_payload: bool = True
    # Ultra_small route budget floor (lo end of its adaptive range); raised
    # without code changes for A/B tests (e.g. 2000). Other routes unchanged.
    ultra_small_budget_tokens: int = 1000
    # Experimenter sampling surface (backend.sampling.parse_sampling): the
    # OpenAI-compat sampling fields (temperature, top_p, top_k, min_p,
    # repeat/presence/frequency penalties, stop, seed, mirostat). Empty =
    # backend defaults. Recorded in run reports for reproducibility.
    sampling: dict = field(default_factory=dict)
    # When the backend returns a refusal/hedge ("no information regarding X"),
    # do NOT store it as a chunk: such replies pollute the store and later get
    # retrieved as "context", poisoning retrieval. Filtering them keeps the
    # store fact-bearing (live finding: ~50% of replies were hedges).
    filter_hedge_replies: bool = True
    # --- confirmation gate (S6 — proposed, opt-in) ---
    # Ingestion is a confirmed act: each generation is graded on
    # closeness-to-copy against the *imprint* (fixture answers offline;
    # digest/chronicler live) before it enters the store. reject/flag
    # generations never enter memory (flag = stored but logged for review).
    # Disabled by default: the rule-based hedge filter governs, preserving
    # current behavior (the mechanism-attribution condition).
    gate_enabled: bool = False
    gate_accept_threshold: float = 0.4  # ingestion_ratio >= -> accept (copy)
    gate_flag_threshold: float = 0.2  # ratio < -> reject; between -> flag
    gate_substantive_floor: int = 3  # first-mention replies must add >= this many content terms beyond the query
    # --- comb (surplus SSD tier, P11) ---
    # Opt-in: chunks the store evicts are frozen to disk (per-conversation
    # JSONL) instead of dropped, so a topic that returns long after leaving
    # the budget can be resurrected. Disabled unless comb_dir is set.
    comb_enabled: bool = False
    comb_relevant_only: bool = True  # freeze only chunks the strata once curated
    comb_max_records: int = 2000
    comb_top_k: int = 5  # comb candidates competing for the budget per turn
    comb_dir: Optional[str] = None  # e.g. runs/<ts>/comb; None disables the comb
    # The comb is consulted only when the active store's best match is weak
    # (below this raw-score gate): normal turns pay zero comb cost, and
    # resurrected candidates cannot crowd out a store that already answers.
    # Calibrated by comb_probe + the P11 replay (2026-08-24): the pipeline
    # drone applies vocab_boost (+0.15), so the probe's unboosted 0.7
    # calibration (~97% of return turns) lands at 0.85 with boost; the gate
    # also fires on *query echoes* (Strata._comb_gate_fires) — template-sibling
    # question chunks score ~1.0 but carry no facts and otherwise keep the
    # gate closed on every return turn after the first.
    comb_gate_threshold: float = 0.85
    # The comb forgets too: records unreferenced for more than this many turns
    # are pruned (Postulate 2, much slower than the store's stale wall).
    comb_max_age_turns: int = 1000
    budget_ranges: dict = field(default_factory=lambda: {
        "ultra_small": (1000, 3000),
        "medium": (3000, 5000),
        "escalation": (4000, 6000),
    })

    @classmethod
    def defaults(cls) -> "StrataConfig":
        return cls()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "StrataConfig":
        known = {f.name for f in fields(cls)}
        cleaned = {k: v for k, v in (data or {}).items() if k in known}
        # JSON round-trips tuples as lists; restore the budget-range tuples.
        if "budget_ranges" in cleaned and cleaned["budget_ranges"]:
            cleaned["budget_ranges"] = {
                k: tuple(v) for k, v in cleaned["budget_ranges"].items()
            }
        return cls(**cleaned)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "StrataConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def apply_gatekeeper_overrides(
        self, overrides: dict, seam: Optional[GatekeeperSeam] = None
    ) -> "StrataConfig":
        """Return a new config with Gatekeeper-provided overrides applied safely."""
        seam = seam or GatekeeperSeam()
        merged = seam.merge_config(self.to_dict(), overrides)
        return self.from_dict(merged)
