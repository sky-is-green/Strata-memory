"""Async batch queen.

Runs in a background process (never on the hot path). Evaluates past turns with
an LLM-as-judge to produce ground-truth relevance labels. The queen prompt asks
about *context utilization* rather than answer correctness, to avoid the
parametric-knowledge confound (Pitfall 11).

Designed to be driven asynchronously/batched; the evaluate/run_batch methods are
synchronous for testability, with the LLM injected via ``generate_fn``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class TurnRecord:
    turn: int
    assembled_context: str
    user_query: str
    llm_response: str
    chunk_ids: list = field(default_factory=list)


@dataclass
class QueenLabel:
    turn: int
    context_sufficient: bool
    context_used: list
    missing_context: list
    sufficiency_score: int
    chunk_labels: dict = field(default_factory=dict)


class Queen:
    EVALUATION_PROMPT = """You are evaluating whether an AI assistant had sufficient context.

The assistant was given this context:
---
{context}
---

The user asked: {query}

The assistant responded: {response}

Answer these questions:
1. Was the context sufficient? (yes/no)
2. Which specific pieces of context were used? (list them)
3. What additional context would have helped? (describe)
4. Rate context sufficiency 1-5: ___

Respond ONLY in this JSON shape:
{{"sufficient": true, "used_pieces": [], "missing": [], "score": 4}}"""

    def __init__(self, generate_fn: Callable[[str], str]) -> None:
        """``generate_fn(prompt) -> JSON string`` is the injected LLM."""
        self.generate_fn = generate_fn

    def evaluate_turn(self, turn_data: TurnRecord) -> QueenLabel:
        prompt = self.EVALUATION_PROMPT.format(
            context=turn_data.assembled_context,
            query=turn_data.user_query,
            response=turn_data.llm_response,
        )
        raw = self.generate_fn(prompt)
        parsed = self._extract_json(raw)
        if parsed is None:
            raise ValueError(f"queen returned non-JSON response: {raw[:200]!r}")
        return QueenLabel(
            turn=turn_data.turn,
            context_sufficient=bool(parsed.get("sufficient")),
            context_used=parsed.get("used_pieces", []),
            missing_context=parsed.get("missing", []),
            sufficiency_score=int(parsed.get("score", 0)),
            chunk_labels=self._map_to_chunks(parsed, turn_data.chunk_ids),
        )

    @staticmethod
    def _extract_json(raw: str) -> Optional[dict]:
        """Best-effort parse of an LLM JSON response.

        Real models wrap JSON in markdown fences, prefix/suffix prose, or return
        empty text. This strips fences, tries a direct parse, then falls back to
        extracting the outermost ``{...}`` block. Returns ``None`` when nothing
        parseable is found (callers decide how to treat that).
        """
        if not raw:
            return None
        text = raw.strip()
        # strip markdown code fences (```json ... ```)
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
            text = re.sub(r"\s*```\s*$", "", text).strip()
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        # extract the outermost object block from surrounding prose
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                obj = json.loads(text[start:end + 1])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
        return None

    def run_batch(
        self, conversation_log: list[TurnRecord], sample_rate: float = 0.1
    ) -> list[QueenLabel]:
        """Evaluate a sample of turns. Default 10% (every 10th turn)."""
        step = max(1, int(round(1.0 / sample_rate)))
        return [self.evaluate_turn(t) for t in conversation_log[::step]]

    @staticmethod
    def _map_to_chunks(parsed: dict, chunk_ids: list) -> dict:
        used = {str(u).lower() for u in parsed.get("used_pieces", [])}
        labels = {}
        for cid in chunk_ids:
            labels[cid] = any(u in str(cid).lower() for u in used) or not used
        return labels
