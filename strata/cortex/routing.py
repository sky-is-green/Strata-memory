"""Heuristic routing + confidence-based escalation (S1.3).

Phase 1 routes each query to a drone tier via surface heuristics. When the
router selects "escalation", or the ultra-small drone returns uncertain scores,
the EscalationHandler re-scores ONLY the uncertain chunks with the medium drone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RoutingDecision:
    route_to: str  # "ultra_small" | "medium" | "escalation"
    confidence: float
    reason: str = field(default="default")


class DroneRouter:
    """Routes each query to a drone tier using heuristic surface signals."""

    HEURISTIC_RULES = {
        "complex_keywords": [
            "refactor", "architecture", "debug", "explain", "analyze",
            "compare", "design", "optimize", "review", "audit",
        ],
        "complex_patterns": [
            r"how does .+ work",
            r"why is .+ (broken|failing|slow)",
            r"what (caused|causes) .+",
            r"(connect|relate|depend).+ between",
        ],
        "code_density_threshold": 3,  # 3+ code blocks = complex
        "length_threshold": 500,
        "depth_threshold": 30,
    }

    def route(
        self, query: str, conversation_history: Optional[list] = None
    ) -> RoutingDecision:
        rules = self.HEURISTIC_RULES
        score = 0
        reasons: list[str] = []
        q = query.lower()

        for kw in rules["complex_keywords"]:
            if kw in q:
                score += 1
                reasons.append(f"keyword:{kw}")

        for pat in rules["complex_patterns"]:
            if re.search(pat, q):
                score += 1
                reasons.append(f"pattern:{pat}")

        code_blocks = query.count("```")
        if code_blocks >= rules["code_density_threshold"]:
            score += 2
            reasons.append(f"code_density:{code_blocks}")

        if len(query) > rules["length_threshold"]:
            score += 1
            reasons.append(f"length:{len(query)}")

        history = conversation_history or []
        if len(history) > rules["depth_threshold"]:
            score += 1
            reasons.append(f"depth:{len(history)}")

        if score >= 3:
            return RoutingDecision("escalation", 0.8, ",".join(reasons) or "default")
        if score >= 2:
            return RoutingDecision("medium", 0.7, ",".join(reasons) or "default")
        return RoutingDecision("ultra_small", 0.9, ",".join(reasons) or "default")


class EscalationHandler:
    """Runs ultra-small first; re-scores only uncertain chunks with medium.

    A chunk is escalated when its predicted relevance is high (it would be
    selected) but its confidence is low. This concentrates the medium drone's
    compute where the small drone is least trustworthy.
    """

    UNCERTAINTY_THRESHOLD_SCORE = 0.7
    UNCERTAINTY_THRESHOLD_CONFIDENCE = 0.6

    def process(self, query, chunks, ultra_small, medium):
        initial_scores = ultra_small.score(query, chunks)

        uncertain_indices = [
            i
            for i, s in enumerate(initial_scores)
            if s.relevance_score > self.UNCERTAINTY_THRESHOLD_SCORE
            and s.confidence < self.UNCERTAINTY_THRESHOLD_CONFIDENCE
        ]

        if not uncertain_indices:
            return initial_scores

        uncertain_chunks = [chunks[i] for i in uncertain_indices]
        medium_scores = medium.score(query, uncertain_chunks)

        for idx, medium_score in zip(uncertain_indices, medium_scores):
            initial_scores[idx].relevance_score = medium_score.relevance_score
            initial_scores[idx].confidence = medium_score.confidence
            initial_scores[idx].source = "medium_validated"

        return initial_scores
