"""Adaptive context budget (focal layer).

Computes the token budget for the assembled context based on route complexity and
how many chunks scored highly, capped at the model's max context minus generation
headroom.
"""

from __future__ import annotations


class AdaptiveBudget:
    BUDGET_RANGES = {
        "ultra_small": (1000, 3000),
        "medium": (3000, 5000),
        "escalation": (4000, 6000),
    }
    GENERATION_HEADROOM = 2048  # tokens reserved for output
    DEFAULT_RANGE = (2000, 4000)

    def __init__(self, ultra_small_budget_tokens: int = 1000) -> None:
        # Floor (lo end) of the ultra_small adaptive range; configurable via
        # StrataConfig.ultra_small_budget_tokens for A/B tests. Other routes
        # keep their class-level ranges.
        self.ultra_small_budget_tokens = ultra_small_budget_tokens

    def compute(self, route: str, high_relevance_count: int, max_context: int = 8192) -> int:
        lo, hi = self.BUDGET_RANGES.get(route, self.DEFAULT_RANGE)
        if route == "ultra_small":
            lo = self.ultra_small_budget_tokens
        fill_factor = min(high_relevance_count / 10.0, 1.0)
        budget = int(lo + (hi - lo) * fill_factor)
        return min(budget, max_context - self.GENERATION_HEADROOM)
