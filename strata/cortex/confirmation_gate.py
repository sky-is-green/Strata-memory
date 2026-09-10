"""S6 — Confirmation Gate & Imprint Grading.

Goal(plan §4.8): make *ingestion a confirmed act* rather
than an automatic one. Every generation is graded on how close it is to a
*genetic perfection imprint* — the known-correct facts for the conversation —
before it is stored. A generation that is not a close "copy" of the imprint is
rejected or flagged and never enters memory.

This directly attacks the starvation chain that kept live P2 depressed: the
model's own refusal/hedge replies were stored automatically, later retrieved
*as context*, and perpetuated refusal. S5's rule-based hedge filter
(``cortex.hedges.is_hedge_reply``) stops the worst of it by rule; the gate
replaces the rule with a *graded, confirmed act*.

Grading reuses the deterministic retrieval diagnostic's fact math:
``ingestion_ratio`` = share of imprint facts the generation actually stated.
No LLM queen is required.

Falsifiable hypothesis (P12 — Confirmation-Gate Hypothesis; P11 is the comb):

    Confirming generations against an imprint before storage — grading on
    closeness-to-copy — raises the ingestion of genuine facts and suppresses
    hedge/refusal pollution, so that (a) ingestion_rate and (b) honest
    retrieval recall improve relative to the current rule-based hedge filter
    alone, on the same conversations, at equal run cost.

Imprints:
- ``FixtureImprint`` — offline/synthetic: the fixture's ground-truth answers
  (conversation_id -> query -> expected answer). Deterministic and cheap.
- ``DigestImprint`` — live "chronicler-lite": fact terms from previously
  *confirmed* replies per conversation. First mentions (no established facts)
  fall to a substantive-content floor instead of being graded against nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from cortex.hedges import is_hedge_reply

# Content-term extraction mirrors experiments.retrieval_diagnostic's
# _content_terms/_answer_fact_terms (strata/ is self-contained; the A/B replay
# tool imports from here so both stay in sync).
_WORD_RE = re.compile(r"[a-z0-9_+#./-]+")
_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have how in is it its of on or "
    "that the this to was were what when where which who why with you your we "
    "our i me my they them their he she it would could should will can do does "
    "did not no so than then there here if into over under about more most "
    "other some such only own same too very just also us".split()
)

STOPWORDS = _STOPWORDS  # exported for the A/B tool / tests


def content_terms(text: str) -> set[str]:
    words = set(_WORD_RE.findall((text or "").lower()))
    return {w for w in words if w not in _STOPWORDS}


def fact_terms(query: str, answer: str) -> set[str]:
    """The distinctive facts an answer adds beyond the query."""
    return content_terms(answer) - content_terms(query)


@dataclass
class GateDecision:
    decision: str  # accept | reject | flag
    ingestion_ratio: Optional[float]  # share of imprint facts stated (None = no imprint facts)
    imprint_facts: int
    stated_facts: set[str]
    reply_facts: set[str]  # new facts the reply adds beyond the query (digest feed)
    substantive: bool
    rule_hedge: bool  # would the rule-based hedge filter have rejected this?
    reason: str = ""


class FixtureImprint:
    """Offline imprint: fixture ground-truth answers per conversation.

    ``answer_map`` is {conversation_id: {user_query: expected_answer}} — the
    same shape ``experiments.retrieval_diagnostic._fixture_answer_map`` builds.
    """

    def __init__(self, answer_map: dict[str, dict[str, str]]):
        self._answer_map = answer_map

    def facts_for(self, conversation_id: str, query: str) -> set[str]:
        expected = self._answer_map.get(conversation_id, {}).get(query, "")
        return fact_terms(query, expected) if expected else set()

    def confirm(self, conversation_id: str, reply_facts: set[str]) -> None:
        pass  # the fixture is fixed; nothing to accumulate


class DigestImprint:
    """Chronicler-lite: fact terms from previously *confirmed* replies per
    conversation. A reply is graded against facts already established; first
    mentions with no established facts fall to the substantive floor."""

    def __init__(self) -> None:
        self._facts: dict[str, set[str]] = {}

    def facts_for(self, conversation_id: str, query: str) -> set[str]:
        return set(self._facts.get(conversation_id, ()))

    def confirm(self, conversation_id: str, reply_facts: set[str]) -> None:
        self._facts.setdefault(conversation_id, set()).update(reply_facts)


class ConfirmationGate:
    """Grades each generation against the imprint before it enters the store.

    Thresholds:
    - rule_hedge → reject (parity with the S5 rule: refusals never stored).
    - imprint facts exist: accept if ingestion_ratio >= accept_threshold;
      reject if < flag_threshold; flag between (borderline — stored but
      marked, and logged for the human gate / chronicler review).
    - no imprint facts (first mention / never-established topic): accept only
      if the reply is *substantive* — it adds >= substantive_floor content
      terms beyond the query. Thin or refusal-like replies are rejected.
    """

    def __init__(
        self,
        accept_threshold: float = 0.4,
        flag_threshold: float = 0.2,
        substantive_floor: int = 3,
        hedge_rule: Optional[Callable[[str], bool]] = None,
    ):
        self.accept_threshold = accept_threshold
        self.flag_threshold = flag_threshold
        self.substantive_floor = substantive_floor
        self.hedge_rule = hedge_rule or is_hedge_reply
        self.stats: dict = {
            "accepted": 0, "rejected": 0, "flagged": 0,
            "rule_hedge_rejects": 0, "thin_rejects": 0,
            "stored_refusals": 0,
            "ingestion_ratios": [],
        }

    def decide(
        self, conversation_id: str, query: str, reply: str, imprint
    ) -> GateDecision:
        rule_hedge = self.hedge_rule(reply)
        reply_terms = content_terms(reply)
        q_terms = content_terms(query)
        reply_facts = reply_terms - q_terms
        imprint_facts = imprint.facts_for(conversation_id, query)
        stated = reply_terms & imprint_facts if imprint_facts else set()
        ratio = len(stated) / len(imprint_facts) if imprint_facts else None
        substantive = len(reply_facts) >= self.substantive_floor

        if rule_hedge:
            decision = "reject"
            reason = "rule_hedge"
            self.stats["rule_hedge_rejects"] += 1
        elif imprint_facts:
            if ratio >= self.accept_threshold:
                decision = "accept"
                reason = "copy"
            elif ratio < self.flag_threshold:
                decision = "reject"
                reason = "not_copy"
            else:
                decision = "flag"
                reason = "borderline"
        else:
            if substantive:
                decision = "accept"
                reason = "substantive_first_mention"
            else:
                decision = "reject"
                reason = "thin"
                self.stats["thin_rejects"] += 1

        if decision == "accept":
            self.stats["accepted"] += 1
        elif decision == "reject":
            self.stats["rejected"] += 1
        else:
            self.stats["flagged"] += 1
        if rule_hedge and decision != "reject":
            self.stats["stored_refusals"] += 1
        if ratio is not None:
            self.stats["ingestion_ratios"].append(ratio)

        return GateDecision(
            decision=decision,
            ingestion_ratio=ratio,
            imprint_facts=len(imprint_facts),
            stated_facts=stated,
            reply_facts=reply_facts,
            substantive=substantive,
            rule_hedge=rule_hedge,
            reason=reason,
        )

    def summary(self) -> dict:
        ratios = self.stats["ingestion_ratios"]
        return {
            "accepted": self.stats["accepted"],
            "rejected": self.stats["rejected"],
            "flagged": self.stats["flagged"],
            "rule_hedge_rejects": self.stats["rule_hedge_rejects"],
            "thin_rejects": self.stats["thin_rejects"],
            "stored_refusals": self.stats["stored_refusals"],
            "mean_ingestion_ratio": (
                round(sum(ratios) / len(ratios), 3) if ratios else None
            ),
        }