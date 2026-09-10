"""Lead-anchored refusal/hedge detection (shared by Strata and the S6
Confirmation Gate).

Live runs showed ~50% of replies were "no information regarding X"
boilerplate. Storing them pollutes the store: the hedge is later retrieved
*as context* for the same topic, so the model sees its own refusal instead of
a fact, and keeps refusing. Filtering keeps the store fact-bearing.

Detection is **lead-anchored**: markers are matched only against the opening
of the reply (after contraction normalization), because a refusal always
announces itself at the start, whereas a factual reply may contain an
incidental "I don't have specific details about your setup" caveat mid-text
and must still be stored. (Validated against all 136 live3 replies:
11 true hedges caught — including contraction refusals — 0 false positives.)
"""

from __future__ import annotations

HEDGE_MARKERS = (
    "no information",
    "no specific information",
    "no information available",
    "cannot fulfill",
    "i cannot",
    "cannot show",
    "cannot provide",
    "unable to fulfill",
    "unable to provide",
    "do not have access",
    "does not provide",
    "does not contain",
    "no access to",
    "do not have the",
    "i do not have",
    "do not know",
    "cannot recall",
    "unable to recall",
    "no knowledge of",
    "i am not aware",
    "i am not able",
    "i am unable",
    "no record of",
)

# Normalize contractions so "I don't have access" / "I can't provide" match
# the "do not have" / "cannot" markers above.
HEDGE_CONTRACTIONS = {
    "don't": "do not",
    "don’t": "do not",
    "can't": "cannot",
    "can’t": "cannot",
    "couldn't": "could not",
    "couldn’t": "could not",
    "wouldn't": "would not",
    "wouldn’t": "would not",
    "i'm": "i am",
    "i’m": "i am",
    "i've": "i have",
    "i’ve": "i have",
    "won't": "will not",
    "won’t": "will not",
    "it's": "it is",
    "it’s": "it is",
}

HEDGE_LEAD_WINDOW = 90


def is_hedge_reply(reply: str) -> bool:
    """True when the reply is a refusal/hedge that must not be stored."""
    t = (reply or "").strip().lower()
    if not t:
        return True  # empty replies carry no facts
    for short, long in HEDGE_CONTRACTIONS.items():
        t = t.replace(short, long)
    lead = t[: HEDGE_LEAD_WINDOW]
    return any(m in lead for m in HEDGE_MARKERS)