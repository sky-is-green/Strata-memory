"""Remembrance pass (retention layer).

Intercepts chunks moving toward the deletion boundary. Relevant chunks are
compressed (conversational fluff stripped) and re-injected at the front, with an
exponentially increasing decay multiplier — mathematical friction so chunks
saved many times need ever-stronger relevance to survive again.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from sieve.vocabulary import Vocabulary


@dataclass
class RemembranceResult:
    chunk_id: str
    saved: bool
    relevance_score: float
    compressed_content: Optional[str] = None
    new_decay: Optional[float] = None


class RemembrancePass:
    REMEMBRANCE_THRESHOLD = 0.65

    FILLER_PATTERNS = [
        r"^(I think|I believe|I feel|basically|essentially|actually),?\s*",
        r"(just|really|very|quite|rather)\s+",
        r"(kind of|sort of|a bit|a little)\s+",
        r"(as I mentioned|as discussed|per our conversation),?\s*",
        r"^(so|well|okay|right|sure),?\s*",
    ]

    def __init__(self, vocab: Optional[Vocabulary] = None) -> None:
        self.vocab = vocab if vocab is not None else Vocabulary.load("code", "general")

    def process(
        self, candidates: list, current_topic: str, drone
    ) -> list[RemembranceResult]:
        results: list[RemembranceResult] = []
        for chunk in candidates:
            relevance = drone.score(current_topic, [chunk.content])[0].relevance_score
            if relevance >= self.REMEMBRANCE_THRESHOLD:
                compressed = self._compress(chunk.content)
                new_decay = chunk.decay_multiplier * self._decay_factor(chunk.times_saved)
                chunk.times_saved += 1
                chunk.decay_multiplier = new_decay
                results.append(
                    RemembranceResult(
                        chunk_id=chunk.id,
                        saved=True,
                        relevance_score=relevance,
                        compressed_content=compressed,
                        new_decay=new_decay,
                    )
                )
            else:
                results.append(
                    RemembranceResult(
                        chunk_id=chunk.id, saved=False, relevance_score=relevance
                    )
                )
        return results

    def _compress(self, content: str) -> str:
        """Strip conversational fluff; keep sentences with domain terms."""
        cleaned = content
        # Loop until stable so stacked fillers ("basically I think ...") are all
        # removed even when each only matches at the start of the string.
        changed = True
        while changed:
            changed = False
            for pattern in self.FILLER_PATTERNS:
                new = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
                if new != cleaned:
                    cleaned = new
                    changed = True

        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        important = [s for s in sentences if self._has_domain_terms(s)]
        return " ".join(important) if important else cleaned

    def _has_domain_terms(self, sentence: str) -> bool:
        return self.vocab.matches(sentence)

    def _decay_factor(self, times_saved: int) -> float:
        """Exponential decay factor: 1.8, 2.1, 2.4, ... (S4 tunes this)."""
        return 1.8 + (times_saved * 0.3)
