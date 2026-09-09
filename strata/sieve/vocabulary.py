"""Domain vocabulary for targeted masking / relevance boosting.

Loaded from ``vocab/*.json`` at startup. Chunks containing vocabulary terms get
a relevance-score boost from the ultra-small drone (configurable weight, default
+0.15). One vocabulary per domain keeps the static-vocabulary system simple until
dynamic hot-swapping is proven (Pitfall 4).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_VOCAB_DIR = Path(__file__).resolve().parents[1] / "vocab"


def _collect_terms(data: dict) -> list[str]:
    """Flatten a vocabulary JSON file's groups into a single term list."""
    terms: list[str] = []
    for value in data.values():
        if isinstance(value, list):
            terms.extend(str(v) for v in value)
        elif isinstance(value, str) and value != "domain" and value != "description":
            terms.append(value)
    return terms


class Vocabulary:
    """A set of domain terms with fast case-insensitive membership matching."""

    def __init__(self, terms: Iterable[str]) -> None:
        self.terms = {str(t).strip().lower() for t in terms if str(t).strip()}
        ordered = sorted(self.terms, key=len, reverse=True)
        if ordered:
            self._pattern = re.compile(
                "|".join(re.escape(t) for t in ordered), re.IGNORECASE
            )
        else:
            self._pattern = None

    @classmethod
    def load(
        cls, *domain_names: str, vocab_dir: Optional[str | Path] = None
    ) -> "Vocabulary":
        """Load and merge one or more domain vocabularies from ``vocab/*.json``."""
        vocab_dir = Path(vocab_dir) if vocab_dir else DEFAULT_VOCAB_DIR
        terms: list[str] = []
        for name in domain_names:
            path = vocab_dir / f"{name}.json"
            if not path.exists():
                raise FileNotFoundError(f"vocabulary not found: {path}")
            terms.extend(_collect_terms(json.loads(path.read_text(encoding="utf-8"))))
        return cls(terms)

    def matches(self, text: str) -> bool:
        """True if any vocabulary term appears in *text* (case-insensitive)."""
        if self._pattern is None:
            return False
        return bool(self._pattern.search(text))

    def matched_terms(self, text: str) -> list[str]:
        """Return the unique vocabulary terms found in *text*."""
        if self._pattern is None:
            return []
        return list(dict.fromkeys(self._pattern.findall(text)))

    @property
    def size(self) -> int:
        return len(self.terms)
