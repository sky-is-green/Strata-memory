"""Token counting.

``estimate_tokens`` is the cheap heuristic used everywhere; it consults an
optional *active* real tokenizer when one is set (``set_active_tokenizer``),
so budget/assembly counts become exact for a run without touching call sites.
The real tokenizer is the model's own ``tokenizer.json`` (``tokenizers``
lib — a transformers dependency, already installed), with tiktoken cl100k as
a secondary fallback, and the heuristic when neither is available.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def estimate_tokens(text: str) -> int:
    """Approximate token count (~1 token per 4 chars). No dependencies."""
    if _ACTIVE is not None:
        return _ACTIVE.count(text or "")
    return max(1, len(text or "") // 4)


_ACTIVE: Optional["Tokenizer"] = None


def set_active_tokenizer(tok: Optional["Tokenizer"]) -> None:
    """Set the run-wide real tokenizer (None restores the heuristic)."""
    global _ACTIVE
    _ACTIVE = tok


def active_tokenizer() -> Optional["Tokenizer"]:
    return _ACTIVE


class Tokenizer:
    def __init__(self, use_real: bool = False, tokenizer_path: Optional[str | Path] = None) -> None:
        self.use_real = use_real
        self._tok = None
        self.label = "heuristic"
        if use_real:
            self._load(tokenizer_path)

    def _load(self, tokenizer_path: Optional[str | Path]) -> None:
        if tokenizer_path:
            try:
                from tokenizers import Tokenizer as Tk

                path = Path(tokenizer_path)
                self._tok = Tk.from_file(str(path))
                self.label = str(path.resolve())
                return
            except Exception:  # noqa: BLE001
                self._tok = None
                self.label = "heuristic (tokenizer load failed)"
                return
        try:
            import tiktoken

            self._tok = tiktoken.get_encoding("cl100k_base")
            self.label = "tiktoken/cl100k_base"
        except Exception:  # noqa: BLE001
            self._tok = None
            self.label = "heuristic (no real tokenizer available)"

    @property
    def is_real(self) -> bool:
        return self._tok is not None

    def count(self, text: str) -> int:
        text = text or ""
        if self._tok is not None:
            try:
                if hasattr(self._tok, "encode"):
                    return max(1, len(self._tok.encode(text)))
                return max(1, len(self._tok.encode(text).ids))
            except Exception:  # noqa: BLE001
                pass
        return max(1, len(text) // 4)


def tokenizer_from_model_json(path: str | Path) -> Optional[Tokenizer]:
    """Build a Tokenizer from a HF ``tokenizer.json`` if present.

    ``path`` is the file itself; returns None when unusable (caller falls
    back to the heuristic default).
    """
    try:
        tok = Tokenizer(use_real=True, tokenizer_path=path)
        return tok if tok.is_real else None
    except Exception:  # noqa: BLE001
        return None