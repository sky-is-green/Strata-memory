"""Prompt-injection / context-poisoning sanitization.

The strata re-injects user content as context, so adversarial chunks could try to
hijack the model with fake system instructions or role spoofing. This module
neutralizes such content by wrapping it in explicit data markers and masking
common injection patterns. Applied (optionally) before context is sent to the
LLM.
"""

from __future__ import annotations

import re

# Patterns that indicate an instruction-hijacking attempt.
_INJECTION_PATTERNS = [
    re.compile(r"ignore (all )?(previous|prior) (instructions|prompts)", re.I),
    re.compile(r"forget (everything|all previous)", re.I),
    re.compile(r"system(\s*prompt)?\s*:", re.I),
    re.compile(r"you are now .{0,40}", re.I),
    re.compile(r"^(assistant|user|system)\s*:\s*", re.I | re.M),
    re.compile(r"<\|?(im_start|system|assistant|user)\|?>", re.I),
]

_DATA_OPEN = "<|user_data|>"
_DATA_CLOSE = "<|/user_data|>"


def is_injection_attempt(text: str) -> bool:
    """True if *text* contains a prompt-injection pattern."""
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def sanitize_context(text: str) -> str:
    """Wrap content in data markers and mask injection patterns."""
    cleaned = text
    for pattern in _INJECTION_PATTERNS:
        cleaned = pattern.sub("[neutralized]", cleaned)
    # Escape any existing data markers to prevent marker spoofing.
    cleaned = cleaned.replace(_DATA_OPEN, "[open-marker]").replace(_DATA_CLOSE, "[close-marker]")
    return f"{_DATA_OPEN}\n{cleaned}\n{_DATA_CLOSE}"


def sanitize_chunks(chunks: list[str]) -> list[str]:
    """Sanitize each chunk (used before storing or formatting context)."""
    return [sanitize_context(c) for c in chunks]
