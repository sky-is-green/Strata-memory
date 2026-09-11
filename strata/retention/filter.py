"""Ingest-time boilerplate filter (harness control text).

The proxied conversation carries harness-injected control text (e.g. Unsloth
Studio system-reminders around tool calls) that must never become persistent
"memory" chunks: at recall time such chunks score highly and get re-injected
into the LLM's context as stale instructions.

Detection is a normalized prefix match: each blocklist entry contributes its
first ``PREFIX_WINDOW`` characters as a case-sensitive head, and any line
whose leading-whitespace-stripped form starts with a head is boilerplate.
Only the first ``PREFIX_WINDOW`` chars are compared so that store-truncated
observations (``"... call ed..."``) still match.
"""

from __future__ import annotations

from typing import Optional, Sequence

PREFIX_WINDOW = 80

DEFAULT_INGEST_BLOCK_PREFIXES: tuple[str, ...] = (
    "You have access to enabled tools. If a tool is needed to satisfy the "
    "user's request or complete the action you described, call ed...",
    "You have used all available tool calls. Based on everything you have "
    "found so far, provide your final answer now. Do not call any...",
)


def _heads(prefixes: Sequence[str]) -> list[str]:
    return [p[:PREFIX_WINDOW] for p in (prefixes or []) if p]


def is_boilerplate_line(
    line: str, prefixes: Optional[Sequence[str]] = None,
) -> bool:
    """True when ``line`` (leading whitespace ignored) starts with any
    blocklisted prefix head. Case-sensitive."""
    heads = _heads(
        DEFAULT_INGEST_BLOCK_PREFIXES if prefixes is None else prefixes
    )
    stripped = line.lstrip()
    return any(stripped.startswith(h) for h in heads)


def strip_boilerplate(
    text: str, prefixes: Optional[Sequence[str]] = None,
) -> str:
    """Remove boilerplate lines from ``text``.

    Lines that match the blocklist are dropped; anything else is returned
    byte-identical (no stripping/reflowing), so normal conversational
    content passes through unchanged. Returns ``""`` when every line is
    boilerplate (or the text is blank).
    """
    heads = _heads(
        DEFAULT_INGEST_BLOCK_PREFIXES if prefixes is None else prefixes
    )
    if not heads or not text or not text.strip():
        return "" if not text or not text.strip() else text
    lines = text.splitlines()
    kept = [ln for ln in lines if not ln.lstrip().startswith(tuple(heads))]
    if len(kept) == len(lines):
        return text
    remainder = "\n".join(kept)
    return "" if not remainder.strip() else remainder.strip()
