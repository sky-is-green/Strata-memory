"""Write-side hygiene for the retention layer — one module, one pipeline.

Every piece of text that becomes a persistent chunk passes through this
module, in this order, BEFORE fingerprinting:

  1. Boilerplate strip (P0) — harness-injected control lines (system-reminders
     around tool calls) are dropped; they must never become "memory" because
     at recall time such chunks score highly and get re-injected as stale
     instructions.
  2. Secret/blob sanitization (U2) — credential-shaped strings are redacted,
     oversized base64 blobs collapsed, hard length cap enforced.

Both consumers of the store's write path go through the same composite entry
point (:func:`prepare_for_storage`) — ``ContextStore.add_chunk`` and the
host-side payload-echo fingerprinting (harness ``app.py``) — so the normalized
form stored and the normalized form echoed are always comparable, which is
exactly what makes recency-echo dedup work. Fingerprinting
(:func:`content_fingerprint`) is applied AFTER this pipeline by its caller, so
dedup groups the sanitized form and every downstream tier (active store ->
checkpoints -> comb archives) inherits clean data.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional, Sequence

from retention.codec import normalize_codec


# ---------------------------------------------------------------------------
# 1. Harness boilerplate blocklist (P0 ingest filter)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Store-time hygiene (U2): credentials and oversized blobs must not reach the
# persistent tiers (active store -> checkpoints -> comb SSD archives), where
# they would survive indefinitely and be re-injected into future prompts.
# Applied inside add_chunk(), BEFORE fingerprinting, so dedup groups the
# sanitized form and every downstream consumer inherits clean data.
# ---------------------------------------------------------------------------

_SECRET_RULES = (
    # OpenAI-style keys (sk- followed by 16+ key characters)
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"), "[redacted-secret]"),
    # GitHub tokens (ghp_/gho_/ghu_/ghs_/ghr_)
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "[redacted-secret]"),
    # AWS access key IDs
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[redacted-secret]"),
    # Bearer/Authorization header values (keep the label, drop the secret)
    (
        re.compile(
            r"(?i)\b(authorization\s*[:=]\s*)bearer\s+[A-Za-z0-9._~+/=-]+"
        ),
        r"\1[redacted]",
    ),
    # key=value style assignments for common secret field names (quoted and
    # bare forms; the quotes go with the redacted value)
    (
        re.compile(
            r"(?i)\b(api[_-]?key|apikey|passwd|password|secret|token)"
            r"(\s*[:=]\s*)(\"[^\"]{4,}\"|'[^']{4,}'|[^\s,;\"']{4,})"
        ),
        r"\1\2[redacted]",
    ),
)

_BASE64_BLOB = re.compile(r"[A-Za-z0-9+/]{256,}={0,2}")

TRUNCATION_MARK = "\n…[truncated]"
DEFAULT_MAX_CHUNK_CHARS = 4000


def content_fingerprint(text: str) -> str:
    """12-hex content fingerprint shared by stored chunks and payload dedup.

    Stored chunks (``ContextChunk.fingerprint``) and incoming request message
    texts use this same function so verbatim copies compare equal.
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def sanitize_for_storage(text: str, max_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> str:
    """Redact credential-shaped strings, collapse base64 blobs, and enforce a
    hard length cap. Deterministic on normal prose: text without matches and
    within ``max_chars`` passes through byte-identical."""
    for pattern, replacement in _SECRET_RULES:
        text = pattern.sub(replacement, text)
    text = _BASE64_BLOB.sub("[base64 blob stripped]", text)
    if len(text) > max_chars:
        text = text[:max_chars] + TRUNCATION_MARK
    return text


# ---------------------------------------------------------------------------
# Composite entry point — the single write-side pipeline
# ---------------------------------------------------------------------------

def prepare_for_storage(
    text: str,
    max_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    block_prefixes: Optional[Sequence[str]] = None,
    sanitize: bool = True,
) -> Optional[str]:
    """The single write-side hygiene pipeline in order: boilerplate strip
    (P0), then secret/blob sanitization + length cap (U2).

    Returns ``None`` when nothing storable remains (blank or fully
    boilerplate input), else the cleaned text ready for fingerprinting and
    storage. Both consumers of the store's write path go through this so the
    stored form and the payload-echo form are always comparable:

      * ``ContextStore.add_chunk`` — passes the store's own settings;
      * harness ``app.py`` payload-echo guard — fingerprints incoming request
        texts with the same pipeline before forwarding.
    """
    # Codec normalization first (P1): repair UTF-8 misdecoded as a legacy
    # single-byte codec BEFORE boilerplate strip and sanitization. This is the
    # encoder-agnostic guard: any upstream transport that mangles multibyte
    # characters produces the same mojibake signature, and normalizing here
    # (before fingerprinting) also fixes dedup grouping for clean vs
    # misdecoded copies of identical text.
    cleaned = normalize_codec(text)
    cleaned = strip_boilerplate(cleaned, block_prefixes)
    if not cleaned or not cleaned.strip():
        return None
    if sanitize:
        cleaned = sanitize_for_storage(cleaned, max_chars)
    return cleaned
