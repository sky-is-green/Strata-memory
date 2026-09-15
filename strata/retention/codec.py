"""Codec normalization: repair UTF-8 text misdecoded as a legacy single-byte
codec (the "mojibake" class), encoder-agnostically.

Design rationale (2026-09-14, post-extraction review):
  Mojibake is not one bug at one site — it is a *class* of failure that any
  transport/encoder boundary can reintroduce: an HTTP response without a
  charset parameter (RFC legacy default ISO-8859-1), a terminal locale
  mismatch, a Windows CP1252 pipeline, a model-server swap. Patching the one
  decode call we found today only protects against that encoder.

  This module therefore repairs at the *store boundary* — the single point
  every harness shares — using a general reverse-mapping rather than a
  signature tuned to one incident:

    suspect text --encode(legacy codec)--> bytes --decode(utf-8)--> repaired

  Invariants (all covered by test_codec.py):
    * Clean text passes through byte-identical (fast path, no allocation).
    * A repair is accepted only when the round-trip succeeds AND the result
      has strictly fewer corruption markers than the input. Never destroys
      data: on any ambiguity the original is kept.
    * Idempotent: normalize(normalize(x)) == normalize(x).
    * Bounded: at most _MAX_PASSES reverse passes (handles mojibake-of-
      mojibake from mixed codec chains without unbounded iteration).

  Known limitation (v1): text mixing clean non-Latin-1 characters (CJK,
  emoji) with a corrupted segment is left unchanged rather than partially
  repaired — the whole-string round-trip cannot encode the clean part. The
  observed corruption mode is whole-turn misdecoding, so this is accepted;
  a segmenting repair (ftfy-style) can replace the core later without
  changing the interface.
"""

from __future__ import annotations

import re
from typing import Optional

# Suspect markers: C1 control chars (U+0080-U+009F — UTF-8 continuation bytes
# misread via Latin-1), and lead-byte shards Â/Ã/â (misread lead bytes of the
# E2/E3 families: dashes, arrows, quotes, bullets). Clean prose without these
# skips repair entirely.
_SUSPECT = re.compile("[\u0080-\u009f\u00c2\u00c3\u00e2]")

# Legacy single-byte codecs to try, most specific first. cp1252 covers the
# Windows pipeline; latin-1 is the RFC/HTTP legacy default and maps all 256
# bytes (cp1252 leaves five undefined).
_LEGACY_CODECS = ("cp1252", "latin-1")

_MAX_PASSES = 2


def _corruption_score(text: str) -> int:
    """Count corruption markers; lower is cleaner."""
    return len(_SUSPECT.findall(text)) + text.count("\ufffd")


def _one_reverse_pass(text: str, codec: str) -> Optional[str]:
    """Apply one reverse pass (encode as ``codec``, decode as UTF-8).

    Returns the candidate, or ``None`` when the round-trip is impossible
    (some character is not encodable in ``codec``, or the bytes are not
    valid UTF-8) — i.e. this codec cannot be the one that mangled the text.
    """
    try:
        return text.encode(codec).decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None


def _best_within_depth(text: str) -> str:
    """Breadth-first search over reverse-pass sequences of length up to
    ``_MAX_PASSES``; returns the reachable text with the lowest corruption
    score (ties keep the first found — cp1252 is tried before latin-1).

    BFS rather than greedy stepwise repair: in a mixed-codec chain
    (e.g. latin-1 then cp1252) the intermediate state can have an *equal*
    corruption score to the input, so a strict-improvement gate would stop
    one step short of the original. Enumerating depth-2 paths recovers it.
    """
    best = text
    best_score = _corruption_score(text)
    frontier = [text]
    seen = {text}
    for _ in range(_MAX_PASSES):
        nxt: list[str] = []
        for node in frontier:
            for codec in _LEGACY_CODECS:
                cand = _one_reverse_pass(node, codec)
                if cand is None or cand == node or cand in seen:
                    continue
                seen.add(cand)
                nxt.append(cand)
                score = _corruption_score(cand)
                if score < best_score:
                    best, best_score = cand, score
        frontier = nxt
    return best


# Characters a cp1252 pipeline can produce for bytes 0x80-0x9F (the C2
# range). These are legitimate *members* of a cp1252 mojibake shard (e.g.
# â¬â‚¬Â¢ = mangled U+2022), so segmentation must not split inside them.
def _build_cp1252_c2() -> frozenset:
    chars = set()
    for b in range(0x80, 0xA0):
        try:
            chars.add(bytes([b]).decode("cp1252"))
        except UnicodeDecodeError:
            continue  # five undefined bytes (0x81/8D/8F/90/9D)
    return frozenset(chars)


_CP1252_C2: frozenset = _build_cp1252_c2()


def _repair_to_fixed_point(text: str) -> str:
    """Iterate bounded BFS repair until no further strict improvement."""
    current = text
    while True:
        better = _best_within_depth(current)
        if better == current:
            break
        current = better
    return current


def _segmented_repair(text: str, is_separator) -> str:
    """Split ``text`` on ``is_separator``, repair each segment to a fixed
    point, rejoin. Separators pass through untouched."""
    out: list[str] = []
    buf: list[str] = []
    for ch in text:
        if is_separator(ch):
            if buf:
                out.append(_repair_to_fixed_point("".join(buf)))
                buf = []
            out.append(ch)
        else:
            buf.append(ch)
    if buf:
        out.append(_repair_to_fixed_point("".join(buf)))
    return "".join(out)


def normalize_codec(text: str) -> str:
    """Repair legacy-codec mojibake in ``text``.

    No-op on clean text (fast path). Two segmentation strategies are applied
    and the strictly cleaner result kept, iterated to a fixed point (each
    iteration lowers the corruption score, so it terminates; idempotent):

      * foreign-split: split only at characters no legacy codec can encode
        (above U+00FF and not in the cp1252 C2 range). Keeps whole cp1252
        shards intact for repair; clean CJK/emoji/arrows act as separators.
      * all-high-split: split at everything above U+00FF. Needed when clean
        C2-range characters (a real em-dash, a euro sign) sit next to
        latin-1 shards and block the whole-segment round-trip.

    A segment is only replaced on a strictly cleaner codec round-trip, which
    keeps the never-destroy invariant: ambiguous or partially-lost fragments
    (e.g. a user-retyped 2-byte shard) are left byte-identical.
    """
    if not text or not _SUSPECT.search(text):
        return text

    def _foreign(ch: str) -> bool:
        return ord(ch) > 0xFF and ch not in _CP1252_C2

    def _all_high(ch: str) -> bool:
        return ord(ch) > 0xFF

    current = text
    while True:
        candidates = (
            _segmented_repair(current, _foreign),
            _segmented_repair(current, _all_high),
        )
        best = min(candidates, key=_corruption_score)
        if _corruption_score(best) >= _corruption_score(current):
            break
        current = best
    return current
