"""Sampling parameters — the experimenter's surface for inference settings.

The OpenAI-compatible seam already supports the standard sampling fields; the
CLIs and the sidecar just never exposed them. This module is the single
validation point: parse a ``--sampling`` JSON string or dict, keep only known
fields, and merge into the request's sampling dict (per-call params win).

Known fields (OpenAI-compat + llama.cpp extras):
- temperature (0..2), top_p (0..1), top_k (int), min_p (0..1)
- repeat_penalty (llama.cpp), presence_penalty / frequency_penalty (OpenAI)
- stop (str or list), seed (int)
- mirostat / mirostat_tau / mirostat_eta (llama.cpp)
"""

from __future__ import annotations

import json
from typing import Any, Optional

SAMPLING_FIELDS = frozenset({
    "temperature", "top_p", "top_k", "min_p", "repeat_penalty",
    "presence_penalty", "frequency_penalty", "stop", "seed",
    "mirostat", "mirostat_tau", "mirostat_eta",
})


def parse_sampling(raw: Optional[str | dict]) -> dict:
    """Validate + normalize a sampling spec (JSON string or dict).

    Unknown keys are dropped (typos must not silently break a run); values
    are passed through as-is so the backend's own validation applies.
    """
    if raw is None or raw == "":
        return {}
    data = raw if isinstance(raw, dict) else json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("sampling must be a JSON object")
    out = {k: v for k, v in data.items() if k in SAMPLING_FIELDS}
    return out


def sampling_fingerprint(sampling: dict) -> str:
    """Stable short string for reports/reproducibility."""
    if not sampling:
        return "default"
    return json.dumps({k: sampling[k] for k in sorted(sampling)}, sort_keys=True)


def merge_sampling(base: dict, overrides: dict) -> dict:
    """Merge two sampling dicts; overrides win. Returns a new dict."""
    out = dict(base or {})
    out.update(overrides or {})
    return out