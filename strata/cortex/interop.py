"""Gatekeeper interop seam.

Where Gatekeeper Studio already solves a problem (endpoint resolution, confidence
calibration, reliability tracking, safe config merge), the strata consumes the
documented HOST-SEAM JSON shapes rather than re-implementing the logic.

The full HOST-SEAM.md contract is not present in this repo, so this seam provides
defaults and a single place to wire real Gatekeeper values when available. It
never reaches into Gatekeeper internals.
"""

from __future__ import annotations

from backend.openai_compat import resolve_endpoint

DEFAULT_CONFIDENCE_THRESHOLD = 0.6


class GatekeeperSeam:
    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        endpoint_default: str = "http://localhost:1234",
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.endpoint_default = endpoint_default

    def normalize_lm_endpoint(self, base_url: str = "") -> str:
        """Gatekeeper Resolve-LmEndpoint: host-only/empty/full-URL handling."""
        return resolve_endpoint(base_url or self.endpoint_default)

    def calibrate_confidence(self, predicted: float, measured_pass_rate: float) -> float:
        """Calibrate drone confidence toward the measured pass rate (Gatekeeper
        confidence system). Simple beta-style blend with the threshold as anchor."""
        if measured_pass_rate is None:
            return predicted
        # Pull predicted toward measured pass rate by half the gap.
        return predicted + 0.5 * (measured_pass_rate - predicted)

    def drone_reliability(self, pass_rate: float, over_confidence: float = 0.0) -> float:
        """Gatekeeper Get-DroneReliability: pass rate minus over-confidence penalty."""
        return max(0.0, pass_rate - over_confidence)

    def merge_config(self, defaults: dict, overrides: dict) -> dict:
        """Safe config merge: overrides win, unknown keys kept, defaults preserved."""
        merged = dict(defaults)
        merged.update({k: v for k, v in (overrides or {}).items() if v is not None})
        return merged
