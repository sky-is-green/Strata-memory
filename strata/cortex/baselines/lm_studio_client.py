"""OpenAI-compatible client for local inference backends (LM Studio / llama.cpp).

LM Studio exposes an OpenAI-compatible API on localhost:1234. This client is the
minimal transport used by the S0.5 baseline harness; S3 generalizes it into the
shared ``LLMBackend`` abstraction. Reuses Gatekeeper's endpoint convention:
host-only -> http://localhost:1234, full URL passthrough.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests

from backend.openai_compat import DEFAULT_MAX_TOKENS

DEFAULT_BASE_URL = "http://localhost:1234"


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    model: str


class LMStudioClient:
    """Thin OpenAI-compatible chat-completions client for a local server."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = "",
        api_key: str = "lm-studio",
        timeout: int = 300,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    # ------------------------------------------------------------------
    def health(self) -> Optional[dict]:
        """Return the loaded-models list, or None if the server is unreachable."""
        try:
            resp = requests.get(
                f"{self.base_url}/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return None

    def list_models(self) -> list[str]:
        data = self.health() or {}
        return [m.get("id", "") for m in data.get("data", [])]

    # ------------------------------------------------------------------
    def generate(
        self,
        messages: list[dict],
        sampling_params: Optional[dict] = None,
    ) -> GenerationResult:
        """Send a chat-completions request and time it. Raises on transport error."""
        payload = {
            "model": self.model,
            "messages": messages,
            **(sampling_params or {"temperature": 0.2, "max_tokens": DEFAULT_MAX_TOKENS}),
        }
        start = __import__("time").monotonic()
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        latency_ms = (__import__("time").monotonic() - start) * 1000.0
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage", {})
        return GenerationResult(
            text=data["choices"][0]["message"]["content"],
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
            latency_ms=latency_ms,
            model=data.get("model", self.model),
        )

    def is_oom(self, error: Exception) -> bool:
        """Heuristic: does an exception look like an out-of-memory event?"""
        msg = str(error).lower()
        return any(k in msg for k in ("out of memory", "oom", "cuda out of memory"))