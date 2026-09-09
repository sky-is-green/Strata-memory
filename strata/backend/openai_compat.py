"""OpenAI-compatible backend transport.

Baseline transport for any local OpenAI-compatible server (vLLM and LM Studio).
Passes the strata-assembled context as the system message and the user query as the
user message, replacing the naive rolling window. Endpoint normalization follows
Gatekeeper's Resolve-LmEndpoint convention (see cortex/interop.py).
"""

from __future__ import annotations

import requests

from backend.base import LLMBackend

DEFAULT_BASE_URL = "http://localhost:1234"

# Output ceiling when no explicit sampling_params are given. Reasoning models
# spend output tokens on chain-of-thought before the visible answer, so a small
# default cap silently produces empty replies; 4096 gives them headroom to
# reason *and* answer (max_tokens is a ceiling, not a target).
DEFAULT_MAX_TOKENS = 4096


def resolve_endpoint(base_url: str) -> str:
    """Normalize an LM endpoint.

    - empty / None -> DEFAULT_BASE_URL
    - full URL (http/https) -> passthrough
    - host[:port] -> http://{host[:port]}
    """
    if not base_url:
        return DEFAULT_BASE_URL
    base_url = base_url.strip()
    if base_url.startswith(("http://", "https://")):
        return base_url.rstrip("/")
    return f"http://{base_url}".rstrip("/")


class RequestsTransport:
    """Thin wrapper so backends are testable without real HTTP."""

    def post(self, url, json=None, headers=None, timeout=None):
        return requests.post(url, json=json, headers=headers, timeout=timeout)

    def get(self, url, headers=None, timeout=None):
        return requests.get(url, headers=headers, timeout=timeout)


class OpenAICompatBackend(LLMBackend):
    """Baseline OpenAI-compatible chat-completions backend."""

    supports_surgical_edits = False

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = "",
        api_key: str = "lm-studio",
        timeout: int = 300,
        transport=None,
        pinned_prefix: str = "",
        disable_thinking: bool = False,
        extra_headers: dict | None = None,
    ) -> None:
        self.base_url = resolve_endpoint(base_url)
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.transport = transport if transport is not None else RequestsTransport()
        # Stable leading system context. Kept byte-identical across turns so
        # llama.cpp's automatic prefix cache can reuse its KV (see KVCacheManager).
        self.pinned_prefix = pinned_prefix
        # When True, every request asks the server to disable chain-of-thought
        # (enable_thinking=false). Reasoning models otherwise burn their output
        # budget on hidden reasoning, which (a) wastes time and (b) makes small
        # max_tokens caps yield empty visible replies.
        self.disable_thinking = disable_thinking
        # Extra HTTP headers from the provider config (hosted APIs that need
        # e.g. an org id or a client header). Merged over the auth header.
        self.extra_headers = dict(extra_headers or {})
        # Usage from the most recent completion (prompt/completion/total tokens).
        # Populated only when the server returns a "usage" object (mock transports
        # omit it); lets P1 measure real decode tokens/sec.
        self.last_usage: dict = {}

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", **self.extra_headers}

    def generate(
        self,
        assembled_context: str,
        user_query: str,
        sampling_params: dict | None = None,
    ) -> str:
        # One leading system message. Strict chat templates (e.g. bonsai-27b)
        # reject a system message that is not the very first message, so the
        # pinned prefix and assembled context are merged into a single system
        # message. The pinned prefix stays a byte-identical leading substring
        # across turns, preserving llama.cpp's automatic prefix caching.
        if self.pinned_prefix:
            system = f"{self.pinned_prefix}\n\n{assembled_context}"
        else:
            system = assembled_context
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_query},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            **(sampling_params or {"temperature": 0.2, "max_tokens": DEFAULT_MAX_TOKENS}),
        }
        if self.disable_thinking:
            payload["enable_thinking"] = False
        resp = self.transport.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        self.last_usage = data.get("usage", {}) or {}
        return data["choices"][0]["message"]["content"]

    def models(self) -> list[str]:
        """Sorted model ids advertised by the server's /v1/models."""
        resp = self.transport.get(
            f"{self.base_url}/v1/models", headers=self._headers(), timeout=10
        )
        resp.raise_for_status()
        ids = [m.get("id", "") for m in resp.json().get("data", [])]
        return sorted(i for i in ids if i)

    def health(self) -> bool:
        try:
            resp = self.transport.get(
                f"{self.base_url}/v1/models", headers=self._headers(), timeout=10
            )
            return resp.ok
        except requests.RequestException:
            return False
