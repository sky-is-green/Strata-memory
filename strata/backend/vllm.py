"""vLLM backend.

Inherits the OpenAI-compatible transport; additionally exposes the KV-cache API
for surgical page-level edits (see backend/cache_manager.py). Default serving port
is 8000.
"""

from __future__ import annotations

from backend.openai_compat import OpenAICompatBackend


class VLLMBackend(OpenAICompatBackend):
    supports_surgical_edits = True

    def __init__(
        self, base_url: str = "http://localhost:8000", model: str = "qwen3.8-27b", **kwargs
    ) -> None:
        super().__init__(base_url=base_url, model=model, api_key="vllm", **kwargs)
