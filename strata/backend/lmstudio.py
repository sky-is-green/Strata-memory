"""LM Studio (llama.cpp) backend.

Default port 1234 (matches the running Gatekeeper LM Studio host). No surgical
KV-cache API; always sends the (compressed) context. The model id is resolved via
Gatekeeper's loaded-model list when empty (see cortex/interop.py).
"""

from __future__ import annotations

from backend.openai_compat import OpenAICompatBackend


class LMStudioBackend(OpenAICompatBackend):
    supports_surgical_edits = False

    def __init__(
        self,
        base_url: str = "http://localhost:1234",
        model: str = "",
        api_key: str = "lm-studio",
        **kwargs,
    ) -> None:
        super().__init__(base_url=base_url, model=model, api_key=api_key, **kwargs)
