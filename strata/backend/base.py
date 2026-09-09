"""Abstract LLM backend contract.

Both vLLM and LM Studio (llama.cpp) expose OpenAI-compatible endpoints, so one
contract serves both. The strata only ever talks to an ``LLMBackend``; the concrete
class decides how context is delivered (surgical KV-cache edits for vLLM, full
compressed-context for LM Studio).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class LLMBackend(ABC):
    """Chat-completion backend contract."""

    supports_surgical_edits: bool = False

    @abstractmethod
    def generate(
        self,
        assembled_context: str,
        user_query: str,
        sampling_params: Optional[dict] = None,
    ) -> str:
        """Send the assembled context + query to the LLM and return the reply."""
        raise NotImplementedError

    def health(self) -> bool:
        """Return True if the backend is reachable."""
        raise NotImplementedError
