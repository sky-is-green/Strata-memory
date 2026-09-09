"""Medium drone: domain-specific encoder (default microsoft/graphcodebert-base).

~400MB, GPU-preferred, 20-50ms per query. Only invoked for chunks the ultra-small
drone flagged as uncertain (see EscalationHandler in cortex/routing.py).

Two scoring modes:
  - ``cross`` (default): cross-encoder-style (query, chunk) pair -> [CLS] mean.
  - ``bi``: bi-encoder — embed query and chunk separately, score by cosine
    similarity. Required for code-specialized bi-encoders (e.g. CodeRankEmbed,
    codet5p-embedding, CodeSage).

An ``embed_fn`` or ``score_pair_fn`` may be injected for offline unit tests,
decoupling the wrapper from the heavy torch model.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from sieve.scores import ChunkScore


class MediumDrone:
    """Domain-aware scorer. Same interface as UltraSmallDrone."""

    def __init__(
        self,
        model_name: str = "microsoft/graphcodebert-base",
        device: str = "cpu",
        model=None,
        tokenizer=None,
        score_pair_fn: Optional[Callable[[str, str], float]] = None,
        embed_fn: Optional[Callable[[str], object]] = None,
        mode: str = "cross",
        pooling: str = "mean",
        trust_remote_code: bool = False,
        add_eos_token: bool = False,
        max_length: int = 512,
    ) -> None:
        if mode not in ("cross", "bi"):
            raise ValueError(f"unknown mode: {mode!r}")
        if pooling not in ("mean", "cls"):
            raise ValueError(f"unknown pooling: {pooling!r}")
        self.model_name = model_name
        self.device = device
        self._model = model
        self._tokenizer = tokenizer
        self._score_pair_fn = score_pair_fn
        self._embed_fn = embed_fn
        self.mode = mode
        self.pooling = pooling
        self.trust_remote_code = trust_remote_code
        self.add_eos_token = add_eos_token
        self.max_length = max_length
        self._torch = None

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._score_pair_fn is not None or self._embed_fn is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=self.trust_remote_code,
            add_eos_token=self.add_eos_token,
        )
        self._model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=self.trust_remote_code
        ).to(self.device).eval()

    def score(self, query: str, chunks: list[str]) -> list[ChunkScore]:
        self._ensure_loaded()
        if self._score_pair_fn is not None:
            return [
                ChunkScore(i, self._score_pair_fn(query, chunk), 0.85, source="medium")
                for i, chunk in enumerate(chunks)
            ]
        if self.mode == "bi":
            q = self._embed(query)
            return [
                ChunkScore(i, self._cosine(q, self._embed(chunk)), 0.85, source="medium")
                for i, chunk in enumerate(chunks)
            ]
        return [
            ChunkScore(i, self._score_pair(query, chunk), 0.85, source="medium")
            for i, chunk in enumerate(chunks)
        ]

    def _score_pair(self, query: str, chunk: str) -> float:
        """Cross-encoder relevance: cosine between the pooled [CLS] of the
        (query, chunk) joint input and the pooled [CLS] of the query alone.

        The previous implementation returned ``float(cls_emb.mean())`` — the
        mean of the [CLS] vector, which is a meaningless, near-constant scalar
        (measured: 0.058 for every pair). Cosine of a pooled representation is
        the standard cross-encoder similarity signal.
        """
        q_emb = self._embed(query)
        joint = self._joint_embed(query, chunk)
        return self._cosine(q_emb, joint)

    def _joint_embed(self, query: str, chunk: str) -> np.ndarray:
        inputs = self._tokenizer(
            query, chunk, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        if self.device != "cpu":
            inputs = inputs.to(self.device)
        with self._torch.no_grad():
            hidden = self._model(**inputs).last_hidden_state
        if self.pooling == "mean":
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:  # cls
            pooled = hidden[:, 0, :]
        return pooled.squeeze(0).cpu().numpy()

    def _embed(self, text: str) -> np.ndarray:
        """Return a pooled embedding for *text* (bi-encoder mode)."""
        if self._embed_fn is not None:
            return np.asarray(self._embed_fn(text), dtype=float)
        inputs = self._tokenizer(
            text, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        if self.device != "cpu":
            inputs = inputs.to(self.device)
        with self._torch.no_grad():
            hidden = self._model(**inputs).last_hidden_state
        if self.pooling == "mean":
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:  # cls
            pooled = hidden[:, 0, :]
        return pooled.squeeze(0).cpu().numpy()

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(a @ b / denom) if denom else 0.0
