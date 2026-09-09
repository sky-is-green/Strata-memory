"""Engine profiles — the "LM Studio-like" management surface.

A provider says *where* to talk (base_url/key/model). An engine profile says
*what kind of engine it is and what it can do*: which sampling capabilities
exist, whether the server supports prefix caching / streaming / a reasoning
toggle, and the advisory load configuration (context length, GPU layers,
threads, KV quantization, flash attention) that a Studio shell would apply
when starting or configuring the server.

Honesty note: load options are *advisory* — the live local backend is LM
Studio, whose server settings are GUI-managed and not API-controllable; they
become actionable once the Studio shell controls llama.cpp-server/vLLM launch
config (HARNESS-SPEC.md). Sampling defaults, by contrast, are applied to every
request through the OpenAI-compatible seam.

Config lives in a JSON file (default ``engines.local.json``, gitignored)::

    {
      "engines": [
        {"name": "lmstudio-bonsai", "kind": "lmstudio",
         "base_url": "http://localhost:1234",
         "load_options": {"context": 32768, "gpu_layers": 99},
         "capabilities": ["prefix_caching", "streaming"],
         "sampling": {"temperature": 0.7, "top_p": 0.9}},
        {"name": "vllm-local", "kind": "vllm",
         "load_options": {"max_model_len": 32768, "tensor_parallel": 1},
         "capabilities": ["surgical_kv", "streaming"]}
      ],
      "default": "lmstudio-bonsai"
    }
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from backend.sampling import merge_sampling

DEFAULT_ENGINES_FILE = "engines.local.json"
ENGINES_FILE_ENV = "HARNESS_ENGINES_FILE"

ENGINE_KINDS = ("lmstudio", "llama_cpp", "vllm", "ollama", "hosted")
ENGINE_CAPABILITIES = (
    "prefix_caching", "streaming", "reasoning_toggle", "kv_cache_quant",
    "parallel_slots", "surgical_kv",
)


@dataclass
class EngineProfile:
    """One inference engine the strata can talk to."""

    name: str
    kind: str = "lmstudio"
    base_url: str = ""
    load_options: dict = field(default_factory=dict)  # advisory server config
    capabilities: list[str] = field(default_factory=list)
    sampling: dict = field(default_factory=dict)  # request defaults

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EngineProfile":
        if not isinstance(data, dict):
            raise ValueError("engine entry must be an object")
        name = str(data.get("name") or "").strip()
        if not name:
            raise ValueError("engine entry missing 'name'")
        kind = str(data.get("kind") or "lmstudio").lower()
        if kind not in ENGINE_KINDS:
            raise ValueError(
                f"engine '{name}': unknown kind '{kind}' (known: "
                + ", ".join(ENGINE_KINDS) + ")"
            )
        caps = [str(c).lower() for c in (data.get("capabilities") or [])]
        unknown = [c for c in caps if c not in ENGINE_CAPABILITIES]
        if unknown:
            raise ValueError(
                f"engine '{name}': unknown capabilities {unknown} "
                f"(known: {', '.join(ENGINE_CAPABILITIES)})"
            )
        sampling = data.get("sampling") or {}
        if not isinstance(sampling, dict):
            raise ValueError(f"engine '{name}': sampling must be an object")
        return cls(
            name=name,
            kind=kind,
            base_url=str(data.get("base_url") or ""),
            load_options=dict(data.get("load_options") or {}),
            capabilities=caps,
            sampling=sampling,
        )

    def merged_sampling(self, overrides: Optional[dict]) -> dict:
        """Sampling defaults + per-call overrides (overrides win)."""
        return merge_sampling(self.sampling, overrides or {})


@dataclass
class EngineRegistry:
    engines: list[EngineProfile] = field(default_factory=list)
    default: str = ""

    def resolve(self, name: Optional[str] = None) -> EngineProfile:
        if not self.engines:
            raise LookupError(
                "no engines configured; add them to engines.local.json "
                "or POST /v1/engines"
            )
        wanted = (name or self.default or "").strip().lower()
        if not wanted:
            return self.engines[0]
        for e in self.engines:
            if e.name.lower() == wanted:
                return e
        known = ", ".join(e.name for e in self.engines)
        raise LookupError(f"unknown engine '{name}' (known: {known})")


def engines_path(path: Optional[str | Path] = None) -> Path:
    if path:
        return Path(path)
    env = os.environ.get(ENGINES_FILE_ENV)
    if env:
        return Path(env)
    return Path(DEFAULT_ENGINES_FILE)


def load_engines(path: Optional[str | Path] = None) -> EngineRegistry:
    path = engines_path(path)
    if not path.exists():
        return EngineRegistry()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = data.get("engines", []) if isinstance(data, dict) else data
    return EngineRegistry(
        engines=[EngineProfile.from_dict(e) for e in (entries or [])],
        default=str(data.get("default") or "") if isinstance(data, dict) else "",
    )


def save_engines(reg: EngineRegistry, path: Optional[str | Path] = None) -> Path:
    path = engines_path(path)
    payload = {"engines": [e.to_dict() for e in reg.engines], "default": reg.default}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def redact_engine(profile: EngineProfile) -> dict:
    """Engine profiles hold no secrets today; kept for API symmetry."""
    return copy.deepcopy(profile.to_dict())