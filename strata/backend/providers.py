"""Provider configuration — one OpenAI-compatible endpoint description.

A *provider* is a named {base_url, api_key, model, headers} record. The same
record serves LM Studio / llama.cpp locally and hosted OpenAI-compatible APIs
(DeepSeek, OpenAI, OpenRouter, Groq, ...), so both the sidecar (harness/) and
the experiment CLIs (--provider NAME) talk to any backend through one seam.

Config lives in a JSON file (default ``providers.local.json``, gitignored via
the ``*.local.json`` rule) so keys never enter the repo or the logs::

    {
      "providers": [
        {"name": "lmstudio", "base_url": "http://localhost:1234",
         "api_key": "lm-studio", "model": "", "headers": {}},
        {"name": "deepseek", "base_url": "https://api.deepseek.com",
         "api_key": "sk-...", "model": "deepseek-chat", "headers": {}}
      ],
      "default": "lmstudio"
    }

Secrets handling: api_key values are only ever written to the config file;
``redacted()`` masks them for any API response or report.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_PROVIDERS_FILE = "providers.local.json"
PROVIDERS_FILE_ENV = "HARNESS_PROVIDERS_FILE"
MASK = "***"


@dataclass
class Provider:
    """One OpenAI-compatible endpoint (local server or hosted API)."""

    name: str
    base_url: str
    api_key: str = ""
    model: str = ""
    extra_headers: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        # JSON field name is "headers" (spec §4); the dataclass attribute
        # matches the OpenAICompatBackend parameter.
        out = asdict(self)
        out["headers"] = out.pop("extra_headers")
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "Provider":
        if not isinstance(data, dict):
            raise ValueError("provider entry must be an object")
        name = str(data.get("name") or "").strip()
        base_url = str(data.get("base_url") or "").strip()
        if not name:
            raise ValueError("provider entry missing 'name'")
        if not base_url:
            raise ValueError(f"provider '{name}' missing 'base_url'")
        headers = data.get("headers") if data.get("headers") is not None \
            else data.get("extra_headers")
        return cls(
            name=name,
            base_url=base_url,
            api_key=str(data.get("api_key") or ""),
            model=str(data.get("model") or ""),
            extra_headers={str(k): str(v) for k, v in (headers or {}).items()},
        )


@dataclass
class ProviderRegistry:
    """The loaded provider set plus which one is default."""

    providers: list[Provider] = field(default_factory=list)
    default: str = ""

    def resolve(self, name: Optional[str] = None) -> Provider:
        """Return the named provider (or the default); clear error otherwise."""
        if not self.providers:
            raise LookupError(
                "no providers configured; add them to providers.local.json "
                "(see providers.example.json) or POST /v1/provider/config"
            )
        wanted = (name or self.default or "").strip().lower()
        if not wanted:
            return self.providers[0]
        for p in self.providers:
            if p.name.lower() == wanted:
                return p
        known = ", ".join(p.name for p in self.providers)
        raise LookupError(f"unknown provider '{name}' (known: {known})")

    def redacted(self) -> list[dict]:
        return [redact_provider(p.to_dict()) for p in self.providers]


def redact_provider(provider_dict: dict) -> dict:
    """Copy of a provider dict with the api_key masked (safe for responses)."""
    out = copy.deepcopy(provider_dict)
    key = out.get("api_key")
    if key:
        out["api_key"] = MASK
    return out


def providers_path(path: Optional[str | Path] = None) -> Path:
    """Resolve the config path: explicit arg > env var > ./providers.local.json."""
    if path:
        return Path(path)
    env = os.environ.get(PROVIDERS_FILE_ENV)
    if env:
        return Path(env)
    return Path(DEFAULT_PROVIDERS_FILE)


def load_registry(path: Optional[str | Path] = None) -> ProviderRegistry:
    """Load a registry from JSON. A missing file yields an empty registry."""
    path = providers_path(path)
    if not path.exists():
        return ProviderRegistry()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = data.get("providers", []) if isinstance(data, dict) else data
    reg = ProviderRegistry(
        providers=[Provider.from_dict(e) for e in (entries or [])],
        default=str(data.get("default") or "") if isinstance(data, dict) else "",
    )
    return reg


def save_registry(reg: ProviderRegistry, path: Optional[str | Path] = None) -> Path:
    """Write the registry back as JSON (api keys in plaintext: file stays local)."""
    path = providers_path(path)
    payload = {
        "providers": [p.to_dict() for p in reg.providers],
        "default": reg.default,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def backend_kwargs(provider: Provider) -> dict:
    """Keyword args for OpenAICompatBackend built from a provider."""
    return {
        "base_url": provider.base_url,
        "api_key": provider.api_key or "lm-studio",
        "model": provider.model,
        "extra_headers": dict(provider.extra_headers),
    }


def apply_provider_overrides(parser_defaults: dict, args, provider: Provider) -> None:
    """Fill argparse results from a provider without clobbering explicit flags.

    ``parser_defaults`` is ``vars(parser.parse_args([]))``. Any of --base-url /
    --model still at its default value comes from the provider; an explicitly
    passed flag wins.
    """
    if getattr(args, "base_url", None) == parser_defaults.get("base_url"):
        args.base_url = provider.base_url
    if (
        getattr(args, "model", None) == parser_defaults.get("model")
        and provider.model
    ):
        args.model = provider.model
