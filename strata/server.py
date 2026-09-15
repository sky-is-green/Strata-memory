"""Standalone Strata server: strata's own serving boundary.

This module is the part of the original sidecar that makes strata work
INDEPENDENTLY of any host application. Extracted from the hivebench sidecar
(harness/app.py) on 2026-09-15, keeping only what strata itself needs:

  * conversation loop         /v1/strata/{turn,curate,observe,stream}
  * persistence + lifecycle   conv-*.json store, LRU hives, reset/inspect/state
  * tuning surface            /v1/strata/defaults
  * curated OpenAI passthrough /v1/openai/{models,chat/completions}
                              (X-Strata-Conversation keyed; dsh / opencode /
                              any OpenAI client plugs in here)
  * provider config           /v1/provider/config (providers.local.json)

Everything imported here is strata's own (cortex/, retention/, sieve/,
backend/, logs/) or a third-party dependency. Nothing imports from
hivebench/. The wire contract matches the original sidecar exactly, so
existing clients keep working unchanged.

Run standalone::

    python -m strata.server --port 8765 --state-dir harness_state
    # or after `pip install .[serve]`:
    strata-serve --port 8765

Embed in a host application (the hivebench / DSH plug-in seam)::

    from strata.server import create_app
    app = create_app(state_dir=..., providers_file=...)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import requests
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from backend.cache_manager import KVCacheManager
from backend.openai_compat import OpenAICompatBackend
from backend.providers import (
    MASK,
    Provider,
    ProviderRegistry,
    backend_kwargs,
    load_registry,
    providers_path,
    save_registry,
)
from cortex.config import StrataConfig
from cortex.strata import Strata
from logs.event_logger import EventLogger
from retention.hygiene import (
    DEFAULT_MAX_CHUNK_CHARS,
    content_fingerprint,
    prepare_for_storage,
)
from retention.store import ContextStore

# Repo root (strata-memory/): one level above the strata/ package.
REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Small helpers (extracted verbatim from the sidecar)
# ---------------------------------------------------------------------------

def _cors_origins() -> list[str]:
    """Console origins. Default: localhost dev origins only - agent mode can
    execute code, so blanket CORS (*) is opt-in via HARNESS_CORS_ORIGINS=*."""
    raw = os.environ.get("HARNESS_CORS_ORIGINS", "").strip()
    if raw == "*":
        return ["*"]
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return ["http://localhost:5173", "http://127.0.0.1:5173",
            "http://localhost:3000", "http://127.0.0.1:3000",
            "http://localhost:8765", "http://127.0.0.1:8765"]


def _required_token() -> str:
    """When HARNESS_TOKEN is set, /v1/* requires this bearer token."""
    return os.environ.get("HARNESS_TOKEN", "").strip()


def _list_models(base_url: str) -> list[str]:
    """Model ids from an OpenAI-compatible upstream (for /v1/openai/models)."""
    resp = requests.get(f"{base_url}/v1/models", timeout=10)
    resp.raise_for_status()
    return [m.get("id") for m in resp.json().get("data", []) if m.get("id")]


# ---------------------------------------------------------------------------
# Conversation registry (the sidecar's _State, minus host-app machinery)
# ---------------------------------------------------------------------------

class ConversationRegistry:
    """Mutable app state: hives, locks, providers, factories.

    Extracted from the sidecar with the host-app pieces removed (engine
    profiles, run dirs, llama-server management). Request models still carry
    an ``engine`` field for wire compatibility; it is accepted and ignored
    here because engine sampling profiles are a host concern."""

    def __init__(
        self,
        ultra_factory: Callable[[], object],
        backend_factory: Callable[[Optional[str]], object],
        providers_file: Optional[Path],
        log_dir: str,
        state_dir: Optional[Path] = None,
    ) -> None:
        self.ultra_factory = ultra_factory
        self.backend_factory = backend_factory
        self.providers_file = providers_file
        self.log_dir = log_dir
        # Conversations persist here across restarts; None/empty disables.
        self.state_dir = Path(state_dir) if state_dir else None
        if self.state_dir is not None:
            self.state_dir.mkdir(parents=True, exist_ok=True)
        self.registry = ProviderRegistry()
        self._ultra = None
        self.hives: dict[str, Strata] = {}
        self.locks: dict[str, threading.Lock] = {}
        self.global_lock = threading.Lock()
        # Conversation lifecycle: LRU-bounded so a long-running server cannot
        # accumulate hives/loggers from every session that ever opened.
        self.max_conversations = int(os.environ.get("HARNESS_MAX_CONVERSATIONS", "50"))
        self._last_access: dict[str, float] = {}
        self._inflight: set[str] = set()
        self._loggers: dict[str, EventLogger] = {}
        self._conv_provider: dict[str, str] = {}  # per-conversation override

    def ultra(self):
        if self._ultra is None:
            self._ultra = self.ultra_factory()
        return self._ultra

    def _conv_path(self, conversation_id: str) -> Optional[Path]:
        """Per-conversation state file. Content-hashed name: arbitrary ids
        (session UUIDs, workspace keys, user input) stay safe on disk."""
        if self.state_dir is None:
            return None
        digest = hashlib.md5(conversation_id.encode("utf-8")).hexdigest()[:16]
        return self.state_dir / f"conv-{digest}.json"

    def save_conversation(self, conversation_id: str, strata: Strata) -> None:
        """Persist one conversation atomically (tmp file + os.replace)."""
        path = self._conv_path(conversation_id)
        if path is None:
            return
        payload = {
            "conversation_id": conversation_id,
            "turn": strata.turn,
            "with_backend": strata.backend is not None,
            "config": strata.config.to_dict(),
            "store": strata.store.to_dict(),
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)

    def drop_conversation(self, conversation_id: str) -> None:
        path = self._conv_path(conversation_id)
        if path is not None and path.exists():
            path.unlink()

    def strata_for(
        self, conversation_id: str, config_overrides: dict | None,
        with_backend: bool = True, engine: Optional[str] = None,
    ) -> Strata:
        """Get or lazily create the conversation's strata.

        A conversation not in memory but present in ``state_dir`` restores
        from disk (same serialization as the benchmark's checkpoint/resume),
        so the strata survives server restarts. In-memory hives are
        LRU-bounded (``HARNESS_MAX_CONVERSATIONS``); evicted conversations
        are persisted first and transparently restore on their next touch.

        ``with_backend=False`` (the curate/observe flow, where the caller's
        own shell generates) creates the strata without an LLM backend; a
        conversation is driven either fully (/v1/strata/turn) or externally
        (curate + observe), whichever touches it first wins.
        """
        with self.global_lock:
            strata = self.hives.get(conversation_id)
            if strata is not None:
                self._last_access[conversation_id] = time.monotonic()
                return strata

            def build(cfg: StrataConfig, backend: object | None) -> Strata:
                logger = self._loggers.get(conversation_id)
                if logger is None:
                    logger = EventLogger(log_dir=self.log_dir)
                    self._loggers[conversation_id] = logger
                h = Strata(
                    config=cfg,
                    ultra=self.ultra(),
                    backend=backend,
                    logger=logger,
                )
                self.hives[conversation_id] = h
                self.locks.setdefault(conversation_id, threading.Lock())
                return h

            path = self._conv_path(conversation_id)
            if path is not None and path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    strata = build(StrataConfig.from_dict(data["config"]),
                                   self.backend_factory(None)
                                   if data.get("with_backend") else None)
                    strata.store = ContextStore.from_dict(
                        data["store"], embed_fn=strata.ultra.embed
                    )
                    strata.turn = int(data["turn"])
                    self._last_access[conversation_id] = time.monotonic()
                    self._evict_locked(exclude=conversation_id)
                    return strata
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    print(f"strata-server: restoring {conversation_id} failed "
                          f"({exc}); starting fresh", file=sys.stderr)

            config = StrataConfig(confidence_mode="off")
            if config_overrides:
                merged = {**config.to_dict(), **config_overrides}
                config = StrataConfig.from_dict(merged)
            strata = build(config, self.backend_factory(None) if with_backend else None)
            self._last_access[conversation_id] = time.monotonic()
            self._evict_locked(exclude=conversation_id)
            return strata

    def _evict_locked(self, exclude: str) -> int:
        """LRU-evict idle conversations beyond the cap. Caller holds the
        global lock; in-flight conversations are never evicted, and evicted
        state is persisted first (restore-on-touch keeps it reachable)."""
        evicted = 0
        while len(self.hives) > self.max_conversations:
            candidates = [cid for cid in self.hives
                          if cid != exclude and cid not in self._inflight]
            if not candidates:
                break
            oldest = min(candidates, key=lambda c: self._last_access.get(c, 0.0))
            self.save_conversation(oldest, self.hives[oldest])
            logger = self._loggers.pop(oldest, None)
            if logger is not None:
                try:
                    logger.close()
                except Exception:  # noqa: BLE001 - eviction must not fail
                    pass
            self.hives.pop(oldest, None)
            self.locks.pop(oldest, None)
            self._last_access.pop(oldest, None)
            evicted += 1
        return evicted

    def begin(self, conversation_id: str) -> None:
        self._inflight.add(conversation_id)
        self._last_access[conversation_id] = time.monotonic()

    def end(self, conversation_id: str) -> None:
        self._inflight.discard(conversation_id)

    def drop(self, conversation_id: str) -> None:
        with self.global_lock:
            self.hives.pop(conversation_id, None)
            self.locks.pop(conversation_id, None)
            self._last_access.pop(conversation_id, None)
            logger = self._loggers.pop(conversation_id, None)
        if logger is not None:
            try:
                logger.close()
            except Exception:  # noqa: BLE001
                pass
        self.drop_conversation(conversation_id)

    def lock_for(self, conversation_id: str) -> threading.Lock:
        with self.global_lock:
            return self.locks.setdefault(conversation_id, threading.Lock())


# ---------------------------------------------------------------------------
# Request / response models (wire-compatible with the original sidecar)
# ---------------------------------------------------------------------------

class TurnRequest(BaseModel):
    query: str
    conversation_id: str = "default"
    model: Optional[str] = None  # override the provider's model for this turn's strata
    provider: Optional[str] = None  # per-conversation inference target (multi-model)
    engine: Optional[str] = None  # accepted for wire compat; ignored (host concern)
    config: Optional[dict] = None  # StrataConfig overrides (applied on creation)


class ResetRequest(BaseModel):
    conversation_id: str


class CurateRequest(BaseModel):
    query: str
    conversation_id: str = "default"
    engine: Optional[str] = None  # accepted for wire compat; ignored
    config: Optional[dict] = None


class ObserveRequest(BaseModel):
    conversation_id: str
    reply: str


class StreamTurnRequest(BaseModel):
    query: str
    conversation_id: str = "default"
    engine: Optional[str] = None  # accepted for wire compat; ignored
    config: Optional[dict] = None


class ProviderEntry(BaseModel):
    name: str
    base_url: str
    api_key: str = ""
    model: str = ""
    headers: dict = {}


class ProviderConfigRequest(BaseModel):
    providers: list[ProviderEntry]
    default: str = ""
    persist: bool = True


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    ultra_factory: Optional[Callable[[], object]] = None,
    backend_factory: Optional[Callable[[Optional[str]], object]] = None,
    providers_file: Optional[Path] = None,
    log_dir: str = "logs",
    state_dir: Optional[Path] = None,
) -> FastAPI:
    """Build the standalone strata server.

    ``ultra_factory`` / ``backend_factory`` are injectable for offline tests;
    defaults build the real L3-v2 drone (CPU MiniLM) and a provider-driven
    OpenAI-compat backend from ``providers_file`` (default: this repo's
    providers.local.json). ``state_dir=None`` defaults to ./harness_state
    (conversations survive restarts); passing an empty string disables
    persistence.
    """
    if providers_file is None:
        providers_file = REPO_ROOT / "providers.local.json"
    if state_dir is None:
        state_dir = Path(os.environ.get("STRATA_STATE_DIR", "harness_state"))

    def _default_ultra():
        embedding_backend = os.environ.get("HARNESS_EMBEDDING_BACKEND", "local")
        embedding_url = os.environ.get("HARNESS_EMBEDDING_URL", "")
        embedding_model = os.environ.get("HARNESS_EMBEDDING_MODEL", "default")
        if embedding_backend == "served" and embedding_url:
            from sieve.served import ServedEmbeddingDrone

            return ServedEmbeddingDrone(base_url=embedding_url,
                                        model=embedding_model)
        # CPU by design: the encoder must never contend with llama-server
        # for VRAM. MiniLM-L3 is 12M params; CPU embedding costs ~5ms/turn.
        # Single-threaded on purpose: llama.cpp already holds a large
        # thread pool, and letting PyTorch OpenMP spawn 16 more exhausts
        # the sandbox pids cap (EAGAIN -> silent worker death). One
        # thread is still ~5ms for a 384-dim embedding.
        import torch

        torch.set_num_threads(1)
        from sieve.ultra_small import UltraSmallDrone

        return UltraSmallDrone(confidence_mode="off", device="cpu")

    def _default_backend(model: Optional[str], provider: Optional[str] = None):
        kw = backend_kwargs(st.registry.resolve(provider))
        if model:
            kw["model"] = model
        return OpenAICompatBackend(**kw)

    app = FastAPI(title="Strata Server", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def token_guard(request: Request, call_next):
        required = _required_token()
        if required and request.url.path.startswith("/v1/"):
            supplied = request.headers.get("x-strata-token", "")
            if supplied != required:
                return JSONResponse({"detail": "invalid or missing token"},
                                    status_code=401)
        return await call_next(request)

    st = ConversationRegistry(
        ultra_factory=ultra_factory or _default_ultra,
        backend_factory=backend_factory or _default_backend,
        providers_file=providers_file,
        log_dir=log_dir,
        state_dir=state_dir if state_dir is not None else Path("harness_state"),
    )
    try:
        st.registry = load_registry(providers_file)
    except (ValueError, OSError) as exc:
        print(f"strata-server: ignoring unreadable providers config ({exc})",
              file=sys.stderr)
    app.state.registry = st

    # Eagerly load the encoder at startup, not lazily on first request.
    # By first-request time uvicorn's event loop + connection threads are
    # alive and tip the process over its thread budget, so MiniLM's OpenMP
    # pool fails to spawn (EAGAIN) and the worker dies silently. Loading
    # here - before uvicorn serves - keeps the load in a clean state.
    try:
        st.ultra()
    except Exception as exc:  # noqa: BLE001 - never block startup on encoder
        print(f"strata-server: encoder pre-load failed ({exc}); will retry lazily",
              file=sys.stderr)

    @app.get("/health")
    def health():
        return {"ok": True, "conversations": len(st.hives)}

    # ------------------------------------------------------------------
    # Conversation loop
    # ------------------------------------------------------------------

    @app.post("/v1/strata/turn")
    def strata_turn(req: TurnRequest):
        query = (req.query or "").strip()
        if not query:
            raise HTTPException(422, "query must not be empty")
        strata = st.strata_for(req.conversation_id, req.config, engine=req.engine)
        # Per-conversation inference target: provider and/or model override
        # swaps the conversation's backend (multi-model: pick any loaded one).
        current_provider = st._conv_provider.get(req.conversation_id)
        wants_backend = (req.provider and req.provider != current_provider) \
            or (req.model and isinstance(strata.backend, OpenAICompatBackend)
                and req.model != strata.backend.model)
        if wants_backend and isinstance(strata.backend, OpenAICompatBackend):
            new_backend = st.backend_factory(req.model, provider=req.provider)
            strata.backend = new_backend
            strata.cache = KVCacheManager(new_backend)
            st._conv_provider[req.conversation_id] = req.provider \
                or st.registry.default
        st.begin(req.conversation_id)
        with st.lock_for(req.conversation_id):
            result = strata.process_turn(req.query, conversation_id=req.conversation_id)
            st.save_conversation(req.conversation_id, strata)
        st.end(req.conversation_id)
        assembled = result.assembled
        return {
            "conversation_id": req.conversation_id,
            "turn": result.turn,
            "reply": result.reply,
            "assembled_content": assembled.content if assembled is not None else "",
            "token_count": result.token_count,
            "budget": result.budget,
            "mode": result.mode,
            "error": result.error,
            "timings": result.timings,
            "pes": result.pes,
            "degradation_level": result.degradation_level,
            "inspection": strata.inspect_turn(result),
        }

    @app.get("/v1/strata/inspect/{conversation_id}")
    def strata_inspect(conversation_id: str):
        """Last turn's full curation detail for the prompt inspector."""
        with st.global_lock:
            strata = st.hives.get(conversation_id)
        if strata is None:
            raise HTTPException(404, f"no such conversation: {conversation_id}")
        if not hasattr(strata, "_last_turn_result") or strata._last_turn_result is None:
            raise HTTPException(404, "no turn has been processed yet")
        return strata.inspect_turn(strata._last_turn_result)

    @app.post("/v1/strata/reset")
    def strata_reset(req: ResetRequest):
        st.drop(req.conversation_id)
        return {"ok": True}

    # ------------------------------------------------------------------
    # Curate / observe (Seam A, dsh-strata flow): the caller's own shell
    # generates - the server only assembles context and ingests replies.
    # ------------------------------------------------------------------

    @app.post("/v1/strata/curate")
    def strata_curate(req: CurateRequest):
        query = (req.query or "").strip()
        if not query:
            raise HTTPException(422, "query must not be empty")
        strata = st.strata_for(req.conversation_id, req.config, with_backend=False,
                               engine=req.engine)
        with st.lock_for(req.conversation_id):
            result = strata.process_turn(query, conversation_id=req.conversation_id)
            st.save_conversation(req.conversation_id, strata)
        assembled = result.assembled
        return {
            "conversation_id": req.conversation_id,
            "turn": result.turn,
            "assembled_content": assembled.content if assembled is not None else "",
            "token_count": result.token_count,
            "budget": result.budget,
            "mode": result.mode,
            "error": result.error,
            "timings": result.timings,
            "pes": result.pes,
            "degradation_level": result.degradation_level,
        }

    @app.post("/v1/strata/observe")
    def strata_observe(req: ObserveRequest):
        # lazily create: external integrators may observe before ever calling
        # curate (e.g. feeding back a reply for a session the studio has
        # never seen); the conversation materializes here.
        strata = st.strata_for(req.conversation_id, None, with_backend=False)
        reply = (req.reply or "").strip()
        stored = False
        if reply and not (
            strata.config.filter_hedge_replies and Strata._is_hedge_reply(reply)
        ):
            st.begin(req.conversation_id)
            with st.lock_for(req.conversation_id):
                stored = strata.store.add_chunk(strata.turn, reply) is not None
                if stored:
                    st.save_conversation(req.conversation_id, strata)
            st.end(req.conversation_id)
        return {"ok": True, "stored": stored, "turn": strata.turn}

    @app.post("/v1/strata/stream")
    async def strata_stream(req: StreamTurnRequest):
        query = (req.query or "").strip()
        if not query:
            raise HTTPException(422, "query must not be empty")
        try:
            provider = st.registry.resolve(None)
        except LookupError:
            raise HTTPException(502, "no provider configured; start a local "
                                     "server or configure one")
        base_url = provider.base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {provider.api_key or 'lm-studio'}",
                   **provider.extra_headers}
        strata = st.strata_for(req.conversation_id, req.config, with_backend=False)
        st.begin(req.conversation_id)
        with st.lock_for(req.conversation_id):
            result = strata.process_turn(query, conversation_id=req.conversation_id)
            st.save_conversation(req.conversation_id, strata)
        st.end(req.conversation_id)
        assembled = result.assembled
        curated = assembled.content if assembled is not None else ""
        payload = {
            "model": provider.model or "local",
            "messages": [
                {"role": "system", "content": curated or "You are a helpful assistant."},
                {"role": "user", "content": query},
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
            **(strata.config.sampling or {}),
        }
        if strata.config.max_tokens:
            payload["max_tokens"] = strata.config.max_tokens

        def sse():
            yield "data: " + json.dumps({
                "type": "meta", "turn": result.turn,
                "token_count": result.token_count, "budget": result.budget,
                "curated_chars": len(curated), "mode": result.mode,
            }) + "\n\n"

            started = time.time()
            parts: list[str] = []
            usage: dict = {}
            try:
                resp = requests.post(
                    f"{base_url}/v1/chat/completions", json=payload,
                    headers=headers, stream=True, timeout=600,
                )
                resp.raise_for_status()
                for raw in resp.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    line = raw[6:].strip() if raw.startswith("data:") else raw.strip()
                    if not line or line == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    usage = chunk.get("usage") or usage
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        text = delta.get("content")
                        if text:
                            parts.append(text)
                            yield "data: " + json.dumps({
                                "type": "delta", "text": text}) + "\n\n"
            except Exception as exc:  # noqa: BLE001 - surfaced as an event
                yield "data: " + json.dumps({
                    "type": "error", "error": str(exc)}) + "\n\n"

            reply = "".join(parts)
            stored = False
            if reply.strip() and not (
                strata.config.filter_hedge_replies
                and Strata._is_hedge_reply(reply)
            ):
                stored = strata.store.add_chunk(strata.turn, reply) is not None
                if stored:
                    st.save_conversation(req.conversation_id, strata)
            elapsed = max(time.time() - started, 1e-6)
            completion_tokens = (usage or {}).get("completion_tokens") or 0
            yield "data: " + json.dumps({
                "type": "done", "stored": stored,
                "tokens": completion_tokens,
                "seconds": round(elapsed, 2),
                "tokens_per_sec": round(completion_tokens / elapsed, 1)
                if completion_tokens else None,
            }) + "\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    @app.get("/v1/strata/defaults")
    def strata_defaults():
        """StrataConfig defaults - the source for the UI tuning form. Overrides
        ride each turn request's `config` and apply when a conversation is
        created (reset to re-tune)."""
        return StrataConfig().to_dict()

    @app.get("/v1/strata/state")
    def strata_state(conversation_id: Optional[str] = Query(default=None)):
        def snapshot(h: Strata) -> dict:
            return {
                "turn": h.turn,
                "store_chunks": len(h.store.all_chunks()),
                "comb_stats": dict(h.comb_stats),
            }

        if conversation_id:
            with st.global_lock:
                strata = st.hives.get(conversation_id)
            if strata is None and st.state_dir is not None \
                    and st._conv_path(conversation_id).exists():
                # lazy-restore a persisted conversation so state survives restarts
                strata = st.strata_for(conversation_id, None)
            if strata is None:
                raise HTTPException(404, f"no such conversation: {conversation_id}")
            return {**snapshot(strata), "conversation_id": conversation_id}
        with st.global_lock:
            items = {cid: snapshot(h) for cid, h in st.hives.items()}
        return {"count": len(items), "conversations": items}

    # ------------------------------------------------------------------
    # Curated OpenAI-compatible passthrough (Mode A integration: dsh,
    # opencode, any OpenAI client). Standard /chat/completions wire shape,
    # curated system context, the reply observed back into the store.
    # Conversation key: X-Strata-Conversation header > payload "user"
    # > "default".
    # ------------------------------------------------------------------

    @app.get("/v1/openai/models")
    def openai_models():
        try:
            provider = st.registry.resolve(None)
        except LookupError:
            raise HTTPException(502, "no provider configured")
        try:
            ids = _list_models(provider.base_url.rstrip("/"))
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            raise HTTPException(502, f"cannot list models from upstream: {exc}")
        if not ids and getattr(provider, "model", None):
            ids = [provider.model]
        return {"object": "list",
                "data": [{"id": m, "object": "model"} for m in ids]}

    @app.post("/v1/openai/chat/completions")
    async def openai_chat_completions(request: Request):
        payload = await request.json()
        messages = payload.get("messages") or []
        if not messages:
            raise HTTPException(422, "messages must not be empty")
        query = ""
        for m in reversed(messages):
            content = m.get("content") if m.get("role") == "user" else None
            if isinstance(content, str) and content.strip():
                query = content
                break
        if not query.strip():
            raise HTTPException(422, "no user message with text content")
        cid = (request.headers.get("X-Strata-Conversation")
               or (payload.get("user") or "") or "default")
        try:
            provider = st.registry.resolve(None)
        except LookupError:
            raise HTTPException(
                502, "no provider configured; configure one via /v1/provider/config "
                     "or providers.local.json")
        base_url = provider.base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {provider.api_key or 'lm-studio'}",
                   **provider.extra_headers}
        strata = st.strata_for(cid, payload.get("config"), with_backend=False)
        # Budget guard / forward window: Unsloth Studio proxies its ENTIRE
        # thread history, which can exceed the upstream context (observed:
        # 1.04M tokens vs 74k available -> llama.cpp 400). Curation carries
        # the memory, so oversized payloads are trimmed to the last few
        # turns; small payloads pass through untouched (dsh/opencode manage
        # their own windows and are unaffected). The trim runs BEFORE the
        # echo fingerprints below: only content actually forwarded may
        # suppress a stored chunk from curation - a fact trimmed off the
        # tail must stay retrievable.
        _MAX_FWD_CHARS = 60_000
        system_msg = (
            messages[0]
            if messages and messages[0].get("role") == "system"
            else None
        )
        body_msgs = messages[1:]
        if sum(len(str(m.get("content") or "")) for m in body_msgs) > _MAX_FWD_CHARS:
            body_msgs = body_msgs[-8:]
        # RC2: fingerprint the SAME normalized form the store persists -
        # prepare_for_storage is the store's own write pipeline (boilerplate
        # strip + secret sanitization with the conversation's own ingest
        # settings), so sanitized/truncated stored chunks can no longer
        # escape echo-dedup.
        _store = getattr(strata, "store", None)
        _prefixes = getattr(_store, "ingest_block_prefixes", None)
        _max_chars = getattr(_store, "max_chunk_chars", None) or DEFAULT_MAX_CHUNK_CHARS
        payload_fingerprints = set()
        forwarded_texts = (
            ([system_msg] if system_msg is not None else []) + body_msgs
        )
        for m in forwarded_texts:
            text = m.get("content")
            if not isinstance(text, str) or not text:
                continue
            prepared = prepare_for_storage(text, _max_chars, _prefixes)
            if prepared is not None:
                payload_fingerprints.add(content_fingerprint(prepared))
        with st.lock_for(cid):
            result = strata.process_turn(
                query, conversation_id=cid,
                payload_fingerprints=payload_fingerprints,
            )
            st.save_conversation(cid, strata)
        curated = result.assembled.content if result.assembled is not None else ""
        merged_sys = curated or "You are a helpful assistant."
        if system_msg is not None and system_msg.get("content"):
            merged_sys = merged_sys + "\n\n" + system_msg["content"]
        stream = bool(payload.get("stream"))
        upstream = {
            **payload,
            "model": provider.model or payload.get("model") or "local",
            "stream": stream,
            "messages": [{"role": "system", "content": merged_sys}] + body_msgs,
        }
        upstream.setdefault("stream_options", {"include_usage": True})

        def observe(reply: str) -> bool:
            stored = False
            if reply.strip() and not (
                strata.config.filter_hedge_replies
                and Strata._is_hedge_reply(reply)
            ):
                stored = strata.store.add_chunk(strata.turn, reply) is not None
                if stored:
                    st.save_conversation(cid, strata)
            return stored

        if not stream:
            resp = requests.post(
                f"{base_url}/v1/chat/completions", json=upstream,
                headers=headers, timeout=600,
            )
            resp.raise_for_status()
            data = resp.json()
            try:
                observe(data["choices"][0]["message"]["content"] or "")
            except (KeyError, IndexError):
                pass
            return data

        def sse():
            parts: list[str] = []
            try:
                resp = requests.post(
                    f"{base_url}/v1/chat/completions", json=upstream,
                    headers=headers, stream=True, timeout=600,
                )
                resp.raise_for_status()
                for raw in resp.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    line = raw[6:].strip() if raw.startswith("data:") else raw.strip()
                    if not line:
                        continue
                    if line == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            parts.append(delta["content"])
                    yield "data: " + json.dumps(chunk) + "\n\n"
            except Exception as exc:  # noqa: BLE001 - surfaced as an SSE error event
                yield "data: " + json.dumps({
                    "error": {"message": str(exc), "type": "strata_upstream_error"},
                }) + "\n\n"
            observe("".join(parts))

        return StreamingResponse(sse(), media_type="text/event-stream")

    # ------------------------------------------------------------------
    # Provider configuration (self-service: no file editing required)
    # ------------------------------------------------------------------

    @app.post("/v1/provider/config")
    def set_providers(req: ProviderConfigRequest):
        reg = ProviderRegistry(default=req.default)
        for entry in req.providers:
            data = entry.model_dump()
            if data.get("api_key") == MASK:
                # the UI echoes the mask back for untouched keys - keep the
                # stored secret instead of overwriting it with "***"
                previous = [p for p in st.registry.providers
                            if p.name.lower() == str(data.get("name", "")).lower()]
                data["api_key"] = previous[0].api_key if previous else ""
            try:
                reg.providers.append(Provider.from_dict(data))
            except ValueError as exc:
                raise HTTPException(422, str(exc))
        st.registry = reg
        persisted = None
        if req.persist:
            path = save_registry(reg, st.providers_file)
            persisted = str(path)
        return {"ok": True, "default": reg.default,
                "providers": reg.redacted(), "persisted_to": persisted}

    @app.get("/v1/provider/config")
    def get_providers():
        return {
            "default": st.registry.default,
            "providers": st.registry.redacted(),
            "file": str(providers_path(st.providers_file)),
        }

    return app


def main() -> None:
    """CLI entry point (console script: strata-serve)."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="strata-serve",
        description="Standalone Strata server (conversation loop + curated "
                    "OpenAI passthrough), independent of any host app.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("STRATA_PORT", "8765")))
    parser.add_argument("--state-dir", default=None,
                        help="conversation persistence dir (default: $STRATA_STATE_DIR "
                             "or ./harness_state)")
    parser.add_argument("--providers", default=None,
                        help="providers JSON file (default: <repo>/providers.local.json)")
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()

    import uvicorn

    app = create_app(
        providers_file=Path(args.providers) if args.providers else None,
        log_dir=args.log_dir,
        state_dir=Path(args.state_dir) if args.state_dir else None,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
