# Using strata-memory inside another harness

strata-memory is an **external** context-curation layer: it wraps your existing
LLM backend and curates every conversation into a bounded, high-relevance
context window. It does not replace your harness; it sits in front of the
model. Three integration modes, from zero-code to deep.

## Mode A: OpenAI-compatible swap (zero code, minutes)

Start the studio once; it serves a real OpenAI-compatible endpoint that curates
every request through the strata:

```powershell
.\.venv\Scripts\python -m harness --setup
.\.venv\Scripts\python -m harness        # serves http://127.0.0.1:8765/v1/openai/chat/completions
```

`POST /v1/openai/chat/completions` is a genuine passthrough: it curates the
turn, prepends the curated context to the leading system message, forwards the
request to your provider, and observes the reply back into the conversation
store. (The older `/v1/chat/completions` endpoint is the **mock** demo path:
canned replies, not real model output; do not point a client at it.)

Any client that accepts a `baseURL` + model id can use it. The API key is
ignored locally (send anything, e.g. `lm-studio`). Conversations are keyed by
the `X-Strata-Conversation` header, then the payload's `user` field, then
`"default"`.

### OpenCode

Copy `docs/opencode.strata.example.json` → `opencode.json` (project root or
`~/.config/opencode/`), then restart opencode; config is loaded once at
startup:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "strata-memory": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Strata Memory (curated context)",
      "options": {
        "baseURL": "http://127.0.0.1:8765/v1/openai"
      },
      "models": {
        "prism-ml/bonsai-27b": {
          "name": "bonsai-27b via Strata",
          "tool_call": true
        }
      }
    }
  },
  "model": "strata-memory/prism-ml/bonsai-27b"
}
```

Select the model with `/models` (or set `model` in config). The model id must
be one the studio's backend can serve (list them at
`http://127.0.0.1:8765/v1/models`). `"tool_call": true` declares the model can
emit tool calls, opencode needs this to edit files/run commands. Drop it if
your backend errors on tool calls. For a multi-project setup, give each
project its own `X-Strata-Conversation` header so stores stay isolated.

### dsh / other OpenAI-compatible harnesses

Same pattern: set the LLM provider's base URL to `http://127.0.0.1:8765/v1` and
an api key of `lm-studio`. The studio passes sampling parameters through
(`temperature`, `top_p`, `max_tokens`, …).

### How it behaves

- Every user turn goes through the strata: classify → route → score → assemble →
  generate, inside a per-conversation store.
- Conversations are isolated by `conversation_id`; reset with `POST /v1/strata/reset`.
- Reasoning models: pass `enable_thinking=false` in client params if the model
  honors it, and note that a small `max_tokens` cap yields empty visible replies
  on models that burn output on hidden CoT (see `docs/INSTALL.md` §8).
- Watch it curate in real time: `GET /v1/strata/state` (or the studio UI).

## Mode B, Python facade (build your own)

The system is a normal pip package with one import surface:

```python
from strata import Strata, HiveConfig, UltraSmallDrone, LMStudioBackend

strata = Strata(
    config=HiveConfig(),                      # budgets, decay, drift thresholds
    ultra=UltraSmallDrone(),                  # ~60 MB CPU drone (~5 ms/query)
    backend=LMStudioBackend(base_url="http://localhost:1234"),
)

result = strata.process_turn("what did we decide about auth?")
print(result.reply)          # the model's answer, generated under curated context
print(result.assembled)      # exactly what the model saw (content + token budget)
```

Per-conversation isolation is manual: call `strata.reset_conversation()` between
conversations (one store per conversation). Any OpenAI-compatible backend works
(`OpenAICompatBackend`), including hosted providers, keys live in
`providers.local.json` (gitignored) or the `HARNESS_PROVIDERS_FILE` env var.
`strata/` never imports from the bench or the studio, so it drops into any
project cleanly.

## Mode C, dsh plugin (deepest integration)

For the dsh fork (`deepseek-harness`, pinned `b150a551b8`), the harness spec
defines a full plugin contract; see `HARNESS-SPEC.md` §3.1:

1. The plugin listens at **`agent/pre-step`**: the documented extension point
   for "decides what the model sees".
2. It calls the sidecar `POST /v1/strata/turn` with
   `{query, conversation_id, model?}` and rewrites the system prompt with the
   curated context (single leading system message, strict chat templates
   require it).
3. Model adapters register through **`ctx.llm`**, so the curated context flows
   through the same seam as any provider.
4. The session log (`SessionEventMap`) is the evaluation record: task prompts,
   steps, tool calls, outcomes, agentic completion can be scored from it.

`setup.ps1` automates the whole dsh toolchain on a fresh machine (fork build,
node carrier, SDK editable installs, llama-server), run it once, then
`python -m harness`.

## Config & gotchas

- **Secrets**: provider keys live in `providers.local.json` (gitignored); the
  NDJSON event logger redacts `api_key`/`token`-like values before writing.
- **Local-only by default**: the studio binds `127.0.0.1`; set
  `HARNESS_TOKEN` to require a console password on the web UI.
- **Reasoning models** and the **AMD/no-NVIDIA** notes from `docs/INSTALL.md`
  §8 apply to every integration mode.

## Model quirks to plan for

| Quirk | What you'll see | What to do |
|---|---|---|
| **Hidden chain-of-thought** | qwen3.8/gemma-4 variants ignore `enable_thinking=false` and burn the output budget on hidden reasoning, empty replies under small caps, multi-minute turns | Use a non-reasoning model (bonsai-27b honors the flag) or toggle thinking off in the LM Studio GUI |
| **Strict chat templates** | The model rejects a system message that isn't first | The studio always merges curated context into the leading system message, keep client system messages first too |
| **Tool calls** | Some GGUFs/backends error on `tools:` payloads | Set `tool_call: false` (or drop it) in the client model config; opencode then goes text-only |
| **Single-slot servers** | Requests queue; concurrency doesn't speed up | Raise llama.cpp `-np`/LM Studio parallel slots for multi-request workloads |
| **`max_tokens` is a ceiling, not a target** | Small caps truncate replies (or yield nothing on thinking models) | Keep 400+ tokens for normal turns; the strata's generation headroom reserves 2048 |
| **Prefix caching** | TTFT grows with context when the pinned prefix changes | Keep the studio's pinned prefix stable; don't rewrite the leading system message yourself |
| **Conversation isolation** | One store per conversation id; shared ids bleed context | Always set `X-Strata-Conversation` (or `user`) per project/session |
| **API key is cosmetic locally** | Any string works against LM Studio | Use real keys in `providers.local.json` only for hosted providers |
| **Context window size** | Windows ≥8k verified; the strata's budget (1-6k) never binds | Nothing to do, the budget is inside any modern window |