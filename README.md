# Strata-Memory / HiveBench

[![CI](https://github.com/sky-is-green/strata-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/sky-is-green/strata-memory/actions/workflows/ci.yml)

**Strata-Memory** is an external, multi-agent context-curation layer for
long-horizon LLM conversations. It sits between a user and a local LLM backend,
filtering, scoring, compressing, and reassembling conversation history into a
bounded, high-relevance context window for every turn, so a generative model
performs well over arbitrarily long conversations on consumer hardware.

**HiveBench** is its evaluation suite: unit/integration/benchmark tests, a live
benchmark harness, and the white paper's falsifiable predictions
([P1-P11](STRATA-WHITE-PAPER.md#5-hypotheses-and-predictions)) with measured
verdicts.

## Quickstart

Requires Python 3.10+ and, for live runs, any OpenAI-compatible backend (LM Studio on `localhost:1234`), or nothing at all: the studio can manage a local
`llama-server` for you from GGUF files dropped into `models/gguf/`.

```powershell
git clone https://github.com/sky-is-green/strata-memory.git
cd strata-memory
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[harness,bench]"
.\.venv\Scripts\python -m harness --setup   # creates config, probes backend, warms the drone
.\.venv\Scripts\python -m harness           # studio UI on http://127.0.0.1:8765
```

Linux/macOS: `python3 -m venv .venv && .venv/bin/python -m pip install -e ".[harness,bench]"`.
Narrower layers install cleanly too:

| Install target | What you get |
|---|---|
| `pip install strata-memory` | The system: drones, cortex, retention, backends |
| `pip install "strata-memory[harness]"` | + HiveBench Studio (FastAPI sidecar) |
| `pip install "strata-memory[bench]"` | + the evaluation suite (pytest, ST trainer) |

The full fresh-machine walkthrough (fixtures, live benchmark, troubleshooting)
is `docs/INSTALL.md`; every claim below is reproduced by commands in this repo.

## Why Strata

The core idea is the white paper's *Separation Postulate*: small bidirectional
"drone" encoders (fast, cheap, CPU-friendly) do the *comprehension*, scoring,
filtering, and routing context, while the primary generative LLM does the
*generation*.

**The headline measurement**: same 308+ turn conversations, same model,
strata vs naive FIFO windowing (live run `20260822_211131`):

- **[Strata ≥ FIFO on 85.1% of retrievable turns](STRATA-WHITE-PAPER.md#p3-context-sufficiency-hypothesis) (P3)**;
  the direct head-to-head against the current standard
- **[90.3% of the facts the model stated made it into context](STRATA-WHITE-PAPER.md#p2-retrieval-precision-hypothesis)** (P2), deterministic
  diagnostic, ≥90% target; FIFO truncates and drops facts at its window
  limit
- **[Flat generation speed across 308+ turns](STRATA-WHITE-PAPER.md#p1-constant-throughput-hypothesis)** (P1), 14.5 → 15.5 decode tps
  (+6.7%), no context-bloat slowdown
- All at ~3.4 ms assembly + ~15 ms drone scoring overhead per turn

![Post-run PES: strata 80.0 GREEN vs rolling 12.2 / FIFO 11.6](figures/pes.png)

*PES is the system's own pipeline-efficiency score (retrieval/routing/
latency/throughput/utilization), a health signal, not a measure of answer
quality. The head-to-head evidence above is what the claims rest on.*

**Why Strata is a great addition to LLM use**

- **Bounded cost, always.** The strata caps the context window regardless of
  conversation length (adaptive budget: 1-3k tokens live), so per-turn cost and
  generation time stay flat instead of growing with history. And because KV
  compression is a *precision* axis while curation is a *selection* axis, the
  savings compound rather than compete: paired with a TurboQuant-class KV
  quantizer (~3-4 bits, near-zero loss), a strata-curated context makes a
  50k-token conversation's cache ~150× smaller than raw history, selection
  multiplies precision on the surviving tokens (white paper §1.6).
- **It drops in around your existing backend.** Any OpenAI-compatible endpoint
  (LM Studio, llama.cpp, vLLM, hosted APIs) works, no model retraining, no
  prompt rewrites; the harness exposes it as a drop-in API.
- **Runs on consumer hardware.** The drones are small CPU models (~60 MB,
  ~5 ms/query, no GPU required), verified on an AMD-only rig with no NVIDIA,
  where FP8-attention paths don't exist (the exact gap TurboQuant-class KV
  quantization fills).
- **The efficiency gap is measured, not claimed:**

| Metric | Strata | Status quo (FIFO/rolling window) |
|---|---|---|
| Pipeline efficiency (PES, flagship live run) | **80.0 GREEN** | 12.2 / 11.6 |
| [Decode speed over 308+ turns](STRATA-WHITE-PAPER.md#p1-constant-throughput-hypothesis) (P1) | **Flat** (14.5→15.5 tps, +6.7%) | Slows as context grows, then truncates |
| [Stated-fact recall](STRATA-WHITE-PAPER.md#p2-retrieval-precision-hypothesis) (P2, deterministic) | **90.3%** | Facts dropped at window limit |
| [Turns where strata ≥ FIFO](STRATA-WHITE-PAPER.md#p3-context-sufficiency-hypothesis) (P3) | **85.1%** | - |
| Paired A/B under window pressure (82 turns, live) | **84.1% overall; 87.5% vs 82.1% late-turn, once the window drops facts** | 84.1% while its window still holds everything |
| Context utilization (p50) | **74.5%** | ~40% (fluff) |
| Added latency per turn | **~18 ms** | 0 (but loses the facts) |
| Stability (500-turn run) | **0 OOM**, peak RSS 34.7 MB | - |

All numbers are the live runs recorded in the white paper's measured-outcome
table (§8); PES is defined in §6. The paired A/B row is the fair-selection
live measurement (bonsai-27b, identical replayed history for both arms,
FIFO window capped at 1500 tokens to force truncation): at parity overall,
with strict strata-only wins outnumbering FIFO-only 14:6 once the naive
window starts dropping facts.

![Context tokens delivered per turn: strata stays flat while unbounded history grows to 33k+ tokens](figures/token_growth.svg)

*Median context tokens per user turn across 721 live turns (two run bundles):
the strata delivers a flat ~1.2-1.4k-token window regardless of session length,
while the unbounded history it replaces reaches 33,500+ tokens by turn 40.*

## Why HiveBench

Most evaluation harnesses tell you how a model performs in a sandbox. HiveBench
tells you *whether the context you feed the model is the reason it works*, and
it does it deterministically, offline, and replayably:

- **Falsifiable, not vibes.** The white paper's
  [P1-P11 predictions](STRATA-WHITE-PAPER.md#5-hypotheses-and-predictions) ship as
  executable tests with measured PASS/FAIL verdicts (§8). Every number in this
  README is reproduced by a command in the repo.
- **No LLM-as-judge circularity in the evidence path.** The deterministic
  diagnostics score fact presence against fixture ground truth, stated-facts
  recall, first-mention exclusion, hedge filtering. **The Strata queen**, an
  asynchronous ground-truth layer that labels, after each turn, whether the
  assembled context was actually sufficient for the query, corroborates that
  evidence; because it shares the served model's biases, it never constitutes
  it (§9, Threat 1).
- **The full test suite runs offline in ~30 seconds**: no LLM calls and no API
  keys; CI-friendly via `--mock`. 599 tests: 546 unit, 53 integration,
  plus live-gated MCP batteries (counts as of the 2026-09-12 pass). (Running the system *live*
  does require a local model backend, LM Studio / llama.cpp, which on most
  rigs means a GPU; the drones themselves stay on CPU.)
- **Paired head-to-head A/B** (`hivebench-ab`): the same turns, the same model,
  strata-curated context vs the naive FIFO window, both answers scored
  deterministically (fixture-fact presence + context fidelity), with both
  arms' stores replaying identical history so the comparison isolates
  selection. The scoring path is unit-tested; interim live results are
  recorded per run under `runs/`.
- **Built for long evidence runs.** Checkpointed, resumable live runs survive
  crashes and reboots:

  ```powershell
  .\.venv\Scripts\python -m experiments.paired_ab --live --model prism-ml/bonsai-27b --max-turns 45 --fifo-budget 1500 --checkpoint-every 2 --output runs/paired_ab.json
  # killed mid-run? relaunch with --resume runs/paired_ab_trunc-style checkpoint,
  # or let tools/resume_evidence.ps1 loop until the final report exists.
  ```

  One-command CLIs (`hivebench`, `hivebench-protocol`, `hivebench-diagnostic`,
  …) wrap the rest.
- **Honest by design.** The suite surfaced its own failures first, the
  measurement fixes that made PES trustworthy (latency floor, stated-facts
  reframe, hedge poisoning) are documented in the paper's threats section
  (§9), not hidden.

## FAQ

Objections we actually hear, answered with what ships in this repo.

**Does my data leave my machine?**
Not by default, and the default is the product: conversations are curated by
CPU-resident drones against a backend you host (LM Studio / llama.cpp on
localhost; the studio can manage a local `llama-server` for you), stores and
archives live in local files, and the studio binds `127.0.0.1`. Nothing phones
home. Hosted APIs only come into play if you put keys in
`providers.local.json` (gitignored) yourself, and then only the requests you
point at that provider leave.

**How is this different from mem0 / Letta / Hindsight?**
They are agent-memory frameworks: append experience, retrieve it later, inside
their own runtime. HiveMemory overlaps on the goal (long-horizon context that
stays useful) but differs on three things. It is *evaluation-first*: the
HiveBench protocol publishes falsifiable predictions with measured PASS/FAIL
verdicts, including its own failed ones, instead of demo numbers. Its curation
is a *bounded* relevance-ranked window under decay, dedup, and drift policies,
so per-turn cost stays flat while unbounded memory grows. And it sits in front
of any OpenAI-compatible backend as a drop-in layer or endpoint, no SDK lock-in
and no model changes. Use them when you want managed memory features inside
their runtimes; use this when you want bounded cost and claims you can re-run.

**Why not just use a bigger context window?**
Because window size is not usable-context size: models under-use mid-window
content (lost-in-the-middle), every turn pays for the whole history, and at
the limit a rolling window blindly evicts exactly the early facts long
conversations need (white paper §1.1). A bigger window moves the cliff; the
strata removes the growth, feeding a flat 1-3k curated window at constant decode
speed (P1) while stated-fact recall measures 90.3% (P2).

**Is this just RAG?**
RAG retrieves from an external corpus per query. The strata retrieves from *the
conversation itself*, continuously, through decay/dedup/drift retention
policies, and composes with RAG rather than competing with it (white paper §2).

## Repo layout

| Path | Contents |
|---|---|
| `strata/` | The system: cortex (routing, PES, congestion, e2e), sieve (drones), retention (**hygiene**, store, decay, comb, remembrance), focal (budget/assembly), membrane (dedup/drift), backend (LM Studio / OpenAI-compat / vLLM), queen (async ground truth), mcp (server + tools) |
| `hivebench/` | The evaluation suite: `tests/` (unit/integration/benchmarks), `testing/` (A/B, ablation, MCP battery), `experiments/` (live benchmark, protocol, probes) |
| `harness/` | HiveBench Studio sidecar (FastAPI service over the strata; MCP server mounted here) |
| `docs/` | Install guide + integration guides (`INTEGRATE.md`: drop-in endpoint, Studio, DSH plugin, MCP) |

## What we have now

The pipeline per user turn — **Membrane → Retention → Sieve → Focal** — plus
the services around it:

| Layer | Module | Role |
|---|---|---|
| Membrane | `strata/membrane/` | Semantic dedup + topic-drift detection, before scoring |
| Retention | `strata/retention/` | Chunk store with decay state, remembrance ladder, comb surplus tier (SSD archive) |
| Sieve | `strata/sieve/` | Small CPU "drone" encoders score every candidate (~5 ms/query, no GPU) |
| Focal | `strata/focal/` | Adaptive budget, relevance floor + per-chunk share cap (P1-FLOOR), assembly into a bounded window |
| Cortex | `strata/cortex/` | Routing, congestion control, PES health, checkpoint/resume, e2e engine |
| Queen | `strata/queen/` | Asynchronous ground truth: labels whether the assembled context was sufficient, after each turn |
| MCP | `strata/mcp/` | `strata_search` / `strata_remember` tools on the sidecar; any MCP client (Studio, opencode, DSH) queries the same curated store |

**Write-side hygiene is one pipeline.** Every chunk passes through
`retention/hygiene.py` before fingerprinting: harness boilerplate stripped
(P0) → secrets and base64 blobs redacted, length capped (U2) → 12-hex content
fingerprint. Dedup groups the sanitized form, so every downstream tier —
active store, checkpoints, comb archives — inherits clean data. The same
composite entry point (`prepare_for_storage`) is used by the store's write
path and by the sidecar's payload-echo guard; that shared normalization is
what makes recency-echo dedup compare like-for-like (RC2).

**The sidecar** (`harness/`) exposes the system as a drop-in
OpenAI-compatible endpoint with the Studio UI on `127.0.0.1:8765`; integration
modes are in `docs/INTEGRATE.md`. Sidecar lifetime is bound to Studio
(P1-LIFECYCLE): it starts when the studio starts and dies with it, zero
polling.

## How we got here

- **Launch state** — the white paper system: layered pipeline with P1–P11
  measured on live runs (flagship `20260822_211131`).
- **Integration wave (S1–S3)** — DSH Mode C plugin; MCP server on the sidecar
  (`2c2b6f4`, `aac9019`); the MCP path made a tested feature, live battery:
  recall/precision 95.4% at ~30 ms/query over 90 probes (Round 6).
- **Corruption eradication** — the suite found harness control text, secrets,
  and verbatim duplicate chunks leaking into persistent memory; fixed as a
  stack, each fix fenced by tests: P0 ingest boilerplate filter + P1 payload
  echo dedup (`2644cbd`); RC1 fingerprint guard for duplicates (`c9b2a58`,
  `621f5c2`, `2154dd5`); RC2 shared normalization so scoring and payload
  fingerprints compare like-for-like (`c9b2a58`; host-side trim `84bce97`);
  P1-FLOOR relevance floor + per-chunk window-share cap (`b5b9e66`).
- **P1-LIFECYCLE** — sidecar lifetime bound to Studio (`3bd8499`).
- **This pass (2026-09-12)** — hygiene consolidation: the boilerplate filter
  and U2 sanitizer merged into one `retention/hygiene.py` pipeline with a
  single composite entry point shared by store writes and payload echo;
  `filter.py` retired. Behavior-preserving: suite green before and after.

## Where we're going

- **Codec repair layer (next build)** — self-healing cp1252↔UTF-8 round-trip
  transform for mojibake-corrupted chunks: strict normalization hooked into
  `prepare_for_storage()` pre-fingerprint, read-boundary defense at context
  assembly, one-time scrub of existing stores. Verification plan: idempotency,
  false-positive guard on clean accented text, mojibake fixture round-trips,
  dedup groups the repaired form, legacy poisoned chunks heal on read without
  migration.
- **S4/S5** — Studio provider row → sidecar (recall battery done; provider row
  pending UI confirm); opencode provider config → sidecar with conversation id
  = project name.
- **P12** — store-time fact distillation (white paper, DRAFT, protocol only).

## Tests and evidence

| Suite | Covers | Current state |
|---|---|---|
| `hivebench/tests/unit` | every layer, offline, no LLM calls | 546 tests — 545 pass; 1 env-gated (host missing `zstandard`) |
| `hivebench/tests/integration` | pipeline end-to-end, incl. live-gated MCP suite | 53 tests |
| Live batteries | paired A/B vs FIFO, protocol P1–P11 verdicts, MCP battery | recorded in white paper §8 and per-run reports |

`python -m pytest hivebench/tests/unit -q` — ~25 s offline.

## Use the system in your own project

`strata/` is self-contained; it never imports from the bench or the harness:

```python
from strata import Strata, HiveConfig, UltraSmallDrone, LMStudioBackend

strata = Strata(
    config=HiveConfig(),
    ultra=UltraSmallDrone(),
    backend=LMStudioBackend(base_url="http://localhost:1234"),
)
result = strata.process_turn("what did we decide about auth?")
print(result.reply)
```

## Run the studio (HiveBench Studio)

The two commands from [Quickstart](#quickstart) are the whole story: `--setup`
copies `providers.example.json` → `providers.local.json` if missing, probes for
a reachable backend (LM Studio on `:1234`, or auto-starts the local
`llama-server` from `models/gguf`), and prints the next step. The studio serves
the strata over a FastAPI API, the endpoint contract lives in
`harness/harness/app.py`, and `docs/INTEGRATE.md` shows how to point external
clients at it.

## Run the test suite

The suite is grouped by what it measures (offline; no LLM required):

```powershell
.\.venv\Scripts\python -m tests.run_hive_tests --group maximum   # full suite (default)
.\.venv\Scripts\python -m tests.run_hive_tests --group speed     # latency/PES
.\.venv\Scripts\python -m tests.run_hive_tests --group intelligence  # retrieval/assembly
.\.venv\Scripts\python -m tests.run_hive_tests --group skills    # pipeline/backends
```

## Try it live

The live benchmark talks to an OpenAI-compatible backend (e.g. LM Studio on
`localhost:1234`). A quick resumable iteration run:

```powershell
.\.venv\Scripts\python -m experiments.generate_data --live --no-thinking --confidence off --max-convs 3 --max-turns 10
```

See `docs/INSTALL.md` for the full setup and run guide, and
`STRATA-WHITE-PAPER.md` §8 for the measured-outcome table behind every claim.

## Documentation

- **`docs/INSTALL.md`**, full-stack install guide (system + benchmark + studio, fresh machine)
- **`docs/INTEGRATE.md`**, using strata-memory inside OpenCode, dsh, or your own harness
- **`STRATA-WHITE-PAPER.md`**, the theory: postulates, falsifiable predictions P1-P11 with measured verdicts (§8), the PES metric (§6), KV-compression landscape (§1.6), threats & limitations (§9)
- **`STRATA-DIAGRAMS.md`**, visuals and measured charts
