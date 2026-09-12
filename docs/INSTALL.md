# Installing HiveBench (system + benchmark + studio)

Fresh-machine install of the full stack: the **strata-memory** system (`strata/`),
the **HiveBench** evaluation suite (sibling repo `../hivebench`,
sky-is-green/hivebench), and the **HiveBench Studio**
sidecar (`harness/`). ~5 minutes to a verified install.

## 1. Prerequisites

- **Python 3.10-3.14** (`python --version`; 3.14 verified on Windows)
- **git**
- A local LLM backend, **one of** (in order of least dependency):
  1. **Managed llama.cpp** *(default - nothing else to install)*: drop GGUF
     files into `models/gguf/`; the studio auto-starts `llama-server`
     - **Selectable GPU backends**: `vulkan` (default, works on AMD/NVIDIA/
       Intel), `rocm` (AMD HIP), `cuda`, `cpu` - fetch with
       `tools/fetch_backend.ps1 -Backend rocm`, then launch with
       `POST /v1/server/start {"backend": "rocm", ...}`.
       Actively tested: vulkan + rocm on an RX 7900 XT; cuda/sycl selectable
       but untested here.
  2. **LM Studio** - convenient front-end on `localhost:1234` if you already
     use it
  3. **Hosted APIs** - any OpenAI-compatible provider via
     `providers.local.json` (DeepSeek, OpenRouter, ...)
- Optional: GPU (not required - the drones run on CPU)

## 2. Get the repo

```powershell
git clone https://github.com/sky-is-green/strata-memory.git
cd strata-memory
```

## 3. Create the venv and install

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[harness,bench]"
```

```bash
# Linux / macOS
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[harness,bench]"
```

What you get:

| Extra | Provides |
|---|---|
| (base) | the system: drones, cortex, retention, backends |
| `[harness]` | HiveBench Studio (FastAPI sidecar) |
| `[bench]` | the evaluation suite (pytest, ST trainer) |

> Prefer the pinned manifest? `pip install -r requirements.txt
> -r requirements-dev.txt` then `pip install -e .` installs the same stack.

## 4. Generate the fixture corpora

The synthetic corpora are **generated, not committed** - a fresh clone has none.
The benchmark and live runs need them:

```powershell
.\.venv\Scripts\python -m tests.fixtures.synthetic_conversations.generate                    # code corpus (50 convs)
.\.venv\Scripts\python -m tests.fixtures.synthetic_conversations.generate --prose --horizon --p9 --return-corpus
```

This writes `tests/fixtures/generated*/` in the hivebench checkout. Re-run any time you need a
clean corpus.

## 5. Verify the install

```powershell
.\.venv\Scripts\python -m tests.run_hive_tests --group maximum
```

Expect hundreds of tests passing in under a minute - **no LLM, no GPU, no API
keys** (the suite is fully offline; `--mock` mode covers CI).

## 6. Start the studio

```powershell
.\.venv\Scripts\python -m harness --setup   # creates config, probes backend, warms the drone
.\.venv\Scripts\python -m harness           # open http://127.0.0.1:8765
```

`--setup` copies `providers.example.json` → `providers.local.json`, checks for a
reachable backend, **pre-downloads the default drone** (`paraphrase-MiniLM-L3-v2`,
~60 MB from Hugging Face, automatic on first live use either way), and prints
the next command. The studio serves an OpenAI-compatible endpoint
(`http://127.0.0.1:8765/v1/chat/completions`) that curates every conversation
through the strata, this is the integration point for other harnesses (see
`docs/INTEGRATE.md`).

## 7. Agent mode (optional - the dsh harness brain)

The console's **Agent (dsh)** chat mode runs the full DeepSeek Harness agent
loop (bash/files/code tools, multi-step turns) against the model loaded in
the studio. It needs the pinned dsh fork plus its Python SDK - one command
sets all of it up:

```powershell
# from the strata-memory root; builds the fork at ..\hivebench-studio
powershell -ExecutionPolicy Bypass -File setup.ps1 -SkipLlama
```

This installs corepack/pnpm, builds the fork, stages the SDK's node runtime
carrier, and editable-installs `deepseek-harness-sdk` + runtime. The sidecar
then spawns the agent runtime on demand (first agent message takes ~10 s to
boot it). Requirements: Node 22+/24. Tool calls need a tool-capable loaded
model (Gemma 4 26B-A4B and Qwen3-class work; Gemma 3 does not - no native
tool template).

Without this step everything else (chat, strata curation, benchmarks, model
management) still works.

## 8. Run the live benchmark

```powershell
# quick iteration run (LM Studio on :1234)
.\.venv\Scripts\python -m experiments.generate_data --live --no-thinking --confidence off --max-convs 3 --max-turns 10

# paired strata-vs-FIFO answer A/B (the LLM-performance head-to-head)
.\.venv\Scripts\python -m experiments.paired_ab --live --model prism-ml/bonsai-27b --max-convs 2 --max-turns 45 --confidence off --max-tokens 120 --no-thinking --checkpoint-every 2 --output runs/paired_ab_prose.json --checkpoint runs/paired_ab_prose.ckpt.json
```

See `STRATA-HANDOFF.md` §9 (live benchmark) and §15 (command cheat sheet) for the
full run matrix.

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `LM Studio not reachable` | Start LM Studio with a model loaded, or pass `--mock` for offline runs |
| "No conversation files found" | Run step 4 (fixtures are generated, not committed) |
| Empty replies under `--max-tokens` | The model is a reasoning model burning output on hidden CoT. Pass `--no-thinking` (bonsai-27b honors it; most qwen/gemma variants ignore the flag - use the GUI thinking toggle there) |
| `ImportError ... pyarrow ... Application Control policy` | Windows AppControl blocks pyarrow's parquet DLL; the drone auto-stubs it (inference never touches parquet). Nothing to do |
| Long runs get interrupted | Every live tool checkpoints; resume with `--resume <ckpt>` or use `tools/resume_evidence.ps1`, which relaunches until the report completes |
| AMD / no NVIDIA | vLLM is dormant; LM Studio / llama.cpp (Vulkan) is the live backend - no CUDA anywhere |

## 10. Where to go next

- `docs/INTEGRATE.md` - wire strata-memory into OpenCode, dsh, or your own harness
- `README.md` - why Strata, why HiveBench, measured results
- `STRATA-HANDOFF.md` - the master document: state, roadmap, commands
- `HARNESS-SPEC.md` - the studio sidecar contract