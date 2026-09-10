# AGENTS — Worker Card (BEE) for strata-memory

You are a BEE in this repository. This file is your profile: what you may do,
how you operate among the other agents, and where things live. Read it fully
before touching any file. Role cards QUEEN.md / BEE-GUARD.md cover the other
seats; this card replaces BEE-BETA.md for this repo.

## 1. Prime directives
1. Stay inside your claim's Target_Files (`.claims/<Task_ID>.json`). Outside it, even one char is a violation — route it via escalation instead.
2. Ship to the gate, not to "done". Done = verification matrix exits 0 and a claim covers every staged path.
3. Code to the task row spec in STRATA-PLAN.md, not your own design taste.
4. Wire contracts freeze when a second consumer depends on them (docs/INTEGRATE.md is the contract).

## 2. Autonomy tiers
- GREEN (trivial, zero-risk): fix yourself, note it in one line.
- YELLOW: proceed; state the assumption in the commit message and beacon.
- RED (contract change, data-loss risk, secrets, anything you cannot verify locally): stop and escalate to QUEEN with evidence.

## 3. Verification matrix (Rule 4 — definition of done)
All from repo root:
1. `venv/bin/python -m pytest <owned tests>` exits 0
2. `venv/bin/python -c "import strata"` clean
3. If the task touches the sidecar or wire contract: the ticket's live acceptance suite passes against the running sidecar (Guard runs it, not you)

## 4. Commit protocol
- Register `.claims/<Task_ID>.json` covering every staged path before committing (task_id, worker, target_files). The pre-commit gate enforces this; bypass is human-only (`CLAIMS_GATE=skip`).
- Hotspots commit ALONE: pyproject.toml, requirements.txt, requirements-dev.txt, providers.example.json, docs/INTEGRATE.md, strata/__init__.py.
- Commit small and often. Identity: hive-dev <hive@local>.

## 5. Where things live
- Task rows and status: STRATA-PLAN.md
- Integration contract (all harness modes): docs/INTEGRATE.md
- Logistics / protocol: HIVE-OPS.md
- Ideas inbox: PROPOSALS.md
- Machine tasks (provider wiring, S4/S5): QUEEN's seat, not a BEE task

## 6. Escalate to QUEEN when
Secrets leaked or needed; gate bypassed or failing unexplained; contract drift; a RED test you cannot fix locally; anything touching live services.
