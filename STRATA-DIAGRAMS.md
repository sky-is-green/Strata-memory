# Strata Memory: Why STRATA Improves Over Not Using STRATA

Companion to `STRATA-WHITE-PAPER.md` and `STRATA-HANDOFF.md` (the single master doc: project state, roadmap, usage).

> **Note (2026-08-24):** the diagrams in §1-§5 are now also embedded in the
> white paper (§1.1, §1.2, §4) alongside the measured charts; this file
> remains the standalone visual summary.
This document answers one question with diagrams:

> **Why does the STRATA system improve results over not using STRATA, and why?**

> Rendering: the diagrams below are Mermaid. Open this file in GitHub, VS Code
> (with "Markdown Preview Mermaid Support"), or Obsidian to render them. If
> your viewer does not render Mermaid (especially the `xychart-beta` charts in
> §6), use the rendered images in `figures/`, they are embedded in
> `STRATA-WHITE-PAPER.md` and shown again under each chart below.

---

## 1. The one-sentence argument

The one-sentence argument: **naive systems keep the *most recent* text; STRATA keeps the
*most relevant* text.** Everything else follows from that distinction.

```mermaid
flowchart LR
    NAIVE["Naive: keep the MOST RECENT<br/>blind FIFO eviction<br/>context loss + lost-in-the-middle + quadratic cost"]
    STRATA["STRATA: keep the MOST RELEVANT<br/>relevance-ranked, bounded selection<br/>foundational context survives + flat cost"]
    NAIVE -->|"degrades as conversations grow"| GAP["The gap widens with conversation length"]
    STRATA -->|"stays flat at any length"| GAP
```

---

## 2. Why naive fails, and how STRATA removes each failure mode

The white paper identifies three compounding failure modes in long conversations. STRATA exists
because each one is **engineerable away** rather than an irreducible limit.

```mermaid
flowchart TB
    subgraph FAIL["Without STRATA, the three naive failure modes"]
        direction TB
        F1["Context loss (recall)<br/>FIFO discards foundational rules and<br/>early decisions blindly"]
        F2["Lost-in-the-middle (attention)<br/>model under-uses the middle of<br/>a large raw window"]
        F3["Quadratic cost (compute)<br/>prompt grows every turn, slower<br/>generation, OOM risk on consumer GPUs"]
    end

    subgraph FIX["How STRATA removes each one"]
        direction TB
        M1["Sieve: relevance-score every chunk<br/>vs. the current query, foundational<br/>context keeps scoring high"]
        M2["Focal: assemble a bounded, relevance-<br/>ranked window, the model's attention is<br/>spent only on tokens predicted to matter"]
        M3["Bounded context + stable pinned prefix,<br/>flat decode throughput, no unbounded<br/>KV-cache growth"]
    end

    F1 --> M1
    F2 --> M2
    F3 --> M3
```

---

## 3. The comparison: same conversation, two outcomes

```mermaid
flowchart TB
    START["Same long conversation, 100 to 500+ turns"] --> SPLIT{"How is context<br/>delivered to the LLM?"}

    SPLIT -->|"no curation layer"| N1
    SPLIT -->|"STRATA curation layer"| H1

    subgraph NA["Without STRATA, naive FIFO rolling window"]
        direction TB
        N1["Window fills at 4-8k tokens"]
        N2["Blind FIFO eviction, oldest text dropped"]
        N3["Context loss: foundational rules and early decisions forgotten"]
        N4["Lost-in-the-middle: model under-uses the middle of the window"]
        N5["Quadratic cost: prompt grows, generation slows, OOM risk"]
        N1 --> N2 --> N3 --> N4 --> N5
    end

    subgraph HI["With STRATA, external context-curation layer"]
        direction TB
        H1["Sieve: drone fleet scores every chunk vs. the current query"]
        H2["Membrane: semantic dedup (keep densest) + topic-drift reset"]
        H3["Retention: remembrance pass saves + sharp decay matrix"]
        H4["Focal: assemble a bounded, high-relevance window (1-3k live, 1-6k configured)"]
        H5["LLM receives only curated, bounded context, every turn"]
        H1 --> H2 --> H3 --> H4 --> H5
    end

    N5 --> RES1["Baseline outcome: near-chance retrieval past the window,<br/>PES ~30 (measured: rolling 12.2, FIFO 11.6),<br/>quality decays with length"]
    H5 --> RES2["Strata outcome: flat decode tps (P1),<br/>recall 90.3% on stated facts (P2, deterministic),<br/>post-run PES 80.0 GREEN vs ~12 baselines"]
    RES1 --> TAKE
    RES2 --> TAKE
    TAKE["Takeaway: same or cheaper compute per turn,<br/>strictly better long-run quality, the gap widens as conversations grow"]
```

---

## 4. Why the mechanism wins: the causal chain

### 4.1 Postulates → outcomes (the logical "why")

```mermaid
flowchart TB
    S1["Separation Postulate, cheap bidirectional encoders do<br/>the comprehension; the LLM only generates"] --> B1["Bounded context delivered every turn"]
    C1["Context Curation Postulate, relevance-ranked selection<br/>dominates an equal-size contiguous window"] --> B2["Budget spent on high-mutual-information tokens;<br/>no lost-in-the-middle"]
    M1["Managed Decay Postulate, forgetting is a design<br/>parameter, not a failure mode"] --> B3["Escalating-friction decay beats both<br/>unbounded growth and blind FIFO"]

    B1 --> O1["P1: throughput flat ±10% across 500 turns<br/>(measured: 14.5→15.5 tps over 308+ turns)"]
    B2 --> O2["P2: recall ≥90% met (90.3% live, deterministic);<br/>precision = encoder ceiling (10.7-40%, Threat 6)"]
    B3 --> O3["Decay/remembrance keep old facts retrievable<br/>when re-referenced (P4: domains separate by age;<br/>false-eviction itself is not yet honestly measured)"]

    O1 --> WIN
    O2 --> WIN
    O3 --> WIN
    WIN["STRATA dominates naive precisely in the regime<br/>where naive degrades, long conversations"]
```

### 4.2 Why externalizing comprehension is cheaper and better

STRATA is not "more context", it is **better allocation of the same compute**.

```mermaid
flowchart LR
    subgraph WHERE["Where the work happens"]
        direction TB
        W1["Without STRATA: the LLM must re-discover, on every<br/>turn, which earlier tokens matter, expensive causal<br/>attention over raw, unbounded history"]
        W2["With STRATA: small bidirectional encoders (orders<br/>of magnitude cheaper) do the comparison and<br/>similarity work up front"]
    end
    W1 -->|"same token budget"| C["Relevance-ranked selection concentrates the LLM's<br/>expensive attention on the tokens that actually matter<br/>(P3: equal budget, strata sufficiency higher on ≥80% of turns)"]
    W2 --> C
```

---

## 5. Why the improvement grows with conversation length

```mermaid
flowchart LR
    L1["1-10 turns: naive and STRATA both fit in the window,<br/>small or no difference (STRATA adds a little overhead)"]
    L10["100+ turns: naive has evicted foundational context<br/>and slows; STRATA still feeds the same bounded,<br/>high-relevance window, the difference becomes large"]
    L1 --> L10
    L10 --> E["The longer the conversation, the more STRATA's<br/>relevance-ranked curation matters, which is exactly<br/>where naive systems fall apart"]
```

---

## 6. Measured results (2026-08-23)

The white paper's §8 table carries the numbers; here they are as charts. All
values are measured (see the paper's §5 for protocol, sources, and confound
notes).

### 6.1 P4: decay survival curves (the domain separation)

Answer-fact survival vs the initial decay multiplier, long-horizon replay
sweep (real L3-v2 drone, fixed 1000-token budget). Code facts (young, no
stale penalty) tolerate aggressive decay; prose facts (stale, age > 20) die at
the lowest multiplier: m90 1.8 vs 1.2, gap 0.6 > 0.2 band.

```mermaid
xychart-beta
    title "P4: retrievable answer-fact survival vs decay multiplier"
    x-axis [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5]
    y-axis "Recall %" 0 --> 100
    line "Code" [91.0, 89.3, 87.6, 82.6, 79.8, 75.3, 70.8]
    line "Prose" [15.3, 3.8, 1.0, 1.0, 0.3, 0.3, 0.3]
```

![Rendered (also embedded in the white paper):](figures/p4.png)

### 6.2 Threat 6: every encoder lands on the same top-K curve

Retrieval precision at top-K on the held-out same-domain pairs (264 pairs, 87
relevant): scale (bge-m3, 568M) and task tuning (contrastive) do not break the
ceiling, the curve is data-structural, not encoder-capacity.

```mermaid
xychart-beta
    title "B avenue: top-K retrieval precision, six encoders on one curve"
    x-axis [1, 3, 5, 8]
    y-axis "Precision %" 0 --> 100
    line "all-MiniLM" [84.0, 72.5, 59.1, 46.7]
    line "bge-m3 (B1)" [84.0, 69.6, 59.1, 46.7]
    line "B2 tuned" [72.0, 62.3, 58.1, 47.5]
    line "B3 tuned" [80.0, 71.0, 59.1, 46.7]
```

![Rendered:](figures/b.png)

### 6.3 P9: densest retention wins the duplicate A/B

Sufficiency per 1k tokens: on the informative turns (dense-first order) the
densest-keeping dedup delivers the same fact in ~1.8× less budget; the control
(both policies keep the dense copy) shows no effect, as designed.

```mermaid
xychart-beta
    title "P9: sufficiency per 1k tokens, densest vs recency retention"
    x-axis ["informative", "control"]
    y-axis "fact-presence × 1000 / fact tokens" 0 --> 35
    bar "densest" [32.3, 30.5]
    bar "recency" [17.6, 30.5]
```

![Rendered:](figures/p9.png)

### 6.4 PES: the strata vs the baselines (post-run, 211131)

The bounded-context headline: post-run PES 80.0 GREEN vs the no-strata
baselines on the same conversations.

```mermaid
xychart-beta
    title "Post-run PES: strata vs baselines (run 20260822_211131)"
    x-axis ["strata", "rolling", "fifo"]
    y-axis "PES (0-100)" 0 --> 100
    bar "PES" [80.0, 12.2, 11.6]
```

![Rendered:](figures/pes.png)

### 6.5 Budget ceiling: invariant to the model window (8k-32k)

Replay of 348 fixture turns at `max_context` 8k/16k/32k: byte-identical
budgets (p50 1400), assembled tokens (p50 ~996), utilization (66.9%). The
route-tier ranges bind; the window cap never does.

```mermaid
xychart-beta
    title "Adaptive budget (p50) vs max_context, flat by design"
    x-axis ["8k", "16k", "32k"]
    y-axis "budget tokens (p50)" 0 --> 4000
    bar "budget" [1400, 1400, 1400]
```

![Rendered:](figures/budget.png)

---

> **Bottom line:** STRATA wins because it externalizes the two things a generative model is
> bad at (*remembering* and *selecting*) to components built for them, and spends the
> model's expensive attention budget only on tokens predicted to matter. Naive systems
> degrade on all three axes (recall, attention, compute) as conversations grow; STRATA's
> bounded, relevance-ranked context keeps all three flat, with the honest caveat that
> selection *efficiency* (precision) is an open, documented ceiling (Threat 6) rather than
> a solved one.

