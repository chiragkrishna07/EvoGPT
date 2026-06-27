# EvoGPT — a from-scratch transformer with an autonomous architecture-search loop

EvoGPT is two things in one ~900-line, dependency-light codebase:

1. **A GPT-style language model written from scratch in PyTorch** — no
   `nn.Transformer`, no HuggingFace. Every modern Llama-era component is
   hand-implemented: Rotary Position Embeddings (RoPE), RMSNorm, Grouped-Query
   Attention (GQA), a SwiGLU feed-forward, weight tying, and a **KV-cache** for
   fast autoregressive decoding.

2. **A self-improving evolutionary Neural Architecture Search (NAS) controller**
   that *designs* the network for you. It maintains a population of
   architectures, trains each under a fixed compute budget, scores them with a
   multi-objective fitness (validation loss **+** a parameter-efficiency
   penalty), and breeds the next generation via tournament selection,
   crossover, and mutation — improving measurably, generation over generation.
   No external APIs: the controller's "decisions" are a fully self-contained
   evolutionary algorithm.

The result is a single command that **discovers, trains, and samples from its
own transformer**, with every decision logged to an auditable leaderboard — and a
suite of experiments that hold the method to a real research standard (random-search
baseline, proxy-fidelity study, weight-inheritance ablation).

```bash
pip install -e .                 # install (torch + matplotlib)
python download_data.py          # fetch the tiny-shakespeare corpus (~1.1 MB)
python run_search.py             # evolve an architecture, then train the champion & sample
python -m experiments.run_all    # full research suite: search + 3 experiments + plots
python -m pytest -q              # 49-test correctness suite
python make_report.py            # assemble results/REPORT.md from all artifacts
```

See **[`results/REPORT.md`](results/REPORT.md)** for the headline results,
**[`TECHNICAL.md`](TECHNICAL.md)** for the full write-up, and
**[`WALKTHROUGH.md`](WALKTHROUGH.md)** for a from-zero, beginner-friendly explanation
of every concept (with a line-by-line data-flow trace and a glossary).

---

## Why this is non-trivial

Most "I built a GPT" projects stop at a forward pass. EvoGPT verifies the parts
that are actually hard to get right, and then *searches the design space*:

| Component | Implemented from scratch | Verified by |
|---|---|---|
| Causal self-attention | ✅ | proves no future-token leakage |
| KV-cache incremental decode | ✅ | exact match vs. full forward (greedy) |
| Grouped-Query Attention | ✅ | parameter savings vs. full MHA |
| RoPE / RMSNorm / SwiGLU / weight tying | ✅ | end-to-end convergence + unit tests |
| Weight inheritance (morphism) | ✅ | exact overlapping-slice copy |
| Evolutionary NAS controller | ✅ | **beats random search at equal compute** |

(49 `pytest` tests; `ruff`-clean; GitHub Actions CI on Python 3.10–3.12.)

## How the loop works

```
            ┌─────────────────────────────────────────────┐
            │  population of architecture "genomes"        │
            └───────────────────┬─────────────────────────┘
                                │  train each (fixed budget, MPS)
                                ▼
            ┌─────────────────────────────────────────────┐
            │  fitness = val_loss + λ · (params / 1M)      │
            └───────────────────┬─────────────────────────┘
                                │  select elites + tournament
                                ▼
            ┌─────────────────────────────────────────────┐
            │  crossover + mutate  →  next generation      │
            └───────────────────┬─────────────────────────┘
                                │  repeat for N generations
                                ▼
                 champion → extended training → sample
```

The **genome** spans depth, width, head count, GQA ratio, MLP ratio, context
length, and dropout (`evogpt/evolve.py::SEARCH_SPACE`). Trained candidates are
cached so identical genomes (e.g. carried-over elites) are never retrained.

## Repository layout

```
evogpt/
  model.py      # the from-scratch transformer (RoPE, RMSNorm, GQA, SwiGLU, KV-cache)
  data.py       # char-level corpus + batching
  train.py      # single-candidate train/eval under a compute budget (the fitness fn)
  evolve.py     # evolutionary NAS controller: genome, mutation, crossover, selection
  morph.py      # weight inheritance (network-morphism slice copy) for Lamarckian evolution
experiments/
  run_all.py        # the full research suite (search + 3 experiments + plots)
  compare_v2.py     # tuned evolution vs. random search (sample efficiency + pop-mean)
  viz_lineage.py    # genealogy DAG of the search
  _common.py        # Spearman/Pearson, best-so-far
run_search.py   # quick entry point: search -> train champion -> sample
analyze.py      # leaderboard -> fitness curve + Pareto plot
make_report.py  # assemble results/REPORT.md from all artifacts
sample.py       # load champion checkpoint and generate
tests/          # 49-test pytest suite
results/        # champion.json/.pt, plots, REPORT.md (generated)
```

## Hardware

Built and trained on an **Apple M3 (16 GB) using the PyTorch MPS (Metal)
backend** — the whole search runs locally in minutes, no GPU cloud required.

## Results (tiny-shakespeare, char-level, Apple M3 / MPS)

The search explores **~6,900 architectures**. The discovered champion
(`3 layers, d=192, 8 heads, GQA n_kv=2, ctx 128`, **954K params**), trained out,
reaches **val loss 1.52 · perplexity 4.56 · 2.19 bits-per-char** and writes
coherent Shakespeare:

```
Thou art a holy subjects for Rome, so near
To accident the crown to hear him within ...
```

But the headline is the **rigor**, not just the champion:

**1. Evolution beats random search at equal compute.** Random search is a famously
strong NAS baseline ([Li & Talwalkar, 2019](#)). A naïve high-mutation EA *tied/lost*
to it — so I diagnosed it (mutation ≈ a random walk), lowered mutation 0.34→0.18 and
strengthened selection. The tuned EA then **wins: best fitness 1.9246 vs random's
1.9606**, reaches random's *final* quality in **1.8× fewer evaluations**, and improves
the whole **population mean (2.077→1.959)** — which random search cannot do.

**2. The cheap proxy is trustworthy.** The search ranks candidates by 200-step
training; vs 700-step ground truth this gives **Spearman ρ = 0.70, Pearson r = 0.93**
— a good (if imperfect) ranker, which is *why* the search works.

**3. Lamarckian weight inheritance helps.** Warm-starting bred children from the
champion's overlapping weights (vs from scratch, same budget) improves final val loss
**2.098 → 1.836**.

| Experiment | Result |
|---|---|
| Champion language model | val 1.52 · ppl 4.56 · 2.19 bpc |
| Evolution vs. random (equal compute) | **+0.036 fitness, 1.8× more sample-efficient** |
| Proxy fidelity (200 vs 700 steps) | Spearman ρ = 0.70, Pearson r = 0.93 |
| Weight inheritance A/B | val 2.098 → **1.836** |

Full numbers, figures, and discussion: **[`results/REPORT.md`](results/REPORT.md)** ·
**[`TECHNICAL.md`](TECHNICAL.md)**.

![fitness curve](results/fitness_curve.png)
![evolution vs random](results/experiments/compare_search.png)
