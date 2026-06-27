# EvoGPT — résumé bullets

Drop-in bullets for the **Projects** section. Pick 3–4; numbers come from the
actual runs in `results/REPORT.md`.

**EvoGPT — Self-Optimizing Transformer with Evolutionary Architecture Search** | PyTorch, MPS

- Implemented a GPT-style language model **from scratch** in PyTorch (no
  HuggingFace / `nn.Transformer`) with modern Llama-era components hand-coded:
  **Rotary Position Embeddings, RMSNorm, Grouped-Query Attention, SwiGLU, weight
  tying, and a KV-cache** for O(1)-per-token decoding; champion model reaches
  **perplexity 4.56 / 2.19 bits-per-char** on char-level tiny-shakespeare.
- Built an **autonomous evolutionary Neural Architecture Search** over ~6,900
  architectures that designs the network itself — population of genomes, fixed
  per-candidate training budget, **multi-objective fitness (val loss + parameter
  efficiency)**, tournament selection, crossover, and mutation, with weight-sharing
  cache and full lineage logging.
- **Benchmarked the search against a random-search baseline** (the standard strong
  NAS baseline): diagnosed that a high-mutation EA only matched random, then tuned
  it to **beat random search at equal compute (+0.036 fitness) and reach random's
  final quality in 1.8× fewer evaluations** while improving the whole population's
  mean fitness.
- Added **Lamarckian weight inheritance** (network-morphism slice copy) that
  warm-starts bred children from the champion's weights, improving final validation
  loss **2.10 → 1.84** at equal budget; validated the search's cheap proxy against
  full training (**Spearman ρ = 0.70**).
- Hardened to production standard: **49-test `pytest` suite** (proves causal-mask
  integrity and exact KV-cache/full-forward equivalence), `ruff`-clean, **GitHub
  Actions CI** (Python 3.10–3.12), packaged with `pyproject.toml` — all trained
  locally on **Apple M3 via the PyTorch MPS/Metal backend**.

## One-liner (summary / LinkedIn headline)

> Built EvoGPT: a from-scratch GPT (RoPE, GQA, SwiGLU, KV-cache) wrapped in an
> evolutionary architecture-search loop that **beats random search at equal compute
> (1.8× more sample-efficient)** — with weight-inheritance, proxy-fidelity, and
> random-baseline ablations, 49 tests, and CI. Trained on Apple-Silicon MPS.

## Interview talking points

- **Why GQA + KV-cache?** Decode-time memory/bandwidth — the `n_rep` repeat, and why
  `is_causal` must flip off during incremental decode (`model.py`).
- **Why benchmark against random search?** Random search is a famously strong NAS
  baseline (Li & Talwalkar, 2019); claiming a search "improves over generations"
  without it is meaningless. Showing the *first* EA lost, then fixing it, is the
  honest engineering story.
- **Why a parameter penalty in the fitness?** Multi-objective: the champion sits at
  the efficiency knee of the val-loss-vs-params Pareto front; the largest models
  scored *worse*.
- **Is the cheap proxy valid?** ρ = 0.70 between 200- and 700-step rankings — good but
  imperfect; the gap is exactly the proxy noise that more eval iters would reduce.
- **What's Lamarckian inheritance?** Children inherit overlapping weight sub-blocks
  from the champion, turning each brief training into a fine-tune — sharper selection
  under a tiny budget.
