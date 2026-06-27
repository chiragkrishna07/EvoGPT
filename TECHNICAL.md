# EvoGPT: A From-Scratch Transformer That Designs Itself

EvoGPT is two things welded together. The first is a GPT-style transformer written from
first principles in PyTorch — no `nn.Transformer`, no HuggingFace, just tensors, attention,
and a pile of correctness tests. The second is an evolutionary Neural Architecture Search
(NAS) loop that treats the network's own design as the thing to optimize. The whole system
trains char-level on tiny-shakespeare, end to end on a single Apple M3 (16GB) through the
PyTorch MPS/Metal backend, with no external APIs in the loop. This document explains how
each piece works, and — more importantly — how I checked that each piece actually does what
it claims.

---

## Why this project

The cheap version of "ML engineering" today is gluing a hosted LLM behind an API and calling
it a system. That skill is real but shallow: it never forces you to understand what a
transformer *is*, why a particular normalization or positional scheme was chosen, or how to
tell a working attention mask from a subtly broken one.

EvoGPT deliberately fills that gap. Every architectural component is hand-written and pinned
by a test. And rather than hand-tuning hyperparameters by intuition, the project poses
architecture design as a formal optimization problem and then asks the uncomfortable
question most NAS write-ups dodge: *is the search actually doing anything a coin flip
wouldn't?* The honest answer — and the iteration it took to earn a "yes" — is the heart of
this write-up.

---

## Architecture from scratch

The model is a decoder-only transformer in the modern Llama-ish style. Each component was
chosen for a concrete reason, and each is held in place by a test.

**Token embedding with weight tying.** The input embedding matrix and the output LM-head
projection share the same weights. This is a well-established trick (it cuts parameters and
regularizes the output distribution toward the input geometry). The test suite verifies the
tie at the *storage* level — the two tensors are literally the same object — so the
optimization can never silently desynchronize them.

**Pre-norm RMSNorm.** Normalization is applied *before* each sublayer (pre-norm), which
keeps a clean residual highway and makes deep stacks trainable without warmup gymnastics. I
use RMSNorm rather than LayerNorm: it drops the mean-centering term and only rescales by the
root-mean-square, which is cheaper and empirically just as stable.

**Rotary Position Embeddings (RoPE).** Position is injected by rotating the query and key
vectors as a function of their absolute position, so that the dot product between two tokens
depends only on their *relative* offset. There are no learned position embeddings to run out
of — RoPE extrapolates by construction. The rotation is applied to Q and K only, never to V.

**Causal multi-head attention with Grouped-Query Attention (GQA).** Standard causal
self-attention, but the number of key/value heads is configurable and can be smaller than
the number of query heads. With `n_kv_head < n_head`, several query heads share one K/V head,
shrinking the KV projections and (critically) the KV-cache. A `kv_ratio` of 4 means a quarter
the K/V heads. The test suite asserts the actual parameter savings match the expected ratio,
so GQA can't degrade into plain multi-head attention without the tests noticing.

**SwiGLU feed-forward.** The MLP uses the SwiGLU gated activation rather than a plain
GELU/ReLU MLP. The hidden width is `mlp_ratio * d_model`, rounded to a multiple of 32 so the
matmuls stay hardware-friendly:

```
hidden = round_to_multiple_of_32(mlp_ratio * d_model)
FFN(x)  = W_down( SiLU(W_gate x) ⊙ (W_up x) )
```

**KV-cache for O(1)-per-token decoding.** During autoregressive generation, previously
computed keys and values are cached so each new token costs one step of attention against the
cache rather than a full re-encode of the prefix. One subtlety the tests pin down: the causal
mask (`is_causal`) must be *on* during the prefill/full forward but *off* during single-token
incremental decode, because a length-1 query against a cached prefix has nothing to mask. The
suite checks **exact greedy equivalence** between cached incremental decoding and a plain full
forward — same tokens, every time.

**GPT-2-style scaled residual init.** Residual-projection weights are initialized scaled by
`1/sqrt(2 * n_layer)` so the residual stream's variance doesn't blow up with depth.

### The tests that pin it down

Correctness here isn't a vibe; it's a 49-test pytest suite. The load-bearing ones:

- **Causal mask integrity:** perturbing a *future* token must not change the logits at any
  *earlier* position. This is the single most important invariant in a causal LM, and it's
  the easiest to break silently.
- **KV-cache vs full-forward equivalence:** incremental greedy decoding produces byte-for-byte
  the same sequence as the full forward pass.
- **GQA parameter savings:** measured K/V parameter count matches the `kv_ratio`.
- **Weight-tying storage identity:** embedding and LM head are the same underlying tensor.

When a refactor breaks one of these, you find out in seconds rather than after a wasted
training run.

---

## The search as an optimization problem

With a correct, configurable model in hand, architecture design becomes optimization over a
discrete space. Each candidate is a **genome** — a fixed set of genes:

```
genome = {
  n_layer    ∈ {2, 3, 4, 5},
  d_model    ∈ [48 .. 192],
  n_head     ∈ {2, 4, 8},
  kv_ratio   ∈ {1, 2, 4},        # n_kv_head = n_head / kv_ratio
  mlp_ratio  ∈ [2.0 .. 4.0],
  block_size ∈ {64, 96, 128},
  dropout    ∈ {0.0, 0.1},
}
```

That's roughly a **6,900-architecture** space — small enough to reason about, large enough
that exhaustive enumeration under real training budgets is impractical.

**Multi-objective fitness.** A pure validation-loss objective would happily crown the
biggest model that fits in memory. Instead, fitness is loss plus a parameter-efficiency
penalty, kept in the natural units of the loss (nats):

```
fitness = val_loss + 0.05 * (params / 1e6)        # lower is better
```

The `0.05`-nats-per-million-params tax pushes selection toward the *efficiency knee* rather
than raw capacity, which is exactly where the champion ends up (see `results/pareto.png`).

**Evolutionary operators.** Each generation:

1. **Train** every candidate under a fixed, cheap compute budget (a few hundred steps).
   Already-evaluated genomes are **cached**, so re-encountering an architecture is free.
2. **Score** by the fitness above.
3. **Keep elites** (the top genomes survive unchanged).
4. **Breed** the next generation via **tournament selection** (sample `k`, keep the best),
   **uniform crossover** (each child gene drawn from one of two parents), and **per-gene
   mutation**.

**Parent ids are logged**, so the full genealogy is a plottable DAG (`results/experiments/lineage.png`).

![Fitness per generation](results/fitness_curve.png)

---

## Results

The search converges on a compact, efficient champion:

```
champion = {
  n_layer: 3, d_model: 192, n_head: 8, n_kv_head: 2 (kv_ratio 4),
  mlp_ratio: 2.0, block_size: 128, dropout: 0.1
}
# ≈ 954K parameters
```

Note the aggressive GQA (kv_ratio 4) and the *minimum* mlp_ratio — the parameter penalty
clearly shaped this design toward doing more with attention and less with a fat MLP. After
extended training (1500 steps), the champion reaches:

| Metric           | Value  |
|------------------|--------|
| Validation loss  | 1.5177 |
| Perplexity       | 4.56   |
| Bits per char    | 2.190  |

It generates coherent (if charmingly nonsensical) Shakespearean text — correctly formatted
speaker tags, plausible word shapes, and Early-Modern-English cadence — which is about the
ceiling for a ~1M-parameter char-level model on this corpus.

During the search itself, under the cheap proxy budget, the best fitness improved from
**1.8612** (generation 0) to **1.8049** — roughly a 3% gain. That number is modest on
purpose: the search optimizes a cheap proxy, and the real payoff is the *architecture* it
selects, which is then trained properly.

![Pareto front: val loss vs params](results/pareto.png)

---

## Is the search actually doing anything? Evolution vs random search

This is the section most NAS projects quietly skip, and it's the one that matters most.

Li & Talwalkar (2019) made the field uncomfortable by showing that **random search is a
shockingly strong NAS baseline** — many published "clever" searches barely beat random
sampling at equal compute. So the only honest way to claim an evolutionary algorithm is
*working* is to race it against random search under an identical compute budget.

**First attempt — and an honest failure.** My initial EA used aggressive mutation: a per-gene
rate of **0.34**, flipping about **2.4 genes per child**. Against random search it **tied or
lost** (advantage ≈ **−0.015**). The diagnosis is straightforward: with that much mutation,
each child barely resembles its parents. Selection's signal gets washed out, and the EA
degenerates into something statistically indistinguishable from random sampling. The
algorithm wasn't *evolving* so much as *re-rolling*.

**The fix.** Three changes, all in the direction of more exploitation and less random walk:

- drop per-gene mutation **0.34 → 0.18**,
- raise tournament size **k → 4** (sharper selection pressure),
- add generations (more time to actually exploit good genes).

**Re-measured result.** The tuned EA reaches a best fitness of **1.9246**, versus a
random-search mean of **1.9606**. It even beats the *luckier* of two random seeds (**1.9342**),
for an advantage of **+0.036**. Two further pieces of evidence make the case stronger than a
single number:

- **Efficiency:** the EA reaches random search's *final* quality in **12 of 21** evaluations
  — about **1.75× fewer** trials.
- **Population improvement:** the population **mean** fitness improves **2.0774 → 1.9591**
  across generations. Random search *cannot* do this by construction — it has no memory, so
  its population mean is stationary. A rising mean is direct evidence of heritable selection.

I'm presenting this as an iteration story rather than a clean win because that's what it was:
the first version didn't beat the baseline, I diagnosed *why* (mutation drowning selection),
fixed it, and re-ran the comparison. A search that beats random search only after you tune it
to actually exploit is a more credible result than one that claims victory out of the box.

![Evolution vs random search](results/experiments/compare_search.png)

![Lineage DAG](results/experiments/lineage.png)

---

## Can we trust the cheap proxy?

The whole search rests on a gamble: that ranking candidates by **short** training (200 steps)
predicts their ranking under **real** training. If the proxy is noise, the search optimizes
noise.

So I validated it. I took 10 architectures, ranked them by the 200-step proxy, then trained
each to a **700-step "ground truth"** and compared:

| Correlation        | Value |
|--------------------|-------|
| Spearman ρ (rank)  | 0.697 |
| Pearson r (linear) | 0.933 |

The honest reading: the proxy is a **good-but-imperfect ranker**. The strong Pearson r means
the *magnitudes* line up tightly — a candidate that looks much better usually is. The more
moderate Spearman ρ means the *fine-grained ordering* gets shuffled, especially among
near-ties. That's exactly the regime where the search succeeds without being an oracle: it
reliably separates good from bad, but won't perfectly resolve two similar genomes. More
evaluation iterations and longer proxy training would tighten ρ — at proportionally higher
search cost.

![Proxy fidelity](results/experiments/proxy_fidelity.png)

---

## Lamarckian inheritance

A standard EA throws away everything a parent learned: each child trains from scratch. But the
genome space has structure — a child often shares `d_model`, head counts, and layer shapes
with a parent, which means their weight tensors *overlap*.

So I ran an A/B test on **Lamarckian weight inheritance**: bred children warm-start by copying
the champion's overlapping weight sub-blocks (a network-morphism-style slice copy) instead of
initializing from scratch, under the same per-candidate compute budget.

| Init strategy        | Final best val loss |
|----------------------|---------------------|
| From scratch         | 2.0984              |
| Inherited (warm-start)| 1.8359             |

A large improvement — and the mechanism is the interesting part. Each candidate only gets a
tiny training budget. From scratch, a genuinely good architecture may not have *time* to
reveal its potential, so selection is partly judging which random init got lucky. Warm-started
candidates skip that burn-in and expose their true quality within the budget, which **sharpens
selection** rather than just speeding up individual training.

![Lamarckian A/B](results/experiments/inheritance_ab.png)

---

## Engineering

The research only counts if the code is trustworthy:

- **49 pytest tests** covering the invariants above (causal masking, KV-cache equivalence, GQA
  savings, weight tying) plus shape/contract checks across the model.
- **ruff** for linting and formatting — a clean, consistent codebase.
- **GitHub Actions CI** running the suite across a Python version matrix, so a regression on
  any supported interpreter blocks the merge.
- **Packaging** so the project installs and runs as a proper module, not a pile of scripts.

Everything runs locally on one M3 via MPS. No cloud, no API keys, no hidden compute.

---

## Limitations & future work

I'd rather state these plainly than have a reader find them.

- **Toy corpus, char-level.** tiny-shakespeare is tiny and character-level. Conclusions about
  *architecture* should transfer; absolute quality numbers obviously won't scale to real LM
  tasks. A **BPE tokenizer** is the natural next step.
- **Single GPU, small budgets.** Per-candidate training is deliberately cheap. Larger budgets
  would reduce proxy noise but cost search throughput.
- **Proxy noise (ρ ≈ 0.70).** The ranker is good, not perfect. More eval iterations / longer
  proxy steps would tighten it.
- **Single-objective scalarization.** The parameter penalty collapses two objectives into one
  scalar. A true **NSGA-II Pareto front** would expose the loss/size trade-off without a
  hand-picked penalty weight.
- **No predictor-based acceleration.** A learned performance **predictor** (predictor-based
  NAS) could replace much of the proxy training entirely.
- **No variance bands.** The evolution-vs-random comparison would be even more convincing with
  **multi-seed variance bands** rather than two seeds.

None of these undercut the core claim — a correct from-scratch transformer, and an
evolutionary search that provably beats random search at equal compute — but they're the
honest frontier of what's been shown.

---

## References

- Vaswani et al., *Attention Is All You Need*, 2017.
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021.
- Zhang & Sennrich, *Root Mean Square Layer Normalization*, 2019.
- Shazeer, *GLU Variants Improve Transformer*, 2020.
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, 2023.
- Li & Talwalkar, *Random Search and Reproducibility for Neural Architecture Search*, 2019.
- Real et al., *Regularized Evolution for Image Classifier Architecture Search* (AmoebaNet), 2019.
