# EvoGPT — The Complete Beginner-to-Expert Walkthrough

> A from-the-ground-up explanation of **everything** in this project — assuming you
> know how to read but nothing about AI. We start with "what is a language model"
> and end tracing real numbers through every layer of the transformer.
>
> Read top-to-bottom for the full story, or jump via the table of contents.

## Table of contents
1. [The 60-second version](#1-the-60-second-version)
2. [Background: what a language model actually is](#2-background-what-a-language-model-actually-is)
3. [Background: what a transformer is](#3-background-what-a-transformer-is)
4. [The architecture, component by component](#4-the-architecture-component-by-component)
5. [The full data flow, traced with real tensor shapes](#5-the-full-data-flow-traced-with-real-tensor-shapes)
6. [How the model learns (training)](#6-how-the-model-learns-training)
7. [The evolutionary architecture search (the "crazy" part)](#7-the-evolutionary-architecture-search-the-crazy-part)
8. [The experiments (how we prove it works)](#8-the-experiments-how-we-prove-it-works)
9. [The code map (every file)](#9-the-code-map-every-file)
10. [The results](#10-the-results)
11. [Glossary of every term](#11-glossary-of-every-term)
12. [Interview Q&A](#12-interview-qa)

---

## 1. The 60-second version

**EvoGPT does two things:**

1. **Builds a mini "ChatGPT" from scratch** — a small AI that learns to write
   Shakespeare one character at a time, with every modern component
   (the same ones inside Meta's Llama) hand-written in PyTorch.

2. **Evolves the design of that AI automatically** — a second program breeds
   hundreds of different model designs using survival-of-the-fittest, finds the
   best one, and then *proves with experiments* that the evolution genuinely
   works (it beats random guessing, the scoring is trustworthy, and an advanced
   "inheritance" trick helps).

Everything runs locally on an Apple M3 laptop. The result: a 954,000-parameter
model that writes coherent Shakespearean text, discovered automatically.

```
Thou art a holy subjects for Rome, so near
To accident the crown to hear him within...
        ↑ written by the model EvoGPT designed and trained
```

---

## 2. Background: what a language model actually is

A language model is a **next-character guesser**. That is the entire idea.

- You show it some text: `"To be or not to b"`
- It outputs a **probability for every possible next character**:
  `'e' → 71%, 'a' → 4%, ' ' → 3%, ...`
- Pick the likely one (`'e'`), stick it on the end, and ask again.
- Repeat → it writes whole sentences, paragraphs, plays.

That's how ChatGPT works too — just bigger. **If you can predict the next
character/word well, you can generate language.**

### Turning text into numbers (tokenization)

Computers do math on numbers, not letters. So we build a dictionary mapping each
unique character to a number:

```
' ' → 0   '!' → 1   ',' → 2  ...  'a' → 39   'b' → 40  ...  'z' → 64
```

Shakespeare's text uses **65 distinct characters**, so every character becomes a
number from 0 to 64. This is **character-level tokenization** — the simplest kind.
(Large models use "BPE" which maps common word-chunks, but char-level is ideal for
learning the fundamentals and keeps the model tiny.)

> **In the code:** `evogpt/data.py` → `CharDataset`.
> `.encode("hi")` → `[44, 45]`. `.decode([44, 45])` → `"hi"`.
> `.vocab_size` → `65`.

### Where the "correct answers" come from (self-supervision)

We never hand-label data. The label is **free**: the correct next character is
simply the character that actually comes next in Shakespeare. So from one line of
text we get hundreds of (input → correct-next-char) training examples. This is
called **self-supervised learning**, and it's why language models can train on raw
internet text.

```
Input:   "To be or not to b"
Target:  "o be or not to be"     ← the same text shifted left by one
                                    (each position's label = the next character)
```

---

## 3. Background: what a transformer is

A **transformer** is a specific design of neural network — the "T" in GPT
(Generative **Pre-trained Transformer**). It powers essentially all modern AI.

The core insight: to predict the next word, a model must **pay attention to the
right earlier words**.

> *"The cat that the dog chased was scared."*
> To know **who** was scared, you must link "scared" → "cat" (not "dog").

The mechanism that lets every position look back and decide *which earlier
positions matter* is called **attention**. A transformer is mostly just attention,
stacked and repeated.

```
characters
   │
   ▼
┌──────────────────────────────────────────────┐
│ EMBEDDING: each character-number → a vector   │   "give each char meaning"
└──────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────┐
│ BLOCK 1:  Attention → SwiGLU                  │ ┐
├──────────────────────────────────────────────┤ │ repeated
│ BLOCK 2:  Attention → SwiGLU                  │ │ N times
├──────────────────────────────────────────────┤ │ (champion: N = 3)
│ BLOCK 3:  Attention → SwiGLU                  │ ┘
└──────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────┐
│ FINAL NORM → predict next-char probabilities  │
└──────────────────────────────────────────────┘
   │
   ▼
probability for each of the 65 possible next characters
```

What makes **this** project special: we implemented every box by hand, using the
same techniques as state-of-the-art 2023–2024 models — no `nn.Transformer`, no
HuggingFace. Now let's open each box.

---

## 4. The architecture, component by component

All of this lives in **`evogpt/model.py`**. We'll explain each piece three ways:
**(a)** a kid-simple analogy, **(b)** what it actually does, **(c)** why it matters.

### 4.0 The config that defines a model — `GPTConfig`

Every model is described by a handful of numbers (the "genome" later evolution
will tune). The **champion** EvoGPT discovered:

```python
GPTConfig(
    vocab_size = 65,     # number of distinct characters
    block_size = 128,    # context length: how many past chars it can see
    n_layer    = 3,      # number of stacked Blocks
    d_model    = 192,    # width: size of each character's vector
    n_head     = 8,      # number of attention "heads"
    n_kv_head  = 2,      # GQA: number of shared key/value heads
    mlp_ratio  = 2.0,    # SwiGLU hidden size = 2.0 × d_model
    dropout    = 0.1,    # regularization (randomly zero 10% of signals in training)
)
# head_dim = d_model / n_head = 192 / 8 = 24
```

These knobs control the trade-off between **smart** (bigger numbers) and
**fast/small** (smaller numbers). Keep them in mind; the shapes below use them.

### 4.1 Embedding — turning numbers into meaning

- **(a) Analogy:** Instead of describing a person by a single ID number, you
  describe them by a list of traits `[height, age, mood, ...]`. One number is
  meaningless; a list of numbers can capture nuance.
- **(b) What it does:** A lookup table with one row per character. Character `7`
  grabs row 7 — a list of **192 numbers** (because `d_model = 192`). These 192
  numbers are **learned** during training to encode useful properties ("is this a
  vowel? start of a name? punctuation?").
- **(c) Why it matters:** Gives the model a rich, trainable representation to do
  math on. `nn.Embedding(65, 192)` = a 65×192 table.

### 4.2 RoPE — Rotary Position Embeddings (giving the model a sense of order)

- **(a) Analogy:** Give every word a clock hand. Word #1's hand points at 12:00,
  word #2 is rotated a little, word #3 more, and so on. By comparing the **angle
  between two hands**, the model feels how far apart two words are.
- **(b) What it does:** Plain attention is *order-blind* — "dog bites man" and "man
  bites dog" look identical to it. RoPE fixes this by **rotating** each position's
  Query and Key vectors by an angle proportional to the position. Nearby positions
  get similar rotations; distant ones differ a lot.
- **(c) Why it matters:** It's the **modern** way to encode position (used in Llama,
  Mistral, etc.), better than the old "add a position vector" approach, and it
  generalizes to longer sequences gracefully.

> **In the code:** `build_rope_cache()` pre-computes the rotation angles (as
> cosines and sines) once; `apply_rope()` applies them to Q and K inside attention.

### 4.3 Attention with GQA — the heart of the transformer

This is where each position looks back at earlier positions. For every position,
the model computes three vectors:

| Vector | Role | Analogy (a search engine) |
|---|---|---|
| **Query (Q)** | "what I'm looking for" | your search box text |
| **Key (K)** | "what I contain" | each web page's title |
| **Value (V)** | "the info I hand over if picked" | each web page's contents |

**The mechanism:** each position's **Query** is compared (dot product) with every
earlier position's **Key**. A strong match = high "attention weight" = that
position's **Value** gets pulled in heavily. The position's new representation is a
weighted blend of the Values it chose to attend to.

```
Position 5's Query  ·  Keys of positions 1..5   →  attention weights (how much to listen)
                                                    ↓
                          new vector for position 5 = weighted sum of Values 1..5
```

**Multi-head (`n_head = 8`):** the model does this **8 times in parallel**, each
"head" with its own Q/K/V, so different heads can track different relationships
(one head tracks grammar, another tracks which name a pronoun refers to, etc.).
Each head works in a smaller `head_dim = 24` slice.

**Causal masking:** Position 5 may attend to positions 1–5 only — **never the
future** (6, 7, ...). Otherwise it would peek at the answer it's supposed to
predict. We force this with `is_causal=True`.
✅ **Verified** by `tests/` (`test_causality`): changing a *future* character does
**not** change earlier positions' outputs — proving no information leaks backward.

**GQA — Grouped-Query Attention (`n_kv_head = 2`):** Normally all 8 heads each
carry their own Key and Value, which is memory-hungry. GQA lets the 8 Query-heads
**share just 2 Key/Value sets** (each K/V set is reused by 8/2 = 4 query-heads).
Nearly the same quality, much less memory — a 2023 production trick.
✅ **Verified** by `test_gqa_param_savings`: fewer KV-heads → fewer parameters.

**The fast math:** internally we use PyTorch's
`scaled_dot_product_attention`, the optimized fused implementation of all of the
above.

### 4.4 SwiGLU — the "think hard" feed-forward network

- **(a) Analogy:** Attention *gathers* the relevant context; SwiGLU *thinks about*
  what was gathered.
- **(b) What it does:** A small 2-layer neural network applied at each position,
  with a smart **gating** mechanism: it computes two projections, passes one
  through a smooth activation (SiLU), and **multiplies** them together (the "gate")
  before projecting back. Hidden size = `mlp_ratio × d_model = 2.0 × 192 = 384`
  (rounded to a multiple of 32 for hardware efficiency).
- **(c) Why it matters:** SwiGLU (and GLU-variants generally) train better than the
  old plain feed-forward; it's the modern default in Llama-class models.

### 4.5 RMSNorm — the automatic volume knob

- **(a) Analogy:** A volume normalizer. As signals pass through many layers they
  can blow up or fade out; RMSNorm rescales them to a steady level at each step.
- **(b) What it does:** Divides each position's vector by its root-mean-square
  magnitude, then scales by a learned weight. "Pre-norm" = applied **before** each
  sub-step (attention and SwiGLU).
- **(c) Why it matters:** Keeps training **stable** in deep networks. RMSNorm is the
  lightweight modern variant (used in Llama) of the older LayerNorm.

### 4.6 Residual connections — "adjust, don't replace"

Each block does `x = x + attention(norm(x))` and `x = x + swiglu(norm(x))`. The
`x +` part is a **residual (skip) connection**: each block **nudges** the running
signal rather than overwriting it. This is what allows deep networks to train at
all (gradients flow cleanly through the `+`).

### 4.7 KV-cache — making text generation fast

- **(a) Analogy:** When writing, you don't re-read the entire book to add each new
  word — you keep notes and glance at them. The cache is those notes.
- **(b) The problem:** Generating one character at a time, a naive model would
  re-process the **entire** sentence-so-far for every new character. Character #500
  would redo all 499 before it — wasteful and slow.
- **(c) The fix:** The KV-cache **stores** the Keys and Values already computed for
  earlier characters. Each new character only computes **its own** K/V and reads the
  rest from the cache.
- ⚠️ **Subtlety:** during this one-character-at-a-time decoding, the causal mask
  must be turned **off** (`is_causal=False`) because the single new query is
  *supposed* to see all cached past keys. Getting this wrong is a classic bug.
- ✅ **Verified** by `test_kv_cache_matches_full`: generating with the fast cache
  produces **exactly** the same characters as the slow full recomputation.

### 4.8 Weight tying — sharing the input and output tables

The first step turns characters → vectors (embedding). The last step turns vectors
→ character-scores (the "LM head"). These are mirror images, so we make them
**share the same weight numbers** (`lm_head.weight = tok_emb.weight`) instead of
learning two separate tables. Saves memory and usually improves quality.
✅ **Verified** by a test that the two share the same storage.

### 4.9 The Block, assembled

```
        x  ──────────────┐ (residual: keep a copy)
        │                │
   ┌────▼─────┐          │
   │ RMSNorm  │          │
   └────┬─────┘          │
   ┌────▼─────────────┐  │
   │ Attention        │  │
   │  (RoPE + GQA     │  │
   │   + KV-cache)    │  │
   └────┬─────────────┘  │
        │                │
        +◄───────────────┘  x = x + attention(norm(x))
        │
        x ───────────────┐ (residual again)
        │                │
   ┌────▼─────┐          │
   │ RMSNorm  │          │
   └────┬─────┘          │
   ┌────▼─────┐          │
   │ SwiGLU   │          │
   └────┬─────┘          │
        +◄───────────────┘  x = x + swiglu(norm(x))
        │
        ▼  (out to next block)
```

Three of these stacked = the champion model.

---

## 5. The full data flow, traced with real tensor shapes

Let's push **one batch** through the champion model and watch the shapes. A
"tensor" is just a multi-dimensional array of numbers; its **shape** is the size of
each dimension.

**Setup:** batch of `B = 32` text snippets, each `T = 128` characters long.
Champion config: `d_model=192, n_head=8, head_dim=24, n_kv_head=2, vocab=65,
SwiGLU hidden=384`.

| Step | Operation | Output shape | Plain meaning |
|---|---|---|---|
| 0 | input characters `idx` | `(32, 128)` | 32 snippets × 128 char-numbers |
| 1 | token embedding | `(32, 128, 192)` | each char → a 192-number vector |
| 2 | make Q | `(32, 128, 192)` → `(32, 8, 128, 24)` | 8 heads × 24 dims each |
| 2 | make K, V | `(32, 128, 48)` → `(32, 2, 128, 24)` | only **2** KV-heads (GQA) |
| 3 | apply RoPE to Q, K | unchanged | inject position via rotation |
| 4 | GQA: repeat KV ×4 | K,V → `(32, 8, 128, 24)` | share 2 KV sets across 8 heads |
| 5 | attention (causal) | `(32, 8, 128, 24)` | each position blends earlier values |
| 6 | merge heads, project | `(32, 128, 192)` | back to model width |
| 7 | residual add | `(32, 128, 192)` | x = x + attention |
| 8 | SwiGLU: up-project | `(32, 128, 384)` | widen to hidden size |
| 9 | SwiGLU: gate + down | `(32, 128, 192)` | think, then back to width |
| 10 | residual add | `(32, 128, 192)` | x = x + swiglu |
| — | **(steps 2–10 repeat for all 3 blocks)** | `(32, 128, 192)` | deeper understanding each block |
| 11 | final RMSNorm | `(32, 128, 192)` | stabilize |
| 12 | LM head (tied) | `(32, 128, 65)` | score for each of 65 next-chars, at every position |
| 13 | softmax | `(32, 128, 65)` | scores → probabilities |
| 14 | cross-entropy vs targets | a single number | "how wrong was it" = **loss** |

**The punchline of step 12:** the model predicts the next character **at every one
of the 128 positions simultaneously** during training (efficient!). During
*generation* we only care about the last position's prediction, then we append it
and roll forward (using the KV-cache).

---

## 6. How the model learns (training)

Learning = adjusting the model's ~954,000 internal numbers ("weights") so the loss
goes down. Lives in **`evogpt/train.py`**, function `train_candidate`. The loop:

```
repeat for max_steps:
    1. get_batch()            → grab random Shakespeare snippets + their targets
    2. logits, loss = model(x, y)   → forward pass, measure wrongness
    3. loss.backward()        → backprop: compute how to nudge every weight
    4. optimizer.step()       → nudge all weights a tiny bit downhill
    5. (occasionally) evaluate on held-out validation text
```

### The key ideas

- **Cross-entropy loss:** high when the model gave the *correct* next character a
  low probability; low when it was confident and right. **This single number is
  what we minimize.**
- **Backpropagation (`loss.backward()`):** calculus (the chain rule) automatically
  computes, for every weight, the direction that reduces loss. PyTorch does this
  for us via "autograd."
- **AdamW optimizer (`optimizer.step()`):** a smart rule for *how* to nudge weights
  using those directions. Plus **weight decay** (gently shrinks weights → less
  overfitting).
- **Learning-rate schedule (`_lr_at`):** the nudge size over time — start tiny
  (**warmup**), rise, then smoothly shrink (**cosine decay**). Standard recipe for
  stable-then-refined training.
- **Train vs validation split:** we hold out 10% of the text the model never trains
  on, to measure honest performance (catch "memorizing vs learning").
- **Gradient clipping:** caps the nudge size so a freak huge gradient can't blow up
  training.

### Reading the score: loss, perplexity, bits-per-char

The same quality, three units:

- **Validation loss = 1.52** (raw cross-entropy; lower is better).
- **Perplexity = e^loss = 4.56** — intuitively, "at each character the model is about
  as unsure as if choosing between ~4.6 equally-likely options" (out of 65). Lower
  = smarter.
- **Bits-per-char = loss / ln(2) = 2.19** — the same thing in information-theory
  units (how many bits to encode each character).

### The crucial mental model

> `train_candidate(design)` returns **how good that design is** (its loss).
> That makes it a **scoring function** — exactly what the evolution needs to compare
> designs. Hold this thought going into Part 7.

---

## 7. The evolutionary architecture search (the "crazy" part)

This is the leap from "I built a GPT" to a **research project**. Lives in
**`evogpt/evolve.py`**.

### 7.1 The problem it solves

To design a transformer you must choose: how many layers? how wide? how many heads?
how much GQA sharing? etc. There are about **6,900 valid combinations** in our
search space — far too many to try by hand. **Neural Architecture Search (NAS)**
means letting an *algorithm* find the best design.

### 7.2 The Darwinian idea

We treat each design like a creature and run **evolution**:

- **Genome** = one design's settings (its "DNA"). Each setting is a "gene":
  ```python
  {n_layer: 3, d_model: 192, n_head: 8, kv_ratio: 4,
   mlp_ratio: 2.0, block_size: 128, dropout: 0.1}
  ```
  The full menu of genes (`SEARCH_SPACE`):
  ```
  n_layer    ∈ {2, 3, 4, 5}
  d_model    ∈ {48, 64, 96, 128, 160, 192}
  n_head     ∈ {2, 4, 8}
  kv_ratio   ∈ {1, 2, 4}      # heads ÷ kv_ratio = n_kv_head (GQA sharing)
  mlp_ratio  ∈ {2.0, 2.5, 2.667, 3.0, 3.5, 4.0}
  block_size ∈ {64, 96, 128}
  dropout    ∈ {0.0, 0.1}
  ```

- **Population:** start with 8 designs.
- **Fitness** (`fitness()`): train each design briefly, measure its loss. Lower =
  fitter. **But also penalize size** (see 7.3).
- **Selection:** the best survive ("elites"); the rest are candidates to be
  replaced.
- **Breeding** the next generation:
  - **Tournament selection:** pick a few random designs, keep the best — repeat to
    choose parents. (Good designs get picked more often, but not *only* the very
    best — preserves diversity.)
  - **Crossover** (`crossover`): mix two parents' genes — `n_layer` from one,
    `d_model` from the other, etc. (a child inheriting traits from both parents).
  - **Mutation** (`mutate`): randomly change a gene or two for fresh variety
    ("try 4 layers instead of 3").
- **Repeat** for several **generations**. The population improves over time, like
  selectively breeding faster racehorses.

```
   ┌─────────────────────────────────────────────────┐
   │  POPULATION of 8 architecture genomes            │
   └───────────────────────┬─────────────────────────┘
                           │ train each briefly (the proxy), score it
                           ▼
   ┌─────────────────────────────────────────────────┐
   │  FITNESS = validation_loss + 0.05·(params / 1M)  │
   └───────────────────────┬─────────────────────────┘
                           │ keep elites; pick parents by tournament
                           ▼
   ┌─────────────────────────────────────────────────┐
   │  CROSSOVER + MUTATION  →  next generation         │
   └───────────────────────┬─────────────────────────┘
                           │ repeat for N generations
                           ▼
              CHAMPION → train it fully → generate text
```

### 7.3 The smart part: multi-objective fitness

```
fitness = validation_loss  +  0.05 × (parameters / 1,000,000)
          └── be accurate ──┘     └──── but also be small ────┘
```

The size penalty (in "nats per million params") means the search prefers a small
model when quality is comparable. Without it, evolution would just bloat the model
forever. **With it, the champion lands at the efficiency sweet spot** — and indeed
your `pareto.png` plot shows the *biggest* candidates actually scored *worse*.

### 7.4 Speed tricks that make this feasible on a laptop

- **Proxy training:** we train each candidate only ~200–300 steps to *estimate* its
  quality quickly (full training of hundreds of models would take days). We later
  **prove** this estimate is trustworthy (Part 8.2).
- **Caching:** identical genomes (e.g., elites carried forward) are never retrained.
- **Lineage logging:** we record each design's parents, enabling a **family tree**
  visualization (`experiments/viz_lineage.py` → `results/experiments/lineage.png`).
- **Weight inheritance** (optional, Part 8.3): children can start from a parent's
  learned weights instead of scratch.

### 7.5 One command, end-to-end (`run_search.py`)

`run_evolution()` runs the loop above → picks the **champion** → trains it longer
(1500 steps) → **generates a Shakespeare sample** → saves the trained weights
(`results/champion.pt`), config, plots, and reports.

---

## 8. The experiments (how we prove it works)

Anyone can *claim* their search works. A serious engineer **proves** it. This is
the most credible part of the project, and your best interview material.

### 8.1 "Does evolution actually beat random guessing?" — the honesty test

**Why it matters:** A famous result (Li & Talwalkar, 2019) showed **random search
is a shockingly strong baseline** in NAS — often as good as fancy methods. So if you
don't compare against random, your "smart" search might be smart for nothing.

**The real story (and it's a great one):**
1. **First attempt: evolution LOST to random search.** Our mutation rate was too
   high (0.34), so children barely resembled their parents — evolution was
   effectively *doing* random search.
2. **Diagnosis + fix:** lowered mutation to 0.18 (children resemble parents →
   genuine "hill-climbing"), strengthened selection, ran more generations.
3. **Result: evolution now WINS.** It found a better design (**+0.036 fitness**),
   reached random search's *best* result in **1.8× fewer evaluations** (more
   sample-efficient), and steadily improved the **whole population's average** —
   something random search structurally cannot do.

**Plot `compare_search.png`:** the blue (evolution) line starts *behind* random
early (the honest disadvantage), crosses over, then pulls decisively ahead.

> **This is gold in interviews:** you showed a negative result, understood *why*,
> fixed it, and re-measured. That's exactly how real ML research works.

### 8.2 "Can we trust the quick scoring?" — proxy fidelity

**The worry:** we rank designs using only ~200 steps of training. What if a design
that looks good after 200 steps is actually bad after full training? Then the whole
search is built on noise.

**The test:** take 10 designs; score each two ways — quick (200 steps) and thorough
(700 steps); check whether the **rankings agree**.

**Result:** **Spearman rank correlation = 0.70**, **Pearson = 0.93**. Translation:
the quick score is a **good (not perfect) predictor** of the true score — good
enough that the search works, imperfect enough that we report the limitation
honestly. (`proxy_fidelity.png` = a scatter where up-and-right trend = agreement.)

### 8.3 "Can children inherit learned skills?" — Lamarckian weight inheritance

Lives in **`evogpt/morph.py`**.

**Normal evolution:** each child design trains **from scratch** (random weights).

**The clever idea:** have a child **inherit the actual learned weights** from the
champion parent — copying the overlapping parts (e.g., a parent's 192-wide layer
seeds the top-left 128×128 corner of a child's 128-wide layer). The child starts
*partly trained* instead of blank. (Called "Lamarckian" — inheriting acquired
traits.)

**The A/B test:** run the search with inheritance ON vs OFF, same compute.
**Result:** final loss **2.098 (scratch) → 1.836 (inherit)** — a big improvement,
because warm-started children reveal their potential within the tiny per-candidate
budget, so the search makes better decisions. (`inheritance_ab.png`.)

---

## 9. The code map (every file)

```
EvoGPT/
├── evogpt/                  ← THE CORE LIBRARY (brain + breeder)
│   ├── model.py             ← the transformer: embedding, RoPE, Attention(GQA),
│   │                          SwiGLU, RMSNorm, KV-cache, generate(), weight tying
│   ├── data.py              ← CharDataset: text ↔ numbers, serves training batches
│   ├── train.py             ← train_candidate(): trains ONE design, returns its score
│   │                          (the "fitness function" the search calls); LR schedule
│   ├── evolve.py            ← genomes, SEARCH_SPACE, mutate, crossover, fitness,
│   │                          run_evolution() (selection + breeding + lineage)
│   └── morph.py             ← weight inheritance (copy overlapping weights to children)
│
├── experiments/             ← THE PROOF
│   ├── run_all.py           ← runs the whole research suite end-to-end
│   ├── compare_v2.py        ← evolution vs random search (the headline experiment)
│   ├── viz_lineage.py       ← draws the family-tree (genealogy DAG) of the search
│   └── _common.py           ← stats helpers: Spearman, Pearson, best-so-far
│
├── run_search.py            ← simple entry point: search → train champion → sample
├── sample.py                ← load saved champion, generate text (--prompt, --tokens)
├── analyze.py               ← search log → fitness curve + Pareto plot
├── make_report.py           ← assemble all results into results/REPORT.md
│
├── tests/                   ← 49 automated correctness tests (causality, KV-cache, …)
├── results/                 ← outputs: champion.pt/.json, plots, REPORT.md
├── data/shakespeare.txt     ← the 1.1 MB training corpus
│
├── README.md                ← project overview (start here)
├── TECHNICAL.md             ← deep technical write-up
├── WALKTHROUGH.md           ← this document
├── RESUME_BULLETS.md        ← ready-to-paste resume lines + interview answers
├── pyproject.toml           ← makes it a real installable Python package
└── Makefile · LICENSE · CONTRIBUTING.md · .github/workflows/ci.yml   ← pro polish
```

**Control flow in one sentence:** `run_search.py` → `run_evolution` (evolve.py) →
repeatedly calls `train_candidate` (train.py) → which builds an `EvoGPT` (model.py)
and feeds it batches from `CharDataset` (data.py); the winner is trained longer and
saved; `experiments/` then stress-tests the whole method.

### Commands you can run

```bash
pip install -e .                 # install the package
python run_search.py             # evolve → train champion → sample (quick)
python -m experiments.run_all    # the FULL research suite + all plots (~slow)
python -m pytest -q              # run the 49 correctness tests
python sample.py --prompt "ROMEO:" --tokens 300   # generate from the champion
python make_report.py            # rebuild results/REPORT.md
```

---

## 10. The results

| What | Result | Meaning |
|---|---|---|
| **Champion model** | val loss 1.52 · perplexity 4.56 · 2.19 bits/char | Writes coherent Shakespeare-ish text, tiny (954K params) |
| **Evolution vs random** | **+0.036 fitness, 1.8× more sample-efficient** | The search genuinely beats the strong baseline |
| **Proxy fidelity** | Spearman 0.70, Pearson 0.93 | The fast scoring used in search is trustworthy |
| **Weight inheritance** | val 2.098 → **1.836** | An advanced technique that measurably helps |
| **Engineering** | 49 tests pass · ruff-clean · CI on Py 3.10–3.12 | Production-quality, not a throwaway notebook |

All trained **locally on an Apple M3 (16 GB)** using the Mac GPU via PyTorch's
**MPS (Metal)** backend — no cloud, no GPU budget.

Figures (in `results/` and `results/experiments/`):
- `fitness_curve.png` — best & average fitness improving across generations
- `pareto.png` — quality vs size; champion sits at the efficiency knee
- `lineage.png` — the family tree of the search
- `compare_search.png` — evolution overtaking random search
- `proxy_fidelity.png` — quick-score vs true-score agreement
- `inheritance_ab.png` — inheritance vs from-scratch

---

## 11. Glossary of every term

| Term | Plain-English meaning |
|---|---|
| **Token** | One unit of text the model reads; here, a single character. |
| **Vocabulary (vocab)** | The set of distinct tokens. Here, 65 characters. |
| **Embedding** | A learned lookup table turning each token into a vector of numbers. |
| **Vector / tensor** | A list (or grid) of numbers the model does math on. |
| **d_model** | The "width" — how many numbers represent each token (192). |
| **Transformer** | The neural-network design built around attention. |
| **Attention** | Mechanism letting each position look back at earlier ones and weight them. |
| **Query / Key / Value (Q/K/V)** | Search-text / page-titles / page-contents inside attention. |
| **Head (n_head)** | One parallel attention computation; 8 heads track 8 relationship types. |
| **head_dim** | Size of each head's slice (d_model / n_head = 24). |
| **Causal mask** | Rule that a position may only see the past, never the future. |
| **GQA** | Grouped-Query Attention: heads share Key/Value sets to save memory. |
| **RoPE** | Rotary Position Embeddings: encodes word order by rotating Q/K vectors. |
| **RMSNorm** | A normalization that keeps signal magnitudes stable across layers. |
| **SwiGLU** | A modern gated feed-forward sub-network ("the thinking part"). |
| **Residual connection** | `x = x + f(x)`; blocks adjust the signal instead of replacing it. |
| **Block / layer** | One Attention + SwiGLU unit; stacked n_layer times. |
| **KV-cache** | Stored Keys/Values so generation doesn't recompute the whole past. |
| **Weight tying** | Sharing the input-embedding and output-prediction tables. |
| **block_size** | Context length: how many past tokens the model can attend to (128). |
| **Parameter / weight** | One of the model's learnable numbers (~954,000 total). |
| **Forward pass** | Running input through the model to get a prediction. |
| **Loss** | A single number measuring how wrong the prediction was (we minimize it). |
| **Cross-entropy** | The specific loss for "predict the right category" problems. |
| **Backpropagation** | Calculus that tells each weight which way to move to lower loss. |
| **Optimizer (AdamW)** | The rule that nudges weights using backprop's directions. |
| **Learning rate** | How big each nudge is. |
| **Epoch / step** | One training iteration on one batch. |
| **Overfitting** | Memorizing training data instead of learning general patterns. |
| **Validation set** | Held-out data to measure honest performance. |
| **Perplexity** | e^loss; "effective number of choices" the model is unsure between. |
| **Bits-per-char** | loss / ln(2); information needed per character. |
| **NAS** | Neural Architecture Search: letting an algorithm design the network. |
| **Genome** | One architecture's settings, treated as DNA for evolution. |
| **Fitness** | A design's score (val loss + size penalty); lower is better. |
| **Generation** | One round of the evolution loop. |
| **Elite** | A top design preserved unchanged into the next generation. |
| **Tournament selection** | Pick a few designs at random, keep the best, to choose parents. |
| **Crossover** | Mixing two parents' genes to form a child design. |
| **Mutation** | Randomly tweaking a gene for variety. |
| **Proxy training** | Brief training used to estimate a design's quality cheaply. |
| **Random search baseline** | Trying random designs; the standard "is your method real?" comparison. |
| **Spearman / Pearson** | Statistics measuring how well two rankings/values agree. |
| **Lamarckian inheritance** | Children inherit a parent's learned weights, not just its design. |
| **MPS / Metal** | Apple's GPU backend that PyTorch uses to train fast on a Mac. |

---

## 12. Interview Q&A

**Q: In one breath, what is EvoGPT?**
A from-scratch GPT-style transformer (RoPE, GQA, SwiGLU, KV-cache — all hand-written)
wrapped in an evolutionary architecture-search loop that automatically discovers a
good model, validated with experiments showing it beats random search, that its
cheap proxy scoring is trustworthy, and that weight inheritance helps.

**Q: Why GQA and a KV-cache?**
Both are about efficiency. GQA shares Key/Value sets across attention heads to cut
memory; the KV-cache stores past Keys/Values so generating each new token is O(1)
work instead of reprocessing the whole sequence. The subtle bit: the causal mask has
to be disabled during single-token decoding, which I verified with an exact-match
test against a full forward pass.

**Q: Why compare against random search?**
Because random search is a famously strong NAS baseline (Li & Talwalkar, 2019).
Without it, "my search improves over generations" is meaningless. My first EA
actually lost to random — the mutation rate was so high it was basically random
sampling — so I lowered mutation, strengthened selection, and then it won on both
final quality and sample efficiency. Reporting that honestly is the point.

**Q: Why a parameter penalty in the fitness?**
To make it multi-objective: accurate *and* small. The champion sits at the
quality-vs-size Pareto knee; the largest candidates actually scored worse, so the
penalty steered the search toward an efficient design.

**Q: Is the cheap proxy valid?**
Spearman 0.70 between 200-step and 700-step rankings — a good but imperfect ranker.
That's exactly why the search works yet isn't oracle-perfect; more eval steps would
tighten it.

**Q: What's the hardest bug you fixed?**
The KV-cache causal-masking subtlety, and separately, diagnosing why evolution
wasn't beating random search (too-high mutation acting as a random walk).

**Q: What would you do next?**
A real BPE tokenizer on a larger corpus, a learned performance predictor to skip
training weak candidates, NSGA-II for a true multi-objective Pareto front, and
multi-seed variance bands on every result.

---

*Written as the companion teaching doc for EvoGPT. For the terse overview see
[`README.md`](README.md); for the formal write-up see [`TECHNICAL.md`](TECHNICAL.md).*
