"""Tuned evolution vs. random search (equal compute), with sample-efficiency and
population-mean metrics. Overwrites results/experiments/compare_search.{png,json}.

The first pass showed random search edging out a high-mutation EA — the standard
'random search is a strong NAS baseline' result. Here the EA exploits more (lower
mutation, more generations, stronger tournament) and we measure not just final
best but *sample efficiency* (evals to reach a quality bar) and *population-mean
convergence* (which random search structurally lacks).

Run:  python -m experiments.compare_v2          # full
      EVO_QUICK=1 python -m experiments.compare_v2
"""
import os
import json
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from evogpt.data import CharDataset, load_corpus
from evogpt.train import TrainBudget, train_candidate, get_device
from evogpt.evolve import (EvolveConfig, run_evolution, genome_to_config,
                           random_genome, fitness, _genome_key)
from experiments._common import best_so_far

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(HERE, "results", "experiments")
os.makedirs(EXP, exist_ok=True)
QUICK = os.environ.get("EVO_QUICK") == "1"


def random_search(dataset, budget, device, n_eval, seed, cache):
    rng = random.Random(seed)
    fits, seen = [], set()
    while len(fits) < n_eval:
        g = random_genome(rng); key = _genome_key(g)
        if key in seen:
            continue
        seen.add(key)
        if key not in cache:
            cfg = genome_to_config(g, dataset.vocab_size)
            cache[key] = train_candidate(cfg, dataset, budget, device)
        fits.append(fitness(cache[key]))
        print(f"    [rand s{seed}] {len(fits):2d}/{n_eval}  fit {fits[-1]:.4f}", flush=True)
    return fits


def main():
    device = get_device(); torch.manual_seed(1337)
    ds = CharDataset(load_corpus(os.path.join(HERE, "data")), block_size=256, device=device)
    steps = 30 if QUICK else 200
    pop, gens, seeds = (4, 3, (11,)) if QUICK else (6, 7, (101, 202))
    budget = TrainBudget(max_steps=steps, batch_size=32, lr=3e-3, eval_iters=20, eval_every=steps)

    # --- tuned evolutionary search (exploit-heavy) ---
    ec = EvolveConfig(pop_size=pop, generations=gens, elite=2, tournament_k=4,
                      param_penalty=0.05, seed=1337, seed_mode="random", mutation_rate=0.18)
    evo_dir = os.path.join(HERE, "runs", "compare_evo")
    run_evolution(ds, budget, device, ec, evo_dir, log=lambda *a: None)
    rows = [json.loads(l) for l in open(os.path.join(evo_dir, "leaderboard.jsonl"))]
    cache = {}  # seed random's cache with EA-trained genomes (fair reuse)
    for r in rows:
        if r["tag"] == "trained":
            cache[_genome_key(r["genome"])] = {"val_loss": r["val_loss"], "n_params": r["n_params"],
                                               "diverged": r["diverged"]}
    evo_trained = [r["fitness"] for r in rows if r["tag"] == "trained"]
    n_eval = len(evo_trained)
    evo_best = best_so_far(evo_trained)
    # population mean fitness per generation (EA refines the whole population)
    pop_mean = []
    for g in sorted({r["gen"] for r in rows}):
        fs = [r["fitness"] for r in rows if r["gen"] == g and r["fitness"] != float("inf")]
        pop_mean.append(sum(fs) / len(fs))
    print(f"  EA trained {n_eval} archs | final best {evo_best[-1]:.4f} | "
          f"pop-mean {pop_mean[0]:.4f} -> {pop_mean[-1]:.4f}", flush=True)

    # --- random search at equal compute ---
    rand_runs = [random_search(ds, budget, device, n_eval, s, cache) for s in seeds]
    rand_curves = [best_so_far(r) for r in rand_runs]
    rand_mean = [sum(c[i] for c in rand_curves) / len(rand_curves) for i in range(n_eval)]
    rand_min = [min(c[i] for c in rand_curves) for i in range(n_eval)]
    rand_max = [max(c[i] for c in rand_curves) for i in range(n_eval)]

    # --- sample efficiency: evals for EA to reach random's FINAL mean-best ---
    target = rand_mean[-1]
    ea_evals = next((i + 1 for i, v in enumerate(evo_best) if v <= target), None)
    speedup = (n_eval / ea_evals) if ea_evals else None

    advantage = rand_mean[-1] - evo_best[-1]
    xs = list(range(1, n_eval + 1))
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot(xs, evo_best, "o-", color="C0", lw=2, label="evolution (best-so-far)")
    ax.plot(xs, rand_mean, "s--", color="C1", lw=2, label="random search (mean)")
    ax.fill_between(xs, rand_min, rand_max, color="C1", alpha=0.15, label="random search (range)")
    if ea_evals:
        ax.axhline(target, color="grey", ls=":", lw=1)
        ax.axvline(ea_evals, color="C0", ls=":", lw=1)
        ax.annotate(f"EA reaches random's final quality\nin {ea_evals} evals ({speedup:.1f}x fewer)",
                    xy=(ea_evals, target), xytext=(ea_evals + 1, target + 0.04),
                    fontsize=8, color="C0")
    ax.set_xlabel("# architectures trained"); ax.set_ylabel("best fitness so far")
    ax.set_title("Tuned evolution vs. random search (equal compute)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(EXP, "compare_search.png"), dpi=130)

    out = {
        "n_eval": n_eval, "evolution_best": evo_best[-1],
        "random_best_mean": rand_mean[-1], "random_best_min": rand_min[-1],
        "advantage": advantage, "ea_evals_to_match_random_final": ea_evals,
        "sample_efficiency_x": speedup,
        "pop_mean_first": pop_mean[0], "pop_mean_last": pop_mean[-1],
        "pop_mean_per_gen": pop_mean,
        "evolution_curve": evo_best, "random_mean_curve": rand_mean,
        "config": {"pop": pop, "gens": gens, "mutation_rate": 0.18,
                   "tournament_k": 4, "steps": steps, "random_seeds": list(seeds)},
    }
    json.dump(out, open(os.path.join(EXP, "compare_search.json"), "w"), indent=2)
    print(f"\n  RESULT: EA best {evo_best[-1]:.4f} vs random mean {rand_mean[-1]:.4f} "
          f"(min {rand_min[-1]:.4f}) -> advantage {advantage:+.4f}")
    if ea_evals:
        print(f"  sample efficiency: EA matches random's final quality in {ea_evals}/{n_eval} "
              f"evals ({speedup:.1f}x fewer)")
    print(f"  population mean: {pop_mean[0]:.4f} -> {pop_mean[-1]:.4f} "
          f"(random search has no population-mean improvement by construction)")


if __name__ == "__main__":
    main()
