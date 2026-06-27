"""Run the full EvoGPT experiment suite as one background job.

Produces everything under results/ and results/experiments/:
  1. Production evolutionary search  -> champion + leaderboard
  2. Champion extended training       -> champion.pt/.json + sample
  3. Random-search baseline           -> evolution-vs-random comparison
  4. Proxy-fidelity study             -> Spearman(short, long) rank correlation
  5. Weight-inheritance A/B           -> Lamarckian convergence speedup

Single process so candidates train serially (no MPS contention).
Run:  python -m experiments.run_all
"""
import os
import json
import random
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from evogpt.data import CharDataset, load_corpus
from evogpt.train import TrainBudget, train_candidate, get_device
from evogpt.evolve import (EvolveConfig, run_evolution, genome_to_config,
                           random_genome, fitness, _genome_key, _valid, SEARCH_SPACE)
from experiments._common import spearman, pearson, best_so_far

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
EXP = os.path.join(RESULTS, "experiments")
os.makedirs(EXP, exist_ok=True)

QUICK = os.environ.get("EVO_QUICK") == "1"   # tiny budgets to smoke-test the whole pipeline
# (production_steps, pop, gens, rand_seeds, proxy_n, proxy_short, proxy_long, inh_steps, inh_pop, inh_gens, champ_steps)
K = dict(steps=30, pop=4, gens=2, rand_seeds=(11,), proxy_n=3, p_short=30, p_long=50,
         inh_steps=20, inh_pop=4, inh_gens=2, champ_steps=40) if QUICK else \
    dict(steps=300, pop=8, gens=5, rand_seeds=(11, 22, 33), proxy_n=10, p_short=200, p_long=700,
         inh_steps=120, inh_pop=6, inh_gens=5, champ_steps=1500)


def banner(msg):
    print(f"\n{'='*64}\n{msg}\n{'='*64}", flush=True)


# --------------------------------------------------------------------------- #
def random_search(dataset, budget, device, n_eval, seed, cache, log=print):
    """Uniform random architecture search; returns fitness in evaluation order."""
    rng = random.Random(seed)
    fits, seen = [], set()
    while len(fits) < n_eval:
        g = random_genome(rng)
        key = _genome_key(g)
        if key in seen:
            continue
        seen.add(key)
        if key in cache:
            res = cache[key]
        else:
            cfg = genome_to_config(g, dataset.vocab_size)
            res = train_candidate(cfg, dataset, budget, device)
            cache[key] = res
        fits.append(fitness(res))
        log(f"    [rand s{seed}] eval {len(fits):2d}/{n_eval}  fit {fits[-1]:.4f}")
    return fits


# --------------------------------------------------------------------------- #
def exp_production_and_compare(dataset, device, shared_cache):
    banner("EXPERIMENT 1+3: Production evolution  &  random-search baseline")
    # --- production evolution (the headline champion) ---
    budget = TrainBudget(max_steps=K["steps"], batch_size=32, lr=3e-3, eval_iters=20,
                         eval_every=max(K["steps"] // 2, 1))
    # Random gen-0 (== random search's first samples), then guided refinement —
    # the fair, standard EA-vs-random comparison. In a ~7,800-arch space, pop=8
    # won't hit the optimum at gen 0, so the best still improves over generations.
    ec = EvolveConfig(pop_size=K["pop"], generations=K["gens"], elite=2, tournament_k=3,
                      param_penalty=0.05, seed=1337, seed_mode="random")
    evo_dir = os.path.join(HERE, "runs", "search")
    summary = run_evolution(dataset, budget, device, ec, evo_dir)
    # seed shared cache with everything evolution trained (fair reuse for random)
    rows = [json.loads(l) for l in open(os.path.join(evo_dir, "leaderboard.jsonl"))]
    evo_trained_fits = [r["fitness"] for r in rows if r["tag"] == "trained"]
    n_eval = len(evo_trained_fits)
    print(f"  evolution trained {n_eval} unique architectures")

    # --- random-search baseline at equal compute (3 seeds for variance) ---
    rand_runs = []
    for s in K["rand_seeds"]:
        rand_runs.append(random_search(dataset, budget, device, n_eval, s, shared_cache))

    evo_curve = best_so_far(evo_trained_fits)
    rand_curves = [best_so_far(r) for r in rand_runs]
    rand_mean = [sum(c[i] for c in rand_curves) / len(rand_curves) for i in range(n_eval)]
    rand_min = [min(c[i] for c in rand_curves) for i in range(n_eval)]
    rand_max = [max(c[i] for c in rand_curves) for i in range(n_eval)]

    xs = list(range(1, n_eval + 1))
    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    ax.plot(xs, evo_curve, "o-", color="C0", lw=2, label="evolutionary search")
    ax.plot(xs, rand_mean, "s--", color="C1", lw=2, label="random search (mean of 3)")
    ax.fill_between(xs, rand_min, rand_max, color="C1", alpha=0.15, label="random search (range)")
    ax.set_xlabel("# architectures trained"); ax.set_ylabel("best fitness so far")
    ax.set_title("Evolution vs. random search (equal compute)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(EXP, "compare_search.png"), dpi=130)

    result = {
        "n_eval": n_eval,
        "evolution_best": evo_curve[-1],
        "random_best_mean": rand_mean[-1],
        "random_best_min": rand_min[-1],
        "evolution_final_curve": evo_curve,
        "random_mean_curve": rand_mean,
        "advantage": rand_mean[-1] - evo_curve[-1],
        "champion": summary["global_best"],
    }
    json.dump(result, open(os.path.join(EXP, "compare_search.json"), "w"), indent=2)
    print(f"  evolution best {evo_curve[-1]:.4f}  vs  random mean {rand_mean[-1]:.4f} "
          f"(min {rand_min[-1]:.4f}) -> advantage {result['advantage']:+.4f}")
    return summary, budget


# --------------------------------------------------------------------------- #
def exp_proxy_fidelity(dataset, device, shared_cache):
    banner("EXPERIMENT 4: Proxy fidelity — does short training rank like long?")
    rng = random.Random(7)
    archs, seen = [], set()
    while len(archs) < K["proxy_n"]:
        g = random_genome(rng); k = _genome_key(g)
        if k not in seen:
            seen.add(k); archs.append(g)

    short = TrainBudget(max_steps=K["p_short"], batch_size=32, eval_iters=20, eval_every=K["p_short"])
    long = TrainBudget(max_steps=K["p_long"], batch_size=32, eval_iters=25, eval_every=max(K["p_long"] // 2, 1))
    pairs = []
    for i, g in enumerate(archs):
        cfg = genome_to_config(g, dataset.vocab_size)
        rs = train_candidate(cfg, dataset, short, device)
        rl = train_candidate(cfg, dataset, long, device)
        pairs.append((rs["val_loss"], rl["val_loss"]))
        print(f"    [{i+1:2d}/{K['proxy_n']}] proxy {rs['val_loss']:.4f}  ground-truth {rl['val_loss']:.4f}")

    proxy = [p[0] for p in pairs]
    truth = [p[1] for p in pairs]
    rho = spearman(proxy, truth)
    r = pearson(proxy, truth)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(proxy, truth, s=70, color="C2", edgecolor="k")
    ax.set_xlabel("proxy val loss (200 steps)"); ax.set_ylabel("ground-truth val loss (700 steps)")
    ax.set_title(f"Proxy fidelity:  Spearman ρ = {rho:.3f}")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(EXP, "proxy_fidelity.png"), dpi=130)

    res = {"spearman": rho, "pearson": r, "n": len(pairs),
           "proxy_steps": 200, "truth_steps": 700, "pairs": pairs}
    json.dump(res, open(os.path.join(EXP, "proxy_fidelity.json"), "w"), indent=2)
    print(f"  Spearman ρ = {rho:.3f}  Pearson r = {r:.3f}  (n={len(pairs)})")
    return res


# --------------------------------------------------------------------------- #
def exp_inheritance_ab(dataset, device):
    banner("EXPERIMENT 5: Weight inheritance (Lamarckian) A/B")
    budget = TrainBudget(max_steps=K["inh_steps"], batch_size=32, lr=3e-3, eval_iters=15,
                         eval_every=K["inh_steps"])
    out = {}
    for mode, inherit in (("scratch", False), ("inherit", True)):
        ec = EvolveConfig(pop_size=K["inh_pop"], generations=K["inh_gens"], elite=2,
                          tournament_k=3, seed=2024, seed_mode="minimal", inherit=inherit)
        d = os.path.join(HERE, "runs", f"inherit_{mode}")
        t0 = time.time()
        summ = run_evolution(dataset, budget, device, ec, d, log=lambda *a: None)
        rows = [json.loads(l) for l in open(os.path.join(d, "leaderboard.jsonl"))]
        per_gen = []
        for gnum in sorted({r["gen"] for r in rows}):
            vs = [r["fitness"] for r in rows if r["gen"] == gnum and r["fitness"] != float("inf")]
            per_gen.append(min(vs))
        out[mode] = {"per_gen_best": per_gen,
                     "final_best": summ["global_best"]["fitness"],
                     "final_val": summ["global_best"]["val_loss"],
                     "wall_s": round(time.time() - t0, 1)}
        print(f"    {mode:8s}: final fit {out[mode]['final_best']:.4f}  "
              f"val {out[mode]['final_val']:.4f}  ({out[mode]['wall_s']}s)")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for mode, style in (("scratch", "s--"), ("inherit", "o-")):
        ax.plot(range(len(out[mode]["per_gen_best"])), out[mode]["per_gen_best"],
                style, lw=2, label=f"{mode} (final val {out[mode]['final_val']:.3f})")
    ax.set_xlabel("generation"); ax.set_ylabel("best fitness")
    ax.set_title("Lamarckian weight inheritance vs. from-scratch")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(EXP, "inheritance_ab.png"), dpi=130)
    json.dump(out, open(os.path.join(EXP, "inheritance_ab.json"), "w"), indent=2)
    return out


# --------------------------------------------------------------------------- #
def train_champion(dataset, device, summary):
    banner("EXPERIMENT 2: Train champion (extended budget) + sample")
    best = summary["global_best"]
    cfg = genome_to_config(best["genome"], dataset.vocab_size)
    fb = TrainBudget(max_steps=K["champ_steps"], batch_size=48, lr=3e-3, eval_iters=25,
                     eval_every=max(K["champ_steps"] // 5, 1))
    res, model = train_candidate(cfg, dataset, fb, device, log_fn=print, return_model=True)
    prompt = dataset.encode("\n").unsqueeze(0).to(device)
    out = model.generate(prompt, max_new_tokens=400, temperature=0.8, top_k=40, top_p=0.95)
    sample = dataset.decode(out[0].tolist())
    print("\n---- SAMPLE ----\n" + sample + "\n----------------")
    json.dump({"genome": best["genome"], "final_val_loss": res["val_loss"],
               "final_val_ppl": res["val_ppl"], "final_bpc": res["bits_per_char"],
               "n_params": res["n_params"], "sample": sample},
              open(os.path.join(RESULTS, "champion.json"), "w"), indent=2)
    torch.save({"config": cfg.to_dict(), "state_dict": model.state_dict()},
               os.path.join(RESULTS, "champion.pt"))
    print(f"  champion val {res['val_loss']:.4f} ppl {res['val_ppl']:.2f} "
          f"bpc {res['bits_per_char']:.3f}")
    return res


# --------------------------------------------------------------------------- #
def main():
    device = get_device()
    torch.manual_seed(1337)
    print(f"device = {device}  | search space ~ {_space_size()} architectures", flush=True)
    dataset = CharDataset(load_corpus(os.path.join(HERE, "data")), block_size=256, device=device)
    print(f"corpus {len(dataset.train_data)+len(dataset.val_data):,} chars | vocab {dataset.vocab_size}")

    shared_cache = {}
    t0 = time.time()
    summary, _ = exp_production_and_compare(dataset, device, shared_cache)
    champ = train_champion(dataset, device, summary)
    proxy = exp_proxy_fidelity(dataset, device, shared_cache)
    inh = exp_inheritance_ab(dataset, device)

    banner("DONE")
    print(f"total wall: {(time.time()-t0)/60:.1f} min")
    # consolidated experiment summary
    json.dump({
        "champion_val_loss": champ["val_loss"], "champion_ppl": champ["val_ppl"],
        "champion_bpc": champ["bits_per_char"],
        "proxy_spearman": proxy["spearman"],
        "inherit_scratch_val": inh["scratch"]["final_val"],
        "inherit_val": inh["inherit"]["final_val"],
    }, open(os.path.join(EXP, "summary.json"), "w"), indent=2)


def _space_size():
    n = 0
    for nl in SEARCH_SPACE["n_layer"]:
        for dm in SEARCH_SPACE["d_model"]:
            for nh in SEARCH_SPACE["n_head"]:
                for kv in SEARCH_SPACE["kv_ratio"]:
                    for ml in SEARCH_SPACE["mlp_ratio"]:
                        for bs in SEARCH_SPACE["block_size"]:
                            for dr in SEARCH_SPACE["dropout"]:
                                if _valid({"d_model": dm, "n_head": nh, "kv_ratio": kv}):
                                    n += 1
    return n


if __name__ == "__main__":
    main()
