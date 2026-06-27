"""Evolutionary Neural Architecture Search (the self-improvement loop).

A controller maintains a population of transformer architectures. Each
generation it:
  1. trains every candidate under a fixed compute budget (train.py),
  2. scores them with a multi-objective fitness (val loss + parameter
     efficiency),
  3. selects the elites, and
  4. breeds the next generation via tournament selection, crossover, and
     mutation of the architecture genome.

Everything is logged to a JSONL leaderboard so the improvement across
generations is auditable and plottable. No external APIs — the controller's
"decisions" are a self-contained evolutionary algorithm.
"""
from __future__ import annotations

import os
import json
import copy
import random
from dataclasses import dataclass

from .model import GPTConfig
from .train import train_candidate, TrainBudget
from .data import CharDataset


# --------------------------------------------------------------------------- #
# Search space (the "genome")
# --------------------------------------------------------------------------- #
SEARCH_SPACE = {
    "n_layer":   [2, 3, 4, 5],
    "d_model":   [48, 64, 96, 128, 160, 192],
    "n_head":    [2, 4, 8],
    "kv_ratio":  [1, 2, 4],          # n_head / n_kv_head
    "mlp_ratio": [2.0, 2.5, 2.667, 3.0, 3.5, 4.0],
    "block_size":[64, 96, 128],
    "dropout":   [0.0, 0.1],
}
# ~7,800 valid architectures — large enough that uniform random search covers
# only a fraction in a budget of ~25 evaluations, giving the guided evolutionary
# search a real edge to demonstrate (see experiments/compare_search.py).


def _valid(genome: dict) -> bool:
    if genome["d_model"] % genome["n_head"] != 0:
        return False
    n_kv = genome["n_head"] // genome["kv_ratio"]
    if n_kv < 1 or genome["n_head"] % n_kv != 0:
        return False
    return True


def random_genome(rng: random.Random) -> dict:
    while True:
        g = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
        if _valid(g):
            return g


def minimal_genome() -> dict:
    """The smallest valid architecture — the 'naive baseline' the search starts
    from in minimal-seed mode, so improvement across generations is real."""
    return {"n_layer": 2, "d_model": 64, "n_head": 2, "kv_ratio": 2,
            "mlp_ratio": 2.0, "block_size": 64, "dropout": 0.0}


def genome_to_config(g: dict, vocab_size: int) -> GPTConfig:
    return GPTConfig(
        vocab_size=vocab_size,
        block_size=g["block_size"],
        n_layer=g["n_layer"],
        n_head=g["n_head"],
        n_kv_head=g["n_head"] // g["kv_ratio"],
        d_model=g["d_model"],
        mlp_ratio=g["mlp_ratio"],
        dropout=g["dropout"],
    )


def mutate(g: dict, rng: random.Random, rate: float = 0.34) -> dict:
    for _ in range(20):
        child = copy.deepcopy(g)
        for k in SEARCH_SPACE:
            if rng.random() < rate:
                child[k] = rng.choice(SEARCH_SPACE[k])
        if _valid(child):
            return child
    return g


def crossover(a: dict, b: dict, rng: random.Random) -> dict:
    for _ in range(20):
        child = {k: (a[k] if rng.random() < 0.5 else b[k]) for k in SEARCH_SPACE}
        if _valid(child):
            return child
    return a


# --------------------------------------------------------------------------- #
# Fitness — multi-objective: language-modeling quality + param efficiency
# --------------------------------------------------------------------------- #
def fitness(result: dict, param_penalty: float = 0.05) -> float:
    """Lower is better. val_loss plus a soft penalty (in nats) for size, so the
    search prefers small models when quality is comparable."""
    if result["diverged"] or result["val_loss"] == float("inf"):
        return float("inf")
    size_term = param_penalty * (result["n_params"] / 1e6)  # nats per 1M params
    return result["val_loss"] + size_term


@dataclass
class EvolveConfig:
    pop_size: int = 8
    generations: int = 5
    elite: int = 2
    tournament_k: int = 3
    param_penalty: float = 0.05
    seed: int = 1337
    seed_mode: str = "random"   # "random" | "minimal" (start small, evolve up)
    inherit: bool = False       # Lamarckian: warm-start children from the champion
    mutation_rate: float = 0.25  # per-gene flip prob; lower = more exploitation


def _genome_key(g: dict) -> str:
    return json.dumps(g, sort_keys=True)


def run_evolution(dataset: CharDataset, budget: TrainBudget, device: str,
                  ec: EvolveConfig, out_dir: str, log=print):
    rng = random.Random(ec.seed)
    os.makedirs(out_dir, exist_ok=True)
    ledger_path = os.path.join(out_dir, "leaderboard.jsonl")
    ledger = open(ledger_path, "w")

    cache: dict[str, dict] = {}           # genome -> result (avoid retraining)
    # population members carry lineage: {"genome": ..., "parents": [id, ...]}
    population = []
    if ec.seed_mode == "minimal":
        # Start from the minimal architecture, lightly diversified, so the search
        # must *discover* better designs (a genuine improvement curve).
        base = minimal_genome()
        population.append({"genome": base, "parents": []})
        while len(population) < ec.pop_size:
            population.append({"genome": mutate(base, rng, rate=0.2), "parents": []})
    else:
        while len(population) < ec.pop_size:
            population.append({"genome": random_genome(rng), "parents": []})

    gen_best = []
    global_best = None
    best_state = None          # champion weights for Lamarckian warm-start
    n_trained = 0

    for gen in range(ec.generations):
        log(f"\n=== Generation {gen} ===")
        scored = []
        for i, member in enumerate(population):
            g, parents = member["genome"], member["parents"]
            cid = f"{gen}.{i}"
            key = _genome_key(g)
            if key in cache:
                res = cache[key]
                tag = "cached"
            else:
                cfg = genome_to_config(g, dataset.vocab_size)
                log(f"  [{cid}] training {g['n_layer']}L d{g['d_model']} "
                    f"h{g['n_head']}/kv{g['n_head'] // g['kv_ratio']} "
                    f"mlp{g['mlp_ratio']} ctx{g['block_size']}")
                # Lamarckian: warm-start new children from the champion's weights.
                warm = best_state if (ec.inherit and gen > 0) else None
                out = train_candidate(cfg, dataset, budget, device,
                                      init_state=warm, return_model=ec.inherit)
                res, model = out if ec.inherit else (out, None)
                cache[key] = res
                tag = "trained"
                n_trained += 1
                if ec.inherit and model is not None:
                    fit_now = fitness(res, ec.param_penalty)
                    if global_best is None or fit_now < global_best["fitness"]:
                        best_state = {k: v.detach().cpu().clone()
                                      for k, v in model.state_dict().items()}
                    del model
            fit = fitness(res, ec.param_penalty)
            record = {
                "id": cid, "gen": gen, "idx": i, "parents": parents, "genome": g,
                "fitness": fit, "val_loss": res["val_loss"], "val_ppl": res["val_ppl"],
                "bits_per_char": res.get("bits_per_char", float("inf")),
                "n_params": res["n_params"], "wall_s": res["wall_s"],
                "inherited_frac": res.get("inherited_frac", 0.0),
                "diverged": res["diverged"], "tag": tag,
            }
            scored.append(record)
            ledger.write(json.dumps(record) + "\n")
            ledger.flush()
            log(f"        -> val {res['val_loss']:.4f} ppl {res['val_ppl']:.2f} "
                f"params {res['n_params']/1e3:.0f}K fit {fit:.4f} ({tag}, {res['wall_s']}s)")

        scored.sort(key=lambda r: r["fitness"])
        best = scored[0]
        gen_best.append(best)
        if global_best is None or best["fitness"] < global_best["fitness"]:
            global_best = best
        log(f"  >> gen {gen} best: val {best['val_loss']:.4f} ppl {best['val_ppl']:.2f} "
            f"params {best['n_params']/1e3:.0f}K  genome={best['genome']}")

        # ---- breed next generation ----
        if gen < ec.generations - 1:
            elites = [{"genome": copy.deepcopy(r["genome"]), "parents": [r["id"]]}
                      for r in scored[:ec.elite]]
            next_pop = elites
            while len(next_pop) < ec.pop_size:
                def tourn():
                    cands = rng.sample(scored, min(ec.tournament_k, len(scored)))
                    return min(cands, key=lambda r: r["fitness"])
                pa, pb = tourn(), tourn()
                child = crossover(pa["genome"], pb["genome"], rng)
                child = mutate(child, rng, rate=ec.mutation_rate)
                next_pop.append({"genome": child, "parents": [pa["id"], pb["id"]]})
            population = next_pop

    ledger.close()

    summary = {
        "global_best": global_best,
        "gen_best": gen_best,
        "n_evaluated": len(cache),
        "n_trained": n_trained,
        "search_space": SEARCH_SPACE,
        "evolve_config": ec.__dict__,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
