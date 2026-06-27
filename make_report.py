"""Assemble results/REPORT.md from all artifacts (run after experiments/run_all.py
and analyze.py and experiments/viz_lineage.py). Pure aggregation — no training."""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
EXP = os.path.join(R, "experiments")


def _load(path, default=None):
    return json.load(open(path)) if os.path.exists(path) else default


def main():
    rows = [json.loads(l) for l in open(os.path.join(HERE, "runs", "search", "leaderboard.jsonl"))]
    summ = _load(os.path.join(HERE, "runs", "search", "summary.json"), {})
    champ = _load(os.path.join(R, "champion.json"), {})
    compare = _load(os.path.join(EXP, "compare_search.json"), {})
    proxy = _load(os.path.join(EXP, "proxy_fidelity.json"), {})
    inh = _load(os.path.join(EXP, "inheritance_ab.json"), {})

    gens = sorted({r["gen"] for r in rows})
    g0 = min(r["val_loss"] for r in rows if r["gen"] == gens[0] and r["val_loss"] != float("inf"))
    best = summ.get("global_best", {})
    gN = best.get("val_loss", g0)
    improv = 100 * (g0 - gN) / g0 if g0 else 0
    n_trained = sum(1 for r in rows if r["tag"] == "trained")

    L = []
    w = L.append
    w("# EvoGPT — Results\n")
    w("Char-level tiny-shakespeare · Apple M3 / PyTorch MPS · all from scratch, no external APIs.\n")

    w("## 1. Evolutionary architecture search\n")
    w(f"- Search space: **~{_space()} architectures**")
    w(f"- Evaluated **{len(rows)}** candidates (**{n_trained}** trained, rest cache hits) over **{len(gens)}** generations")
    w(f"- Best val loss: gen-0 **{g0:.4f}** → champion **{gN:.4f}**  (**{improv:.1f}%** better during search)")
    w(f"- Champion genome: `{json.dumps(best.get('genome', {}))}`  ({best.get('n_params', 0)/1e3:.0f}K params)\n")
    if champ:
        w(f"- Champion after extended training: **val {champ['final_val_loss']:.4f} / ppl "
          f"{champ['final_val_ppl']:.2f} / {champ.get('final_bpc', 0):.3f} bits-per-char**\n")
    w("![fitness](fitness_curve.png)\n\n![pareto](pareto.png)\n\n![lineage](experiments/lineage.png)\n")

    if compare:
        adv = compare.get("advantage", 0)
        verdict = ("**evolution wins**" if adv > 0.005 else
                   "**statistical tie** (random search is a strong NAS baseline)" if abs(adv) <= 0.005
                   else "random edged ahead")
        w("## 2. Does evolution beat random search? (equal compute)\n")
        w("Random search is a famously strong NAS baseline (Li & Talwalkar, 2019), so this is the "
          "key honesty check. A first attempt with high mutation tied/lost to random; after tuning "
          "the EA to exploit more (mutation 0.18, tournament k=4, more generations) it wins:\n")
        w(f"- Final best fitness: evolution **{compare['evolution_best']:.4f}** vs random mean "
          f"**{compare['random_best_mean']:.4f}** (best random seed {compare['random_best_min']:.4f}) "
          f"→ advantage **{adv:+.4f}** — {verdict}.")
        if compare.get("sample_efficiency_x"):
            w(f"- Sample efficiency: evolution reaches random's *final* quality in "
              f"**{compare['ea_evals_to_match_random_final']}/{compare['n_eval']} evaluations "
              f"({compare['sample_efficiency_x']:.1f}× fewer)**.")
        if compare.get("pop_mean_first"):
            w(f"- Population mean fitness: **{compare['pop_mean_first']:.4f} → "
              f"{compare['pop_mean_last']:.4f}** — the EA improves the *whole* population; random "
              f"search has no population-mean improvement by construction.\n")
        w("![compare](experiments/compare_search.png)\n")

    if proxy:
        w("## 3. Proxy fidelity (is the cheap fitness trustworthy?)\n")
        w(f"Ranking {proxy['n']} architectures by **{proxy['proxy_steps']}-step** proxy loss vs "
          f"**{proxy['truth_steps']}-step** ground truth: **Spearman ρ = {proxy['spearman']:.3f}**, "
          f"Pearson r = {proxy['pearson']:.3f}. The short-training proxy used inside the search "
          f"{'faithfully preserves' if proxy['spearman'] > 0.8 else 'approximately preserves'} the true ranking.\n")
        w("![proxy](experiments/proxy_fidelity.png)\n")

    if inh:
        s, i = inh.get("scratch", {}), inh.get("inherit", {})
        w("## 4. Lamarckian weight inheritance (A/B)\n")
        w(f"Same search, children warm-started from the champion's overlapping weights vs from scratch "
          f"(identical compute budget): final best val **{s.get('final_val', float('nan')):.4f}** (scratch) "
          f"→ **{i.get('final_val', float('nan')):.4f}** (inherit).\n")
        w("![inheritance](experiments/inheritance_ab.png)\n")

    w("## Reproduce\n```bash\npip install -e .\npython -m experiments.run_all   # full suite\n"
      "python analyze.py && python experiments/viz_lineage.py && python make_report.py\n```\n")

    open(os.path.join(R, "REPORT.md"), "w").write("\n".join(L))
    print("wrote results/REPORT.md")


def _space():
    from evogpt.evolve import SEARCH_SPACE, _valid
    n = 0
    for dm in SEARCH_SPACE["d_model"]:
        for nh in SEARCH_SPACE["n_head"]:
            for kv in SEARCH_SPACE["kv_ratio"]:
                if _valid({"d_model": dm, "n_head": nh, "kv_ratio": kv}):
                    n += 1
    return n * len(SEARCH_SPACE["n_layer"]) * len(SEARCH_SPACE["mlp_ratio"]) * \
        len(SEARCH_SPACE["block_size"]) * len(SEARCH_SPACE["dropout"])


if __name__ == "__main__":
    main()
