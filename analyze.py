"""Read the search ledger and produce result plots + a markdown report.
Run after run_search.py completes:  python analyze.py
"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "runs", "search")
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


def load_ledger():
    rows = []
    with open(os.path.join(RUN, "leaderboard.jsonl")) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    rows = load_ledger()
    summary = json.load(open(os.path.join(RUN, "summary.json")))
    gens = sorted({r["gen"] for r in rows})

    # ---- Plot 1: best & mean fitness per generation (the improvement curve) ----
    best_per_gen, mean_per_gen = [], []
    for g in gens:
        fits = [r["fitness"] for r in rows if r["gen"] == g and r["fitness"] != float("inf")]
        best_per_gen.append(min(fits))
        mean_per_gen.append(sum(fits) / len(fits))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(gens, best_per_gen, "o-", label="best fitness", linewidth=2)
    ax.plot(gens, mean_per_gen, "s--", label="population mean", alpha=0.7)
    ax.set_xlabel("generation"); ax.set_ylabel("fitness (val loss + size penalty)")
    ax.set_title("EvoGPT — architecture search improvement")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fitness_curve.png"), dpi=130)

    # ---- Plot 2: Pareto view — val loss vs params, colored by generation ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    valid = [r for r in rows if r["val_loss"] != float("inf")]
    sc = ax.scatter([r["n_params"] / 1e6 for r in valid],
                    [r["val_loss"] for r in valid],
                    c=[r["gen"] for r in valid], cmap="viridis", s=60, edgecolor="k", linewidth=0.4)
    best = summary["global_best"]
    ax.scatter([best["n_params"] / 1e6], [best["val_loss"]], marker="*",
               s=400, color="red", edgecolor="k", label="champion", zorder=5)
    ax.set_xlabel("parameters (millions)"); ax.set_ylabel("validation loss")
    ax.set_title("Quality vs. size (each point = one trained architecture)")
    plt.colorbar(sc, label="generation"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "pareto.png"), dpi=130)

    # ---- Markdown report ----
    n_trained = sum(1 for r in rows if r["tag"] == "trained")
    g0_best = min(r["val_loss"] for r in rows if r["gen"] == gens[0] and r["val_loss"] != float("inf"))
    gN_best = best["val_loss"]
    improv = 100 * (g0_best - gN_best) / g0_best
    champ = json.load(open(os.path.join(RESULTS, "champion.json"))) \
        if os.path.exists(os.path.join(RESULTS, "champion.json")) else None

    lines = [
        "# EvoGPT — Results",
        "",
        f"- Architectures evaluated: **{len(rows)}** ({n_trained} trained, rest cache hits)",
        f"- Generations: **{len(gens)}**",
        f"- Gen-0 best val loss: **{g0_best:.4f}**  →  champion val loss: **{gN_best:.4f}**",
        f"- Search-time improvement: **{improv:.1f}%**",
        "",
        "## Champion architecture",
        "```json",
        json.dumps(best["genome"], indent=2),
        "```",
        f"- Parameters: **{best['n_params']/1e3:.0f}K**",
        f"- Search val loss / ppl: **{best['val_loss']:.4f} / {best['val_ppl']:.2f}**",
    ]
    if champ:
        lines += [
            f"- Champion after extended training: val loss **{champ['final_val_loss']:.4f}** "
            f"(ppl **{champ['final_val_ppl']:.2f}**)",
            "",
            "## Sample (champion, extended training)",
            "```",
            champ["sample"].strip(),
            "```",
        ]
    lines += ["", "## Figures", "![fitness](fitness_curve.png)", "", "![pareto](pareto.png)"]
    with open(os.path.join(RESULTS, "REPORT.md"), "w") as f:
        f.write("\n".join(lines))

    print(f"gen-0 best {g0_best:.4f} -> champion {gN_best:.4f}  ({improv:.1f}% better)")
    print("wrote results/fitness_curve.png, results/pareto.png, results/REPORT.md")


if __name__ == "__main__":
    main()
