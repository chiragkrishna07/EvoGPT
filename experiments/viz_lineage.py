"""Visualize the genealogy / family tree of an EvoGPT architecture search.

Reads the JSON-Lines leaderboard ledger, builds a DAG (parent -> child edges),
and lays it out by hand: x = generation, y = index within generation. Nodes are
scatter points colored by fitness (viridis; diverged nodes rendered grey) and
sized by parameter count. The global-best (lowest finite fitness) node is starred.
Also prints the "champion lineage": the best node traced back through its first
parent to a generation-0 ancestor.

matplotlib only (Agg backend); no networkx / graphviz.

Run:
  python experiments/viz_lineage.py
  python experiments/viz_lineage.py --ledger runs/search/leaderboard.jsonl --out results/experiments/lineage.png
"""
from __future__ import annotations

import os
import json
import math
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Project root = parent of the experiments/ directory that holds this file.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DIVERGED_COLOR = "#9e9e9e"  # distinct grey for inf/NaN fitness


def _is_finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def load_ledger(path):
    """Parse JSONL; return a list of node dicts (blank/garbage lines skipped)."""
    nodes = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                nodes.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return nodes


def build_index(nodes):
    """Map id -> node, filling in any missing gen/idx defensively."""
    by_id = {}
    for n in nodes:
        nid = n.get("id")
        if nid is None:
            continue
        by_id[nid] = n
    return by_id


def node_position(n):
    """(x, y) layout: x = generation, y = index within generation."""
    return float(n.get("gen", 0)), float(n.get("idx", 0))


def champion_lineage(best_id, by_id):
    """Trace from the best node back to a gen-0 ancestor via the FIRST parent."""
    chain = []
    seen = set()
    cur = best_id
    while cur is not None and cur in by_id and cur not in seen:
        seen.add(cur)
        node = by_id[cur]
        chain.append(node)
        parents = node.get("parents") or []
        cur = parents[0] if parents else None
    chain.reverse()  # gen-0 ancestor first, champion last
    return chain


def render(nodes, by_id, out_path):
    """Draw the lineage DAG to out_path; return the best (champion) node."""
    # --- fitness range over finite nodes (for colormap normalization) -------- #
    finite_fits = [float(n["fitness"]) for n in nodes
                   if "fitness" in n and _is_finite(n.get("fitness"))]
    if finite_fits:
        fmin, fmax = min(finite_fits), max(finite_fits)
    else:
        fmin, fmax = 0.0, 1.0
    if fmax <= fmin:
        fmax = fmin + 1.0

    # --- parameter range (for node sizing) ----------------------------------- #
    params = [float(n.get("n_params", 0) or 0) for n in nodes]
    pmin, pmax = (min(params), max(params)) if params else (0.0, 1.0)
    if pmax <= pmin:
        pmax = pmin + 1.0

    def size_of(n):
        p = float(n.get("n_params", 0) or 0)
        frac = (p - pmin) / (pmax - pmin)
        return 40.0 + 360.0 * frac  # 40..400 pt^2

    fig, ax = plt.subplots(figsize=(12, 7.5))

    # --- edges: parent -> child (light grey), skip dangling parent ids ------- #
    for n in nodes:
        cx, cy = node_position(n)
        for pid in (n.get("parents") or []):
            parent = by_id.get(pid)
            if parent is None:
                continue  # referenced parent not in ledger -> skip edge
            px, py = node_position(parent)
            ax.plot([px, cx], [py, cy], color="#cfcfcf", lw=0.8,
                    zorder=1, solid_capstyle="round")

    # --- nodes --------------------------------------------------------------- #
    fin_x, fin_y, fin_c, fin_s = [], [], [], []
    div_x, div_y, div_s = [], [], []
    for n in nodes:
        x, y = node_position(n)
        if _is_finite(n.get("fitness")):
            fin_x.append(x)
            fin_y.append(y)
            fin_c.append(float(n["fitness"]))
            fin_s.append(size_of(n))
        else:
            div_x.append(x)
            div_y.append(y)
            div_s.append(size_of(n))

    sc = None
    if fin_x:
        sc = ax.scatter(fin_x, fin_y, c=fin_c, s=fin_s, cmap="viridis",
                        vmin=fmin, vmax=fmax, edgecolors="white", linewidths=0.6,
                        zorder=3)
    if div_x:
        ax.scatter(div_x, div_y, c=DIVERGED_COLOR, s=div_s,
                   edgecolors="white", linewidths=0.6, zorder=3,
                   label="diverged (inf/NaN)")

    # --- global best (lowest finite fitness): outline + star ----------------- #
    best = None
    if finite_fits:
        best = min((n for n in nodes if _is_finite(n.get("fitness"))),
                   key=lambda n: float(n["fitness"]))
        bx, by = node_position(best)
        ax.scatter([bx], [by], s=size_of(best) + 140, facecolors="none",
                   edgecolors="crimson", linewidths=2.0, zorder=4)
        ax.scatter([bx], [by], marker="*", s=240, color="crimson",
                   edgecolors="white", linewidths=0.6, zorder=5)

    # --- colorbar ------------------------------------------------------------ #
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("fitness (lower is better)")

    # --- axes / labels ------------------------------------------------------- #
    gens = sorted({int(n.get("gen", 0)) for n in nodes})
    if gens:
        ax.set_xticks(gens)
        ax.set_xticklabels([f"gen {g}" for g in gens])
        ax.set_xlim(min(gens) - 0.5, max(gens) + 0.5)
    ax.set_xlabel("generation")
    ax.set_ylabel("index within generation")
    ax.set_title("EvoGPT architecture-search lineage\n"
                 "(node color = fitness, size = #params, star = global best)")
    ax.grid(True, axis="x", color="#eeeeee", zorder=0)

    # --- legend (size + diverged + best) ------------------------------------- #
    handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="crimson",
               markeredgecolor="white", markersize=15, label="global best"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#440154",
               markeredgecolor="white", markersize=6, label="small model"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#440154",
               markeredgecolor="white", markersize=13, label="large model"),
    ]
    if div_x:
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=DIVERGED_COLOR,
                              markeredgecolor="white", markersize=9,
                              label="diverged (inf/NaN)"))
    ax.legend(handles=handles, loc="upper left", framealpha=0.9, fontsize=9)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return best


def print_lineage(best, by_id):
    """Print the champion's lineage from gen-0 ancestor down to the champion."""
    print("\n" + "=" * 64)
    print("CHAMPION LINEAGE  (gen-0 ancestor -> champion, via first parent)")
    print("=" * 64)
    if best is None:
        print("No finite-fitness node found; cannot trace a lineage.")
        return
    chain = champion_lineage(best.get("id"), by_id)
    for depth, node in enumerate(chain):
        g = node.get("genome", {}) or {}
        nid = node.get("id", "?")
        vl = node.get("val_loss")
        vl_s = f"{vl:.4f}" if _is_finite(vl) else str(vl)
        arrow = "   " * depth + ("|- " if depth else "")
        print(f"{arrow}id={nid:<6} val_loss={vl_s:<10} "
              f"n_layer={g.get('n_layer')}  d_model={g.get('d_model')}  "
              f"n_head={g.get('n_head')}")
    champ = chain[-1] if chain else best
    cf = champ.get("fitness")
    cf_s = f"{cf:.4f}" if _is_finite(cf) else str(cf)
    print(f"\nChampion: id={champ.get('id')}  fitness={cf_s}  "
          f"val_ppl={champ.get('val_ppl')}  n_params={champ.get('n_params')}")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default="runs/search/leaderboard.jsonl",
                    help="path to the JSONL leaderboard (relative to project root)")
    ap.add_argument("--out", default="results/experiments/lineage.png",
                    help="output PNG path (relative to project root)")
    args = ap.parse_args()

    ledger = args.ledger if os.path.isabs(args.ledger) else os.path.join(ROOT, args.ledger)
    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)

    if not os.path.exists(ledger):
        raise SystemExit(f"Ledger not found: {ledger}\n"
                         "(The search may still be running; pass --ledger.)")

    nodes = load_ledger(ledger)
    if not nodes:
        raise SystemExit(f"No valid records found in ledger: {ledger}")

    by_id = build_index(nodes)
    best = render(nodes, by_id, out)
    print(f"Wrote lineage figure: {out}  ({len(nodes)} nodes)")
    print_lineage(best, by_id)


if __name__ == "__main__":
    main()
