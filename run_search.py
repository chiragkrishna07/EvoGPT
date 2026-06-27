"""Entry point: run the evolutionary architecture search on tiny-shakespeare,
then train the discovered best architecture longer and sample from it.

Usage:
  python run_search.py --quick     # tiny budget, smoke-test the full loop
  python run_search.py             # full search
"""
import os
import json
import argparse
import torch

from evogpt.data import CharDataset, load_corpus
from evogpt.train import TrainBudget, train_candidate, get_device
from evogpt.evolve import (EvolveConfig, run_evolution, genome_to_config)

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "search"))
    args = ap.parse_args()

    device = get_device()
    torch.manual_seed(1337)
    print(f"device = {device}")

    text = load_corpus(os.path.join(HERE, "data"))
    # block_size here is just for dataset init; candidates override per-genome.
    dataset = CharDataset(text, block_size=256, device=device)
    print(f"corpus chars = {len(text):,} | vocab = {dataset.vocab_size}")

    if args.quick:
        budget = TrainBudget(max_steps=60, batch_size=16, eval_iters=8, eval_every=30)
        ec = EvolveConfig(pop_size=4, generations=2, elite=1, seed=1337)
    else:
        budget = TrainBudget(max_steps=300, batch_size=32, lr=3e-3,
                             eval_iters=20, eval_every=150)
        ec = EvolveConfig(pop_size=8, generations=5, elite=2,
                          tournament_k=3, param_penalty=0.05, seed=1337,
                          seed_mode="minimal")

    os.makedirs(args.out, exist_ok=True)
    summary = run_evolution(dataset, budget, device, ec, args.out)

    best = summary["global_best"]
    print("\n==================== SEARCH COMPLETE ====================")
    print(f"best genome : {best['genome']}")
    print(f"val loss    : {best['val_loss']:.4f}  (ppl {best['val_ppl']:.2f})")
    print(f"params      : {best['n_params']/1e3:.0f}K")

    # ---- Train the champion longer for a real sample ----
    print("\nTraining champion architecture (extended budget)...")
    cfg = genome_to_config(best["genome"], dataset.vocab_size)
    final_budget = TrainBudget(
        max_steps=120 if args.quick else 1500,
        batch_size=48, lr=3e-3, eval_iters=25, eval_every=100,
    )
    res, model = train_candidate(cfg, dataset, final_budget, device,
                                 log_fn=print, return_model=True)
    print(f"champion final val loss: {res['val_loss']:.4f} (ppl {res['val_ppl']:.2f})")

    # ---- Sample ----
    prompt = dataset.encode("\n").unsqueeze(0).to(device)
    out = model.generate(prompt, max_new_tokens=400, temperature=0.8, top_k=40)
    sample = dataset.decode(out[0].tolist())
    print("\n---------------- SAMPLE ----------------")
    print(sample)
    print("----------------------------------------")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "champion.json"), "w") as f:
        json.dump({"genome": best["genome"], "final_val_loss": res["val_loss"],
                   "final_val_ppl": res["val_ppl"], "n_params": res["n_params"],
                   "sample": sample}, f, indent=2)
    torch.save({"config": cfg.to_dict(), "state_dict": model.state_dict()},
               os.path.join(HERE, "results", "champion.pt"))
    print("\nsaved -> results/champion.json, results/champion.pt")


if __name__ == "__main__":
    main()
