"""Training / evaluation harness for a single candidate architecture.

`train_candidate` is the fitness function the evolutionary search calls: it
trains a model under a fixed compute budget and returns validation loss plus
efficiency metadata. Kept deliberately fast (a few hundred steps) so a whole
population can be evaluated in minutes.
"""
from __future__ import annotations

import time
import math
import torch

from .model import EvoGPT, GPTConfig
from .data import CharDataset
from .morph import inherit_weights

LN2 = math.log(2.0)


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def estimate_loss(model, dataset, batch_size, eval_iters, device, block_size):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = dataset.get_batch(split, batch_size, block_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


class TrainBudget:
    """Compute budget per candidate — small so search stays fast."""
    def __init__(self, max_steps=400, batch_size=32, lr=3e-3,
                 warmup=40, eval_iters=20, eval_every=100, grad_clip=1.0):
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.lr = lr
        self.warmup = warmup
        self.eval_iters = eval_iters
        self.eval_every = eval_every
        self.grad_clip = grad_clip


def _lr_at(step, budget):
    if step < budget.warmup:
        return budget.lr * (step + 1) / budget.warmup
    progress = (step - budget.warmup) / max(1, budget.max_steps - budget.warmup)
    return 0.1 * budget.lr + 0.5 * (budget.lr - 0.1 * budget.lr) * (1 + math.cos(math.pi * progress))


def train_candidate(cfg: GPTConfig, dataset: CharDataset, budget: TrainBudget,
                    device: str, log_fn=None, return_model: bool = False,
                    seed: int = 1337, init_state: dict | None = None):
    """Train one architecture and return a metrics dict (the fitness signal).

    `seed` fixes init + batch order so the architecture is the only variable
    (fair comparison across candidates). `init_state` warm-starts the model via
    weight inheritance (Lamarckian evolution)."""
    torch.manual_seed(seed)
    model = EvoGPT(cfg).to(device)
    inherited_frac = 0.0
    if init_state is not None:
        inherited_frac = inherit_weights(model, init_state)
    n_params = model.num_params()
    opt = torch.optim.AdamW(model.parameters(), lr=budget.lr,
                            betas=(0.9, 0.95), weight_decay=0.1)

    t0 = time.time()
    history = []
    best_val = float("inf")
    diverged = False
    for step in range(budget.max_steps):
        lr = _lr_at(step, budget)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = dataset.get_batch("train", budget.batch_size, cfg.block_size)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            diverged = True
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), budget.grad_clip)
        opt.step()

        if step % budget.eval_every == 0 or step == budget.max_steps - 1:
            m = estimate_loss(model, dataset, budget.batch_size, budget.eval_iters, device, cfg.block_size)
            best_val = min(best_val, m["val"])
            history.append({"step": step, "train": m["train"], "val": m["val"]})
            if log_fn:
                log_fn(f"    step {step:4d} | train {m['train']:.4f} | val {m['val']:.4f} | lr {lr:.1e}")

    wall = time.time() - t0
    ok = not diverged and best_val < 20
    result = {
        "val_loss": best_val if not diverged else float("inf"),
        "val_ppl": math.exp(best_val) if ok else float("inf"),
        "bits_per_char": best_val / LN2 if ok else float("inf"),
        "n_params": n_params,
        "wall_s": round(wall, 1),
        "diverged": diverged,
        "inherited_frac": round(inherited_frac, 3),
        "history": history,
        "config": cfg.to_dict(),
    }
    if return_model:
        return result, model
    return result
