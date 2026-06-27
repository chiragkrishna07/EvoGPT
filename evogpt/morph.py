"""Network-morphism-style weight inheritance for Lamarckian evolution.

When the search breeds a child architecture, instead of training it from random
init we warm-start it from a parent's weights: every parameter that exists in
both nets has its *overlapping* leading sub-block copied (e.g. a parent's
192-wide attention projection seeds the top-left 128x128 of a child's 128-wide
one). Mismatched / new parameters keep the child's fresh init.

This is "Lamarckian" because learned weights are inherited across generations,
not just the architecture genome — turning each candidate's brief training into
a fine-tune rather than a cold start.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def inherit_weights(child: torch.nn.Module, parent_state: dict, verbose: bool = False) -> float:
    """Copy overlapping weight slices from `parent_state` into `child` in place.
    Returns the fraction of child parameters that received inherited values."""
    child_state = child.state_dict()
    copied, total = 0, 0
    for name, cp in child_state.items():
        total += cp.numel()
        pp = parent_state.get(name)
        if pp is None or pp.dim() != cp.dim():
            continue
        # overlapping leading sub-block along every dimension
        sl = tuple(slice(0, min(c, p)) for c, p in zip(cp.shape, pp.shape))
        cp[sl].copy_(pp[sl].to(cp.dtype).to(cp.device))
        copied += cp[sl].numel()
    if verbose:
        print(f"    inherited {copied}/{total} ({100 * copied / max(total, 1):.0f}%) params")
    return copied / max(total, 1)
