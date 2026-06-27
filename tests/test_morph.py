"""Tests for evogpt.morph.inherit_weights (network-morphism weight inheritance)."""
from __future__ import annotations

import torch
import torch.nn as nn

from evogpt.model import EvoGPT

from conftest import DEVICE, tiny_config


def test_identical_models_inherit_fraction_one_and_equal():
    cfg = tiny_config()
    parent = EvoGPT(cfg).to(DEVICE)
    child = EvoGPT(cfg).to(DEVICE)
    # Confirm they start different (random init differs without same seed path).
    parent_state = {k: v.clone() for k, v in parent.state_dict().items()}
    frac = __import__("evogpt.morph", fromlist=["inherit_weights"]).inherit_weights(
        child, parent_state)
    assert frac == 1.0
    for k, v in child.state_dict().items():
        assert torch.equal(v, parent_state[k]), k


def test_larger_parent_into_smaller_child_copies_slices():
    from evogpt.morph import inherit_weights

    child_cfg = tiny_config(d_model=32, n_head=4, n_kv_head=4)
    parent_cfg = tiny_config(d_model=64, n_head=4, n_kv_head=4)
    child = EvoGPT(child_cfg).to(DEVICE)
    parent = EvoGPT(parent_cfg).to(DEVICE)
    parent_state = {k: v.clone() for k, v in parent.state_dict().items()}

    frac = inherit_weights(child, parent_state)
    assert frac > 0.0

    # Spot-check an overlapping leading sub-block was copied exactly.
    cs = child.state_dict()
    name = "tok_emb.weight"  # (vocab, d_model): vocab same, d_model 32 vs 64
    c = cs[name]
    p = parent_state[name]
    sl = tuple(slice(0, min(a, b)) for a, b in zip(c.shape, p.shape))
    assert torch.equal(c[sl], p[sl].to(c.dtype))


def test_smaller_parent_into_larger_child_partial_and_preserves_rest():
    from evogpt.morph import inherit_weights

    child_cfg = tiny_config(d_model=64)
    parent_cfg = tiny_config(d_model=32)
    child = EvoGPT(child_cfg).to(DEVICE)
    parent = EvoGPT(parent_cfg).to(DEVICE)
    parent_state = {k: v.clone() for k, v in parent.state_dict().items()}

    before = {k: v.clone() for k, v in child.state_dict().items()}
    frac = inherit_weights(child, parent_state)
    assert 0.0 < frac < 1.0

    # Overlapping slice of tok_emb now matches the parent...
    cs = child.state_dict()
    c, p = cs["tok_emb.weight"], parent_state["tok_emb.weight"]
    sl = tuple(slice(0, min(a, b)) for a, b in zip(c.shape, p.shape))
    assert torch.equal(c[sl], p[sl].to(c.dtype))
    # ...but the non-overlapping tail kept the child's fresh init.
    tail = c[:, p.shape[1]:]
    assert torch.equal(tail, before["tok_emb.weight"][:, p.shape[1]:])


def test_mismatched_ndim_params_skipped():
    from evogpt.morph import inherit_weights

    class Child(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(4, 4))

    child = Child()
    # Same name, but parent param has a different number of dims -> must be skipped.
    parent_state = {"w": torch.ones(4)}
    before = child.w.detach().clone()
    frac = inherit_weights(child, parent_state)
    assert frac == 0.0
    assert torch.equal(child.w.detach(), before)


def test_missing_name_skipped_safely():
    from evogpt.morph import inherit_weights

    class Child(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(3, 3))

    child = Child()
    before = child.w.detach().clone()
    frac = inherit_weights(child, {"other": torch.ones(3, 3)})
    assert frac == 0.0
    assert torch.equal(child.w.detach(), before)
