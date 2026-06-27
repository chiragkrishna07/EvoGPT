"""Tests for evogpt.model: shapes, loss, causality, KV-cache, GQA, tying, RoPE."""
from __future__ import annotations

import math

import torch

from evogpt.model import (
    EvoGPT,
    GPTConfig,
    apply_rope,
    build_rope_cache,
)

from conftest import DEVICE, tiny_config, tiny_model


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_config_head_dim_and_to_dict():
    cfg = tiny_config(d_model=64, n_head=4)
    assert cfg.head_dim == 16
    d = cfg.to_dict()
    assert d["d_model"] == 64
    assert d["n_head"] == 4
    assert d["vocab_size"] == cfg.vocab_size


# --------------------------------------------------------------------------- #
# Forward / loss
# --------------------------------------------------------------------------- #
def test_forward_output_shapes(model, cfg):
    B, T = 3, 16
    x = torch.randint(0, cfg.vocab_size, (B, T), device=DEVICE)
    logits, loss = model(x)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert loss is None


def test_loss_none_without_targets(model, cfg):
    x = torch.randint(0, cfg.vocab_size, (2, 8), device=DEVICE)
    _, loss = model(x)
    assert loss is None


def test_loss_positive_scalar_with_targets(model, cfg):
    x = torch.randint(0, cfg.vocab_size, (2, 8), device=DEVICE)
    y = torch.randint(0, cfg.vocab_size, (2, 8), device=DEVICE)
    logits, loss = model(x, y)
    assert loss is not None
    assert loss.dim() == 0  # scalar
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


# --------------------------------------------------------------------------- #
# Causality
# --------------------------------------------------------------------------- #
def test_causal_mask_no_future_leakage():
    """Changing the LAST input token must not change logits at earlier positions."""
    model = tiny_model().eval()
    x = torch.randint(0, model.cfg.vocab_size, (1, 16), device=DEVICE)
    with torch.no_grad():
        a, _ = model(x)
        x2 = x.clone()
        x2[0, -1] = (x2[0, -1] + 1) % model.cfg.vocab_size
        b, _ = model(x2)
    assert torch.allclose(a[0, :-1], b[0, :-1], atol=1e-5)
    # And the last position SHOULD change (sanity: the model is not constant).
    assert not torch.allclose(a[0, -1], b[0, -1], atol=1e-5)


# --------------------------------------------------------------------------- #
# KV-cache equivalence
# --------------------------------------------------------------------------- #
def test_kv_cache_greedy_matches_full_forward():
    """Incremental KV-cache decode must equal greedy next-token from a full pass."""
    model = tiny_model(block_size=32).eval()
    torch.manual_seed(0)
    prompt = torch.randint(0, model.cfg.vocab_size, (1, 8), device=DEVICE)
    with torch.no_grad():
        gen = model.generate(prompt.clone(), max_new_tokens=10,
                              temperature=1e-9, top_k=1)
        full_logits, _ = model(gen)
    pred = full_logits[0, 7:-1].argmax(-1)
    assert torch.equal(pred, gen[0, 8:])


# --------------------------------------------------------------------------- #
# GQA param savings
# --------------------------------------------------------------------------- #
def test_gqa_reduces_params_vs_mha():
    base = GPTConfig(vocab_size=65, block_size=32, n_layer=2,
                     n_head=4, n_kv_head=4, d_model=64)   # full MHA
    gqa = GPTConfig(vocab_size=65, block_size=32, n_layer=2,
                    n_head=4, n_kv_head=1, d_model=64)     # grouped-query
    assert EvoGPT(gqa).num_params() < EvoGPT(base).num_params()


# --------------------------------------------------------------------------- #
# Weight tying
# --------------------------------------------------------------------------- #
def test_weight_tying_same_storage(model):
    assert model.lm_head.weight is model.tok_emb.weight
    # Same underlying storage: mutating one mutates the other.
    with torch.no_grad():
        model.tok_emb.weight[0, 0] += 1.0
    assert model.lm_head.weight[0, 0].item() == model.tok_emb.weight[0, 0].item()


def test_num_params_non_embedding_excludes_tied_table(model):
    full = model.num_params(non_embedding=False)
    non_emb = model.num_params(non_embedding=True)
    assert non_emb == full - model.tok_emb.weight.numel()
    assert non_emb < full


# --------------------------------------------------------------------------- #
# Overfit a single repeated batch
# --------------------------------------------------------------------------- #
def test_overfit_single_batch():
    """Loss must drop >50% when training on one fixed batch."""
    model = tiny_model(block_size=16)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, model.cfg.vocab_size, (4, 16), device=DEVICE)
    y = torch.randint(0, model.cfg.vocab_size, (4, 16), device=DEVICE)
    first = None
    last = None
    for _ in range(200):
        _, loss = model(x, y)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = loss.item()
    assert last < first * 0.5, f"{first:.3f} -> {last:.3f}"


# --------------------------------------------------------------------------- #
# generate(): max_new_tokens, top_k, top_p
# --------------------------------------------------------------------------- #
def test_generate_respects_max_new_tokens():
    model = tiny_model(block_size=32).eval()
    prompt = torch.randint(0, model.cfg.vocab_size, (1, 4), device=DEVICE)
    out = model.generate(prompt.clone(), max_new_tokens=6)
    assert out.shape == (1, 4 + 6)
    # original prompt is preserved as a prefix
    assert torch.equal(out[0, :4], prompt[0])


def test_generate_with_top_k():
    model = tiny_model(block_size=32).eval()
    prompt = torch.randint(0, model.cfg.vocab_size, (1, 4), device=DEVICE)
    out = model.generate(prompt.clone(), max_new_tokens=5, top_k=3)
    assert out.shape == (1, 9)
    assert out.min().item() >= 0
    assert out.max().item() < model.cfg.vocab_size


def test_generate_with_top_p():
    model = tiny_model(block_size=32).eval()
    prompt = torch.randint(0, model.cfg.vocab_size, (1, 4), device=DEVICE)
    out = model.generate(prompt.clone(), max_new_tokens=5, top_p=0.9)
    assert out.shape == (1, 9)
    assert out.min().item() >= 0
    assert out.max().item() < model.cfg.vocab_size


def test_generate_stops_at_block_size():
    """The context-window guard caps total length at block_size."""
    model = tiny_model(block_size=32).eval()
    prompt = torch.randint(0, model.cfg.vocab_size, (1, 4), device=DEVICE)
    out = model.generate(prompt.clone(), max_new_tokens=1000, top_k=1)
    assert out.shape[1] <= 4 + model.cfg.block_size


# --------------------------------------------------------------------------- #
# RoPE preserves norms (it is a rotation)
# --------------------------------------------------------------------------- #
def test_apply_rope_preserves_vector_norms():
    torch.manual_seed(0)
    B, H, T, hd = 2, 4, 16, 16
    x = torch.randn(B, H, T, hd)
    cos, sin = build_rope_cache(T, hd, theta=10000.0,
                                device=torch.device("cpu"), dtype=torch.float32)
    rotated = apply_rope(x, cos, sin)
    n_before = x.norm(dim=-1)
    n_after = rotated.norm(dim=-1)
    assert torch.allclose(n_before, n_after, atol=1e-4)


def test_apply_rope_position_zero_is_identity():
    """At position 0 the rotation angle is 0, so RoPE is the identity there."""
    T, hd = 4, 16
    x = torch.randn(1, 1, T, hd)
    cos, sin = build_rope_cache(T, hd, theta=10000.0,
                                device=torch.device("cpu"), dtype=torch.float32)
    rotated = apply_rope(x, cos, sin)
    assert torch.allclose(rotated[:, :, 0, :], x[:, :, 0, :], atol=1e-5)
