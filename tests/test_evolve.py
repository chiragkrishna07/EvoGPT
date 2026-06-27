"""Tests for evogpt.evolve: genome validity, mutation/crossover, config, fitness."""
from __future__ import annotations

import math
import random

from evogpt.evolve import (
    SEARCH_SPACE,
    _valid,
    crossover,
    fitness,
    genome_to_config,
    minimal_genome,
    mutate,
    random_genome,
)


# --------------------------------------------------------------------------- #
# Genome validity
# --------------------------------------------------------------------------- #
def test_random_genome_is_valid():
    rng = random.Random(0)
    for _ in range(200):
        g = random_genome(rng)
        assert _valid(g)
        # keys cover the whole search space
        assert set(g) == set(SEARCH_SPACE)


def test_minimal_genome_is_valid():
    g = minimal_genome()
    assert _valid(g)


def test_mutate_always_valid_many_seeds():
    for seed in range(100):
        rng = random.Random(seed)
        g = random_genome(rng)
        child = mutate(g, rng)
        assert _valid(child)


def test_crossover_always_valid_many_seeds():
    for seed in range(100):
        rng = random.Random(seed)
        a = random_genome(rng)
        b = random_genome(rng)
        child = crossover(a, b, rng)
        assert _valid(child)
        # every gene comes from one of the parents
        for k in SEARCH_SPACE:
            assert child[k] in (a[k], b[k])


# --------------------------------------------------------------------------- #
# genome_to_config
# --------------------------------------------------------------------------- #
def test_genome_to_config_sets_kv_head():
    g = minimal_genome()  # n_head=2, kv_ratio=2 -> n_kv_head=1
    cfg = genome_to_config(g, vocab_size=65)
    assert cfg.n_kv_head == g["n_head"] // g["kv_ratio"]
    assert cfg.n_head == g["n_head"]
    assert cfg.d_model == g["d_model"]
    assert cfg.block_size == g["block_size"]
    assert cfg.mlp_ratio == g["mlp_ratio"]
    assert cfg.dropout == g["dropout"]
    assert cfg.vocab_size == 65


def test_genome_to_config_kv_ratio_one_is_full_mha():
    g = dict(minimal_genome())
    g.update(n_head=4, kv_ratio=1, d_model=64)
    cfg = genome_to_config(g, vocab_size=65)
    assert cfg.n_kv_head == cfg.n_head == 4


def test_genome_to_config_random_consistency():
    rng = random.Random(7)
    for _ in range(50):
        g = random_genome(rng)
        cfg = genome_to_config(g, vocab_size=65)
        assert cfg.n_kv_head == g["n_head"] // g["kv_ratio"]
        # config constraints hold
        assert cfg.d_model % cfg.n_head == 0
        assert cfg.n_head % cfg.n_kv_head == 0


# --------------------------------------------------------------------------- #
# Fitness
# --------------------------------------------------------------------------- #
def _result(val_loss, n_params=1_000_000, diverged=False):
    return {"val_loss": val_loss, "n_params": n_params, "diverged": diverged}


def test_fitness_inf_for_diverged():
    assert fitness(_result(2.0, diverged=True)) == float("inf")


def test_fitness_inf_for_inf_val_loss():
    assert fitness(_result(float("inf"))) == float("inf")


def test_fitness_orders_lower_val_loss_lower():
    better = fitness(_result(1.0, n_params=1_000_000))
    worse = fitness(_result(2.0, n_params=1_000_000))
    assert better < worse


def test_param_penalty_increases_fitness_with_size():
    small = fitness(_result(1.0, n_params=500_000), param_penalty=0.05)
    large = fitness(_result(1.0, n_params=5_000_000), param_penalty=0.05)
    assert large > small


def test_param_penalty_zero_ignores_size():
    a = fitness(_result(1.0, n_params=500_000), param_penalty=0.0)
    b = fitness(_result(1.0, n_params=5_000_000), param_penalty=0.0)
    assert a == b == 1.0


def test_fitness_value_formula():
    # val_loss + penalty * (n_params / 1e6)
    f = fitness(_result(1.5, n_params=2_000_000), param_penalty=0.1)
    assert math.isclose(f, 1.5 + 0.1 * 2.0)
