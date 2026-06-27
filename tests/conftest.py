"""Shared fixtures and helpers for the EvoGPT test suite.

Every test runs on CPU with tiny models so the whole suite finishes in seconds
and never contends with a parallel GPU (MPS) training job.
"""
from __future__ import annotations

import os
import random

import pytest
import torch

from evogpt.model import EvoGPT, GPTConfig

# Tiny ceilings enforced by the suite: d_model<=64, n_layer<=2, n_head<=4,
# vocab<=65, block_size<=32.
DEVICE = "cpu"
SEED = 1337


@pytest.fixture(autouse=True)
def _deterministic():
    """Make every test deterministic and pinned to CPU."""
    torch.manual_seed(SEED)
    random.seed(SEED)
    # Belt-and-suspenders: hide CUDA so nothing wanders onto a GPU.
    prev = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    yield
    if prev is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = prev


def tiny_config(**overrides) -> GPTConfig:
    """A small, valid config respecting the suite's size ceilings."""
    base = dict(
        vocab_size=65,
        block_size=32,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        d_model=64,
        mlp_ratio=2.0,
        dropout=0.0,
    )
    base.update(overrides)
    return GPTConfig(**base)


def tiny_model(**overrides) -> EvoGPT:
    """Build a tiny EvoGPT on CPU in eval-or-train default state."""
    return EvoGPT(tiny_config(**overrides)).to(DEVICE)


@pytest.fixture
def cfg() -> GPTConfig:
    return tiny_config()


@pytest.fixture
def model(cfg) -> EvoGPT:
    return EvoGPT(cfg).to(DEVICE)


# A small but structured text so a CharDataset has a stable, >1 vocab and
# enough length to slice batches from.
SAMPLE_TEXT = ("abcdefghijklmnopqrstuvwxyz\n" * 40) + ("hello world! 0123456789. " * 40)


@pytest.fixture
def sample_text() -> str:
    return SAMPLE_TEXT
