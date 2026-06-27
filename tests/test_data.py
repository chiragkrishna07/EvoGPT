"""Tests for evogpt.data.CharDataset and load_corpus."""
from __future__ import annotations

import os

import pytest
import torch

from evogpt.data import CharDataset, load_corpus

from conftest import DEVICE


def test_encode_decode_round_trip(sample_text):
    ds = CharDataset(sample_text, block_size=16, device=DEVICE)
    s = "hello world"
    enc = ds.encode(s)
    assert isinstance(enc, torch.Tensor)
    assert enc.dtype == torch.long
    assert ds.decode(enc) == s


def test_vocab_size_matches_unique_chars(sample_text):
    ds = CharDataset(sample_text, block_size=16, device=DEVICE)
    assert ds.vocab_size == len(set(sample_text))
    assert ds.vocab_size == len(ds.stoi) == len(ds.itos)


def test_stoi_itos_are_inverses(sample_text):
    ds = CharDataset(sample_text, block_size=16, device=DEVICE)
    for ch, i in ds.stoi.items():
        assert ds.itos[i] == ch


def test_get_batch_shapes_and_offset(sample_text):
    block = 16
    ds = CharDataset(sample_text, block_size=block, device=DEVICE)
    B = 8
    x, y = ds.get_batch("train", batch_size=B)
    assert x.shape == (B, block)
    assert y.shape == (B, block)
    # y is x shifted by one: y[:, :-1] == x[:, 1:]
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_get_batch_respects_explicit_block_size(sample_text):
    ds = CharDataset(sample_text, block_size=16, device=DEVICE)
    x, y = ds.get_batch("train", batch_size=4, block_size=8)
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)


def test_get_batch_val_split(sample_text):
    ds = CharDataset(sample_text, block_size=16, device=DEVICE)
    x, y = ds.get_batch("val", batch_size=4, block_size=8)
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)


def test_split_partitions_data(sample_text):
    ds = CharDataset(sample_text, block_size=16, split=0.9, device=DEVICE)
    total = len(ds.train_data) + len(ds.val_data)
    assert total == len(sample_text)
    assert len(ds.train_data) > len(ds.val_data)


def test_tokens_in_vocab_range(sample_text):
    ds = CharDataset(sample_text, block_size=16, device=DEVICE)
    x, y = ds.get_batch("train", batch_size=4, block_size=8)
    assert x.min().item() >= 0
    assert x.max().item() < ds.vocab_size
    assert y.max().item() < ds.vocab_size


# --------------------------------------------------------------------------- #
# load_corpus reads data/shakespeare.txt
# --------------------------------------------------------------------------- #
def test_load_corpus_reads_shakespeare():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(here, "data")
    # The corpus is fetched by download_data.py, not committed (see .gitignore),
    # so skip cleanly if it isn't present in this environment.
    if not os.path.exists(os.path.join(data_dir, "shakespeare.txt")):
        pytest.skip("corpus not downloaded (run: python download_data.py)")
    text = load_corpus(data_dir)
    assert isinstance(text, str)
    assert len(text) > 0


def test_load_corpus_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus(str(tmp_path))
