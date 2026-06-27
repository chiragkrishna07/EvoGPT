"""Char-level data pipeline. Self-contained: builds the vocab from the corpus,
encodes to a uint16 tensor, and serves random contiguous batches.

Char-level keeps the vocab tiny (~65 tokens) so candidate models in the
evolutionary search train in seconds on Apple-Silicon MPS.
"""
from __future__ import annotations

import os
import json
import torch


class CharDataset:
    def __init__(self, text: str, block_size: int, split: float = 0.9, device="cpu"):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
        self.block_size = block_size
        self.device = device

        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(split * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split: str, batch_size: int, block_size: int | None = None):
        data = self.train_data if split == "train" else self.val_data
        bs = min(block_size or self.block_size, len(data) - 1)
        ix = torch.randint(len(data) - bs - 1, (batch_size,))
        x = torch.stack([data[i:i + bs] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + bs] for i in ix])
        return x.to(self.device), y.to(self.device)

    def encode(self, s: str):
        return torch.tensor([self.stoi.get(c, 0) for c in s], dtype=torch.long)

    def decode(self, t) -> str:
        return "".join(self.itos[int(i)] for i in t)

    def save_meta(self, path: str):
        with open(path, "w") as f:
            json.dump({"stoi": self.stoi, "itos": self.itos,
                       "vocab_size": self.vocab_size}, f)


def load_corpus(data_dir: str) -> str:
    path = os.path.join(data_dir, "shakespeare.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    # Fallback: synthetic but learnable structured corpus (offline-safe).
    raise FileNotFoundError(f"No corpus at {path}")
