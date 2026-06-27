"""Shared helpers for the experiment suite (no third-party stats deps)."""
from __future__ import annotations

import random
import math


def rankdata(values):
    """Average-rank of each element (ties share the mean rank). 1-indexed."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-indexed average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a, b) -> float:
    """Spearman rank correlation between two equal-length sequences."""
    ra, rb = rankdata(a), rankdata(b)
    n = len(a)
    mean_a, mean_b = sum(ra) / n, sum(rb) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - mean_a) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mean_b) ** 2 for y in rb))
    return cov / (va * vb) if va > 0 and vb > 0 else 0.0


def pearson(a, b) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = math.sqrt(sum((x - ma) ** 2 for x in a))
    vb = math.sqrt(sum((y - mb) ** 2 for y in b))
    return cov / (va * vb) if va > 0 and vb > 0 else 0.0


def best_so_far(seq):
    """Running minimum — for 'best fitness vs evaluations' curves."""
    out, cur = [], float("inf")
    for v in seq:
        cur = min(cur, v)
        out.append(cur)
    return out
