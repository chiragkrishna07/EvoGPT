"""Fast correctness checks: shapes, causality, KV-cache equivalence, GQA, overfit.
Run: python -m tests.test_smoke
"""
import torch
from evogpt.model import EvoGPT, GPTConfig


def test_forward_shapes():
    cfg = GPTConfig(vocab_size=65, block_size=32, n_layer=2, n_head=4, n_kv_head=2, d_model=64)
    m = EvoGPT(cfg)
    x = torch.randint(0, 65, (3, 16))
    logits, loss = m(x, x)
    assert logits.shape == (3, 16, 65), logits.shape
    assert loss.item() > 0
    print("  ok forward shapes + loss")


def test_causality():
    """Changing a future token must not change earlier logits (causal mask works)."""
    cfg = GPTConfig(vocab_size=65, block_size=32, n_layer=2, n_head=4, n_kv_head=4, d_model=64)
    m = EvoGPT(cfg).eval()
    x = torch.randint(0, 65, (1, 16))
    with torch.no_grad():
        a, _ = m(x)
        x2 = x.clone(); x2[0, -1] = (x2[0, -1] + 1) % 65
        b, _ = m(x2)
    assert torch.allclose(a[0, :-1], b[0, :-1], atol=1e-5), "future token leaked into past"
    print("  ok causality (no future leakage)")


def test_kv_cache_matches_full():
    """Incremental decode with KV-cache must equal a full forward pass."""
    cfg = GPTConfig(vocab_size=65, block_size=64, n_layer=2, n_head=4, n_kv_head=2, d_model=64)
    m = EvoGPT(cfg).eval()
    torch.manual_seed(0)
    prompt = torch.randint(0, 65, (1, 8))
    with torch.no_grad():
        gen = m.generate(prompt.clone(), max_new_tokens=10, temperature=1e-9, top_k=1)
        full_logits, _ = m(gen)
    # greedy next-token from full pass should match the generated continuation
    pred = full_logits[0, 7:-1].argmax(-1)
    assert torch.equal(pred, gen[0, 8:]), "KV-cache decode diverged from full forward"
    print("  ok KV-cache == full forward (greedy)")


def test_gqa_param_savings():
    """GQA (fewer kv heads) must reduce parameter count vs full MHA."""
    base = GPTConfig(vocab_size=65, n_head=8, n_kv_head=8, d_model=128)
    gqa = GPTConfig(vocab_size=65, n_head=8, n_kv_head=2, d_model=128)
    assert EvoGPT(gqa).num_params() < EvoGPT(base).num_params()
    print("  ok GQA reduces params")


def test_can_overfit():
    """A tiny model must drive loss far down on a single repeated batch."""
    cfg = GPTConfig(vocab_size=65, block_size=16, n_layer=2, n_head=4, n_kv_head=4, d_model=64)
    m = EvoGPT(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = torch.randint(0, 65, (4, 16)); y = torch.randint(0, 65, (4, 16))
    first = None
    for _ in range(200):
        _, loss = m(x, y)
        if first is None:
            first = loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first * 0.5, f"failed to overfit: {first:.3f} -> {loss.item():.3f}"
    print(f"  ok overfit ({first:.3f} -> {loss.item():.3f})")


if __name__ == "__main__":
    print("Running smoke tests...")
    test_forward_shapes()
    test_causality()
    test_kv_cache_matches_full()
    test_gqa_param_savings()
    test_can_overfit()
    print("ALL PASSED")
