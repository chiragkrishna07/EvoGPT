"""Load the evolved champion checkpoint and generate text.
Usage:  python sample.py --prompt "ROMEO:" --tokens 500
"""
import os
import argparse
import torch

from evogpt.model import EvoGPT, GPTConfig
from evogpt.data import CharDataset, load_corpus
from evogpt.train import get_device

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--tokens", type=int, default=500)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    args = ap.parse_args()

    device = get_device()
    ckpt = torch.load(os.path.join(HERE, "results", "champion.pt"), map_location=device)
    cfg = GPTConfig(**ckpt["config"])
    model = EvoGPT(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # rebuild the char vocab from the same corpus
    dataset = CharDataset(load_corpus(os.path.join(HERE, "data")),
                          block_size=cfg.block_size, device=device)

    idx = dataset.encode(args.prompt).unsqueeze(0).to(device)
    out = model.generate(idx, max_new_tokens=args.tokens, temperature=args.temp, top_k=args.top_k)
    print(dataset.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
