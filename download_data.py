"""Fetch the tiny-shakespeare training corpus into data/shakespeare.txt.

The corpus is not committed to the repo (see .gitignore); run this once after
cloning. Public-domain text, ~1.1 MB.

    python download_data.py
"""
import os
import urllib.request

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "shakespeare.txt")


def main():
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    if os.path.exists(DEST) and os.path.getsize(DEST) > 0:
        print(f"already present: {DEST} ({os.path.getsize(DEST):,} bytes)")
        return
    print(f"downloading -> {DEST}")
    urllib.request.urlretrieve(URL, DEST)
    print(f"done ({os.path.getsize(DEST):,} bytes)")


if __name__ == "__main__":
    main()
