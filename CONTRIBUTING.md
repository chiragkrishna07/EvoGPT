# Contributing to EvoGPT

Thanks for your interest in contributing! This guide covers the basics of
getting set up and the conventions the project follows.

## Development setup

Requires Python >= 3.10.

```bash
# Clone, then create/activate a virtual environment, then:
make install        # pip install -e .[dev]
```

This installs EvoGPT in editable mode along with the dev dependencies
(`pytest` and `ruff`).

> Note: on CI we install a CPU-only build of PyTorch to keep things light:
> `pip install --index-url https://download.pytorch.org/whl/cpu torch`.
> You can do the same locally if you don't need GPU support.

## Running the tests

```bash
make test           # pytest -q
```

## Linting and formatting

```bash
make lint           # ruff check .
make format         # ruff format .
```

Please make sure `ruff check .` passes before opening a pull request.

## Running the project

```bash
make search         # python run_search.py   -> evolutionary architecture search
make experiments    # python -m experiments.run_all
```

## Project layout

```
evogpt/             Core importable package
  model.py          EvoGPT model + GPTConfig
  data.py           CharDataset, load_corpus
  train.py          train_candidate, TrainBudget, get_device
  evolve.py         Evolutionary search loop
  morph.py          Architecture mutation / morphing helpers
experiments/        Reproducible experiment runners (run_all.py, _common.py)
tests/              pytest test suite
run_search.py       Entry point for the evolutionary search
sample.py           Sample text from a trained model
analyze.py          Analysis / plotting utilities
```

## Pull requests

1. Fork and branch off `main`.
2. Make your change with accompanying tests where appropriate.
3. Ensure `make lint` and `make test` pass.
4. Open a PR against `main` with a clear description.
