.PHONY: install test lint format search experiments clean

install:
	pip install -e .[dev]

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format .

search:
	python run_search.py

experiments:
	python -m experiments.run_all

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -rf runs
	rm -f results/*.pt
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
