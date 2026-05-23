.PHONY: setup test smoke eval lint

setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

test:
	pytest -q

smoke:
	python scripts/run_retrieval.py --config configs/smoke.yaml --input data/smoke/examples.jsonl --output data/runs/smoke/results.jsonl

eval:
	python scripts/evaluate_retrieval.py --input data/smoke/examples.jsonl --predictions data/runs/smoke/results.jsonl --output data/runs/smoke/metrics.json

lint:
	ruff check src tests scripts
