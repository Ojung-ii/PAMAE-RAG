# Experiment protocol

Run in this order.

## 1. Smoke test

```bash
pytest -q
make smoke
make eval
```

## 2. Dataset validation

```bash
python scripts/validate_data.py --input data/processed/<dataset>/examples.jsonl
```

## 3. Retrieval-only experiment

```bash
python scripts/run_retrieval.py \
  --config configs/default.yaml \
  --input data/processed/<dataset>/examples.jsonl \
  --output data/runs/<dataset>/pamae_rag_v1_results.jsonl

python scripts/evaluate_retrieval.py \
  --input data/processed/<dataset>/examples.jsonl \
  --predictions data/runs/<dataset>/pamae_rag_v1_results.jsonl \
  --output data/runs/<dataset>/pamae_rag_v1_metrics.json
```

## 4. Diagnostics to report before answer generation

1. `V_q` gold carrier recall.
2. anchor support recall.
3. rendered context support recall.
4. objective decrease from Phase I full validation.
5. objective decrease from Phase II refinement.
6. average context tokens.
7. retrieval latency.
