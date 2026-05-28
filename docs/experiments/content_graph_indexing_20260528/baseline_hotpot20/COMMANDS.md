# Baseline HotpotQA 20

```bash
conda run -n pamae-rag python scripts/run_retrieval.py \
  --config configs/ablations_dq_connectivity_retrieval/hotpotqa_dq_hybrid_mutual_knn8_07sem_03graph_refine_cell.yaml \
  --input data/processed/hotpotqa/examples_100.jsonl \
  --output docs/experiments/content_graph_indexing_20260528/baseline_hotpot20/retrieval.jsonl \
  --limit 20

conda run -n pamae-rag python scripts/evaluate_retrieval.py \
  --input data/processed/hotpotqa/examples_100.jsonl \
  --predictions docs/experiments/content_graph_indexing_20260528/baseline_hotpot20/retrieval.jsonl \
  --output docs/experiments/content_graph_indexing_20260528/baseline_hotpot20/retrieval_metrics.json \
  --limit 20

conda run -n pamae-rag python scripts/run_qa.py \
  --input data/processed/hotpotqa/examples_100.jsonl \
  --predictions docs/experiments/content_graph_indexing_20260528/baseline_hotpot20/retrieval.jsonl \
  --output docs/experiments/content_graph_indexing_20260528/baseline_hotpot20/qa.jsonl \
  --metrics-output docs/experiments/content_graph_indexing_20260528/baseline_hotpot20/qa_metrics.json \
  --limit 20
```

| run | graph_mode | oracle | candidate_recall | projected_recall | post_refine_recall | rendered_recall | context_f1 | avg_context_tokens | retrieval_ms | generation_ms | EM | F1 | oracle_gap | risk_decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline_hotpot20 | legacy_hybrid_sem_graph | false | 0.9750 | n/a | 0.1500 | 0.7250 | 0.3327 | 504.4 | 419.5 | 0.5 | 0.0000 | 0.0672 | 0.0254 | measurement_only |
