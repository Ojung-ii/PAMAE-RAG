# Baseline 2Wiki 20

Current baseline uses the existing legacy graph-aware retrieval config:

```bash
conda run -n pamae-rag python scripts/run_retrieval.py \
  --config configs/ablations_dq_connectivity_retrieval/2wikimultihopqa_dq_hybrid_mutual_knn8_07sem_03graph_refine_cell.yaml \
  --input data/processed/2wikimultihopqa/examples_100.jsonl \
  --output docs/experiments/content_graph_indexing_20260528/baseline_2wiki20/retrieval.jsonl \
  --limit 20

conda run -n pamae-rag python scripts/evaluate_retrieval.py \
  --input data/processed/2wikimultihopqa/examples_100.jsonl \
  --predictions docs/experiments/content_graph_indexing_20260528/baseline_2wiki20/retrieval.jsonl \
  --output docs/experiments/content_graph_indexing_20260528/baseline_2wiki20/retrieval_metrics.json \
  --limit 20

conda run -n pamae-rag python scripts/run_qa.py \
  --input data/processed/2wikimultihopqa/examples_100.jsonl \
  --predictions docs/experiments/content_graph_indexing_20260528/baseline_2wiki20/retrieval.jsonl \
  --output docs/experiments/content_graph_indexing_20260528/baseline_2wiki20/qa.jsonl \
  --metrics-output docs/experiments/content_graph_indexing_20260528/baseline_2wiki20/qa_metrics.json \
  --limit 20
```

Summary:

| run | graph_mode | oracle | candidate_recall | projected_recall | post_refine_recall | rendered_recall | context_f1 | avg_context_tokens | retrieval_ms | generation_ms | EM | F1 | oracle_gap | risk_decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline_2wiki20 | legacy_hybrid_sem_graph | false | 0.8625 | n/a | 0.0250 | 0.5000 | 0.2347 | 497.1 | 381.9 | 0.5 | 0.0000 | 0.0355 | 0.0140 | measurement_only |

Notes:

- `projected_recall` is not yet available because content graph projection is not connected in this step.
- This baseline does not claim success; it establishes the fixed sample, command shape, QA generator, and metric before oracle/context-graph work.
- Stage-wise diagnostics show candidate recall `0.8625`, local refinement survival `0.0250`, rendered recall `0.5000`, and support-fact rendered survival `0.5000`.
