# Content Graph 2Wiki 20

Content graph retrieval uses the dependency-free text-derived graph source:

```bash
conda run -n pamae-rag python scripts/run_retrieval.py \
  --config configs/content_graph/2wikimultihopqa_content_graph_refine_cell.yaml \
  --input data/processed/2wikimultihopqa/examples_100.jsonl \
  --output docs/experiments/content_graph_indexing_20260528/content_graph_2wiki20/retrieval.jsonl \
  --limit 20

conda run -n pamae-rag python scripts/evaluate_retrieval.py \
  --input data/processed/2wikimultihopqa/examples_100.jsonl \
  --predictions docs/experiments/content_graph_indexing_20260528/content_graph_2wiki20/retrieval.jsonl \
  --output docs/experiments/content_graph_indexing_20260528/content_graph_2wiki20/retrieval_metrics.json \
  --limit 20

conda run -n pamae-rag python scripts/run_qa.py \
  --input data/processed/2wikimultihopqa/examples_100.jsonl \
  --predictions docs/experiments/content_graph_indexing_20260528/content_graph_2wiki20/retrieval.jsonl \
  --output docs/experiments/content_graph_indexing_20260528/content_graph_2wiki20/qa.jsonl \
  --metrics-output docs/experiments/content_graph_indexing_20260528/content_graph_2wiki20/qa_metrics.json \
  --limit 20
```

Summary:

| run | graph_mode | oracle | candidate_recall | projected_recall | post_refine_recall | rendered_recall | context_f1 | avg_context_tokens | retrieval_ms | generation_ms | EM | F1 | oracle_gap | risk_decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| content_graph_2wiki20 | content_hybrid_sem_graph | false | 0.8625 | 0.8000 | 0.3500 | 0.6000 | 0.2843 | 350.1 | 1260.7 | 0.3 | 0.0000 | 0.0580 | -0.0085 | measurement_limited |

Notes:

- Content graph projection is now active and preserves projected support at `0.8000`.
- Local refinement/reranking support survival improves from the baseline `0.0250` to `0.3500`.
- Rendered recall improves from `0.5000` to `0.6000`.
- Content graph projection adds `1140.6 ms` average latency inside retrieval.
- QA F1 exceeds the current deterministic oracle F1 (`0.0495`), which means the offline generator is not a strict oracle-upper-bound generator. Treat this as a measurement limitation and do not claim oracle-gap success from this run.
