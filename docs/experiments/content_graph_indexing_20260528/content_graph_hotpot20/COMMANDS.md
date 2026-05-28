# Content Graph HotpotQA 20

```bash
conda run -n pamae-rag python scripts/run_retrieval.py \
  --config configs/content_graph/hotpotqa_content_graph_refine_cell.yaml \
  --input data/processed/hotpotqa/examples_100.jsonl \
  --output docs/experiments/content_graph_indexing_20260528/content_graph_hotpot20/retrieval.jsonl \
  --limit 20

conda run -n pamae-rag python scripts/evaluate_retrieval.py \
  --input data/processed/hotpotqa/examples_100.jsonl \
  --predictions docs/experiments/content_graph_indexing_20260528/content_graph_hotpot20/retrieval.jsonl \
  --output docs/experiments/content_graph_indexing_20260528/content_graph_hotpot20/retrieval_metrics.json \
  --limit 20

conda run -n pamae-rag python scripts/run_qa.py \
  --input data/processed/hotpotqa/examples_100.jsonl \
  --predictions docs/experiments/content_graph_indexing_20260528/content_graph_hotpot20/retrieval.jsonl \
  --output docs/experiments/content_graph_indexing_20260528/content_graph_hotpot20/qa.jsonl \
  --metrics-output docs/experiments/content_graph_indexing_20260528/content_graph_hotpot20/qa_metrics.json \
  --limit 20
```

| run | graph_mode | oracle | candidate_recall | projected_recall | post_refine_recall | rendered_recall | context_f1 | avg_context_tokens | retrieval_ms | generation_ms | EM | F1 | oracle_gap | risk_decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| content_graph_hotpot20 | content_hybrid_sem_graph | false | 0.9750 | 0.9250 | 0.3750 | 0.6750 | 0.2871 | 478.8 | 676.9 | 0.4 | 0.0000 | 0.0624 | 0.0087 | no_adoption |

Notes:

- Content projection and refinement survival improve, but rendered recall and QA F1 decline relative to the baseline.
- This fails the QA-gated adoption criterion and should not be treated as a successful performance improvement.
