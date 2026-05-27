# Oracle HotpotQA 20

```bash
conda run -n pamae-rag python scripts/run_qa.py \
  --input data/processed/hotpotqa/examples_100.jsonl \
  --oracle-context \
  --corpus data/processed/hotpotqa_corpus.json \
  --output docs/experiments/content_graph_indexing_20260528/oracle_hotpot20/qa.jsonl \
  --metrics-output docs/experiments/content_graph_indexing_20260528/oracle_hotpot20/qa_metrics.json \
  --limit 20
```

| run | graph_mode | oracle | candidate_recall | projected_recall | post_refine_recall | rendered_recall | context_f1 | avg_context_tokens | retrieval_ms | generation_ms | EM | F1 | oracle_gap | risk_decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| oracle_hotpot20 | direct_gold_context | true | n/a | n/a | n/a | 1.0000 | 1.0000 | 153.0 | 0.0 | 0.1 | 0.0000 | 0.0711 | 0.0000 | measurement_only |
