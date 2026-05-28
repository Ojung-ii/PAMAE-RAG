# Oracle 2Wiki 20

Oracle QA uses gold supporting evidence as final QA context only. When `support_facts` sentence metadata resolves completely, the oracle context is the gold support sentences. It does not alter retrieval, candidate generation, scoring, or graph construction.

```bash
conda run -n pamae-rag python scripts/run_qa.py \
  --input data/processed/2wikimultihopqa/examples_100.jsonl \
  --oracle-context \
  --corpus data/processed/2wikimultihopqa_corpus.json \
  --output docs/experiments/content_graph_indexing_20260528/oracle_2wiki20/qa.jsonl \
  --metrics-output docs/experiments/content_graph_indexing_20260528/oracle_2wiki20/qa_metrics.json \
  --limit 20
```

Summary:

| run | graph_mode | oracle | candidate_recall | projected_recall | post_refine_recall | rendered_recall | context_f1 | avg_context_tokens | retrieval_ms | generation_ms | EM | F1 | oracle_gap | risk_decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| oracle_2wiki20 | direct_gold_context | true | n/a | n/a | n/a | 1.0000 | 1.0000 | 47.4 | 0.0 | 0.1 | 0.0000 | 0.0495 | 0.0000 | measurement_only |

Baseline gap under the fixed generator:

- baseline F1: `0.0355`
- oracle F1: `0.0495`
- baseline oracle gap: `0.0140`
