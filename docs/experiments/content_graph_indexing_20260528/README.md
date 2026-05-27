# Content Graph Indexing QA Diagnostics

This experiment series redesigns graph indexing toward dependency-free content-derived entity/fact graphs, then adds stage-wise diagnostics before any performance-oriented change.

Fixed QA measurement settings introduced for this series:

- generator: `deterministic_extractive_sentence_v1`
- prompt: `closed_book_free_context_qa_v1`
- metric: `squad_normalized_em_f1_v1`
- sample: first 20 rows from `data/processed/2wikimultihopqa/examples_100.jsonl`
- seed: config seed `42`
- oracle corpus: `data/processed/2wikimultihopqa_corpus.json`

The default generator is an offline deterministic extractive sentence generator. It is not an LLM upper bound; all comparisons in this series must use the same generator and metric.

Current 2Wiki20 QA measurements:

| run | graph_mode | oracle | candidate_recall | projected_recall | post_refine_recall | rendered_recall | context_f1 | avg_context_tokens | retrieval_ms | generation_ms | EM | F1 | oracle_gap | risk_decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline_2wiki20 | legacy_hybrid_sem_graph | false | n/a | n/a | n/a | 0.5000 | 0.2347 | 497.1 | 322.6 | 0.4 | 0.0000 | 0.0355 | 0.0140 | measurement_only |
| oracle_2wiki20 | direct_gold_context | true | n/a | n/a | n/a | 1.0000 | 1.0000 | 279.7 | 0.0 | 0.2 | 0.0000 | 0.0495 | 0.0000 | measurement_only |
