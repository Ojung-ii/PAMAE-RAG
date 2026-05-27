# Content Graph Indexing QA Diagnostics

This experiment series redesigns graph indexing toward dependency-free content-derived entity/fact graphs, then adds stage-wise diagnostics before any performance-oriented change.

Fixed QA measurement settings introduced for this series:

- generator: `deterministic_extractive_sentence_v1`
- prompt: `closed_book_free_context_qa_v1`
- metric: `squad_normalized_em_f1_v1`
- sample: first 20 rows from `data/processed/2wikimultihopqa/examples_100.jsonl`
- seed: config seed `42`

The default generator is an offline deterministic extractive sentence generator. It is not an LLM upper bound; all comparisons in this series must use the same generator and metric.
