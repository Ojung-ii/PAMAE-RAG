# Semantic Embedding-Space Rerun Report

- Branch: `experiment/semantic-embedding-space-rerun`
- Commit: `996182d`
- Gate outcome: **STOP_BEFORE_100**
- Final decision: **STOP**

Previous semantic STOP summary: the first semantic carrier run stopped because query embeddings were absent. This rerun audits provenance, rejects legacy 128D chunks, and uses one compatible local encoder when available.

Theory: raw cosine is not treated as a PAMAE distance. Semantic diagnostics use normalized angular distance, and semantic renderers are graph-constrained to `T_q union S1`; the entity-chunk graph retrieval core remains unchanged.

## 2wikimultihopqa

- Gate decision: **STOP_BEFORE_100**
- Reason: semantic attribution does not separate current-only answer chunks from non-answer chunks
- Model: `nvidia/NV-Embed-v2`
- Revision: `3fa59658547db50a1e8e3346cf057fd0c77ed6ef`
- Dim: `4096`
- Query coverage: 1.0000
- Chunk coverage: 1.0000
- Normalized: `True`
- Text format: chunk `Title: <title>
Text: <chunk_text>`, query `raw_question`

### Semantic Attribution

- mean d_ang(q,u), current-only answer: 0.4368
- mean d_ang(q,u), current-only non-answer: 0.4301
- median d_ang(q,u), current-only answer: 0.4496
- median d_ang(q,u), current-only non-answer: 0.4344
- mean d_ang(u,T_q), current-only answer: 0.3690
- mean d_ang(u,T_q), current-only non-answer: 0.3820
- semantic_separation_query: -0.0067
- semantic_separation_tree: 0.0130

### Pool Sizes

- avg strict tree chunks: 4.5000
- avg shell1 chunks: 73.0700
- avg shell2 chunks: 0.0000
- answer on support tree rate: 0.4000
- answer in shell1 rate: 0.2600
- answer in shell2 rate: 0.0000

### Variant Table

| renderer | oracle | diagnostic | answer_in_context | rendered_recall | context_f1 | qa_f1 | avg_tokens | retrieval_ms | shell1 | rendered_shell1 | missing_rate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_renderer | False | False | 0.4300 | 0.5750 | 0.2646 | 0.0814 | 333.3900 | 763.2947 | 0.0000 | 0.0000 | 0.0000 |
| metric_path_carrier | False | False | 0.3600 | 0.4550 | 0.3362 | 0.0801 | 192.3300 | 760.4666 | 0.0000 | 0.0000 | 0.0000 |
| tree_shell1_graph_order | False | False | 0.4000 | 0.5050 | 0.2385 | 0.0831 | 420.4500 | 866.3531 | 73.0700 | 3.2200 | 0.0000 |
| tree_shell1_semantic_query_order | False | False | 0.4600 | 0.6225 | 0.2876 | 0.0849 | 357.6700 | 862.8903 | 73.0700 | 3.4300 | 0.0000 |
| tree_shell1_semantic_tree_order | False | False | 0.4400 | 0.5450 | 0.2516 | 0.0805 | 315.6500 | 871.1978 | 73.0700 | 3.3900 | 0.0000 |
| semantic_weighted_tree_diagnostic | False | True | 0.4100 | 0.4825 | 0.2680 | 0.0793 | 305.8700 | 1398.2412 | 0.0000 | 0.0000 | 0.0000 |
| current_answer_role_oracle | True | False | 0.4300 | 0.1525 | 0.4837 | 0.1160 | 43.4100 | 773.7429 | 0.0000 | 0.0000 | 0.0000 |
| tree_answer_oracle | True | True | 0.4000 | 0.1400 | 0.4817 | 0.1130 | 33.3300 | 759.3738 | 0.0000 | 0.0000 | 0.0000 |
| shell1_answer_oracle | True | False | 0.2500 | 0.0625 | 0.3022 | 0.0167 | 50.8000 | 876.7611 | 73.0700 | 0.3800 | 0.0000 |

## Expert Panel Rules

- GraphRAG expert: reject semantic gains outside graph-defined candidates.
- IR expert: semantic adequacy is justified only if answer/non-answer separation appears inside the graph shell.
- Graph theory expert: reject if angular metric or positive edge constraints fail.
- NLP expert: stop if query similarity selects topical non-answer chunks.
- RAG expert: if coverage improves without QA, inspect formatting before retrieval changes.
- Systems expert: do not adopt if retrieval time increases too much.
- Professor/meta-reviewer: one-dataset or weight/threshold-dependent gains are local-minimum risk.
