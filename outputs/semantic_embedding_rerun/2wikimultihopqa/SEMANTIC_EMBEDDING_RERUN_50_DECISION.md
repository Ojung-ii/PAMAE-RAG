# Semantic Embedding-Space Rerun Report

- Branch: `experiment/semantic-embedding-space-rerun`
- Commit: `515af2e`
- Gate outcome: **GO_TO_100**
- Final decision: **DIAGNOSTIC_ONLY**

Previous semantic STOP summary: the first semantic carrier run stopped because query embeddings were absent. This rerun audits provenance, rejects legacy 128D chunks, and uses one compatible local encoder when available.

Theory: raw cosine is not treated as a PAMAE distance. Semantic diagnostics use normalized angular distance, and semantic renderers are graph-constrained to `T_q union S1`; the entity-chunk graph retrieval core remains unchanged.

## 2wikimultihopqa

- Gate decision: **GO_TO_100**
- Reason: semantic attribution is meaningful and 2Wiki smoke preserves coverage metrics
- Model: `nvidia/NV-Embed-v2`
- Revision: `3fa59658547db50a1e8e3346cf057fd0c77ed6ef`
- Dim: `4096`
- Query coverage: 1.0000
- Chunk coverage: 1.0000
- Normalized: `True`
- Text format: chunk `Title: <title>
Text: <chunk_text>`, query `raw_question`

### Semantic Attribution

- mean d_ang(q,u), current-only answer: 0.3982
- mean d_ang(q,u), current-only non-answer: 0.4347
- median d_ang(q,u), current-only answer: 0.3982
- median d_ang(q,u), current-only non-answer: 0.4416
- mean d_ang(u,T_q), current-only answer: 0.3023
- mean d_ang(u,T_q), current-only non-answer: 0.3838
- semantic_separation_query: 0.0365
- semantic_separation_tree: 0.0814

### Pool Sizes

- avg strict tree chunks: 4.6200
- avg shell1 chunks: 76.0800
- avg shell2 chunks: 0.0000
- answer on support tree rate: 0.3600
- answer in shell1 rate: 0.2800
- answer in shell2 rate: 0.0000

### Variant Table

| renderer | oracle | diagnostic | answer_in_context | rendered_recall | context_f1 | qa_f1 | avg_tokens | retrieval_ms | shell1 | rendered_shell1 | missing_rate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_renderer | False | False | 0.4000 | 0.5700 | 0.2683 | 0.0610 | 351.0400 | 758.4953 | 0.0000 | 0.0000 | 0.0000 |
| metric_path_carrier | False | False | 0.3200 | 0.4550 | 0.3428 | 0.0693 | 200.7600 | 801.0776 | 0.0000 | 0.0000 | 0.0000 |
| tree_shell1_graph_order | False | False | 0.3600 | 0.5250 | 0.2567 | 0.0663 | 419.6400 | 859.3931 | 76.0800 | 3.1000 | 0.0000 |
| tree_shell1_semantic_query_order | False | False | 0.4600 | 0.6350 | 0.2998 | 0.0610 | 370.7800 | 820.8651 | 76.0800 | 3.3200 | 0.0000 |
| tree_shell1_semantic_tree_order | False | False | 0.4200 | 0.5800 | 0.2742 | 0.0610 | 332.6200 | 827.8948 | 76.0800 | 3.3600 | 0.0000 |
| semantic_weighted_tree_diagnostic | False | True | 0.3800 | 0.5000 | 0.2736 | 0.0657 | 328.9000 | 1229.8977 | 0.0000 | 0.0000 | 0.0000 |
| current_answer_role_oracle | True | False | 0.4000 | 0.1500 | 0.5217 | 0.0923 | 39.9400 | 759.0249 | 0.0000 | 0.0000 | 0.0000 |
| tree_answer_oracle | True | True | 0.3600 | 0.1300 | 0.5093 | 0.0930 | 32.2600 | 841.0313 | 0.0000 | 0.0000 | 0.0000 |
| shell1_answer_oracle | True | False | 0.2600 | 0.0750 | 0.3846 | 0.0174 | 50.1600 | 840.2112 | 76.0800 | 0.3200 | 0.0000 |

## Expert Panel Rules

- GraphRAG expert: reject semantic gains outside graph-defined candidates.
- IR expert: semantic adequacy is justified only if answer/non-answer separation appears inside the graph shell.
- Graph theory expert: reject if angular metric or positive edge constraints fail.
- NLP expert: stop if query similarity selects topical non-answer chunks.
- RAG expert: if coverage improves without QA, inspect formatting before retrieval changes.
- Systems expert: do not adopt if retrieval time increases too much.
- Professor/meta-reviewer: one-dataset or weight/threshold-dependent gains are local-minimum risk.
