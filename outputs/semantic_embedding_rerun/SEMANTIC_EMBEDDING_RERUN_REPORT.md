# Semantic Embedding-Space Rerun Report

- Branch: `experiment/semantic-embedding-space-rerun`
- Commit: `837b34a`
- Gate outcome: **STOP_BEFORE_100**
- Final decision: **STOP**

Previous semantic STOP summary: the first semantic carrier run stopped because query embeddings were absent. This rerun audits provenance, rejects legacy 128D chunks, and uses one compatible local encoder when available.

Theory: raw cosine is not treated as a PAMAE distance. Semantic diagnostics use normalized angular distance, and semantic renderers are graph-constrained to `T_q union S1`; the entity-chunk graph retrieval core remains unchanged.

## Verification

- `python -m compileall src tests scripts`: passed.
- `pytest -q`: 145 passed.
- Non-oracle renderers kept `oracle_leakage_count = 0` and `score_mixing_detected = false`.
- Embedding coverage was complete for the diagnostic pools in both datasets: query coverage 1.0, chunk coverage 1.0, dimension match true, normalized true.

## Embedding Provenance Audit

- Existing processed chunk embeddings were found, but they were legacy 128D vectors with no model id, revision, pooling, normalization metadata, or matching query embeddings.
- The legacy cache was rejected for query/chunk semantic distance because provenance was unknown and query vectors were absent.
- Local NV-Embed-v2 was valid and selected: `nvidia/NV-Embed-v2`, revision `3fa59658547db50a1e8e3346cf057fd0c77ed6ef`.
- Qwen/Qwen3-Embedding-8B was not used because a valid local NV-Embed-v2 snapshot was available and Qwen3 was not present in the local Hugging Face cache.
- Fixed formats were used: chunks as `Title: <title>\nText: <chunk_text>`, queries as raw question text. No instruction/template variants were tuned.

## Compatibility Evidence

- One encoder produced both query and chunk embeddings for both datasets.
- Output dimension was 4096 for all cached query/chunk vectors.
- Vectors were L2-normalized before angular distance.
- No random/fake fallback vectors were used; missing diagnostic embeddings were logged instead of silently replaced.
- The semantic stage remained graph-constrained: renderers drew from `T_q` or `T_q union S1`; no global dense top-k retrieval was added.

## Gate Summary

- 2Wiki 50 opened the overnight gate: valid embeddings, no invariant failures, and positive initial semantic separation.
- 2Wiki 100 did not preserve the attribution evidence: current-only answer chunks were not closer to the query than current-only non-answer chunks (`semantic_separation_query = -0.0067`).
- Hotpot 100 showed strong positive separation and the semantic-query renderer improved answer coverage, rendered recall, context F1, QA F1, and stayed within the 1.10 token budget allowance.
- Because the attribution signal is not stable across 2Wiki and Hotpot, this is not an adoption candidate. The final decision is **STOP** under the stated stop condition: semantic attribution does not separate answer from non-answer chunks reliably.

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

## hotpotqa

- Gate decision: **GO_TO_100**
- Reason: semantic attribution is meaningful and the non-oracle semantic-query renderer preserves or improves Hotpot coverage metrics
- Model: `nvidia/NV-Embed-v2`
- Revision: `3fa59658547db50a1e8e3346cf057fd0c77ed6ef`
- Dim: `4096`
- Query coverage: 1.0000
- Chunk coverage: 1.0000
- Normalized: `True`
- Text format: chunk `Title: <title>
Text: <chunk_text>`, query `raw_question`

### Semantic Attribution

- mean d_ang(q,u), current-only answer: 0.3562
- mean d_ang(q,u), current-only non-answer: 0.4167
- median d_ang(q,u), current-only answer: 0.3522
- median d_ang(q,u), current-only non-answer: 0.4203
- mean d_ang(u,T_q), current-only answer: 0.3381
- mean d_ang(u,T_q), current-only non-answer: 0.3810
- semantic_separation_query: 0.0605
- semantic_separation_tree: 0.0429

### Pool Sizes

- avg strict tree chunks: 4.5200
- avg shell1 chunks: 77.0900
- avg shell2 chunks: 0.0000
- answer on support tree rate: 0.5700
- answer in shell1 rate: 0.4500
- answer in shell2 rate: 0.0000

### Variant Table

| renderer | oracle | diagnostic | answer_in_context | rendered_recall | context_f1 | qa_f1 | avg_tokens | retrieval_ms | shell1 | rendered_shell1 | missing_rate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_renderer | False | False | 0.6500 | 0.6700 | 0.2883 | 0.0731 | 480.9300 | 722.3412 | 0.0000 | 0.0000 | 0.0000 |
| metric_path_carrier | False | False | 0.5500 | 0.5000 | 0.3359 | 0.0684 | 272.8200 | 713.2558 | 0.0000 | 0.0000 | 0.0000 |
| tree_shell1_graph_order | False | False | 0.5900 | 0.5550 | 0.2461 | 0.0665 | 493.8200 | 859.3053 | 77.0900 | 2.8000 | 0.0001 |
| tree_shell1_semantic_query_order | False | False | 0.7600 | 0.7800 | 0.3352 | 0.0789 | 485.2400 | 841.5643 | 77.0900 | 2.9300 | 0.0001 |
| tree_shell1_semantic_tree_order | False | False | 0.6600 | 0.6800 | 0.2971 | 0.0733 | 477.3700 | 866.8465 | 77.0900 | 2.8200 | 0.0001 |
| semantic_weighted_tree_diagnostic | False | True | 0.5900 | 0.5350 | 0.2674 | 0.0721 | 428.9800 | 1681.5198 | 0.0000 | 0.0000 | 0.0000 |
| current_answer_role_oracle | True | False | 0.6500 | 0.3650 | 0.6293 | 0.1020 | 72.3300 | 717.0125 | 0.0000 | 0.0000 | 0.0000 |
| tree_answer_oracle | True | True | 0.5700 | 0.2900 | 0.6105 | 0.0813 | 56.1700 | 704.8970 | 0.0000 | 0.0000 | 0.0000 |
| shell1_answer_oracle | True | False | 0.4500 | 0.1800 | 0.4530 | 0.0696 | 77.3800 | 877.2259 | 77.0900 | 0.8200 | 0.0000 |

## Expert Panel Rules

- GraphRAG expert: reject semantic gains outside graph-defined candidates.
- IR expert: semantic adequacy is justified only if answer/non-answer separation appears inside the graph shell.
- Graph theory expert: reject if angular metric or positive edge constraints fail.
- NLP expert: stop if query similarity selects topical non-answer chunks.
- RAG expert: if coverage improves without QA, inspect formatting before retrieval changes.
- Systems expert: do not adopt if retrieval time increases too much.
- Professor/meta-reviewer: one-dataset or weight/threshold-dependent gains are local-minimum risk.

## Adoption Gate Result

- `tree_shell1_semantic_query_order` passed the metric-style comparison on Hotpot and improved 2Wiki renderer metrics, but the 2Wiki 100 attribution check failed.
- The semantic-weighted tree diagnostic was not adoptable and was slower, especially on Hotpot (`retrieval_ms = 1681.5198` vs current `722.3412`).
- Shell-1 pools were large (`73.0700` average chunks on 2Wiki, `77.0900` on Hotpot), so any future semantic carrier method needs an explicit principle for graph-shell budget allocation rather than an unconstrained expansion.
- Final decision: **STOP**.

## Next Recommendation

Do not adopt the semantic layer yet. The next experiment should isolate why `tree_shell1_semantic_query_order` improves coverage despite weak 2Wiki attribution: compare query-angular ordering against a non-answer topicality control inside `T_q union S1`, and require stable answer/non-answer separation before treating semantic adequacy as a method.
