# Semantic Effect Decomposition Report

- Branch: `diagnose/semantic-effect-decomposition`
- Commit: `3fb1164`
- Fixed common_qa prompt hash: `31e4b446be8b00a4989078fb4a957bc61b19bf4b8014674e2baad4612cc4396d`
- Final decision: **ADOPTION_CANDIDATE**
- Adoption-candidate renderers: `tree_shell1_semantic_query_order`
- Next recommendation: Treat tree_shell1_semantic_query_order as an adoption candidate, not an automatic adoption: it passed both 100-query datasets under common_qa, but should get larger-sample validation and a bridge-carrier safety check before paper-method promotion.

Previous semantic rerun summary: valid local NV-Embed-v2 embeddings replaced legacy 128D vectors, but the earlier rerun stopped because the semantic attribution direction was not stable across 2Wiki and Hotpot. This run reuses the validated embedding cache and decomposes shell expansion from semantic ordering under a fixed prompt.

Theory boundary: the PAMAE entity-chunk retrieval core, graph-metric medoid selection, local refinement, support-tree construction, context budget, generator, evaluator, dataset order, and embedding cache remain unchanged. Semantic distance is normalized angular distance and is used only inside graph-defined candidates or the diagnostic semantic-weighted tree.

## 2wikimultihopqa

- Decision: **DIAGNOSTIC_ONLY**
- Reason: diagnostics valid; adoption requires both datasets to pass gates
- Dominant answer-coverage effect vs strict tree: **query_semantic_ordering**
- Same sample: `True` (identical query IDs and order)
- Prompt: `common_qa` hash `31e4b446be8b00a4989078fb4a957bc61b19bf4b8014674e2baad4612cc4396d` exact `True`
- Embedding: `nvidia/NV-Embed-v2` dim `4096` query coverage 1.0000, chunk coverage 1.0000

### Variant Table

| renderer | answer_in_context | rendered_recall | context_f1 | qa_f1 | avg_tokens | retrieval_ms | generation_ms | total_ms | support_tree_answer | shell1_answer | shell1_rendered_answer | bridge_retained | bridge_cut |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_renderer | 0.4300 | 0.5750 | 0.2646 | 0.0814 | 333.3900 | 990.2503 | 0.1763 | 990.4266 | 0.4000 | 0.0000 | 0.0000 | 0.2400 | 0.0100 |
| metric_path_carrier | 0.3600 | 0.4550 | 0.3362 | 0.0801 | 192.3300 | 977.8895 | 0.1063 | 977.9958 | 0.4000 | 0.0000 | 0.0000 | 0.2100 | 0.0100 |
| tree_shell1_graph_order | 0.4000 | 0.5050 | 0.2385 | 0.0831 | 420.4500 | 1066.6469 | 0.2134 | 1066.8602 | 0.4000 | 0.2600 | 0.0200 | 0.2400 | 0.0100 |
| tree_shell1_semantic_query_order | 0.4600 | 0.6225 | 0.2876 | 0.0849 | 357.6700 | 1063.3666 | 0.1855 | 1063.5520 | 0.4000 | 0.2600 | 0.0900 | 0.2400 | 0.0100 |
| tree_shell1_semantic_tree_order | 0.4400 | 0.5450 | 0.2516 | 0.0805 | 315.6500 | 1137.0288 | 0.1703 | 1137.1991 | 0.4000 | 0.2600 | 0.0700 | 0.2400 | 0.0100 |
| semantic_weighted_tree_diagnostic | 0.4100 | 0.4825 | 0.2680 | 0.0793 | 305.8700 | 1740.9676 | 0.1629 | 1741.1306 | 0.4000 | 0.0000 | 0.0000 | 0.2200 | 0.0300 |

### Decomposition Deltas

| effect | answer_in_context | rendered_recall | context_f1 | qa_f1 | tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| delta_shell_B1_minus_A1 | 0.0400 | 0.0500 | -0.0978 | 0.0030 | 228.1200 |
| delta_query_semantic_B2_minus_B1 | 0.0600 | 0.1175 | 0.0492 | 0.0017 | -62.7800 |
| delta_tree_semantic_B3_minus_B1 | 0.0400 | 0.0400 | 0.0131 | -0.0027 | -104.8000 |

### Semantic Attribution

- semantic_separation_query_current_only: -0.0067
- semantic_separation_tree_current_only: 0.0130
- semantic_separation_query_shell1: 0.0088
- semantic_separation_tree_shell1: 0.0026

| group | count | mean d_ang(q,u) | median d_ang(q,u) | mean d_ang(u,T_q) | median d_ang(u,T_q) |
| --- | ---: | ---: | ---: | ---: | ---: |
| current_only_answer | 10 | 0.4368 | 0.4496 | 0.3690 | 0.3708 |
| current_only_non_answer | 338 | 0.4301 | 0.4344 | 0.3820 | 0.3922 |
| shell1_answer | 91 | 0.4429 | 0.4512 | 0.4170 | 0.4384 |
| shell1_non_answer | 7216 | 0.4518 | 0.4521 | 0.4196 | 0.4263 |
| tree_answer | 45 | 0.3783 | 0.3754 | 0.0000 | 0.0000 |
| tree_non_answer | 405 | 0.4108 | 0.4205 | 0.0000 | 0.0000 |
| projected_nonrendered_answer | 266 | 0.4424 | 0.4515 | 0.4155 | 0.4392 |

### Pool And Bridge Diagnostics

- avg strict tree chunks: 4.5000
- avg shell1 chunks: 73.0700
- avg shell2 chunks: 0.0000
- answer on support tree rate: 0.4000
- answer in shell1 rate: 0.2600
- answer in shell2 rate: 0.0000

### Adoption Gates

- `tree_shell1_graph_order`: pass `False`, blockers `answer_in_context_regression, rendered_recall_regression, context_f1_regression, token_gate`
- `tree_shell1_semantic_query_order`: pass `True`, blockers `none`
- `tree_shell1_semantic_tree_order`: pass `False`, blockers `qa_f1_regression, rendered_recall_regression, context_f1_regression`

## hotpotqa

- Decision: **DIAGNOSTIC_ONLY**
- Reason: diagnostics valid; adoption requires both datasets to pass gates
- Dominant answer-coverage effect vs strict tree: **query_semantic_ordering**
- Same sample: `True` (identical query IDs and order)
- Prompt: `common_qa` hash `31e4b446be8b00a4989078fb4a957bc61b19bf4b8014674e2baad4612cc4396d` exact `True`
- Embedding: `nvidia/NV-Embed-v2` dim `4096` query coverage 1.0000, chunk coverage 1.0000

### Variant Table

| renderer | answer_in_context | rendered_recall | context_f1 | qa_f1 | avg_tokens | retrieval_ms | generation_ms | total_ms | support_tree_answer | shell1_answer | shell1_rendered_answer | bridge_retained | bridge_cut |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_renderer | 0.6500 | 0.6700 | 0.2883 | 0.0731 | 480.9300 | 843.1438 | 0.2473 | 843.3911 | 0.5700 | 0.0000 | 0.0000 | 0.2100 | 0.0200 |
| metric_path_carrier | 0.5500 | 0.5000 | 0.3359 | 0.0684 | 272.8200 | 848.1731 | 0.1433 | 848.3164 | 0.5700 | 0.0000 | 0.0000 | 0.1900 | 0.0000 |
| tree_shell1_graph_order | 0.5900 | 0.5550 | 0.2461 | 0.0665 | 493.8200 | 1039.8663 | 0.2572 | 1040.1236 | 0.5700 | 0.4500 | 0.0500 | 0.2200 | 0.0100 |
| tree_shell1_semantic_query_order | 0.7600 | 0.7800 | 0.3352 | 0.0789 | 485.2400 | 1023.0308 | 0.2451 | 1023.2759 | 0.5700 | 0.4500 | 0.3300 | 0.2000 | 0.0300 |
| tree_shell1_semantic_tree_order | 0.6600 | 0.6800 | 0.2971 | 0.0733 | 477.3700 | 1065.2311 | 0.2385 | 1065.4696 | 0.5700 | 0.4500 | 0.2000 | 0.2200 | 0.0100 |
| semantic_weighted_tree_diagnostic | 0.5900 | 0.5350 | 0.2674 | 0.0721 | 428.9800 | 2022.7616 | 0.2177 | 2022.9793 | 0.5700 | 0.0000 | 0.0000 | 0.2000 | 0.0600 |

### Decomposition Deltas

| effect | answer_in_context | rendered_recall | context_f1 | qa_f1 | tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| delta_shell_B1_minus_A1 | 0.0400 | 0.0550 | -0.0898 | -0.0018 | 221.0000 |
| delta_query_semantic_B2_minus_B1 | 0.1700 | 0.2250 | 0.0892 | 0.0124 | -8.5800 |
| delta_tree_semantic_B3_minus_B1 | 0.0700 | 0.1250 | 0.0511 | 0.0067 | -16.4500 |

### Semantic Attribution

- semantic_separation_query_current_only: 0.0605
- semantic_separation_tree_current_only: 0.0429
- semantic_separation_query_shell1: 0.0572
- semantic_separation_tree_shell1: 0.0547

| group | count | mean d_ang(q,u) | median d_ang(q,u) | mean d_ang(u,T_q) | median d_ang(u,T_q) |
| --- | ---: | ---: | ---: | ---: | ---: |
| current_only_answer | 26 | 0.3562 | 0.3522 | 0.3381 | 0.3325 |
| current_only_non_answer | 273 | 0.4167 | 0.4203 | 0.3810 | 0.3852 |
| shell1_answer | 94 | 0.4033 | 0.3974 | 0.3820 | 0.3810 |
| shell1_non_answer | 7615 | 0.4606 | 0.4639 | 0.4366 | 0.4427 |
| tree_answer | 77 | 0.3525 | 0.3469 | 0.0000 | 0.0000 |
| tree_non_answer | 375 | 0.4090 | 0.4148 | 0.0000 | 0.0000 |
| projected_nonrendered_answer | 310 | 0.4134 | 0.4142 | 0.3823 | 0.3975 |

### Pool And Bridge Diagnostics

- avg strict tree chunks: 4.5200
- avg shell1 chunks: 77.0900
- avg shell2 chunks: 0.0000
- answer on support tree rate: 0.5700
- answer in shell1 rate: 0.4500
- answer in shell2 rate: 0.0000

### Adoption Gates

- `tree_shell1_graph_order`: pass `False`, blockers `qa_f1_regression, answer_in_context_regression, rendered_recall_regression, context_f1_regression`
- `tree_shell1_semantic_query_order`: pass `True`, blockers `none`
- `tree_shell1_semantic_tree_order`: pass `False`, blockers `retrieval_time_gate`

## Cross-Dataset Interpretation

- Query shell1 attribution signs: `positive, positive`
- Tree shell1 attribution signs: `positive, positive`
- Attribution direction consistent: `True`
- Next recommendation: Treat tree_shell1_semantic_query_order as an adoption candidate, not an automatic adoption: it passed both 100-query datasets under common_qa, but should get larger-sample validation and a bridge-carrier safety check before paper-method promotion.

Expert-panel read: if B1 improves over A1 but B2/B3 do not improve over B1, the gain is graph shell expansion rather than semantic ordering. If query-semantic ordering damages bridge/path retention, it is risky for GraphRAG even when Hotpot improves. If B4 improves alone, semantic edge lengths remain diagnostic-only.
