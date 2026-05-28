# Path Carrier Completion Diagnostic

- Branch: `experiment/path-carrier-completion`
- Commit: `c3c87b1`
- 100-query gate outcome: **DIAGNOSTIC_ONLY_100**
- Final decision: **DIAGNOSTIC_ONLY**

PAMAE principle check: Phase I/II retrieval is unchanged. The non-oracle renderer uses `SPClosure(A_q + Theta_refined)`, graph shortest-path distance, deterministic ordering, and the existing context budget. It does not use scalar score mixing, dense/BM25/LLM reranking, answer-aware retrieval, or gold-aware retrieval.

Formal support-tree definition: `T_q = SPClosure(A_q union Theta_refined)`, where `A_q` is the query anchor entity set, `Theta_refined` is the refined chunk medoid set from the unchanged entity-chunk retriever, and `SPClosure` is deterministic shortest-path closure under the graph metric. `T_q` may contain entity and chunk nodes, but only chunk nodes are rendered as text.

Previous answer-carrier attribution summary: on the prior 2Wiki diagnostic, current answer-in-context was `0.4300`, selected medoid answer availability was `0.1800`, and the current-minus-medoid gap was `0.2500`. This experiment tested whether metric support-tree carriers explain that gap without copying hidden current-renderer behavior.

2Wiki 50 decision: **DIAGNOSTIC_ONLY_100**. The 50-query run was interpretable and invariant-clean, so 100-query 2Wiki and Hotpot runs were used only to test stability, not adoption.

## 2wikimultihopqa

- Decision: **DIAGNOSTIC_ONLY_100**
- Reason: support tree explains the gap, but deterministic rendering order/budget loses answers
- current answer-in-context: 0.4300
- selected medoid answer availability: 0.1800
- current-minus-medoid answer gap: 0.2500
- answer on refined support tree: 0.4000
- answer rendered by metric path carrier: 0.3600
- current-minus-metric-path answer gap: 0.0700

### Answer Role Distribution

`{"anchor_medoid_path": 36, "budget_cutoff": 1, "current_only_hidden_recovery": 7, "medoid": 18, "medoid_medoid_path": 12}`

### Path Carrier Failure Taxonomy

`{"A_answer_not_projected": 24, "B_answer_not_on_support_tree": 36, "C_answer_on_tree_not_rendered": 1, "D_answer_rendered_by_path_carrier": 0, "E_current_only_hidden_recovery": 3, "F_metric_path_adds_answer": 0, "G_answer_rendered_qa_fail": 15, "H_success": 21}`

### Variant Table

| run | renderer_mode | oracle | answer_in_context | qa_f1 | rendered_recall | context_f1 | avg_context_tokens | retrieval_ms | triangle | oracle_leakage |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| entity_chunk_reference_current_renderer | current_renderer | False | 0.4300 | 0.0814 | 0.5750 | 0.2646 | 333.3900 | 937.8755 | 0 | 0 |
| entity_chunk_reference_metric_path_carrier | metric_path_carrier | False | 0.3600 | 0.0801 | 0.4550 | 0.3362 | 192.3300 | 910.9705 | 0 | 0 |
| entity_chunk_reference_metric_path_carrier_no_medoids | metric_path_carrier_no_medoids | False | 0.2200 | 0.0537 | 0.1550 | 0.2626 | 85.9600 | 949.4766 | 0 | 0 |
| entity_chunk_reference_metric_path_carrier_medoids_first | metric_path_carrier_medoids_first | False | 0.3600 | 0.0801 | 0.4550 | 0.3362 | 192.3300 | 938.5374 | 0 | 0 |
| entity_chunk_reference_current_answer_role_oracle | current_answer_role_oracle | True | 0.4300 | 0.1160 | 0.1525 | 0.4837 | 43.4100 | 962.8197 | 0 | 0 |
| entity_chunk_reference_support_tree_answer_oracle | support_tree_answer_oracle | True | 0.4000 | 0.1130 | 0.1400 | 0.4817 | 33.3300 | 883.3550 | 0 | 0 |

### Oracle Comparison

| oracle | qa_f1 | answer_in_context | avg_context_tokens |
| --- | ---: | ---: | ---: |
| current_answer_role_oracle | 0.1160 | 0.4300 | 43.4100 |
| support_tree_answer_oracle | 0.1130 | 0.4000 | 33.3300 |

## hotpotqa

- Decision: **DIAGNOSTIC_ONLY_100**
- Reason: support tree explains the gap, but deterministic rendering order/budget loses answers
- current answer-in-context: 0.6500
- selected medoid answer availability: 0.4100
- current-minus-medoid answer gap: 0.2400
- answer on refined support tree: 0.5700
- answer rendered by metric path carrier: 0.5500
- current-minus-metric-path answer gap: 0.1000

### Answer Role Distribution

`{"anchor_medoid_path": 52, "budget_cutoff": 0, "current_only_hidden_recovery": 11, "medoid": 41, "medoid_medoid_path": 26}`

### Path Carrier Failure Taxonomy

`{"A_answer_not_projected": 6, "B_answer_not_on_support_tree": 37, "C_answer_on_tree_not_rendered": 0, "D_answer_rendered_by_path_carrier": 0, "E_current_only_hidden_recovery": 2, "F_metric_path_adds_answer": 1, "G_answer_rendered_qa_fail": 23, "H_success": 31}`

### Variant Table

| run | renderer_mode | oracle | answer_in_context | qa_f1 | rendered_recall | context_f1 | avg_context_tokens | retrieval_ms | triangle | oracle_leakage |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| entity_chunk_reference_current_renderer | current_renderer | False | 0.6500 | 0.0731 | 0.6700 | 0.2883 | 480.9300 | 782.6615 | 0 | 0 |
| entity_chunk_reference_metric_path_carrier | metric_path_carrier | False | 0.5500 | 0.0684 | 0.5000 | 0.3359 | 272.8200 | 780.8693 | 0 | 0 |
| entity_chunk_reference_metric_path_carrier_no_medoids | metric_path_carrier_no_medoids | False | 0.1900 | 0.0398 | 0.1550 | 0.2215 | 84.5200 | 781.8008 | 0 | 0 |
| entity_chunk_reference_metric_path_carrier_medoids_first | metric_path_carrier_medoids_first | False | 0.5500 | 0.0684 | 0.5000 | 0.3359 | 272.8200 | 778.9641 | 0 | 0 |
| entity_chunk_reference_current_answer_role_oracle | current_answer_role_oracle | True | 0.6500 | 0.1020 | 0.3650 | 0.6293 | 72.3300 | 790.6076 | 0 | 0 |
| entity_chunk_reference_support_tree_answer_oracle | support_tree_answer_oracle | True | 0.5700 | 0.0813 | 0.2900 | 0.6105 | 56.1700 | 819.3141 | 0 | 0 |

### Oracle Comparison

| oracle | qa_f1 | answer_in_context | avg_context_tokens |
| --- | ---: | ---: | ---: |
| current_answer_role_oracle | 0.1020 | 0.6500 | 72.3300 |
| support_tree_answer_oracle | 0.0813 | 0.5700 | 56.1700 |

## Adoption Decision

Do not adopt `metric_path_carrier`. It preserved the PAMAE boundary and reduced token count, but it failed the non-oracle adoption gates on both datasets: answer-in-context and rendered recall were lower than `current_renderer`, and QA F1 did not improve.

The support tree explains much of the current-minus-medoid gap, but not all of current renderer recovery. On 2Wiki, refined support-tree answer coverage was `0.4000` versus current `0.4300`; on Hotpot it was `0.5700` versus current `0.6500`. The remaining gap is consistent with current-only hidden recovery and extra non-medoid chunks rather than pure `SPClosure(A_q union Theta_refined)`.

## Local-Minimum Guard

- Did we simply add more chunks? No. The non-oracle renderer used the fixed metric closure `SPClosure(A_q union Theta_refined)` and kept the existing context budget.
- Did we use only graph metric closure? Yes for the non-oracle path carrier renderer; answer strings appeared only in oracle diagnostics.
- Did support-tree carriers explain the current answer recovery? Mostly, but not completely.
- Did the method preserve answer coverage and rendered recall? No. It lowered both on 2Wiki and Hotpot.
- Did it remain stable on Hotpot? Diagnostically yes, but Hotpot repeated the same coverage loss.
- Did it reduce hidden renderer behavior? Partially. It exposed the residual current-only recovery rather than reproducing it.
- Did token count stay controlled? Yes. Tokens dropped from `333.39` to `192.33` on 2Wiki and from `480.93` to `272.82` on Hotpot.

Conclusion: **IMPLICIT_RENDERER_HEURISTIC_BOTTLENECK**. Metric support-tree carrier completion is a valid diagnostic object, but the current renderer's best answer recovery is not fully explained by refined anchor-medoid support-tree closure. There is also a secondary path-carrier rendering bottleneck: deterministic `SPClosure` rendering loses answer coverage and rendered recall relative to the current renderer. Next recommended experiment: characterize the residual current-only extra non-medoid chunks and determine whether they can be formalized as metric-constrained terminal/carrier additions, before changing retrieval or adopting a renderer.
