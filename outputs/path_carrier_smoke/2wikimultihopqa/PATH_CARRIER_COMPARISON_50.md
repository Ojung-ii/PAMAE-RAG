# Path Carrier Completion Diagnostic

- Branch: `experiment/path-carrier-completion`
- Commit: `30f1931`
- Final decision: **DIAGNOSTIC_ONLY_100**

PAMAE principle check: Phase I/II retrieval is unchanged. The non-oracle renderer uses `SPClosure(A_q + Theta_refined)`, graph shortest-path distance, deterministic ordering, and the existing context budget. It does not use scalar score mixing, dense/BM25/LLM reranking, answer-aware retrieval, or gold-aware retrieval.

## 2wikimultihopqa

- Decision: **DIAGNOSTIC_ONLY_100**
- Reason: support tree explains the gap, but deterministic rendering order/budget loses answers
- current answer-in-context: 0.4000
- selected medoid answer availability: 0.1200
- current-minus-medoid answer gap: 0.2800
- answer on refined support tree: 0.3600
- answer rendered by metric path carrier: 0.3200
- current-minus-metric-path answer gap: 0.0800

### Answer Role Distribution

`{"anchor_medoid_path": 16, "budget_cutoff": 1, "current_only_hidden_recovery": 4, "medoid": 6, "medoid_medoid_path": 3}`

### Path Carrier Failure Taxonomy

`{"A_answer_not_projected": 9, "B_answer_not_on_support_tree": 23, "C_answer_on_tree_not_rendered": 1, "D_answer_rendered_by_path_carrier": 0, "E_current_only_hidden_recovery": 1, "F_metric_path_adds_answer": 0, "G_answer_rendered_qa_fail": 7, "H_success": 9}`

### Variant Table

| run | renderer_mode | oracle | answer_in_context | qa_f1 | rendered_recall | context_f1 | avg_context_tokens | retrieval_ms | triangle | oracle_leakage |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| entity_chunk_reference_current_renderer | current_renderer | False | 0.4000 | 0.0610 | 0.5700 | 0.2683 | 351.0400 | 924.8146 | 0 | 0 |
| entity_chunk_reference_metric_path_carrier | metric_path_carrier | False | 0.3200 | 0.0693 | 0.4550 | 0.3428 | 200.7600 | 939.3328 | 0 | 0 |
| entity_chunk_reference_metric_path_carrier_no_medoids | metric_path_carrier_no_medoids | False | 0.2200 | 0.0516 | 0.1700 | 0.2916 | 95.6600 | 893.4477 | 0 | 0 |
| entity_chunk_reference_metric_path_carrier_medoids_first | metric_path_carrier_medoids_first | False | 0.3200 | 0.0693 | 0.4550 | 0.3428 | 200.7600 | 910.0448 | 0 | 0 |
| entity_chunk_reference_current_answer_role_oracle | current_answer_role_oracle | True | 0.4000 | 0.0923 | 0.1500 | 0.5217 | 39.9400 | 906.2944 | 0 | 0 |
| entity_chunk_reference_support_tree_answer_oracle | support_tree_answer_oracle | True | 0.3600 | 0.0930 | 0.1300 | 0.5093 | 32.2600 | 961.1939 | 0 | 0 |

### Oracle Comparison

| oracle | qa_f1 | answer_in_context | avg_context_tokens |
| --- | ---: | ---: | ---: |
| current_answer_role_oracle | 0.0923 | 0.4000 | 39.9400 |
| support_tree_answer_oracle | 0.0930 | 0.3600 | 32.2600 |
