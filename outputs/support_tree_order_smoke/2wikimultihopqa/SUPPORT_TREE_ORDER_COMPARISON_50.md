# Support-Tree Order/Budget Diagnostic

- Branch: `diagnose/support-tree-order-budget`
- Commit: `7f1a3ab`
- 100-query gate outcome: **DIAGNOSTIC_ONLY_100**
- Final bottleneck classification: **HIDDEN_NON_TREE_RECOVERY**

Previous path-carrier summary: `metric_path_carrier` preserved the graph-metric closure boundary and reduced tokens, but lost answer-in-context and rendered recall versus `current_renderer` on both 2Wiki and Hotpot. This diagnostic decomposes that loss into support-tree membership, path role, render order, budget cutoff, and current-only hidden recovery.

Theoretical boundary: `T_q = SPClosure(A_q union Theta_refined)`. Diagnostic renderers are not adoption candidates; oracle renderers use answer strings only for upper-bound analysis.

## 2wikimultihopqa

- Decision: **DIAGNOSTIC_ONLY_100**
- Reason: current-minus-metric gap is interpretable by order/budget or hidden non-tree recovery
- Final bottleneck classification: **HIDDEN_NON_TREE_RECOVERY**
- current answer-in-context: 0.4000
- metric_path_carrier answer-in-context: 0.3200
- current-minus-metric answer gap: 0.0800
- answer on support tree rate: 0.3600
- answer near support tree distance 1 rate: 0.6000
- answer near support tree distance 2 rate: 0.6000
- answer on tree but cut by budget rate: 0.0200
- answer current-only non-tree rate: 0.0800
- answer current-tree intersection rate: 0.3400
- answer tree-only rate: 0.0200

### Diagnostic Renderer Answer-In-Context

- tree_all_no_budget: 0.3600
- tree_current_budget_order: 0.3400
- current_tree_intersection_only: 0.3400
- current_only_non_tree: 0.0800
- tree_answer_oracle: 0.3600

### Failure Taxonomy

`{"A_answer_not_projected": 9, "B_answer_not_on_tree": 20, "C_answer_on_tree_cut_by_budget": 1, "D_answer_on_tree_bad_order": 1, "E_current_hidden_non_tree": 4, "F_tree_contains_answer_current_misses": 0, "G_answer_rendered_qa_fail": 7, "H_success": 8}`

### Variant Table

| run | renderer_mode | oracle | diagnostic | answer_in_context | qa_f1 | rendered_recall | context_f1 | avg_context_tokens |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| entity_chunk_reference_current_renderer | current_renderer | False | False | 0.4000 | 0.0610 | 0.5700 | 0.2683 | 351.0400 |
| entity_chunk_reference_metric_path_carrier | metric_path_carrier | False | False | 0.3200 | 0.0693 | 0.4550 | 0.3428 | 200.7600 |
| entity_chunk_reference_tree_all_no_budget | tree_all_no_budget | False | True | 0.3600 | 0.0671 | 0.5350 | 0.3703 | 267.3000 |
| entity_chunk_reference_tree_current_budget_order | tree_current_budget_order | False | True | 0.3400 | 0.0663 | 0.4950 | 0.3486 | 219.6600 |
| entity_chunk_reference_current_tree_intersection_only | current_tree_intersection_only | False | True | 0.3400 | 0.0663 | 0.4950 | 0.3486 | 215.6800 |
| entity_chunk_reference_current_only_non_tree | current_only_non_tree | False | True | 0.0800 | 0.0413 | 0.0750 | 0.0564 | 135.3600 |
| entity_chunk_reference_tree_answer_oracle | tree_answer_oracle | True | True | 0.3600 | 0.0930 | 0.1300 | 0.5093 | 32.2600 |

## Local-Minimum Guard

- Did we simply add more chunks? Diagnostic-only variants include no-budget and subset ablations, but no renderer is proposed for adoption.
- Did we use current renderer behavior as a method? Only in diagnostic ablations that are explicitly non-adoptable.
- Did we use answer/gold outside diagnostics? No; answer strings appear only in oracle/diagnostic attribution outputs.
- Did support tree membership explain answer recovery? See the per-dataset support-tree rates above.
- Did order/budget explain metric_path_carrier failure? See budget cutoff and tree-but-not-rendered rates above.
- Did hidden non-tree chunks explain current renderer recovery? See current-only non-tree and current-vs-tree diff rates above.
- Would a corridor objective be justified, or is that premature? Treat it as premature unless distance-1/2 near-tree rates dominate the residual gap across both datasets.

Final decision: **DIAGNOSTIC_ONLY**. This round is for attribution, not adoption.
