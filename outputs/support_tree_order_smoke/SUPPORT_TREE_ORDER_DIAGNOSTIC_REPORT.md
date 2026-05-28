# Support-Tree Order/Budget Diagnostic

- Branch: `diagnose/support-tree-order-budget`
- Commit: `dbc7394`
- 100-query gate outcome: **DIAGNOSTIC_ONLY_100**
- Final bottleneck classification: **HIDDEN_NON_TREE_RECOVERY**

Previous path-carrier summary: `metric_path_carrier` preserved the graph-metric closure boundary and reduced tokens, but lost answer-in-context and rendered recall versus `current_renderer` on both 2Wiki and Hotpot. This diagnostic decomposes that loss into support-tree membership, path role, render order, budget cutoff, and current-only hidden recovery.

Theoretical boundary: `T_q = SPClosure(A_q union Theta_refined)`. Diagnostic renderers are not adoption candidates; oracle renderers use answer strings only for upper-bound analysis.

## Support-Tree Definition

The support tree is the deterministic shortest-path closure over query anchor entity nodes and refined chunk medoids from the unchanged entity-chunk retriever. Only chunk nodes in `T_q` are rendered as text. Entity nodes remain structural carriers only.

This round did not change graph construction, candidate generation, medoid selection, refinement, prompt, generator, evaluator, context budget, or scalar scoring. Non-oracle diagnostics use graph membership, shortest paths, render order, and budget metadata only.

## Diagnostic Renderer Definitions

- `metric_path_carrier`: non-oracle path carrier renderer over chunk nodes in `T_q`.
- `tree_all_no_budget`: diagnostic no-budget upper bound over all chunk nodes in `T_q`.
- `tree_current_budget_order`: diagnostic tree-only renderer using current order for overlapping chunks, then deterministic tree order.
- `current_tree_intersection_only`: diagnostic renderer for `C_current intersection T_q`.
- `current_only_non_tree`: diagnostic renderer for `C_current - T_q`.
- `tree_answer_oracle`: oracle-only answer-containing chunk on `T_q`, never an adoption candidate.

## 2Wiki 50 Gate

The 2Wiki 50 smoke passed verification and invariants, then selected **DIAGNOSTIC_ONLY_100**. Current answer-in-context was 0.4000, `metric_path_carrier` was 0.3200, `tree_all_no_budget` and `tree_answer_oracle` both capped at 0.3600, and `current_only_non_tree` reached 0.0800. That made the decomposition interpretable enough to run 100-query stability checks, but not adoption evidence.

## 2wikimultihopqa

- Decision: **DIAGNOSTIC_ONLY_100**
- Reason: current-minus-metric gap is interpretable by order/budget or hidden non-tree recovery
- Final bottleneck classification: **HIDDEN_NON_TREE_RECOVERY**
- current answer-in-context: 0.4300
- metric_path_carrier answer-in-context: 0.3600
- current-minus-metric answer gap: 0.0700
- answer on support tree rate: 0.4000
- answer near support tree distance 1 rate: 0.5900
- answer near support tree distance 2 rate: 0.5900
- answer on tree but cut by budget rate: 0.0100
- answer current-only non-tree rate: 0.0800
- answer current-tree intersection rate: 0.3900
- answer tree-only rate: 0.0100

### Diagnostic Renderer Answer-In-Context

- tree_all_no_budget: 0.4000
- tree_current_budget_order: 0.3900
- current_tree_intersection_only: 0.3900
- current_only_non_tree: 0.0800
- tree_answer_oracle: 0.4000

### Failure Taxonomy

`{"A_answer_not_projected": 24, "B_answer_not_on_tree": 32, "C_answer_on_tree_cut_by_budget": 1, "D_answer_on_tree_bad_order": 2, "E_current_hidden_non_tree": 8, "F_tree_contains_answer_current_misses": 0, "G_answer_rendered_qa_fail": 14, "H_success": 19}`

### Variant Table

| run | renderer_mode | oracle | diagnostic | answer_in_context | qa_f1 | rendered_recall | context_f1 | avg_context_tokens |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| entity_chunk_reference_current_renderer | current_renderer | False | False | 0.4300 | 0.0814 | 0.5750 | 0.2646 | 333.3900 |
| entity_chunk_reference_metric_path_carrier | metric_path_carrier | False | False | 0.3600 | 0.0801 | 0.4550 | 0.3362 | 192.3300 |
| entity_chunk_reference_tree_all_no_budget | tree_all_no_budget | False | True | 0.4000 | 0.0874 | 0.5200 | 0.3614 | 241.5600 |
| entity_chunk_reference_tree_current_budget_order | tree_current_budget_order | False | True | 0.3900 | 0.0869 | 0.5000 | 0.3530 | 200.1200 |
| entity_chunk_reference_current_tree_intersection_only | current_tree_intersection_only | False | True | 0.3900 | 0.0869 | 0.5000 | 0.3532 | 197.5600 |
| entity_chunk_reference_current_only_non_tree | current_only_non_tree | False | True | 0.0800 | 0.0439 | 0.0750 | 0.0610 | 135.8300 |
| entity_chunk_reference_tree_answer_oracle | tree_answer_oracle | True | True | 0.4000 | 0.1130 | 0.1400 | 0.4817 | 33.3300 |

## hotpotqa

- Decision: **DIAGNOSTIC_ONLY_100**
- Reason: current-minus-metric gap is interpretable by order/budget or hidden non-tree recovery
- Final bottleneck classification: **HIDDEN_NON_TREE_RECOVERY**
- current answer-in-context: 0.6500
- metric_path_carrier answer-in-context: 0.5500
- current-minus-metric answer gap: 0.1000
- answer on support tree rate: 0.5700
- answer near support tree distance 1 rate: 0.8000
- answer near support tree distance 2 rate: 0.8000
- answer on tree but cut by budget rate: 0.0000
- answer current-only non-tree rate: 0.2400
- answer current-tree intersection rate: 0.5500
- answer tree-only rate: 0.0200

### Diagnostic Renderer Answer-In-Context

- tree_all_no_budget: 0.5700
- tree_current_budget_order: 0.5500
- current_tree_intersection_only: 0.5500
- current_only_non_tree: 0.2400
- tree_answer_oracle: 0.5700

### Failure Taxonomy

`{"A_answer_not_projected": 6, "B_answer_not_on_tree": 28, "C_answer_on_tree_cut_by_budget": 0, "D_answer_on_tree_bad_order": 2, "E_current_hidden_non_tree": 24, "F_tree_contains_answer_current_misses": 1, "G_answer_rendered_qa_fail": 19, "H_success": 20}`

### Variant Table

| run | renderer_mode | oracle | diagnostic | answer_in_context | qa_f1 | rendered_recall | context_f1 | avg_context_tokens |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| entity_chunk_reference_current_renderer | current_renderer | False | False | 0.6500 | 0.0731 | 0.6700 | 0.2883 | 480.9300 |
| entity_chunk_reference_metric_path_carrier | metric_path_carrier | False | False | 0.5500 | 0.0684 | 0.5000 | 0.3359 | 272.8200 |
| entity_chunk_reference_tree_all_no_budget | tree_all_no_budget | False | True | 0.5700 | 0.0650 | 0.5450 | 0.3410 | 309.0900 |
| entity_chunk_reference_tree_current_budget_order | tree_current_budget_order | False | True | 0.5500 | 0.0639 | 0.5300 | 0.3358 | 297.4800 |
| entity_chunk_reference_current_tree_intersection_only | current_tree_intersection_only | False | True | 0.5500 | 0.0644 | 0.5150 | 0.3322 | 289.1000 |
| entity_chunk_reference_current_only_non_tree | current_only_non_tree | False | True | 0.2400 | 0.0557 | 0.1550 | 0.1191 | 191.8300 |
| entity_chunk_reference_tree_answer_oracle | tree_answer_oracle | True | True | 0.5700 | 0.0813 | 0.2900 | 0.6105 | 56.1700 |

## Local-Minimum Guard

- Did we simply add more chunks? No adoption candidate did; no-budget expansion was diagnostic-only.
- Did we use current renderer behavior as a method? No. Current-order and current-only variants are explicitly diagnostic and non-adoptable.
- Did we use answer/gold outside diagnostics? No. Answer strings appear only in attribution/oracle diagnostics.
- Did support tree membership explain answer recovery? Partially. `T_q` explains 0.3900/0.5500 current-tree intersection on 2Wiki/Hotpot, but current answer-in-context is 0.4300/0.6500.
- Did order/budget explain metric_path_carrier failure? No. Budget cutoff is 0.0100 on 2Wiki and 0.0000 on Hotpot, and current-order tree rendering remains below current.
- Did hidden non-tree chunks explain current renderer recovery? Yes. `current_only_non_tree` reaches 0.0800 on 2Wiki and 0.2400 on Hotpot, while tree oracles cap below current.
- Would a corridor objective be justified, or is that premature? A corridor is now a justified next diagnostic, not an adoption step: answer-near-tree distance-1 rates are 0.5900 and 0.8000, but the radius principle still needs to be defined without tuning.

## Conclusion

Final decision: **DIAGNOSTIC_ONLY**.

Final bottleneck classification: **HIDDEN_NON_TREE_RECOVERY**.

`metric_path_carrier` is not adopted because it lowers answer-in-context and rendered recall on both datasets. The support-tree object is meaningful, but strict `T_q` cannot reproduce current recovery: `tree_answer_oracle` and `tree_all_no_budget` cap below current on both datasets. The next recommended experiment is to formalize a non-tuned metric corridor around `T_q` or otherwise explain `C_current - T_q` under a graph-metric objective before changing any production renderer.
