# Support-Tree Order/Budget 2Wiki 50 Decision

- Branch: `diagnose/support-tree-order-budget`
- Commit: `7f1a3ab`
- Verification: `python -m compileall src tests scripts` passed; `/home/ojungii/miniconda3/envs/pamae-rag/bin/python -m pytest -q` passed with 129 tests.
- Invariants: non-oracle `metric_path_carrier` and diagnostic ablations do not use answer/gold labels; `tree_answer_oracle` is marked oracle-only; `oracle_leakage_count = 0`.

## Headline

Decision: **DIAGNOSTIC_ONLY_100**

Reason: the 2Wiki 50 current-minus-metric answer gap is interpretable. It is mostly explained by current-only non-tree recovery, with a smaller tree-order/budget residual. This is useful enough to test stability at 100 queries, but it is not adoption evidence.

## Core Metrics

| metric | value |
| --- | ---: |
| current answer-in-context | 0.4000 |
| metric_path_carrier answer-in-context | 0.3200 |
| current-minus-metric answer gap | 0.0800 |
| answer on support tree | 0.3600 |
| answer near support tree distance 1 | 0.6000 |
| answer near support tree distance 2 | 0.6000 |
| answer on tree but metric not rendered | 0.0400 |
| answer cut by metric budget | 0.0200 |
| answer current-only non-tree | 0.0800 |
| answer current-tree intersection | 0.3400 |
| answer tree-only | 0.0200 |

## Diagnostic Renderers

| renderer | answer_in_context | qa_f1 | rendered_recall | avg_context_tokens |
| --- | ---: | ---: | ---: | ---: |
| tree_all_no_budget | 0.3600 | 0.0671 | 0.5350 | 267.3000 |
| tree_current_budget_order | 0.3400 | 0.0663 | 0.4950 | 219.6600 |
| current_tree_intersection_only | 0.3400 | 0.0663 | 0.4950 | 215.6800 |
| current_only_non_tree | 0.0800 | 0.0413 | 0.0750 | 135.3600 |
| tree_answer_oracle | 0.3600 | 0.0930 | 0.1300 | 32.2600 |

## Failure Taxonomy

| type | count |
| --- | ---: |
| A_answer_not_projected | 9 |
| B_answer_not_on_tree | 20 |
| C_answer_on_tree_cut_by_budget | 1 |
| D_answer_on_tree_bad_order | 1 |
| E_current_hidden_non_tree | 4 |
| F_tree_contains_answer_current_misses | 0 |
| G_answer_rendered_qa_fail | 7 |
| H_success | 8 |

## Interpretation

`tree_all_no_budget` and `tree_answer_oracle` both cap at 0.3600 answer-in-context, below current renderer's 0.4000. That means the support tree is not merely suffering from metric-path order or budget; the current renderer has a small but real non-tree recovery component.

`current_tree_intersection_only` reaches 0.3400 and `current_only_non_tree` reaches 0.0800. Together they explain the current renderer's answer coverage. The metric renderer's 0.3200 is close to the tree-intersection result but misses a small support-tree/order residual.

## Decision

**DIAGNOSTIC_ONLY_100**

Run 2Wiki 100 and Hotpot 100 to test whether this decomposition is stable. Do not adopt any renderer from this result.
