# Connectivity Backbone Failure Analysis

Failure analysis compares:

- top-rho baseline
- semantic refine-cell baseline
- best graph-aware refine-cell per dataset

Best graph-aware setting used here:

- HotpotQA: `hybrid_mutual_knn8_07sem_03graph`
- 2WikiMultiHopQA: `hybrid_mutual_knn8_07sem_03graph`

## HotpotQA

| group | count | avg_context_overlap | avg_query_span_count | avg_largest_component_ratio | avg_disconnected_pair_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| top_rho_success_graph_refine_cell_fail | 4 | 1.50 | 2.50 | 0.9604 | 0.2561 |
| top_rho_fail_graph_refine_cell_success | 2 | 2.00 | 1.00 | 0.9325 | 0.4904 |
| both_success | 89 | 2.45 | 2.51 | 0.9186 | 0.4241 |
| both_fail | 5 | 2.00 | 1.40 | 0.9421 | 0.4056 |

HotpotQA has a small positive subset: graph-aware refine-cell succeeds on 2 queries where top-rho fails. The larger pattern still favors top-rho, which succeeds on 4 queries where graph-aware refine-cell fails.

## 2WikiMultiHopQA

| group | count | avg_context_overlap | avg_query_span_count | avg_largest_component_ratio | avg_disconnected_pair_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| top_rho_success_graph_refine_cell_fail | 7 | 0.86 | 1.29 | 0.9381 | 0.4420 |
| top_rho_fail_graph_refine_cell_success | 2 | 0.50 | 1.00 | 0.9235 | 0.4197 |
| both_success | 90 | 3.14 | 2.01 | 0.9275 | 0.3705 |
| both_fail | 1 | 1.00 | 3.00 | 0.9296 | 0.5907 |

2Wiki also has 2 graph-only wins, but top-rho has more unique wins and much higher precision/F1 overall.

Detailed per-dataset files:

- `hotpotqa_failure_analysis.md`
- `2wikimultihopqa_failure_analysis.md`
