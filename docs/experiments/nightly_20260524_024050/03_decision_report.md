# Decision Report

Generated during the 2026-05-24 nightly compact run on branch `experiment/nightly_pamae_compact_20260524_024050`.

## Gold Universe Status

| dataset | num_queries | gold_total | gold_in_nodes | gold_universe_recall | all_gold_covered_ratio | status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| hotpotqa | 100 | 200 | 196 | 0.9800 | 0.9600 | evaluation-capable |
| 2wikimultihopqa | 100 | 246 | 215 | 0.8740 | 0.7100 | evaluation-capable, universe-limited |

HotpotQA and 2Wiki support label extraction are no longer failing with `gold_total=0`. The 2Wiki universe remains partially limited by candidate construction, but no gold evidence was injected into the retrieval universe.

## Budget Validity

| dataset | variant set | context_node_budget_satisfied_rate | context_token_budget_satisfied_rate | decision |
| --- | --- | ---: | ---: | --- |
| musique | top_rho_compact rerun | 1.0 | 1.0 | budget enforcement repaired |
| hotpotqa | component compact grid | 1.0 for all variants | 1.0 for all variants | valid compact comparison |
| 2wikimultihopqa | component compact grid | 1.0 for all variants | 1.0 for all variants | valid compact comparison |

## Component Comparisons

### HotpotQA

| comparison | result |
| --- | --- |
| `refine_cell_compact` vs `top_rho_compact` | Case C: `top_rho` is clearly better. F1 0.3697 vs 0.2547, recall 0.7150 vs 0.5600, recall/node 0.1262 vs 0.0828, recall/1k tokens 1.4140 vs 1.1129. |
| `full_validation` vs `sample_only` | `sample_only` is better on recall and F1: 0.6150/0.2702 vs 0.5950/0.2635. |
| `refine` vs `full_validation` | Refinement reduces objective and context size but hurts support metrics: F1 0.2547 vs 0.2635. |
| `refine_cell` vs `refine` | Same output under this compact config. |
| objective-support alignment | Weak. Top-rho Spearman is 0.0447; full validation and sample-only are negative; refined variants are only 0.1208. |

### 2WikiMultiHopQA

| comparison | result |
| --- | --- |
| `refine_cell_compact` vs `top_rho_compact` | Case C: `top_rho` is clearly better. F1 0.3692 vs 0.2724, recall 0.6350 vs 0.5675, recall/node 0.1168 vs 0.0774, recall/1k tokens 1.2532 vs 1.1673. |
| `full_validation` vs `sample_only` | Mixed: `sample_only` has higher recall, while `full_validation` has slightly higher F1. |
| `refine` vs `full_validation` | Refinement reduces objective and context size but hurts support metrics: F1 0.2724 vs 0.2755. |
| `refine_cell` vs `refine` | Same output under this compact config. |
| objective-support alignment | Positive but not decisive. Refined variants reach 0.1987, but top-rho still wins support metrics. |

## Decision

Case C holds on both HotpotQA and 2Wiki: `top_rho_compact` is clearly better than `refine_cell_compact` on F1 and recall-per-node, while all compact budgets are satisfied.

Do not launch 500-query or full-dataset runs from this v1 compact configuration before improving the retrieval signal. The current result says the clean PAMAE v1 scaffold is useful for controlled comparison, but support retrieval quality is still dominated by the relevance mass ranking. The next work should improve `rho_q` and then revisit `d_q`; graph-aware distance and terminal-conditioned posterior remain next-step ideas, not overnight changes.

## Sensitivity Decision

k-sensitivity was skipped. It was before the 08:20 cutoff, but the component grids already met Case C on both repaired datasets. Running sensitivity would consume time without addressing the main failure mode: objective/support alignment is too weak for PAMAE refinement to beat the compact top-rho baseline.
