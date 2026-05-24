# PAMAE-RAG v1 Compact Negative Result

This note freezes the conclusion from the compact nightly run on branch
`experiment/nightly_pamae_compact_20260524_024050`.

## Support Label Repair

HotpotQA and 2WikiMultiHopQA support labels were repaired before component ablations were interpreted.

| dataset | gold_total | gold_universe_recall | all_gold_covered_ratio | status |
| --- | ---: | ---: | ---: | --- |
| HotpotQA | 200 | 0.9800 | 0.9600 | repaired and evaluation-capable |
| 2WikiMultiHopQA | 246 | 0.8740 | 0.7100 | repaired, universe-limited |

2Wiki remains partially limited by candidate universe construction, but the previous `gold_total=0` adapter failure is gone.

## Budget Validity

All compact variants satisfied both node and token budgets after the strict renderer budget fix.

| dataset | context_node_budget_satisfied_rate | context_token_budget_satisfied_rate |
| --- | ---: | ---: |
| HotpotQA compact component grid | 1.0 | 1.0 |
| 2Wiki compact component grid | 1.0 | 1.0 |
| MuSiQue top-rho budget check | 1.0 | 1.0 |

The compact comparison is therefore not confounded by one method using a wider context.

## HotpotQA Component Results

| variant | recall | precision | F1 | recall/node | recall/1k tokens | avg nodes | avg tokens | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| top_rho | 0.7150 | 0.2523 | 0.3697 | 0.1262 | 1.4140 | 5.80 | 505.99 | 0.0447 |
| sample_only | 0.6150 | 0.1739 | 0.2702 | 0.0870 | 1.2267 | 7.10 | 502.56 | -0.0934 |
| full_validation | 0.5950 | 0.1698 | 0.2635 | 0.0849 | 1.1848 | 7.00 | 504.05 | -0.1009 |
| refine_cell | 0.5600 | 0.1655 | 0.2547 | 0.0828 | 1.1129 | 6.63 | 504.64 | 0.1208 |

## 2WikiMultiHopQA Component Results

| variant | recall | precision | F1 | recall/node | recall/1k tokens | avg nodes | avg tokens | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| top_rho | 0.6350 | 0.2732 | 0.3692 | 0.1168 | 1.2532 | 5.78 | 507.77 | 0.0817 |
| sample_only | 0.5900 | 0.1836 | 0.2746 | 0.0773 | 1.2420 | 7.50 | 485.55 | 0.2214 |
| full_validation | 0.5800 | 0.1854 | 0.2755 | 0.0776 | 1.2111 | 7.41 | 488.92 | 0.2229 |
| refine_cell | 0.5675 | 0.1849 | 0.2724 | 0.0774 | 1.1673 | 7.29 | 494.85 | 0.1987 |

## Interpretation

`top_rho` is stronger than `refine_cell` on both repaired datasets. The simplest explanation is that current `rho_q` ranks support evidence better than the medoid objective can recover after clustering. PAMAE refinement does reduce the PAMAE objective, but support F1 drops. That means the objective is internally consistent but weakly aligned with support retrieval quality under the current relevance mass and distance.

This is not a renderer or budget-control failure. It is an alignment failure between:

- `rho_q`, the query-conditioned relevance mass used by the objective.
- `d_q`, the query-conditioned distance matrix used for medoid coverage.
- downstream support evidence quality.

## Why 500/Full Runs Were Held

The 100-query compact result already gives a stable decision: Case C on both HotpotQA and 2Wiki. Scaling this exact v1 setting would mostly spend time confirming that compact top-rho is better. The correct next step is to improve `rho_q` and then revisit `d_q`, not to push a known-losing v1 configuration into larger runs.

## Conclusion

PAMAE-RAG v1 scaffold succeeded:

- schema and evaluator are aligned,
- support labels are repaired,
- compact budgets are enforced,
- renderer variants are comparable,
- component ablations are reproducible.

But with the current `rho_q/d_q`, compact top-rho is superior to PAMAE refinement. v1.1 should focus on gold-free relevance alignment diagnostics and improved relevance modes before any 500-query or full-dataset expansion.

## Directions To Avoid

Do not repair this by adding:

- bridge bonuses inside the PAMAE objective,
- dataset-specific boosting,
- gold leakage through labels, support flags, answers, objects, or oracle evidence,
- early terminal posterior integration,
- refinement-only heuristic acceptance rules,
- support/gold-aware graph edges.

The next branch should keep the PAMAE core fixed and improve the fixed input signals.
