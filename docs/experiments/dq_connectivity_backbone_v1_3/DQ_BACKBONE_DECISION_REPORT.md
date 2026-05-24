# d_q Connectivity Backbone Decision Report

## 1. Symbolic-Only Failure Recap

The minimal symbolic graph was too fragmented for retrieval. It produced many small components, so graph shortest-path distances would mostly collapse to fallback distances.

In the new diagnostics, symbolic-only still fails the connectivity gate:

| dataset | disconnected_pair_rate | largest_component_ratio | gold_support_connected_rate |
| --- | ---: | ---: | ---: |
| HotpotQA symbolic-only | 0.9014 | 0.2228 | 0.7895 |
| 2Wiki symbolic-only | 0.8770 | 0.1769 | 0.5067 |

## 2. Connectivity Diagnostics

The semantic kNN backbone fixes connectivity without using support labels, gold node ids, answer/object fields, or dataset shortcuts.

| dataset | config | disconnected | largest comp. | gold connected | avg degree | avg edges | selected |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| HotpotQA | mutual_knn4 | 0.5102 | 0.6711 | 0.9263 | 6.24 | 1657.98 | no |
| HotpotQA | mutual_knn8 | 0.1478 | 0.9217 | 0.9579 | 8.23 | 2189.25 | yes |
| HotpotQA | knn4 | 0.0063 | 0.9968 | 1.0000 | 9.58 | 2548.90 | yes |
| HotpotQA | knn8 | 0.0009 | 0.9995 | 1.0000 | 14.38 | 3823.60 | no, dense |
| 2Wiki | mutual_knn4 | 0.5350 | 0.6531 | 0.8133 | 6.48 | 1658.51 | no |
| 2Wiki | mutual_knn8 | 0.1357 | 0.9282 | 0.9733 | 8.56 | 2167.88 | yes |
| 2Wiki | knn4 | 0.0079 | 0.9959 | 1.0000 | 9.68 | 2454.84 | yes |
| 2Wiki | knn8 | 0.0014 | 0.9993 | 1.0000 | 14.39 | 3616.07 | no, dense |

Selected settings:

- `mutual_knn8`: best sparse mutual-kNN connectivity.
- `knn4`: simple directed-kNN backbone that nearly connects the graph while keeping average degree below 10.

## 3. Retrieval Results

All runs satisfy node and token budgets.

### HotpotQA

| variant | recall | precision | F1 | hit | avg nodes | recall/1k tokens | objective spearman |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| top_rho | 0.7150 | 0.2523 | 0.3697 | 0.93 | 5.80 | 1.4140 | 0.0447 |
| semantic refine_cell | 0.5600 | 0.1655 | 0.2547 | 0.82 | 6.63 | 1.1129 | 0.1208 |
| hybrid mutual_knn8 refine_cell | 0.6150 | 0.1877 | 0.2863 | 0.91 | 6.53 | 1.2215 | 0.0641 |
| graph_sp mutual_knn8 refine_cell | 0.5200 | 0.1570 | 0.2403 | 0.75 | 6.42 | 1.0259 | -0.2245 |
| hybrid knn4 refine_cell | 0.5250 | 0.1595 | 0.2436 | 0.79 | 6.44 | 1.0424 | -0.0750 |
| graph_sp knn4 refine_cell | 0.4950 | 0.1563 | 0.2366 | 0.80 | 6.33 | 0.9799 | -0.0060 |

### 2WikiMultiHopQA

| variant | recall | precision | F1 | hit | avg nodes | recall/1k tokens | objective spearman |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| top_rho | 0.6350 | 0.2732 | 0.3692 | 0.97 | 5.78 | 1.2532 | 0.0817 |
| semantic refine_cell | 0.5675 | 0.1849 | 0.2724 | 0.92 | 7.29 | 1.1673 | 0.1987 |
| hybrid mutual_knn8 refine_cell | 0.5650 | 0.1842 | 0.2723 | 0.92 | 7.16 | 1.1526 | 0.2106 |
| graph_sp mutual_knn8 refine_cell | 0.4750 | 0.1627 | 0.2366 | 0.78 | 6.79 | 0.9559 | 0.0131 |
| hybrid knn4 refine_cell | 0.5400 | 0.1787 | 0.2630 | 0.91 | 6.96 | 1.0953 | 0.2273 |
| graph_sp knn4 refine_cell | 0.5025 | 0.1755 | 0.2540 | 0.86 | 6.77 | 1.0083 | 0.1329 |

## 4. Semantic vs Graph-Aware refine_cell

HotpotQA shows a real positive signal:

- semantic refine_cell F1: 0.2547
- best graph-aware F1: 0.2863
- gain: +0.0316 F1
- recall gain: +0.0550
- hit gain: +0.09

2Wiki does not show improvement:

- semantic refine_cell F1: 0.2724
- best graph-aware F1: 0.2723
- change: effectively flat, slightly lower

Pure `graph_sp` is worse than semantic on both datasets. The useful signal is the hybrid distance, especially sparse `mutual_knn8`.

## 5. Graph-Aware refine_cell vs top_rho

Graph-aware refine_cell does not beat top-rho.

| dataset | best graph-aware F1 | top-rho F1 | remaining gap |
| --- | ---: | ---: | ---: |
| HotpotQA | 0.2863 | 0.3697 | 0.0834 |
| 2Wiki | 0.2723 | 0.3692 | 0.0969 |

The HotpotQA gap narrows compared with semantic refine-cell, but not enough to make PAMAE the leading compact method.

## 6. Subset / Failure Analysis

HotpotQA:

- top-rho success / graph fail: 4
- top-rho fail / graph success: 2
- both success: 89
- both fail: 5

2Wiki:

- top-rho success / graph fail: 7
- top-rho fail / graph success: 2
- both success: 90
- both fail: 1

There are small graph-only win subsets, but top-rho still has more unique wins and better aggregate F1.

## 7. Decision

The connectivity backbone succeeds as a graph-construction repair. It turns the symbolic graph from fragmented into a meaningful query-local graph without gold leakage.

The retrieval result is mixed:

- Positive: HotpotQA hybrid `mutual_knn8` improves over semantic refine-cell.
- Negative: 2Wiki does not improve, and top-rho remains clearly stronger on both datasets.
- Negative: pure graph shortest-path distance hurts both datasets.

Decision: the `d_q` direction has partial promise, but graph-aware distance alone is not sufficient for a paper-grade PAMAE-RAG main result yet. Continue only with hybrid sparse mutual-kNN as a diagnostic direction. Do not scale to 500/full until this signal is replicated or paired with a stronger, still gold-free evidence posterior.

## 8. Next Steps

1. Keep PAMAE objective unchanged.
2. Do not use pure `graph_sp` as the default.
3. If continuing `d_q`, focus on hybrid semantic/graph distance with sparse mutual-kNN.
4. Investigate the HotpotQA graph-only success subset before broadening experiments.
5. Consider terminal-conditioned posterior only if hybrid graph-aware `d_q` shows repeatable partial gains; otherwise demote PAMAE to diagnostic/future work.
