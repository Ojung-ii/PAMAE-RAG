# rho_q Alignment Decision Report

Run directory: `docs/experiments/rho_alignment_20260524_105205`

This run evaluated final-safe `rho_q` modes only:

- `current`
- `title_aware`
- `entity_title_aware`
- `hybrid_title_semantic`

`diagnostic_subject_title` was not run. PAMAE objective, sample search, full-universe validation, refinement, and `d_q` were unchanged.

## Relevance Rank Diagnostics

### HotpotQA

| mode | gold_top10_rate | gold_top20_rate | gold_rank_median | mean_gold_relevance | relevance_label_auc |
| --- | ---: | ---: | ---: | ---: | ---: |
| current | 0.7950 | 0.8550 | 3.0 | 0.2971 | 0.9754 |
| title_aware | 0.7350 | 0.7950 | 2.0 | 0.3625 | 0.9431 |
| entity_title_aware | 0.6600 | 0.7100 | 2.0 | 0.4316 | 0.9366 |
| hybrid_title_semantic | 0.7350 | 0.7950 | 2.0 | 0.3625 | 0.9431 |

HotpotQA rank diagnostics favor `current` for top10/top20 and AUC. `title_aware` improves median rank and mean gold relevance, but loses top-k coverage. `entity_title_aware` raises mean gold relevance while hurting top10/top20 coverage, so it is more concentrated but less robust.

### 2WikiMultiHopQA

| mode | gold_top10_rate | gold_top20_rate | gold_rank_median | mean_gold_relevance | relevance_label_auc |
| --- | ---: | ---: | ---: | ---: | ---: |
| current | 0.6179 | 0.6585 | 3 | 0.2122 | 0.9152 |
| title_aware | 0.6382 | 0.6545 | 2 | 0.4053 | 0.8910 |
| entity_title_aware | 0.6138 | 0.6260 | 2 | 0.4714 | 0.8831 |
| hybrid_title_semantic | 0.6382 | 0.6545 | 2 | 0.4053 | 0.8910 |

2Wiki rank diagnostics are mixed. `title_aware` and `hybrid_title_semantic` improve top10 and median rank, while `current` has slightly better top20 and AUC. `entity_title_aware` again increases mean gold relevance but lowers top-k coverage.

## Retrieval Results

### HotpotQA top_rho

| mode | recall | precision | F1 | recall/1k tokens | objective_support_spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| current | 0.7150 | 0.2022 | 0.3141 | 1.5087 | -0.0531 |
| title_aware | 0.7150 | 0.2523 | 0.3697 | 1.4140 | 0.0447 |
| entity_title_aware | 0.6100 | 0.1921 | 0.2901 | 1.2167 | -0.1570 |
| hybrid_title_semantic | 0.7150 | 0.2523 | 0.3697 | 1.4140 | 0.0447 |

Best HotpotQA `top_rho` by F1: `title_aware` / `hybrid_title_semantic`, F1 0.3697.

Best HotpotQA `top_rho` by recall/1k tokens: `current`, 1.5087, but with much lower precision and F1.

### HotpotQA refine_cell

| mode | recall | precision | F1 | recall/1k tokens | objective_support_spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| current | 0.6350 | 0.1694 | 0.2670 | 1.3445 | -0.0093 |
| title_aware | 0.5600 | 0.1655 | 0.2547 | 1.1129 | 0.1208 |
| entity_title_aware | 0.5350 | 0.1532 | 0.2375 | 1.0714 | -0.1006 |
| hybrid_title_semantic | 0.5600 | 0.1655 | 0.2547 | 1.1129 | 0.1208 |

Best HotpotQA `refine_cell` by F1: `current`, F1 0.2670.

Best HotpotQA comparison:

- best `top_rho`: F1 0.3697 (`title_aware` / `hybrid_title_semantic`)
- best `refine_cell`: F1 0.2670 (`current`)
- gap: 0.1027 F1

### 2Wiki top_rho

| mode | recall | precision | F1 | recall/1k tokens | objective_support_spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| current | 0.5725 | 0.1834 | 0.2712 | 1.8973 | -0.0836 |
| title_aware | 0.6350 | 0.2732 | 0.3692 | 1.2532 | 0.0817 |
| entity_title_aware | 0.6225 | 0.2430 | 0.3364 | 1.3030 | 0.0672 |
| hybrid_title_semantic | 0.6350 | 0.2732 | 0.3692 | 1.2532 | 0.0817 |

Best 2Wiki `top_rho` by F1: `title_aware` / `hybrid_title_semantic`, F1 0.3692.

Best 2Wiki `top_rho` by recall/1k tokens: `current`, 1.8973, but with much lower precision and F1.

### 2Wiki refine_cell

| mode | recall | precision | F1 | recall/1k tokens | objective_support_spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| current | 0.5675 | 0.1720 | 0.2594 | 1.8701 | 0.2421 |
| title_aware | 0.5675 | 0.1849 | 0.2724 | 1.1673 | 0.1987 |
| entity_title_aware | 0.5800 | 0.1805 | 0.2706 | 1.2430 | 0.2069 |
| hybrid_title_semantic | 0.5675 | 0.1849 | 0.2724 | 1.1673 | 0.1987 |

Best 2Wiki `refine_cell` by F1: `title_aware` / `hybrid_title_semantic`, F1 0.2724.

Best 2Wiki comparison:

- best `top_rho`: F1 0.3692 (`title_aware` / `hybrid_title_semantic`)
- best `refine_cell`: F1 0.2724 (`title_aware` / `hybrid_title_semantic`)
- gap: 0.0968 F1

## Judgment Criteria

### A. Does improved rho_q beat existing title_aware top_rho?

No by F1.

- HotpotQA: `title_aware` / `hybrid_title_semantic` remain best by F1 at 0.3697.
- 2Wiki: `title_aware` / `hybrid_title_semantic` remain best by F1 at 0.3692.

`current` improves recall/1k tokens on both datasets, but this comes with a large precision/F1 drop. This is not a clean relevance improvement for compact retrieval.

### B. Does improved rho_q reduce the refine_cell vs top_rho gap?

Only slightly, and not enough.

- HotpotQA: best gap is 0.1027 F1 (`title_aware` top_rho 0.3697 vs `current` refine_cell 0.2670).
- 2Wiki: best gap is 0.0968 F1 (`title_aware` top_rho 0.3692 vs `title_aware` refine_cell 0.2724).

`refine_cell` remains clearly behind `top_rho`.

### C. Does objective_support_spearman improve?

Mixed, but not decisive.

- HotpotQA best Spearman remains `title_aware` / `hybrid_title_semantic` refine_cell at 0.1208.
- 2Wiki best Spearman is `current` refine_cell at 0.2421.

The Spearman gains do not translate into `refine_cell` beating or approaching `top_rho` on F1.

### D. Does gold top10/top20 improve?

No clear new-mode win.

- HotpotQA: `current` has the best top10/top20 and AUC.
- 2Wiki: `title_aware` has the best top10, but `current` has slightly better top20 and AUC.
- `entity_title_aware` does not improve top10/top20 on either dataset.
- `hybrid_title_semantic` equals `title_aware`, consistent with query embeddings being unavailable and semantic fallback taking effect.

## Decision

The tested gold-free `rho_q` modes do not provide a clean improvement over the existing `title_aware` compact baseline.

`entity_title_aware` is not ready: it increases mean gold relevance but reduces top-k coverage and hurts retrieval F1. `hybrid_title_semantic` is currently a title-aware fallback because query embeddings are unavailable in the processed examples.

Recommended next step: move to `d_q` graph metric work, or first revise query-title/entity grounding before running more PAMAE component ablations. Do not scale these rho modes to 500-query or full-dataset runs yet.
