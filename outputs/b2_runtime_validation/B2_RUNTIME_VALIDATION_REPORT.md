# B2 Runtime Validation Report

- Branch: `optimize-b2-runtime-validation`
- Commit: `5af5971`
- Fixed prompt hash: `31e4b446be8b00a4989078fb4a957bc61b19bf4b8014674e2baad4612cc4396d`
- Method: fixed `tree_shell1_semantic_query_order`; production mode may remove diagnostics only.

## Previous Robustness Context

- Previous B2 validation was stopped because Hotpot quality improved but the retrieval-time gate failed.
- This run keeps the B2 method fixed and separates diagnostic overhead from production-path overhead.

## Fixed Method Definition

- PAMAE-style entity-chunk retrieval, graph-metric medoid selection, local refinement, support-tree construction, prompt, generator, evaluator, context budget, sample order, and NV-Embed-v2 cache remain fixed.
- B2 candidate pool remains `T_q union S1`, where `T_q = SPClosure(A_q union Theta_refined)` and `S1 = {u in U_q : d_G(u,T_q)=1}`.
- B2 ordering remains lexicographic: graph role, graph distance to `T_q`, query angular-distance order, chunk id.
- No global dense retrieval, scalar score mixing, BM25/LLM reranking, semantic kNN edges, new thresholds, or shell-radius changes were introduced.

## 2wikimultihopqa

- Equivalence rendered match: `1.0000`
- Equivalence context hash match: `1.0000`
- Quality match: `not_applicable_different_sample_size`
- Gate pass: `True` blockers: `none`
- B2/current retrieval time ratio: `0.5018`
- Embedding cache: `nvidia/NV-Embed-v2` dim `4096` normalized `True`
- B2 embedding missing rate: `0.0000`
- B2 query embedding cache hit rate: `1.0000`

| variant | mode | em | qa_f1 | answer_in_context | rendered_recall | context_f1 | avg_context_tokens | retrieval_ms | generation_ms | support_tree_chunk_count | shell1_chunk_count | candidate_pool_size | rendered_shell1_chunk_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_renderer | production | 0.0020 | 0.0790 | 0.4700 | 0.5905 | 0.2668 | 350.0860 | 1835.3773 | 0.2004 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tree_shell1_semantic_query_order | diagnostic | 0.0000 | 0.0610 | 0.4800 | 0.6450 | 0.2993 | 375.9400 | 2026.8736 | 0.3728 | 4.5400 | 76.9200 | 81.4600 | 3.4600 |
| tree_shell1_semantic_query_order | production | 0.0020 | 0.0811 | 0.5060 | 0.6385 | 0.2906 | 372.0160 | 921.0132 | 0.1897 | 4.5400 | 75.3220 | 79.8620 | 3.4180 |

### Gate Checks

| check | pass |
| --- | --- |
| qa_f1 | True |
| answer_in_context | True |
| rendered_recall | True |
| context_f1 | True |
| tokens | True |
| time | True |
| prompt | True |
| oracle_leakage | True |
| score_mixing | True |
| equivalence | True |

### B2 Production Timing
- time_anchor_ms: 818.0207 ms
- time_core_retrieval_ms: 788.0101 ms

## hotpotqa

- Equivalence rendered match: `1.0000`
- Equivalence context hash match: `1.0000`
- Quality match: `not_applicable_different_sample_size`
- Gate pass: `True` blockers: `none`
- B2/current retrieval time ratio: `0.9946`
- Embedding cache: `nvidia/NV-Embed-v2` dim `4096` normalized `True`
- B2 embedding missing rate: `0.0000`
- B2 query embedding cache hit rate: `1.0000`

| variant | mode | em | qa_f1 | answer_in_context | rendered_recall | context_f1 | avg_context_tokens | retrieval_ms | generation_ms | support_tree_chunk_count | shell1_chunk_count | candidate_pool_size | rendered_shell1_chunk_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_renderer | production | 0.0020 | 0.0719 | 0.6120 | 0.6280 | 0.2685 | 480.6920 | 908.5848 | 0.2923 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tree_shell1_semantic_query_order | diagnostic | 0.0000 | 0.0878 | 0.7800 | 0.8100 | 0.3509 | 482.6000 | 948.8296 | 0.2620 | 4.3600 | 75.5800 | 79.9400 | 2.9600 |
| tree_shell1_semantic_query_order | production | 0.0020 | 0.0743 | 0.7300 | 0.7770 | 0.3366 | 480.4460 | 903.7098 | 0.2380 | 4.5240 | 78.2340 | 82.7580 | 2.7840 |

### Gate Checks

| check | pass |
| --- | --- |
| qa_f1 | True |
| answer_in_context | True |
| rendered_recall | True |
| context_f1 | True |
| tokens | True |
| time | True |
| prompt | True |
| oracle_leakage | True |
| score_mixing | True |
| equivalence | True |

### B2 Production Timing
- time_anchor_ms: 759.2522 ms
- time_core_retrieval_ms: 752.4709 ms

## Local-Minimum Guard

- Did we change the PAMAE core retrieval objective? No.
- Did we change B2 candidate pool? No.
- Did we change B2 ordering semantics? No.
- Did we use global dense retrieval? No.
- Did we use score mixing? No.
- Did production mode produce identical contexts? See equivalence rows above.
- Did B2 preserve answer coverage and rendered recall? See quality gates above.
- Did B2 satisfy the time gate? See time ratios above.
- Did optimization hide diagnostic costs rather than real retrieval costs? Production mode removes diagnostics and reports stage timings separately.

Final decision: **ADOPTION_CANDIDATE_CONFIRMED**

Next recommendation: B2 production mode preserves context and passes the retrieval-time gate.
