# B2 Runtime Validation Report

- Branch: `optimize-b2-runtime-validation`
- Commit: `af77c17`
- Fixed prompt hash: `31e4b446be8b00a4989078fb4a957bc61b19bf4b8014674e2baad4612cc4396d`
- Method: fixed `tree_shell1_semantic_query_order`; production mode may remove diagnostics only.

## 2wikimultihopqa

- Equivalence rendered match: `1.0000`
- Equivalence context hash match: `1.0000`
- Quality match: `True`
- Gate pass: `False` blockers: `baseline_or_b2_production_missing`
- B2/current retrieval time ratio: `0.0000`

| variant | mode | em | qa_f1 | answer_in_context | rendered_recall | context_f1 | avg_context_tokens | retrieval_ms | generation_ms | support_tree_chunk_count | shell1_chunk_count | candidate_pool_size | rendered_shell1_chunk_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tree_shell1_semantic_query_order | production | 0.0000 | 0.0610 | 0.4800 | 0.6450 | 0.2993 | 375.9400 | 1956.9462 | 0.3622 | 4.5400 | 76.9200 | 81.4600 | 3.4600 |
| tree_shell1_semantic_query_order | diagnostic | 0.0000 | 0.0610 | 0.4800 | 0.6450 | 0.2993 | 375.9400 | 2026.8736 | 0.3728 | 4.5400 | 76.9200 | 81.4600 | 3.4600 |

### B2 Production Timing
- time_core_retrieval_ms: 1712.3796 ms
- time_anchor_ms: 1612.2808 ms

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

Final decision: **DIAGNOSTIC_ONLY**

Next recommendation: Equivalence diagnostics were generated; production current-vs-B2 gates were not run in this report.
