# Answer Carrier 50 Decision

- Branch: `diagnose/answer-carrier-attribution`
- Commit: `2bb1aa9`
- Dataset: `2wikimultihopqa`
- Max queries: `50`

## Verification Status

- `python -m compileall src tests scripts`: passed
- `pytest -q`: passed, `117 passed`

## Invariant Status

- `triangle_inequality_violation_count`: `0`
- `local_objective_invalid_count`: `0`
- Oracle renderers are marked `oracle_renderer: true` and are diagnostic-only.
- Current renderer reference uses the same sample and seed schedule as the oracle diagnostic variants.
- Answer strings and gold labels are used only in diagnostics and oracle renderers.
- No new scalar score mixing, dense reranking, BM25 reranking, LLM reranking, answer-aware retrieval, or gold-aware retrieval was added.

## Current Renderer Reference

| metric | value |
| --- | ---: |
| answer_in_context | 0.4000 |
| qa_f1 | 0.0610 |
| avg_context_tokens | 351.0400 |
| selected_medoid_answer_availability | 0.1200 |
| current_minus_medoid_answer_gap | 0.2800 |

## Answer Carrier Stage Attribution

| metric | value |
| --- | ---: |
| answer_chunk_candidate_rate | 0.8200 |
| answer_chunk_projected_rate | 0.8200 |
| answer_chunk_selected_medoid_rate | 0.1200 |
| answer_chunk_post_refine_medoid_rate | 0.1200 |
| answer_chunk_selected_basin_rate | 0.6800 |
| answer_chunk_support_tree_rate | 0.3600 |
| answer_chunk_bridge_rate | 0.2400 |
| answer_chunk_current_rendered_rate | 0.4000 |
| answer_chunk_rendered_nonmedoid_rate | 0.0800 |
| answer_chunk_budget_cutoff_rate | 0.4600 |

## Renderer Role Distribution

- Answer render roles: `{"selected_medoid": 6, "post_refine_medoid": 0, "support_tree_bridge": 12, "path_closure": 0, "basin_member": 0, "extra_nonmedoid": 4, "fallback_or_unknown": 0}`
- Gold render roles: `{"selected_medoid": 34, "post_refine_medoid": 0, "support_tree_bridge": 27, "path_closure": 0, "basin_member": 0, "extra_nonmedoid": 9, "fallback_or_unknown": 0}`

## Oracle Comparison

| renderer | answer_in_context | qa_f1 | avg_context_tokens |
| --- | ---: | ---: | ---: |
| projected_answer_chunk_oracle | 0.8200 | 0.1193 | 145.6000 |
| selected_basin_answer_chunk_oracle | 0.6800 | 0.0959 | 133.0400 |
| current_answer_role_oracle | 0.4000 | 0.0923 | 39.9400 |
| gold_chunk_role_oracle | 0.4000 | 0.0575 | 114.6400 |

## Failure Taxonomy

`{"A_answer_not_in_candidate": 9, "B_answer_lost_at_projection": 0, "C_answer_lost_at_medoid_selection": 0, "D_answer_in_basin_not_medoid": 0, "E_answer_on_path_or_bridge": 9, "F_answer_rendered_nonmedoid": 3, "G_answer_budget_cutoff": 23, "H_answer_rendered_qa_fail": 3, "I_success": 3}`

## Decision

**GO_TO_100**

Exact reason: diagnostics are valid and interpretable. The current renderer's `0.4000` answer-in-context is not coming primarily from selected medoid chunks (`0.1200`). The `0.2800` gap is explained by rendered non-medoid roles, especially support-tree bridge chunks, plus a smaller extra-nonmedoid contribution. Projected and selected-basin answer oracles are much stronger than the current renderer, so a 100-query stability check is meaningful before designing the next method.
