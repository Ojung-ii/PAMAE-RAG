# Answer Carrier Diagnostic Report

- Branch: `diagnose/answer-carrier-attribution`
- Commit before report: `b308374`
- Dataset: `2wikimultihopqa`
- Runs: 50-query gate plus 100-query stability pass

## Previous Local-Surface STOP Summary

The previous local-surface experiment stopped before 100-query runs because selected medoid chunks contained answer-bearing sentences in only `0.1200` of the 2Wiki 50-query smoke. The current renderer had `0.4000` answer-in-context, while the selected-chunk answer sentence oracle also reached only `0.1200`. That ruled out "non-gold local sentence/fact rendering inside selected medoid chunks" as the primary explanation.

## Principle Check

- No retrieval objective, graph construction, medoid selection, refinement, QA prompt, generator, evaluator, or context budget was changed.
- Added tracing is diagnostic-only.
- Answer strings and gold labels are used only in diagnostics and explicitly marked oracle renderers.
- No dense reranking, BM25 reranking, LLM reranking, answer-aware retrieval, gold-aware retrieval, or new scalar score mixing was added.
- Oracle renderers are excluded from adoption decisions.

## 2Wiki 50 Gate

| metric | value |
| --- | ---: |
| current_renderer answer-in-context | 0.4000 |
| selected medoid chunk answer availability | 0.1200 |
| current_minus_medoid_answer_gap | 0.2800 |
| answer_chunk_candidate_rate | 0.8200 |
| answer_chunk_projected_rate | 0.8200 |
| answer_chunk_selected_basin_rate | 0.6800 |
| answer_chunk_support_tree_rate | 0.3600 |
| answer_chunk_current_rendered_rate | 0.4000 |
| answer_chunk_budget_cutoff_rate | 0.4600 |

Decision after 50: **GO_TO_100**. The gap was interpretable: current rendering recovered answer chunks mostly through support-tree bridge roles, with a smaller extra-nonmedoid contribution.

## 2Wiki 100 Stage Table

| metric | value |
| --- | ---: |
| current_renderer answer-in-context | 0.4300 |
| selected medoid chunk answer availability | 0.1800 |
| current_minus_medoid_answer_gap | 0.2500 |
| answer_chunk_candidate_rate | 0.7600 |
| answer_chunk_projected_rate | 0.7600 |
| answer_chunk_selected_medoid_rate | 0.1800 |
| answer_chunk_post_refine_medoid_rate | 0.1800 |
| answer_chunk_selected_basin_rate | 0.6300 |
| answer_chunk_support_tree_rate | 0.4000 |
| answer_chunk_bridge_rate | 0.2500 |
| answer_chunk_current_rendered_rate | 0.4300 |
| answer_chunk_rendered_nonmedoid_rate | 0.0800 |
| answer_chunk_budget_cutoff_rate | 0.3800 |

## Render Role Distribution

100-query answer render roles:

`{"selected_medoid": 18, "post_refine_medoid": 0, "support_tree_bridge": 26, "path_closure": 0, "basin_member": 0, "extra_nonmedoid": 8, "fallback_or_unknown": 2}`

100-query gold render roles:

`{"selected_medoid": 71, "post_refine_medoid": 0, "support_tree_bridge": 48, "path_closure": 0, "basin_member": 0, "extra_nonmedoid": 19, "fallback_or_unknown": 0}`

Interpretation: current answer recovery is not primarily selected medoids. The largest rendered answer role outside medoids is `support_tree_bridge`, and the explicit extra-nonmedoid contribution is smaller.

## Oracle Comparison

| renderer | answer_in_context | qa_f1 | avg_context_tokens |
| --- | ---: | ---: | ---: |
| current_renderer | 0.4300 | 0.0814 | 333.3900 |
| projected_answer_chunk_oracle | 0.7600 | 0.1413 | 138.8400 |
| selected_basin_answer_chunk_oracle | 0.6300 | 0.1078 | 122.0600 |
| current_answer_role_oracle | 0.4300 | 0.1160 | 43.4100 |
| gold_chunk_role_oracle | 0.4100 | 0.0885 | 101.2200 |

The projected and selected-basin answer oracles are much stronger than final medoids, so global projection often contains the carrier. The current-answer-role oracle preserves current answer coverage with far fewer tokens and better QA F1 than the full current context, suggesting current-rendered answer chunks are themselves sufficient more often than surrounding context is.

## Failure Taxonomy

100-query dominant categories:

`{"A_answer_not_in_candidate": 24, "B_answer_lost_at_projection": 0, "C_answer_lost_at_medoid_selection": 0, "D_answer_in_basin_not_medoid": 0, "E_answer_on_path_or_bridge": 19, "F_answer_rendered_nonmedoid": 6, "G_answer_budget_cutoff": 38, "H_answer_rendered_qa_fail": 4, "I_success": 9}`

The largest bucket is budget cutoff after the renderer's unbudgeted order, followed by answer-not-in-candidate. Among rendered non-medoid recoveries, path/bridge carriers dominate over generic extra nonmedoid chunks.

## Current-Minus-Medoid Gap Analysis

The stable gap is real:

- 50-query gap: `0.4000 - 0.1200 = 0.2800`
- 100-query gap: `0.4300 - 0.1800 = 0.2500`

This confirms that selected medoids are not the main source of current answer-in-context. The current renderer recovers answer carriers through rendered support-tree bridge/path-like chunks and a smaller number of extra nonmedoid chunks. The selected basin oracle also shows that many answer carriers are in selected basins but are not chosen as representatives.

## Conclusion

**PATH_CARRIER_RENDERING_BOTTLENECK**

Secondary signal: basin representative loss is also present, because selected basins contain answer chunks much more often than final medoids do. But the specific mystery this round targeted, "where does current renderer's answer-in-context come from?", is answered most directly by the support-tree bridge/path carrier role distribution.

## Next Recommended Experiment

Design a principle-based path/basin carrier renderer that keeps the entity--chunk backbone fixed and renders metric-justified carrier chunks from selected support-tree paths or selected basins under the same budget. Do not tune by answer strings. The next non-oracle candidate should formalize why path/basin chunks are included, avoid scalar score mixing, and compare against current `cell_top_rho` using answer-in-context, rendered recall, context F1, QA F1, and token count.
