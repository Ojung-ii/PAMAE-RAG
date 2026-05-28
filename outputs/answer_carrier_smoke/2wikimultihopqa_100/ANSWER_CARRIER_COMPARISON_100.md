# Answer Carrier Comparison 50

- Branch: `diagnose/answer-carrier-attribution`
- Commit: `b308374`
- Final decision: **GO_TO_100**
- Reason: current-minus-medoid answer gap is explained by rendered non-medoid/path roles

## Required Headline Metrics

- current_renderer answer-in-context: 0.4300
- selected medoid chunk answer availability: 0.1800
- current_minus_medoid_answer_gap: 0.2500

## Answer Carrier Stage Table

| metric | value |
| --- | ---: |
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

- answer render roles: `{"basin_member": 0, "extra_nonmedoid": 8, "fallback_or_unknown": 2, "path_closure": 0, "post_refine_medoid": 0, "selected_medoid": 18, "support_tree_bridge": 26}`
- gold render roles: `{"basin_member": 0, "extra_nonmedoid": 19, "fallback_or_unknown": 0, "path_closure": 0, "post_refine_medoid": 0, "selected_medoid": 71, "support_tree_bridge": 48}`

## Failure Taxonomy

`{"A_answer_not_in_candidate": 24, "B_answer_lost_at_projection": 0, "C_answer_lost_at_medoid_selection": 0, "D_answer_in_basin_not_medoid": 0, "E_answer_on_path_or_bridge": 19, "F_answer_rendered_nonmedoid": 6, "G_answer_budget_cutoff": 38, "H_answer_rendered_qa_fail": 4, "I_success": 9}`

## Oracle QA F1

| oracle | qa_f1 |
| --- | ---: |
| projected_answer_chunk_oracle_f1 | 0.1413 |
| selected_basin_answer_chunk_oracle_f1 | 0.1078 |
| current_answer_role_oracle_f1 | 0.1160 |
| gold_chunk_role_oracle_f1 | 0.0885 |

## Runs

| run | renderer_mode | oracle | answer_in_context | qa_f1 | avg_context_tokens | triangle_inequality_violation_count | local_objective_invalid_count |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| entity_chunk_reference_current_renderer | current_renderer | False | 0.4300 | 0.0814 | 333.3900 | 0 | 0 |
| entity_chunk_reference_projected_answer_chunk_oracle | projected_answer_chunk_oracle | True | 0.7600 | 0.1413 | 138.8400 | 0 | 0 |
| entity_chunk_reference_selected_basin_answer_chunk_oracle | selected_basin_answer_chunk_oracle | True | 0.6300 | 0.1078 | 122.0600 | 0 | 0 |
| entity_chunk_reference_current_answer_role_oracle | current_answer_role_oracle | True | 0.4300 | 0.1160 | 43.4100 | 0 | 0 |
| entity_chunk_reference_gold_chunk_role_oracle | gold_chunk_role_oracle | True | 0.4100 | 0.0885 | 101.2200 | 0 | 0 |
