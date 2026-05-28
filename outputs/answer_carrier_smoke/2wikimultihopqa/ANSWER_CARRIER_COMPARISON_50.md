# Answer Carrier Comparison 50

- Branch: `diagnose/answer-carrier-attribution`
- Commit: `2bb1aa9`
- Final decision: **GO_TO_100**
- Reason: current-minus-medoid answer gap is explained by rendered non-medoid/path roles

## Required Headline Metrics

- current_renderer answer-in-context: 0.4000
- selected medoid chunk answer availability: 0.1200
- current_minus_medoid_answer_gap: 0.2800

## Answer Carrier Stage Table

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

## Render Role Distribution

- answer render roles: `{"basin_member": 0, "extra_nonmedoid": 4, "fallback_or_unknown": 0, "path_closure": 0, "post_refine_medoid": 0, "selected_medoid": 6, "support_tree_bridge": 12}`
- gold render roles: `{"basin_member": 0, "extra_nonmedoid": 9, "fallback_or_unknown": 0, "path_closure": 0, "post_refine_medoid": 0, "selected_medoid": 34, "support_tree_bridge": 27}`

## Failure Taxonomy

`{"A_answer_not_in_candidate": 9, "B_answer_lost_at_projection": 0, "C_answer_lost_at_medoid_selection": 0, "D_answer_in_basin_not_medoid": 0, "E_answer_on_path_or_bridge": 9, "F_answer_rendered_nonmedoid": 3, "G_answer_budget_cutoff": 23, "H_answer_rendered_qa_fail": 3, "I_success": 3}`

## Oracle QA F1

| oracle | qa_f1 |
| --- | ---: |
| projected_answer_chunk_oracle_f1 | 0.1193 |
| selected_basin_answer_chunk_oracle_f1 | 0.0959 |
| current_answer_role_oracle_f1 | 0.0923 |
| gold_chunk_role_oracle_f1 | 0.0575 |

## Runs

| run | renderer_mode | oracle | answer_in_context | qa_f1 | avg_context_tokens | triangle_inequality_violation_count | local_objective_invalid_count |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| entity_chunk_reference_current_renderer | current_renderer | False | 0.4000 | 0.0610 | 351.0400 | 0 | 0 |
| entity_chunk_reference_projected_answer_chunk_oracle | projected_answer_chunk_oracle | True | 0.8200 | 0.1193 | 145.6000 | 0 | 0 |
| entity_chunk_reference_selected_basin_answer_chunk_oracle | selected_basin_answer_chunk_oracle | True | 0.6800 | 0.0959 | 133.0400 | 0 | 0 |
| entity_chunk_reference_current_answer_role_oracle | current_answer_role_oracle | True | 0.4000 | 0.0923 | 39.9400 | 0 | 0 |
| entity_chunk_reference_gold_chunk_role_oracle | gold_chunk_role_oracle | True | 0.4000 | 0.0575 | 114.6400 | 0 | 0 |
