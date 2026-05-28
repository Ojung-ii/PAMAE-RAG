# Basin Selection Comparison

| run | F1 | oracle_gap | TypeB | selected_basin_hit | rendered_recall | context_f1 | answer_in_context | retrieval_ms | generation_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_content | 0.0731 | 0.0196 | 43 | 0.5700 | 0.6700 | 0.2883 | 0.6500 | 808.6527 | 0.3822 |
| basin_preserving_selection | 0.0708 | 0.0219 | 0 | 1.0000 | 0.6600 | 0.2851 | 0.6400 | 807.4667 | 0.2900 |
| basin_preserving_selection_plus_basin_renderer | 0.0589 | 0.0338 | 0 | 1.0000 | 0.5100 | 0.3296 | 0.5500 | 816.0091 | 0.1834 |

## Oracle Diagnostics

- gold_support_f1: `0.0927`
- answer_containing_f1: `0.3104`
- answer_copy_f1: `0.6535`
- oracle_dominance_valid: `true`
- diagnosis: `oracle_context_construction_weaker_than_answer_copy`

## Risk Gates

- `current_content`: `REFERENCE` blockers=`none`
- `basin_preserving_selection`: `STOP` blockers=`f1_regression, oracle_gap_regression, rendered_recall_regression, context_f1_regression, answer_in_context_regression`
- `basin_preserving_selection_plus_basin_renderer`: `STOP` blockers=`f1_regression, oracle_gap_regression, rendered_recall_regression, answer_in_context_regression`
