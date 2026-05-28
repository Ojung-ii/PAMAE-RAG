# Basin Selection Comparison

| run | F1 | oracle_gap | TypeB | selected_basin_hit | rendered_recall | context_f1 | answer_in_context | retrieval_ms | generation_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_content | 0.0814 | 0.0162 | 35 | 0.6400 | 0.5750 | 0.2646 | 0.4300 | 903.8193 | 0.2763 |
| basin_preserving_selection | 0.0821 | 0.0155 | 0 | 1.0000 | 0.5775 | 0.2679 | 0.4400 | 894.6290 | 0.2874 |
| basin_preserving_selection_plus_basin_renderer | 0.0850 | 0.0127 | 0 | 1.0000 | 0.4925 | 0.3547 | 0.4200 | 968.6804 | 0.1487 |

## Oracle Diagnostics

- gold_support_f1: `0.0976`
- answer_containing_f1: `0.2651`
- answer_copy_f1: `0.6375`
- oracle_dominance_valid: `true`
- diagnosis: `oracle_context_construction_weaker_than_answer_copy`

## Risk Gates

- `current_content`: `REFERENCE` blockers=`none`
- `basin_preserving_selection`: `PASS` blockers=`none`
- `basin_preserving_selection_plus_basin_renderer`: `STOP` blockers=`rendered_recall_regression, answer_in_context_regression`
