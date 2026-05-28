# Path Realizability Comparison

## hotpotqa

| run | F1 | oracle_gap | rendered_recall | context_f1 | answer_in_context | C | D | E | F | G | mean_d_medoid_gold | support_tree_rate | retrieval_ms | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| current_content_current_renderer | 0.0731 | 0.0196 | 0.6700 | 0.2883 | 0.6500 | 4 | 5 | 2 | 56 | 31 | 0.8800 | 0.8200 | 860.1028 | REFERENCE |
| current_content_gold_path_oracle_renderer | 0.0716 | 0.0211 | 0.7150 | 0.4258 | 0.7200 | 4 | 1 | 0 | 59 | 34 | 0.8800 | 0.8200 | 843.1329 | DIAGNOSTIC_ONLY |
| current_content_path_neighborhood_renderer | 0.0750 | 0.0177 | 0.6550 | 0.2834 | 0.6300 | 4 | 6 | 0 | 56 | 32 | 0.8800 | 0.8200 | 805.5300 | STOP (rendered_recall_regression, answer_in_context_regression, context_f1_regression, renderer_sparsity_not_reduced) |
| basin_preserving_selection_current_renderer | 0.0708 | 0.0219 | 0.6600 | 0.2851 | 0.6400 | 5 | 7 | 0 | 56 | 32 | 0.9900 | 0.7900 | 888.9297 | DIAGNOSTIC_ONLY |
| basin_preserving_selection_gold_path_oracle_renderer | 0.0651 | 0.0276 | 0.6900 | 0.4197 | 0.6800 | 5 | 2 | 0 | 62 | 31 | 0.9900 | 0.7900 | 801.1255 | DIAGNOSTIC_ONLY |
| basin_preserving_selection_path_neighborhood_renderer | 0.0689 | 0.0238 | 0.6300 | 0.2736 | 0.6100 | 5 | 8 | 1 | 57 | 29 | 0.9900 | 0.7900 | 836.2045 | STOP (f1_regression, oracle_gap_regression, rendered_recall_regression, answer_in_context_regression, context_f1_regression, representative_mismatch_increase, renderer_sparsity_not_reduced) |

## 2wikimultihopqa

| run | F1 | oracle_gap | rendered_recall | context_f1 | answer_in_context | C | D | E | F | G | mean_d_medoid_gold | support_tree_rate | retrieval_ms | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| current_content_current_renderer | 0.0814 | 0.0162 | 0.5750 | 0.2646 | 0.4300 | 7 | 3 | 2 | 66 | 18 | 0.8925 | 0.8800 | 915.9623 | REFERENCE |
| current_content_gold_path_oracle_renderer | 0.0790 | 0.0186 | 0.6100 | 0.4089 | 0.4700 | 7 | 2 | 0 | 68 | 19 | 0.8925 | 0.8800 | 918.0457 | DIAGNOSTIC_ONLY |
| current_content_path_neighborhood_renderer | 0.0730 | 0.0247 | 0.5625 | 0.2586 | 0.4100 | 7 | 3 | 1 | 70 | 15 | 0.8925 | 0.8800 | 977.3356 | STOP (f1_regression, oracle_gap_regression, rendered_recall_regression, answer_in_context_regression, context_f1_regression, renderer_sparsity_not_reduced) |
| basin_preserving_selection_current_renderer | 0.0821 | 0.0155 | 0.5775 | 0.2679 | 0.4400 | 6 | 5 | 1 | 68 | 19 | 0.9517 | 0.8800 | 992.0563 | DIAGNOSTIC_ONLY |
| basin_preserving_selection_gold_path_oracle_renderer | 0.0837 | 0.0139 | 0.5750 | 0.4032 | 0.4300 | 6 | 5 | 0 | 68 | 20 | 0.9517 | 0.8800 | 1019.9420 | DIAGNOSTIC_ONLY |
| basin_preserving_selection_path_neighborhood_renderer | 0.0777 | 0.0200 | 0.5650 | 0.2627 | 0.4200 | 6 | 6 | 1 | 69 | 17 | 0.9517 | 0.8800 | 1025.7305 | STOP (f1_regression, oracle_gap_regression, rendered_recall_regression, answer_in_context_regression, context_f1_regression, renderer_sparsity_not_reduced) |
