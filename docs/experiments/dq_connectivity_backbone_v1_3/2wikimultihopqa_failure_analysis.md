# d_q Failure Case Analysis

| group | count | avg_context_overlap | avg_query_span_count | avg_largest_component_ratio | avg_disconnected_pair_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| top_rho_success_graph_refine_cell_fail | 7 | 0.8571428571428571 | 1.2857142857142858 | 0.9380597415963652 | 0.44201489456515397 |
| top_rho_fail_graph_refine_cell_success | 2 | 0.5 | 1 | 0.9235144439455172 | 0.41973303363203257 |
| both_success | 90 | 3.1444444444444444 | 2.011111111111111 | 0.9275232935773798 | 0.3705239400653891 |
| both_fail | 1 | 1 | 3 | 0.9295774647887324 | 0.5907233524604416 |

## Sample Query IDs

- top_rho_success_graph_refine_cell_fail: `9fe6a6760baf11ebab90acde48001122`, `265daf200bdc11eba7f7acde48001122`, `1c0dd3b00bdc11eba7f7acde48001122`, `86508dc60bdc11eba7f7acde48001122`, `f05423560bda11eba7f7acde48001122`, `3bb9c0740bb011ebab90acde48001122`, `421d07aa0bb011ebab90acde48001122`
- top_rho_fail_graph_refine_cell_success: `2dc690ba0bdc11eba7f7acde48001122`, `e311a3ba0bdc11eba7f7acde48001122`
- both_success: `83bf3b5a0bd911eba7f7acde48001122`, `a80d84e7096d11ebbdb0ac1f6bf848b6`, `462bb642099211ebbdb0ac1f6bf848b6`, `a1cdb240085811ebbd5bac1f6bf848b6`, `6718770a087311ebbd66ac1f6bf848b6`, `7f7046d308f711ebbdaaac1f6bf848b6`, `551e024408a611ebbd7fac1f6bf848b6`, `33f51d7e0bde11eba7f7acde48001122`, `914b452c0bdc11eba7f7acde48001122`, `d6898f78089511ebbd75ac1f6bf848b6`
- both_fail: `62a9cd640bb011ebab90acde48001122`
