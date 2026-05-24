# Query Graph Diagnostic

- input: `data/processed/2wikimultihopqa/examples_100.jsonl`
- config: `configs/ablations_dq/2wikimultihopqa_dq_graph_sp_refine_cell.yaml`

| metric | value |
| --- | ---: |
| num_queries | 100 |
| avg_num_nodes | 496.21 |
| avg_num_edges | 1230.17 |
| avg_num_connected_components | 366.39 |
| avg_largest_component_ratio | 0.17692870324870516 |
| avg_disconnected_pair_rate | 0.962536393184337 |
| gold_support_connected_rate | 0.47435897435897434 |
| gold_support_avg_shortest_path_distance | 0.5648148148148148 |

## Average Edge Counts

| edge_type | count |
| --- | ---: |
| same_canonical_title | 3.0 |
| shared_query_span | 1151.37 |
| title_mention | 75.8 |

## Problematic Queries

- `83bf3b5a0bd911eba7f7acde48001122`: disconnected_pair_rate=0.9989779447965549, num_edges=107, largest_component_ratio=0.012345679012345678
- `a80d84e7096d11ebbdb0ac1f6bf848b6`: disconnected_pair_rate=0.9950293664548431, num_edges=129, largest_component_ratio=0.08665511265164645
- `2dc690ba0bdc11eba7f7acde48001122`: disconnected_pair_rate=0.9931583320351041, num_edges=94, largest_component_ratio=0.09414758269720101
- `462bb642099211ebbdb0ac1f6bf848b6`: disconnected_pair_rate=0.9982853811597682, num_edges=38, largest_component_ratio=0.02066115702479339
- `a1cdb240085811ebbd5bac1f6bf848b6`: disconnected_pair_rate=0.9948126870281194, num_edges=585, largest_component_ratio=0.07612456747404844
- `6718770a087311ebbd66ac1f6bf848b6`: disconnected_pair_rate=0.9949210475640285, num_edges=75, largest_component_ratio=0.0779896013864818
- `7f7046d308f711ebbdaaac1f6bf848b6`: disconnected_pair_rate=0.9998521331275027, num_edges=19, largest_component_ratio=0.006872852233676976
- `9fe6a6760baf11ebab90acde48001122`: disconnected_pair_rate=0.9969053934571176, num_edges=15, largest_component_ratio=0.03418803418803419
- `551e024408a611ebbd7fac1f6bf848b6`: disconnected_pair_rate=0.9986818980667839, num_edges=93, largest_component_ratio=0.026362038664323375
- `33f51d7e0bde11eba7f7acde48001122`: disconnected_pair_rate=0.9935038934805864, num_edges=125, largest_component_ratio=0.09153713298791019
