# Query Graph Diagnostic

- input: `data/processed/hotpotqa/examples_100.jsonl`
- config: `configs/ablations_dq/hotpotqa_dq_graph_sp_refine_cell.yaml`

| metric | value |
| --- | ---: |
| num_queries | 100 |
| avg_num_nodes | 531.31 |
| avg_num_edges | 1209.47 |
| avg_num_connected_components | 361.58 |
| avg_largest_component_ratio | 0.2227879463733904 |
| avg_disconnected_pair_rate | 0.9642475232324385 |
| gold_support_connected_rate | 0.7684210526315789 |
| gold_support_avg_shortest_path_distance | 0.5616438356164384 |

## Average Edge Counts

| edge_type | count |
| --- | ---: |
| same_canonical_title | 3.92 |
| shared_query_span | 1074.03 |
| title_mention | 131.52 |

## Problematic Queries

- `5abe953b5542993f32c2a170`: disconnected_pair_rate=0.9983864316833103, num_edges=107, largest_component_ratio=0.02936096718480138
- `5ab69f9a554299710c8d1ef8`: disconnected_pair_rate=0.9917891228601727, num_edges=152, largest_component_ratio=0.12
- `5abc030e554299642a094bdc`: disconnected_pair_rate=0.9769912938346389, num_edges=1739, largest_component_ratio=0.25573192239858905
- `5ae6c2285542995703ce8b9a`: disconnected_pair_rate=0.9896395981196424, num_edges=202, largest_component_ratio=0.17162872154115585
- `5ab57fc4554299488d4d99c0`: disconnected_pair_rate=0.9992195901311949, num_edges=84, largest_component_ratio=0.012259194395796848
- `5ade9c9c5542997c77adee8c`: disconnected_pair_rate=0.9986765158510311, num_edges=43, largest_component_ratio=0.024930747922437674
- `5a749af055429979e28829b7`: disconnected_pair_rate=0.9935991120155397, num_edges=259, largest_component_ratio=0.09649122807017543
- `5ae1389655429901ffe4ae05`: disconnected_pair_rate=0.994905269861487, num_edges=328, largest_component_ratio=0.08916083916083917
- `5adcd4325542994d58a2f6ed`: disconnected_pair_rate=0.9861111111111112, num_edges=28, largest_component_ratio=0.07407407407407407
- `5a85c3225542992a431d1b95`: disconnected_pair_rate=0.9960503594479077, num_edges=161, largest_component_ratio=0.05244755244755245
