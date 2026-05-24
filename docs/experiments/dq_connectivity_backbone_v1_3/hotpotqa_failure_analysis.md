# d_q Failure Case Analysis

| group | count | avg_context_overlap | avg_query_span_count | avg_largest_component_ratio | avg_disconnected_pair_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| top_rho_success_graph_refine_cell_fail | 4 | 1.5 | 2.5 | 0.9604297184805111 | 0.25610098327314756 |
| top_rho_fail_graph_refine_cell_success | 2 | 2 | 1 | 0.9325382435912407 | 0.490379578183909 |
| both_success | 89 | 2.449438202247191 | 2.50561797752809 | 0.9186041940590449 | 0.424107236237533 |
| both_fail | 5 | 2 | 1.4 | 0.9420789977979024 | 0.40558099153131577 |

## Sample Query IDs

- top_rho_success_graph_refine_cell_fail: `5ae40d405542996836b02c28`, `5ae163b3554299422ee99678`, `5a8031c1554299485f598585`, `5ae7281c5542991e8301cb69`
- top_rho_fail_graph_refine_cell_success: `5a8efb5a55429918e830d172`, `5a857cf35542994c784ddb1d`
- both_success: `5ab69f9a554299710c8d1ef8`, `5abc030e554299642a094bdc`, `5ae6c2285542995703ce8b9a`, `5ab57fc4554299488d4d99c0`, `5ade9c9c5542997c77adee8c`, `5a749af055429979e28829b7`, `5ae1389655429901ffe4ae05`, `5adcd4325542994d58a2f6ed`, `5a85c3225542992a431d1b95`, `5ab94bc2554299743d22eacf`
- both_fail: `5abe953b5542993f32c2a170`, `5a84973e5542992a431d1a67`, `5ae526c455429908b6326529`, `5adc46665542994650320cc8`, `5ab3b42c5542992ade7c6e4f`
