# Evidence Path-Realizability Diagnostic Report

## Run Identity

- Branch: `diagnose/evidence-path-realizability`
- Code commit before report: `ada298ae78827b6d39f903ce67fe69409989f231`
- Smoke root: `outputs/path_realizability_smoke`
- Tests before smoke: `python -m compileall src tests scripts` passed; `pytest -q` passed with 89 tests.
- Final decision: `STOP`

## Previous STOP Summary

The previous basin-preserving selection round showed that eligible query-anchor basins could be fully covered while HotpotQA still regressed in F1, oracle gap, selected survival, rendered recall, and answer-in-context. The failure was therefore not simply "basin uncovered"; the stricter question became whether selected representatives path-realize indispensable evidence and whether non-gold graph rendering can recover it without heuristic score mixing.

## Oracle Diagnostics

| dataset | method_f1 | gold_support_f1 | answer_containing_f1 | answer_copy_f1 | oracle_dominance_valid | diagnosis |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| hotpotqa | 0.0731 | 0.0927 | 0.3104 | 0.6535 | true | oracle_context_construction_weaker_than_answer_copy |
| 2wikimultihopqa | 0.0814 | 0.0976 | 0.2651 | 0.6375 | true | oracle_context_construction_weaker_than_answer_copy |

Answer-copy remains much stronger than method F1, so this round did not change the prompt, generator, evaluator, dataset split, or graph construction. Gold-support oracle remains weak relative to answer-containing and answer-copy oracles, so gold-support-only counterfactuals are not a reliable adoption signal.

## QA and Gate Table

| dataset | run | F1 | oracle_gap | rendered_recall | context_f1 | answer_in_context | avg_context_tokens | retrieval_ms | decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hotpotqa | current_content_current_renderer | 0.0731 | 0.0196 | 0.6700 | 0.2883 | 0.6500 | 480.93 | 860.10 | REFERENCE |
| hotpotqa | current_content_gold_path_oracle_renderer | 0.0716 | 0.0211 | 0.7150 | 0.4258 | 0.7200 | 322.58 | 843.13 | DIAGNOSTIC_ONLY |
| hotpotqa | current_content_path_neighborhood_renderer | 0.0750 | 0.0177 | 0.6550 | 0.2834 | 0.6300 | 480.67 | 805.53 | STOP |
| hotpotqa | basin_preserving_selection_current_renderer | 0.0708 | 0.0219 | 0.6600 | 0.2851 | 0.6400 | 484.59 | 888.93 | DIAGNOSTIC_ONLY |
| hotpotqa | basin_preserving_selection_gold_path_oracle_renderer | 0.0651 | 0.0276 | 0.6900 | 0.4197 | 0.6800 | 323.37 | 801.13 | DIAGNOSTIC_ONLY |
| hotpotqa | basin_preserving_selection_path_neighborhood_renderer | 0.0689 | 0.0238 | 0.6300 | 0.2736 | 0.6100 | 481.99 | 836.20 | STOP |
| 2wikimultihopqa | current_content_current_renderer | 0.0814 | 0.0162 | 0.5750 | 0.2646 | 0.4300 | 333.39 | 915.96 | REFERENCE |
| 2wikimultihopqa | current_content_gold_path_oracle_renderer | 0.0790 | 0.0186 | 0.6100 | 0.4089 | 0.4700 | 222.33 | 918.05 | DIAGNOSTIC_ONLY |
| 2wikimultihopqa | current_content_path_neighborhood_renderer | 0.0730 | 0.0247 | 0.5625 | 0.2586 | 0.4100 | 350.19 | 977.34 | STOP |
| 2wikimultihopqa | basin_preserving_selection_current_renderer | 0.0821 | 0.0155 | 0.5775 | 0.2679 | 0.4400 | 350.50 | 992.06 | DIAGNOSTIC_ONLY |
| 2wikimultihopqa | basin_preserving_selection_gold_path_oracle_renderer | 0.0837 | 0.0139 | 0.5750 | 0.4032 | 0.4300 | 241.02 | 1019.94 | DIAGNOSTIC_ONLY |
| 2wikimultihopqa | basin_preserving_selection_path_neighborhood_renderer | 0.0777 | 0.0200 | 0.5650 | 0.2627 | 0.4200 | 364.01 | 1025.73 | STOP |

## Representative Taxonomy Counts

| dataset | run | A_projection | B_basin | C_rep_mismatch | D_renderer_sparse | E_budget | F_generator | G_success |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hotpotqa | current_content_current_renderer | 0 | 2 | 4 | 5 | 2 | 56 | 31 |
| hotpotqa | current_content_gold_path_oracle_renderer | 0 | 2 | 4 | 1 | 0 | 59 | 34 |
| hotpotqa | current_content_path_neighborhood_renderer | 0 | 2 | 4 | 6 | 0 | 56 | 32 |
| hotpotqa | basin_preserving_selection_current_renderer | 0 | 0 | 5 | 7 | 0 | 56 | 32 |
| hotpotqa | basin_preserving_selection_gold_path_oracle_renderer | 0 | 0 | 5 | 2 | 0 | 62 | 31 |
| hotpotqa | basin_preserving_selection_path_neighborhood_renderer | 0 | 0 | 5 | 8 | 1 | 57 | 29 |
| 2wikimultihopqa | current_content_current_renderer | 2 | 2 | 7 | 3 | 2 | 66 | 18 |
| 2wikimultihopqa | current_content_gold_path_oracle_renderer | 2 | 2 | 7 | 2 | 0 | 68 | 19 |
| 2wikimultihopqa | current_content_path_neighborhood_renderer | 2 | 2 | 7 | 3 | 1 | 70 | 15 |
| 2wikimultihopqa | basin_preserving_selection_current_renderer | 2 | 0 | 6 | 5 | 1 | 68 | 19 |
| 2wikimultihopqa | basin_preserving_selection_gold_path_oracle_renderer | 2 | 0 | 6 | 5 | 0 | 68 | 20 |
| 2wikimultihopqa | basin_preserving_selection_path_neighborhood_renderer | 2 | 0 | 6 | 6 | 1 | 69 | 17 |

Representative mismatch does not dominate. Most unresolved cases are downstream of evidence being projected and often path-reachable, but answer-bearing context and generator behavior still fail to yield QA gains. This is consistent with the stronger oracle finding that answer-containing contexts are much stronger than gold-support contexts.

## Path-Realizability Aggregate Metrics

| dataset | run | gold_projected | gold_in_selected_basin | medoid_to_gold_path | gold_on_support_tree | gold_rendered | answer_found | answer_projected | answer_rendered |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hotpotqa | current_content_current_renderer | 0.950 | 0.800 | 0.745 | 0.545 | 0.670 | 0.940 | 0.940 | 0.650 |
| hotpotqa | current_content_gold_path_oracle_renderer | 0.950 | 0.800 | 0.745 | 0.545 | 0.715 | 0.940 | 0.940 | 0.720 |
| hotpotqa | current_content_path_neighborhood_renderer | 0.950 | 0.800 | 0.745 | 0.545 | 0.655 | 0.940 | 0.940 | 0.630 |
| hotpotqa | basin_preserving_selection_current_renderer | 0.950 | 0.970 | 0.655 | 0.515 | 0.660 | 0.940 | 0.940 | 0.640 |
| hotpotqa | basin_preserving_selection_gold_path_oracle_renderer | 0.950 | 0.970 | 0.655 | 0.515 | 0.690 | 0.940 | 0.940 | 0.680 |
| hotpotqa | basin_preserving_selection_path_neighborhood_renderer | 0.950 | 0.970 | 0.655 | 0.515 | 0.630 | 0.940 | 0.940 | 0.610 |
| 2wikimultihopqa | current_content_current_renderer | 0.829 | 0.711 | 0.569 | 0.500 | 0.561 | 0.760 | 0.760 | 0.430 |
| 2wikimultihopqa | current_content_gold_path_oracle_renderer | 0.829 | 0.711 | 0.569 | 0.500 | 0.589 | 0.760 | 0.760 | 0.470 |
| 2wikimultihopqa | current_content_path_neighborhood_renderer | 0.829 | 0.711 | 0.569 | 0.500 | 0.549 | 0.760 | 0.760 | 0.410 |
| 2wikimultihopqa | basin_preserving_selection_current_renderer | 0.829 | 0.837 | 0.545 | 0.500 | 0.561 | 0.760 | 0.760 | 0.440 |
| 2wikimultihopqa | basin_preserving_selection_gold_path_oracle_renderer | 0.829 | 0.837 | 0.545 | 0.500 | 0.561 | 0.760 | 0.760 | 0.430 |
| 2wikimultihopqa | basin_preserving_selection_path_neighborhood_renderer | 0.829 | 0.837 | 0.545 | 0.500 | 0.549 | 0.760 | 0.760 | 0.420 |

Basin-preserving selection increases selected-basin membership, especially on HotpotQA, but lowers medoid-to-gold path rate and support-tree rate. That matches the previous STOP report: basin coverage is not equivalent to representative preservation.

## Distance and Query-Level Taxonomy Analysis

The support-tree rate in this section is the query-level taxonomy hit rate: a query is counted if any gold or answer-containing diagnostic row is on the support tree. The path-realizability table above reports stricter per-gold-row rates.

| dataset | run | mean_d_medoid_gold | mean_gold_distance_percentile_within_basin | gold_on_support_tree_rate | path_exists_but_not_rendered_rate | answer_projected_not_rendered_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| hotpotqa | current_content_current_renderer | 0.8800 | 0.3530 | 0.8200 | 0.3300 | 0.2900 |
| hotpotqa | current_content_path_neighborhood_renderer | 0.8800 | 0.3530 | 0.8200 | 0.3000 | 0.3100 |
| hotpotqa | basin_preserving_selection_current_renderer | 0.9900 | 0.3501 | 0.7900 | 0.3300 | 0.3000 |
| hotpotqa | basin_preserving_selection_path_neighborhood_renderer | 0.9900 | 0.3501 | 0.7900 | 0.3200 | 0.3200 |
| 2wikimultihopqa | current_content_current_renderer | 0.8925 | 0.2896 | 0.8800 | 0.2800 | 0.3700 |
| 2wikimultihopqa | current_content_path_neighborhood_renderer | 0.8925 | 0.2896 | 0.8800 | 0.2800 | 0.3900 |
| 2wikimultihopqa | basin_preserving_selection_current_renderer | 0.9517 | 0.2869 | 0.8800 | 0.2900 | 0.3600 |
| 2wikimultihopqa | basin_preserving_selection_path_neighborhood_renderer | 0.9517 | 0.2869 | 0.8800 | 0.2900 | 0.3800 |

The mean medoid-to-gold distance increases under basin-preserving selection on both datasets. This supports the interpretation that selected-basin coverage can move representatives farther from indispensable evidence even when the basin itself is covered.

## Renderer Findings

Gold-path oracle:

- Hotpot current renderer: recall and answer coverage improve, but F1 decreases from 0.0731 to 0.0716.
- Hotpot basin renderer path oracle: recall and answer coverage improve relative to basin current, but F1 decreases from 0.0708 to 0.0651.
- 2Wiki current renderer: recall and answer coverage improve, but F1 decreases from 0.0814 to 0.0790.
- 2Wiki basin renderer path oracle: F1 rises to 0.0837, but rendered recall and answer coverage do not beat the current-content reference.

Non-gold path-neighborhood:

- Hotpot current selection: F1 rises slightly, but rendered recall, context F1, and answer-in-context all decrease. This hits the explicit stop condition.
- Hotpot basin selection: F1, oracle gap, rendered recall, context F1, and answer-in-context all regress.
- 2Wiki current selection: F1, oracle gap, rendered recall, context F1, and answer-in-context all regress.
- 2Wiki basin selection: F1, oracle gap, rendered recall, context F1, and answer-in-context all regress.

## Adoption Gate Decision

Final decision: `STOP`

Reasons:

1. HotpotQA is primary, and both path-neighborhood probes fail required gates.
2. The only Hotpot path-neighborhood F1 increase comes with lower rendered recall and lower answer-in-context.
3. Gold-path oracle does not strongly improve QA despite improving recall and answer coverage.
4. Basin-preserving selection raises selected-basin membership but increases mean medoid-to-gold distance and lowers path/support-tree realization.
5. 2Wiki does not rescue the method: path-neighborhood also regresses there.

No renderer is adopted. Gold-path oracle remains counterfactual-only. Path-neighborhood remains diagnostic-only.

## Stopped Ideas

The following were not implemented:

- Scalar score mixing of graph distance, PPR, BM25, dense scores, or degree penalties.
- Gold-aware or answer-aware retrieval logic.
- Prompt, generator, evaluator, dataset split, graph construction, or context budget changes.
- Dataset-specific thresholds or query-pattern branches.

## Next Bottleneck Recommendation

The next target should not be renderer tuning first. The evidence says:

1. Query anchor basin coverage is too weak as a representative-preservation condition.
2. Gold evidence is often projected and sometimes path-reachable, but answer-containing evidence is still frequently not rendered.
3. Gold-path oracle does not turn added evidence into strong QA gains, while answer-containing and answer-copy oracles remain much stronger.

Recommended next step:

Add diagnostics that compare gold-support context, answer-containing context, and rendered context at the sentence/span level under the fixed generator. Specifically, determine whether gold supporting chunks contain answer-bearing phrasing that the deterministic extractive generator can copy. If answer-containing context remains much stronger than gold-path context, retrieval/rendering should optimize for principle-consistent answer-bearing evidence proxies only after defining non-gold, content-derived support signals that do not use answer strings.
