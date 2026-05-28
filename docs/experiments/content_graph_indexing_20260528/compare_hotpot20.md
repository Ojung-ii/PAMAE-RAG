| run | graph_mode | oracle | candidate_recall | projected_recall | post_refine_recall | rendered_recall | context_f1 | avg_context_tokens | retrieval_ms | generation_ms | EM | F1 | oracle_gap | risk_decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline_hotpot20 | legacy_hybrid_sem_graph | false | 0.9750 | n/a | 0.1500 | 0.7250 | 0.3327 | 504.3500 | 291.9184 | 0.2916 | 0.0000 | 0.0672 | 0.0254 | measurement_only |
| oracle_hotpot20 | direct_gold_context | true | n/a | n/a | n/a | 1.0000 | 1.0000 | 56.1500 | 0.0000 | 0.0631 | 0.0000 | 0.0925 | 0.0000 | measurement_only |
| content_graph_hotpot20 | content_hybrid_sem_graph | false | 0.9750 | 0.9250 | 0.3750 | 0.6750 | 0.2871 | 478.8000 | 638.9325 | 0.2762 | 0.0000 | 0.0624 | 0.0301 | no_adoption |

- oracle_context_complete: `true`
- oracle_answer_coverage: `0.9000`
- oracle_selected_answer_coverage: `0.3500`
- qa_settings_consistent: `true`
- oracle_dominance_valid: `true`
- dominance_violations: `none`

- adoption_gate[baseline_hotpot20]: `false` blockers=`reference_run`
- adoption_gate[content_graph_hotpot20]: `false` blockers=`f1_not_improved, oracle_gap_not_reduced, selected_answer_coverage_regression, rendered_recall_regression, context_f1_regression`
