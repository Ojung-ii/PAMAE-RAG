| run | graph_mode | oracle | candidate_recall | projected_recall | post_refine_recall | rendered_recall | context_f1 | avg_context_tokens | retrieval_ms | generation_ms | EM | F1 | oracle_gap | risk_decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline_2wiki20 | legacy_hybrid_sem_graph | false | 0.8625 | n/a | 0.0250 | 0.5000 | 0.2347 | 497.1000 | 311.7705 | 0.4016 | 0.0000 | 0.0355 | 0.0140 | measurement_only |
| oracle_2wiki20 | direct_gold_context | true | n/a | n/a | n/a | 1.0000 | 1.0000 | 47.4000 | 0.0000 | 0.0584 | 0.0000 | 0.0495 | 0.0000 | measurement_only |
| content_graph_2wiki20 | content_hybrid_sem_graph | false | 0.8625 | 0.8000 | 0.3500 | 0.6000 | 0.2843 | 350.1000 | 967.5797 | 0.2878 | 0.0000 | 0.0580 | -0.0085 | measurement_limited |

- oracle_context_complete: `true`
- oracle_answer_coverage: `0.6000`
- oracle_selected_answer_coverage: `0.1000`
- qa_settings_consistent: `true`
- oracle_dominance_valid: `false`
- dominance_violations: `content_graph_2wiki20`
