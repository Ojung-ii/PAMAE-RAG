# Path Carrier 2Wiki 50 Decision

- Branch: `experiment/path-carrier-completion`
- Commit: `30f1931`
- Verification: `python -m compileall src tests scripts` passed; `/home/ojungii/miniconda3/envs/pamae-rag/bin/python -m pytest -q` passed with 124 tests.
- Invariants: `triangle_inequality_violation_count = 0`; `oracle_leakage_count = 0`; non-oracle `metric_path_carrier` uses graph shortest-path closure only.

## Headline

Decision: **DIAGNOSTIC_ONLY_100**

Reason: the refined support tree explains most of the current-minus-medoid answer gap, but deterministic metric-path rendering does not fully reproduce current answer recovery and lowers rendered recall.

## Current Reference

| metric | value |
| --- | ---: |
| answer_in_context | 0.4000 |
| qa_f1 | 0.0610 |
| rendered_recall | 0.5700 |
| context_f1 | 0.2683 |
| avg_context_tokens | 351.0400 |
| selected_medoid_answer_availability | 0.1200 |
| current_minus_medoid_answer_gap | 0.2800 |
| answer_on_refined_support_tree | 0.3600 |

## Metric Path Carrier

| metric | value |
| --- | ---: |
| answer_in_context | 0.3200 |
| qa_f1 | 0.0693 |
| rendered_recall | 0.4550 |
| context_f1 | 0.3428 |
| avg_context_tokens | 200.7600 |
| answer_metric_path_rendered | 0.3200 |
| current_minus_metric_path_answer_gap | 0.0800 |

## Diagnostic Variants

| renderer | answer_in_context | qa_f1 | rendered_recall | avg_context_tokens |
| --- | ---: | ---: | ---: | ---: |
| metric_path_carrier_no_medoids | 0.2200 | 0.0516 | 0.1700 | 95.6600 |
| metric_path_carrier_medoids_first | 0.3200 | 0.0693 | 0.4550 | 200.7600 |
| current_answer_role_oracle | 0.4000 | 0.0923 | 0.1500 | 39.9400 |
| support_tree_answer_oracle | 0.3600 | 0.0930 | 0.1300 | 32.2600 |

## Interpretation

The support tree is a real carrier object: answer-containing chunks are on `SPClosure(A_q + Theta_refined)` for 0.3600 of queries, compared with only 0.1200 selected-medoid availability. That accounts for most of the current renderer's 0.4000 answer-in-context.

The non-oracle metric path renderer recovers 0.3200 answer-in-context, so it reproduces much, but not all, of current behavior. It improves QA F1 and cuts tokens, but it also reduces answer coverage and rendered recall, so this is not adoption evidence.

The oracle checks are consistent: support-tree answer oracle reaches 0.3600 answer-in-context, while current-answer-role oracle reaches 0.4000. The residual 0.0400 support-tree gap and 0.0800 metric-rendering gap are the reason to continue only as diagnostic stability testing.

## Decision

**DIAGNOSTIC_ONLY_100**

Run 2Wiki 100 and Hotpot 100 only to confirm whether the bottleneck is stable. Do not adopt from this 2Wiki-only result.
