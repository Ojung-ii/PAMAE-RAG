# QA Stage Bottleneck Analysis

## Content Buckets

| bucket | count | mean_f1 | mean_context_recall | mean_answer_coverage | mean_selected_answer_coverage | sample_query_ids |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| answer_absent_from_context | 5 | 0.0789 | 0.5000 | 0.0000 | 0.0000 | `5a7a3a945542996a35c17147`, `5ab57fc4554299488d4d99c0`, `5abc030e554299642a094bdc`, `5ade86b255429939a52fe8e0`, `5ae1f17e554299234fd04349` |
| answer_present_not_selected | 4 | 0.0000 | 0.8750 | 1.0000 | 0.0000 | `5a85c3225542992a431d1b95`, `5a8835dc5542994846c1ce2b`, `5ab3cecd5542992ade7c6eae`, `5ade9c9c5542997c77adee8c` |
| pre_refinement_anchor_loss | 8 | 0.0411 | 0.5625 | 0.5000 | 0.1250 | `5a714dea5542994082a3e7a9`, `5a749af055429979e28829b7`, `5a84574455429933447460e6`, `5a8481945542997175ce1ed3`, `5ab69f9a554299710c8d1ef8`, `5abe953b5542993f32c2a170`, `5adcd4325542994d58a2f6ed`, `5ae6c2285542995703ce8b9a` |
| qa_partial_or_success | 3 | 0.1748 | 1.0000 | 1.0000 | 1.0000 | `5a72b2dc5542992359bc3173`, `5ab94bc2554299743d22eacf`, `5ae1389655429901ffe4ae05` |

## Bucket Transitions

- `answer_absent_from_context->pre_refinement_anchor_loss`: 1
- `answer_present_not_selected->answer_present_not_selected`: 1
- `answer_present_not_selected->pre_refinement_anchor_loss`: 1
- `pre_refinement_anchor_loss->answer_absent_from_context`: 3
- `pre_refinement_anchor_loss->answer_present_not_selected`: 2
- `pre_refinement_anchor_loss->pre_refinement_anchor_loss`: 5
- `pre_refinement_anchor_loss->qa_partial_or_success`: 1
- `qa_partial_or_success->answer_present_not_selected`: 1
- `qa_partial_or_success->qa_partial_or_success`: 2
- `refinement_update_loss->answer_absent_from_context`: 2
- `refinement_update_loss->pre_refinement_anchor_loss`: 1

## Largest Content Regressions

| query_id | baseline_bucket | content_bucket | delta_f1 | baseline_f1 | content_f1 | oracle_f1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `5ab69f9a554299710c8d1ef8` | answer_present_not_selected | pre_refinement_anchor_loss | -0.2000 | 0.2000 | 0.0000 | 0.2000 |
| `5a714dea5542994082a3e7a9` | pre_refinement_anchor_loss | pre_refinement_anchor_loss | -0.1429 | 0.1429 | 0.0000 | 0.1429 |
| `5ade9c9c5542997c77adee8c` | qa_partial_or_success | answer_present_not_selected | -0.0833 | 0.0833 | 0.0000 | 0.0833 |
| `5a72b2dc5542992359bc3173` | pre_refinement_anchor_loss | qa_partial_or_success | 0.0000 | 0.2000 | 0.2000 | 0.2000 |
| `5a7a3a945542996a35c17147` | refinement_update_loss | answer_absent_from_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `5a84574455429933447460e6` | pre_refinement_anchor_loss | pre_refinement_anchor_loss | 0.0000 | 0.0000 | 0.0000 | 0.1111 |
| `5a8481945542997175ce1ed3` | answer_absent_from_context | pre_refinement_anchor_loss | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `5a85c3225542992a431d1b95` | pre_refinement_anchor_loss | answer_present_not_selected | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `5a8835dc5542994846c1ce2b` | pre_refinement_anchor_loss | answer_present_not_selected | 0.0000 | 0.0000 | 0.0000 | 0.2857 |
| `5ab3cecd5542992ade7c6eae` | answer_present_not_selected | answer_present_not_selected | 0.0000 | 0.0000 | 0.0000 | 0.3333 |
