# QA Stage Bottleneck Analysis

## Content Buckets

| bucket | count | mean_f1 | mean_context_recall | mean_answer_coverage | mean_selected_answer_coverage | sample_query_ids |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| answer_absent_from_context | 9 | 0.0308 | 0.5833 | 0.0000 | 0.0000 | `006d81bc0bde11eba7f7acde48001122`, `2d8f3ebe0bda11eba7f7acde48001122`, `551e024408a611ebbd7fac1f6bf848b6`, `83bf3b5a0bd911eba7f7acde48001122`, `914b452c0bdc11eba7f7acde48001122`, `9fe6a6760baf11ebab90acde48001122`, `a80d84e7096d11ebbdb0ac1f6bf848b6`, `bab3b6d00bda11eba7f7acde48001122`, `d6898f78089511ebbd75ac1f6bf848b6` |
| answer_present_not_selected | 3 | 0.0933 | 0.9167 | 1.0000 | 0.0000 | `1c0dd3b00bdc11eba7f7acde48001122`, `7f7046d308f711ebbdaaac1f6bf848b6`, `a1cdb240085811ebbd5bac1f6bf848b6` |
| pre_refinement_anchor_loss | 6 | 0.0290 | 0.5000 | 0.3333 | 0.0000 | `265daf200bdc11eba7f7acde48001122`, `2dc690ba0bdc11eba7f7acde48001122`, `462bb642099211ebbdb0ac1f6bf848b6`, `6718770a087311ebbd66ac1f6bf848b6`, `84b691d8086a11ebbd5fac1f6bf848b6`, `f5e3b9ca0bdb11eba7f7acde48001122` |
| projection_loss | 1 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | `33f51d7e0bde11eba7f7acde48001122` |
| qa_partial_or_success | 1 | 0.4286 | 0.5000 | 1.0000 | 1.0000 | `dcbee4b608b011ebbd85ac1f6bf848b6` |

## Bucket Transitions

- `answer_present_not_selected->answer_absent_from_context`: 1
- `pre_refinement_anchor_loss->answer_absent_from_context`: 6
- `pre_refinement_anchor_loss->answer_present_not_selected`: 3
- `pre_refinement_anchor_loss->pre_refinement_anchor_loss`: 6
- `pre_refinement_anchor_loss->projection_loss`: 1
- `pre_refinement_anchor_loss->qa_partial_or_success`: 1
- `refinement_update_loss->answer_absent_from_context`: 2

## Largest Content Regressions

| query_id | baseline_bucket | content_bucket | delta_f1 | baseline_f1 | content_f1 | oracle_f1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `006d81bc0bde11eba7f7acde48001122` | pre_refinement_anchor_loss | answer_absent_from_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `265daf200bdc11eba7f7acde48001122` | pre_refinement_anchor_loss | pre_refinement_anchor_loss | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `2d8f3ebe0bda11eba7f7acde48001122` | pre_refinement_anchor_loss | answer_absent_from_context | 0.0000 | 0.1176 | 0.1176 | 0.1176 |
| `33f51d7e0bde11eba7f7acde48001122` | pre_refinement_anchor_loss | projection_loss | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `462bb642099211ebbdb0ac1f6bf848b6` | pre_refinement_anchor_loss | pre_refinement_anchor_loss | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `551e024408a611ebbd7fac1f6bf848b6` | pre_refinement_anchor_loss | answer_absent_from_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `6718770a087311ebbd66ac1f6bf848b6` | pre_refinement_anchor_loss | pre_refinement_anchor_loss | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `7f7046d308f711ebbdaaac1f6bf848b6` | pre_refinement_anchor_loss | answer_present_not_selected | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `83bf3b5a0bd911eba7f7acde48001122` | refinement_update_loss | answer_absent_from_context | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `84b691d8086a11ebbd5fac1f6bf848b6` | pre_refinement_anchor_loss | pre_refinement_anchor_loss | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
