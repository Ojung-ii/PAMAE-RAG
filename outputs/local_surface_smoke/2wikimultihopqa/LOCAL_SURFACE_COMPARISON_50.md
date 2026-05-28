# Local Surface Diagnostic Report

- Branch: `experiment/chunk-backbone-local-surface`
- Commit: `3adda83`
- Final decision: **STOP**

## Previous Sentence-Primary STOP Summary

The previous sentence-primary global graph round stopped because neither `entity_sentence` nor `entity_sentence_chunk_hier` beat the entity--chunk reference on answer projection, answer rendering, answer-in-context, rendered recall, or QA F1 across 2Wiki and Hotpot.

## Reason For Returning To Chunk Backbone

This round keeps the stronger entity--chunk retrieval backbone fixed and tests whether answer-bearing sentence surfaces can be selected inside the already selected chunks.

## PAMAE Principle Check

- Global retrieval object remains the chunk.
- Local rendering objects are sentences/fact-grounded sentences inside selected chunks.
- Local medoids use graph shortest-path distance with PPR only as sentence mass.
- Fact-mediated rendering uses deterministic graph closure, not scalar score mixing.
- Oracle renderers use answer/gold labels only for diagnostics and are excluded from adoption gates.

## Selected-Chunk Answer Surface Availability

- `2wikimultihopqa` selected chunks contain answer sentences in 0.1200 of queries and gold support sentences in 0.6400.

## Local-Minimum Guard Answers

- Did this preserve the PAMAE principle? Yes: the global retrieval object stayed chunk-level and local sentence selection used graph distance.
- Did it avoid scalar score mixing? Yes.
- Did it improve both answer-bearing recovery and QA? See tables; adoption requires both datasets and all non-oracle gates.
- Did it reduce or increase context tokens? See `ctx tok`; token increases block adoption.
- Did it reveal selected chunks already contain answer sentences? See availability rates above.
- Did it reveal non-gold local rendering cannot identify those sentences? Compare non-oracle recovery to answer oracle gaps.
- Final decision: **STOP**.

## 2wikimultihopqa

| run | renderer_mode | answer_sentence_available_in_selected_chunks | gold_sentence_available_in_selected_chunks | answer_sentence_rendered | gold_sentence_rendered | answer_in_context | rendered_recall | context_f1 | qa_f1 | oracle_gap_answer_containing | avg_context_tokens | triangle_inequality_violation_count | local_objective_invalid_count | answer_rendered_given_available | gold_rendered_given_available | qa_success_given_answer_rendered | decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| entity_chunk_reference_current_renderer | current_renderer | 0.1200 | 0.6400 | 0.4000 | 0.9200 | 0.4000 | 0.6650 | 0.1231 | 0.0610 | 0.0000 | 351.0400 | 0 | 0 | 1.0000 | 0.9688 | 0.4000 | REFERENCE |
| entity_chunk_reference_local_sentence_medoid | local_sentence_medoid | 0.1200 | 0.6400 | 0.1200 | 0.4800 | 0.1200 | 0.2617 | 0.1585 | 0.0392 | 0.0000 | 89.4800 | 0 | 0 | 1.0000 | 0.7500 | 0.8333 | STOP (qa_f1_regression, answer_in_context_regression, rendered_recall_regression) |
| entity_chunk_reference_fact_mediated_sentence | fact_mediated_sentence | 0.1200 | 0.6400 | 0.1200 | 0.6400 | 0.1200 | 0.3583 | 0.1607 | 0.0370 | 0.0000 | 144.9600 | 0 | 0 | 1.0000 | 1.0000 | 0.8333 | STOP (qa_f1_regression, answer_in_context_regression, rendered_recall_regression) |
| entity_chunk_reference_selected_chunk_answer_sentence_oracle | selected_chunk_answer_sentence_oracle | 0.1200 | 0.6400 | 0.1200 | 0.1000 | 0.1200 | 0.0400 | 0.0480 | 0.0319 | 0.0000 | 5.6200 | 0 | 0 | 1.0000 | 0.1562 | 1.0000 | STOP (oracle_renderer_excluded, qa_f1_regression, answer_in_context_regression, rendered_recall_regression, context_f1_regression) |
| entity_chunk_reference_selected_chunk_gold_sentence_oracle | selected_chunk_gold_sentence_oracle | 0.1200 | 0.6400 | 0.1000 | 0.6400 | 0.1200 | 0.3583 | 0.3354 | 0.0371 | 0.0000 | 31.3000 | 0 | 0 | 0.8333 | 1.0000 | 0.8000 | STOP (oracle_renderer_excluded, qa_f1_regression, answer_in_context_regression, rendered_recall_regression) |

## Adoption Gate Decision

{
  "decision": "STOP",
  "reason": "no non-oracle local renderer passed all available datasets"
}

## Next Recommendation

If selected chunks contain answer sentences but non-oracle local renderers miss them while the answer oracle is strong, the next step is a principled non-gold answer-bearing proxy rather than another graph backbone change.
