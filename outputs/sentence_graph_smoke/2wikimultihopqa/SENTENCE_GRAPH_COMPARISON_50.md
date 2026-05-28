# Sentence Graph Granularity Diagnostic Report

- Branch: `experiment/sentence-primary-graph`
- Commit: `8e38697`
- Final decision: **STOP**

## Why This Was Tested

Previous diagnostics suggested that graph evidence could be projected or path-reachable while answer-bearing surface evidence still failed to render or help QA. This run tests whether making sentences, rather than chunks, the primary PAMAE medoids improves that bottleneck.

## PAMAE Principle Check

- Primary objects are sentence nodes.
- Selected medoids are sentence nodes.
- PPR is used only to define query-conditioned sentence mass.
- Shortest-path distance is the PAMAE metric.
- No dense reranking, LLM reranking, answer-aware retrieval, or scalar score mixing is used.
- `entity_sentence_chunk_hier` stores chunk parents for metadata/rendering; parent edges are excluded from the main metric.

## 2wikimultihopqa

| run | graph | renderer | ans proj | ans rendered | ans ctx | rendered recall | context F1 | QA F1 | ctx tok | map rate | obj inc | tri viol | decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| entity_chunk_reference_current_renderer | entity_chunk_reference | current_renderer | 0.8200 | 0.4000 | 0.4000 | 0.5700 | 0.2683 | 0.0610 | 351.0400 | n/a | 0 | 0 | REFERENCE |
| entity_sentence_sentence_only | entity_sentence | sentence_only | 0.7200 | 0.2800 | 0.2800 | 0.4167 | 0.3224 | 0.0578 | 64.9800 | 0.9000 | 0 | 0 | STOP (answer_projected_regression, answer_rendered_regression, answer_in_context_regression, rendered_recall_regression, qa_f1_regression) |
| entity_sentence_sentence_path | entity_sentence | sentence_path | 0.7200 | 0.2800 | 0.2800 | 0.4417 | 0.2795 | 0.0583 | 124.8000 | 0.9000 | 0 | 0 | STOP (answer_projected_regression, answer_rendered_regression, answer_in_context_regression, rendered_recall_regression, qa_f1_regression) |
| entity_sentence_chunk_hier_sentence_only | entity_sentence_chunk_hier | sentence_only | 0.7200 | 0.2800 | 0.2800 | 0.4167 | 0.3224 | 0.0578 | 64.9800 | 0.9000 | 0 | 0 | STOP (answer_projected_regression, answer_rendered_regression, answer_in_context_regression, rendered_recall_regression, qa_f1_regression) |
| entity_sentence_chunk_hier_sentence_path | entity_sentence_chunk_hier | sentence_path | 0.7200 | 0.2800 | 0.2800 | 0.4417 | 0.2795 | 0.0583 | 124.8000 | 0.9000 | 0 | 0 | STOP (answer_projected_regression, answer_rendered_regression, answer_in_context_regression, rendered_recall_regression, qa_f1_regression) |
| entity_sentence_chunk_hier_sentence_parent_title | entity_sentence_chunk_hier | sentence_parent_title | 0.7200 | 0.2800 | 0.2800 | 0.4417 | 0.2795 | 0.0495 | 142.3400 | 0.9000 | 0 | 0 | STOP (answer_projected_regression, answer_rendered_regression, answer_in_context_regression, rendered_recall_regression, qa_f1_regression) |
| entity_sentence_chunk_hier_sentence_local_window | entity_sentence_chunk_hier | sentence_local_window | 0.7200 | 0.3200 | 0.3200 | 0.5117 | 0.2276 | 0.0565 | 207.7000 | 0.9000 | 0 | 0 | DIAGNOSTIC_ONLY_STOP (answer_projected_regression, answer_rendered_regression, answer_in_context_regression, rendered_recall_regression, context_f1_regression, qa_f1_regression) |
| entity_sentence_chunk_hier_sentence_parent_chunk | entity_sentence_chunk_hier | sentence_parent_chunk | 0.7200 | 0.3600 | 0.3600 | 0.5617 | 0.1560 | 0.0573 | 430.5200 | 0.9000 | 0 | 0 | DIAGNOSTIC_ONLY_STOP (answer_projected_regression, answer_rendered_regression, answer_in_context_regression, rendered_recall_regression, context_f1_regression, qa_f1_regression, context_tokens_over_110pct_reference) |

## Adoption Gate Decision

{
  "decision": "STOP",
  "reason": "no non-diagnostic variant passed all available datasets"
}

## Next Recommendation

If sentence projection improves without QA improvement, focus next on context formatting and narrower parent-context rendering. If neither projection nor rendering improves on Hotpot, graph granularity alone is insufficient and the next diagnostic should target non-gold answer-bearing fact proxies.
