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

## Graph Variants Implemented

- `entity_sentence`: entity nodes plus sentence evidence nodes; entity--sentence mention edges and adjacent sentence edges only; no chunk nodes.
- `entity_sentence_chunk_hier`: entity, sentence, and chunk parent nodes; chunk parent edges are stored for metadata/rendering and excluded from the default metric.
- Main renderers: `sentence_only`, `sentence_path`, and `sentence_parent_title`; diagnostic renderers: fixed `sentence_local_window` and `sentence_parent_chunk`.

## Sentence Splitting And Mapping Coverage

- `2wikimultihopqa`: gold support mapping 0.8725; answer-containing sentence found 0.7800; avg sentences/chunk 4.6792; avg entities/sentence 3.0301; isolated sentence rate 0.0001.
- `hotpotqa`: gold support mapping 0.9783; answer-containing sentence found 0.9100; avg sentences/chunk 4.9380; avg entities/sentence 2.9327; isolated sentence rate 0.0000.

Mapping coverage was high enough to interpret the smoke results; the stop decision is not caused by an obvious sentence-boundary mapping failure.

## Answer Projection And Rendering Analysis

- `2wikimultihopqa` reference: projected 0.7600, rendered 0.4300, in context 0.4300, QA F1 0.0814. Best non-diagnostic sentence variant `entity_sentence_chunk_hier_sentence_parent_title`: projected 0.6600, rendered 0.2900, in context 0.3000, QA F1 0.0629.
- `2wikimultihopqa` best diagnostic renderer `entity_sentence_chunk_hier_sentence_parent_chunk` reached answer-in-context 0.3600 at 441.7000 tokens, still below the reference answer-in-context 0.4300.
- `hotpotqa` reference: projected 0.9400, rendered 0.6500, in context 0.6500, QA F1 0.0731. Best non-diagnostic sentence variant `entity_sentence_chunk_hier_sentence_parent_title`: projected 0.8100, rendered 0.2700, in context 0.3100, QA F1 0.0498.
- `hotpotqa` best diagnostic renderer `entity_sentence_chunk_hier_sentence_parent_chunk` reached answer-in-context 0.4100 at 368.7700 tokens, still below the reference answer-in-context 0.6500.

The sentence-primary graphs did not beat the chunk reference on answer projection, answer rendering, answer-in-context, rendered recall, or QA F1 on either dataset. Local-window and parent-chunk diagnostics recover some surface evidence, but not enough to pass the gates.

## Objective Monotonicity And Metric Validity

- Objective increase count across sentence runs: `0`.
- Triangle inequality violation count across sentence runs: `0`.
- The implementation also enforces sentence-node medoids in the retriever and excludes chunk parent edges from the primary metric by default.

## Renderer Comparison

- `2wikimultihopqa` sentence-only kept context small (65.4800 tokens) but rendered answer sentences only 0.2600 of the time.
- `2wikimultihopqa` best lean renderer by QA was `entity_sentence_sentence_path` with QA F1 0.0766; best answer-context renderer was `entity_sentence_chunk_hier_sentence_parent_chunk` with answer-in-context 0.3600.
- `hotpotqa` sentence-only kept context small (73.6900 tokens) but rendered answer sentences only 0.2000 of the time.
- `hotpotqa` best lean renderer by QA was `entity_sentence_sentence_path` with QA F1 0.0512; best answer-context renderer was `entity_sentence_chunk_hier_sentence_parent_chunk` with answer-in-context 0.4100.

## Chunk Parent Shortcut Risk Analysis

- The hierarchical graph's sentence-only and sentence-path results match the pure `entity_sentence` graph, which is the expected behavior when chunk parent edges are not metric shortcuts.
- `sentence_parent_title` adds parent metadata without full chunk text; `sentence_parent_chunk` is diagnostic-only and was not considered an adoption candidate.
- No result indicates that `entity_sentence_chunk_hier` behaved like a chunk-primary shortcut graph in the main comparison.

## 2wikimultihopqa

| run | graph | renderer | ans proj | ans rendered | ans ctx | rendered recall | context F1 | QA F1 | ctx tok | map rate | obj inc | tri viol | decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| entity_chunk_reference_current_renderer | entity_chunk_reference | current_renderer | 0.7600 | 0.4300 | 0.4300 | 0.5750 | 0.2646 | 0.0814 | 333.3900 | n/a | 0 | 0 | REFERENCE |
| entity_sentence_sentence_only | entity_sentence | sentence_only | 0.6600 | 0.2600 | 0.2600 | 0.4008 | 0.3059 | 0.0763 | 65.4800 | 0.8725 | 0 | 0 | STOP (answer_projected_regression, answer_rendered_regression, answer_in_context_regression, rendered_recall_regression, qa_f1_regression) |
| entity_sentence_sentence_path | entity_sentence | sentence_path | 0.6600 | 0.2900 | 0.2900 | 0.4275 | 0.2579 | 0.0766 | 126.4400 | 0.8725 | 0 | 0 | STOP (answer_projected_regression, answer_rendered_regression, answer_in_context_regression, rendered_recall_regression, context_f1_regression, qa_f1_regression) |
| entity_sentence_chunk_hier_sentence_only | entity_sentence_chunk_hier | sentence_only | 0.6600 | 0.2600 | 0.2600 | 0.4008 | 0.3059 | 0.0763 | 65.4800 | 0.8725 | 0 | 0 | STOP (answer_projected_regression, answer_rendered_regression, answer_in_context_regression, rendered_recall_regression, qa_f1_regression) |
| entity_sentence_chunk_hier_sentence_path | entity_sentence_chunk_hier | sentence_path | 0.6600 | 0.2900 | 0.2900 | 0.4275 | 0.2579 | 0.0766 | 126.4400 | 0.8725 | 0 | 0 | STOP (answer_projected_regression, answer_rendered_regression, answer_in_context_regression, rendered_recall_regression, context_f1_regression, qa_f1_regression) |
| entity_sentence_chunk_hier_sentence_parent_title | entity_sentence_chunk_hier | sentence_parent_title | 0.6600 | 0.2900 | 0.3000 | 0.4275 | 0.2579 | 0.0629 | 143.8900 | 0.8725 | 0 | 0 | STOP (answer_projected_regression, answer_rendered_regression, answer_in_context_regression, rendered_recall_regression, context_f1_regression, qa_f1_regression) |
| entity_sentence_chunk_hier_sentence_local_window | entity_sentence_chunk_hier | sentence_local_window | 0.6600 | 0.3200 | 0.3200 | 0.4775 | 0.2021 | 0.0753 | 211.9800 | 0.8725 | 0 | 0 | DIAGNOSTIC_ONLY_STOP (answer_projected_regression, answer_rendered_regression, answer_in_context_regression, rendered_recall_regression, context_f1_regression, qa_f1_regression) |
| entity_sentence_chunk_hier_sentence_parent_chunk | entity_sentence_chunk_hier | sentence_parent_chunk | 0.6600 | 0.3600 | 0.3600 | 0.5275 | 0.1425 | 0.0753 | 441.7000 | 0.8725 | 0 | 0 | DIAGNOSTIC_ONLY_STOP (answer_projected_regression, answer_rendered_regression, answer_in_context_regression, rendered_recall_regression, context_f1_regression, qa_f1_regression, context_tokens_over_110pct_reference) |

## hotpotqa

| run | graph | renderer | ans proj | ans rendered | ans ctx | rendered recall | context F1 | QA F1 | ctx tok | map rate | obj inc | tri viol | decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| entity_chunk_reference_current_renderer | entity_chunk_reference | current_renderer | 0.9400 | 0.6500 | 0.6500 | 0.6700 | 0.2883 | 0.0731 | 480.9300 | n/a | 0 | 0 | REFERENCE |
| entity_sentence_sentence_only | entity_sentence | sentence_only | 0.8100 | 0.2000 | 0.2000 | 0.2678 | 0.2336 | 0.0431 | 73.6900 | 0.9783 | 0 | 0 | STOP (answer_projected_regression, answer_rendered_regression, answer_in_context_regression, rendered_recall_regression, context_f1_regression, qa_f1_regression) |
| entity_sentence_sentence_path | entity_sentence | sentence_path | 0.8100 | 0.2700 | 0.2700 | 0.3315 | 0.2238 | 0.0512 | 129.4300 | 0.9783 | 0 | 0 | STOP (answer_projected_regression, answer_rendered_regression, answer_in_context_regression, rendered_recall_regression, context_f1_regression, qa_f1_regression) |
| entity_sentence_chunk_hier_sentence_only | entity_sentence_chunk_hier | sentence_only | 0.8100 | 0.2000 | 0.2000 | 0.2678 | 0.2336 | 0.0431 | 73.6900 | 0.9783 | 0 | 0 | STOP (answer_projected_regression, answer_rendered_regression, answer_in_context_regression, rendered_recall_regression, context_f1_regression, qa_f1_regression) |
| entity_sentence_chunk_hier_sentence_path | entity_sentence_chunk_hier | sentence_path | 0.8100 | 0.2700 | 0.2700 | 0.3315 | 0.2238 | 0.0512 | 129.4300 | 0.9783 | 0 | 0 | STOP (answer_projected_regression, answer_rendered_regression, answer_in_context_regression, rendered_recall_regression, context_f1_regression, qa_f1_regression) |
| entity_sentence_chunk_hier_sentence_parent_title | entity_sentence_chunk_hier | sentence_parent_title | 0.8100 | 0.2700 | 0.3100 | 0.3315 | 0.2238 | 0.0498 | 145.7000 | 0.9783 | 0 | 0 | STOP (answer_projected_regression, answer_rendered_regression, answer_in_context_regression, rendered_recall_regression, context_f1_regression, qa_f1_regression) |
| entity_sentence_chunk_hier_sentence_local_window | entity_sentence_chunk_hier | sentence_local_window | 0.8100 | 0.3800 | 0.3800 | 0.4013 | 0.1972 | 0.0531 | 221.1900 | 0.9783 | 0 | 0 | DIAGNOSTIC_ONLY_STOP (answer_projected_regression, answer_rendered_regression, answer_in_context_regression, rendered_recall_regression, context_f1_regression, qa_f1_regression) |
| entity_sentence_chunk_hier_sentence_parent_chunk | entity_sentence_chunk_hier | sentence_parent_chunk | 0.8100 | 0.4000 | 0.4100 | 0.4492 | 0.1664 | 0.0574 | 368.7700 | 0.9783 | 0 | 0 | DIAGNOSTIC_ONLY_STOP (answer_projected_regression, answer_rendered_regression, answer_in_context_regression, rendered_recall_regression, context_f1_regression, qa_f1_regression) |

## Adoption Gate Decision

{
  "decision": "STOP",
  "reason": "no non-diagnostic variant passed all available datasets"
}

## Next Recommendation

If sentence projection improves without QA improvement, focus next on context formatting and narrower parent-context rendering. If neither projection nor rendering improves on Hotpot, graph granularity alone is insufficient and the next diagnostic should target non-gold answer-bearing fact proxies.
