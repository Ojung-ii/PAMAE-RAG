# 2Wiki Local Surface 50-Query Decision

- Branch: `experiment/chunk-backbone-local-surface`
- Commit: `5f4f922`
- Verification: `compileall` passed; `pytest -q` passed with `113 passed`
- Decision: **STOP_BEFORE_100**

## Invariant Status

- `triangle_inequality_violation_count == 0` for all runs.
- `local_objective_invalid_count == 0` for all runs.
- Oracle renderers are excluded from adoption gates.
- The current renderer reference uses the same config, seed, and 50-query sample.
- Gold and answer strings are used only for diagnostics or oracle renderers.
- Non-oracle renderers log `uses_answer_string=false` and `uses_gold_label=false`.
- No scalar score mixing was introduced.

## Metrics

| renderer_mode | answer_sentence_available_in_selected_chunks | gold_sentence_available_in_selected_chunks | answer_sentence_rendered | gold_sentence_rendered | answer_in_context | rendered_recall | context_f1 | qa_f1 | avg_context_tokens | answer_rendered_given_available | gold_rendered_given_available | qa_success_given_answer_rendered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_renderer | 0.1200 | 0.6400 | 0.4000 | 0.9200 | 0.4000 | 0.6650 | 0.1231 | 0.0610 | 351.0400 | 1.0000 | 0.9688 | 0.4000 |
| local_sentence_medoid | 0.1200 | 0.6400 | 0.1200 | 0.4800 | 0.1200 | 0.2617 | 0.1585 | 0.0392 | 89.4800 | 1.0000 | 0.7500 | 0.8333 |
| fact_mediated_sentence | 0.1200 | 0.6400 | 0.1200 | 0.6400 | 0.1200 | 0.3583 | 0.1607 | 0.0370 | 144.9600 | 1.0000 | 1.0000 | 0.8333 |
| selected_chunk_answer_sentence_oracle | 0.1200 | 0.6400 | 0.1200 | 0.1000 | 0.1200 | 0.0400 | 0.0480 | 0.0319 | 5.6200 | 1.0000 | 0.1562 | 1.0000 |

## Decision Rationale

Selected chunks rarely contain answer-bearing sentences: only `0.1200` of queries. The answer-sentence oracle is also capped at `0.1200` answer-in-context, so the weak result is not mainly caused by non-gold local sentence/fact selection failing to identify available answer sentences.

The current renderer reaches `0.4000` answer-in-context because it renders additional chunk context beyond the selected medoid chunks. Restricting local rendering to selected chunks removes that answer-bearing surface.

Exact reason: **STOP_BEFORE_100 because selected chunks rarely contain answer sentences and the answer-sentence oracle is weak.** Running 100-query 2Wiki and Hotpot would not test the intended local-surface bottleneck under the current selected-chunk-only construction.

## Next Recommendation

Diagnose the gap between selected medoid chunks and rendered chunks before tuning local rendering. A meaningful next test should decide whether local surface selection should operate over the current renderer's rendered chunk set, selected basins, or another PAMAE-consistent chunk neighborhood, rather than only over medoid chunks.
