# Basin Selection Diagnostic Report

## Run Identity

- Branch: `improve/basin-preserving-selection`
- Code commit at smoke start: `0e87631897718eb8c6845cf433af822714e9d613`
- Smoke root: `outputs/basin_selection_smoke`
- Test status before smoke: `python -m compileall src tests scripts` passed; `pytest -q` passed with 82 tests.
- Final decision: `STOP`

## Previous Diagnosis Summary

The content-derived graph redesign had high candidate and projection survival, but evidence was lost when the projected graph universe was compressed into selected anchors or medoids. Local refinement did not appear to be the main destructive step, but it also could not recover evidence basins already lost by selection. Rendering remained a secondary bottleneck. This round therefore tested whether basin coverage constraints could preserve query-conditioned evidence mass without adding heuristic score mixing.

## Implemented Design Changes

1. Query-level failure taxonomy
   - Added `projection_miss`, `selection_miss`, `rendering_miss`, `qa_fail`, and `success`.
   - Gold evidence is used only for diagnostics, never for retrieval or scoring.
   - Added aggregate failure counts per run.

2. Stronger oracle diagnostics
   - Added `gold_support`, `answer_containing`, and `answer_copy` oracle modes.
   - These share the same prompt, generator, metric, sample, and seed as method runs.
   - `answer_copy` is used to detect prompt, generator, or evaluator interface failure.

3. Basin-preserving medoid selection
   - Added `basin_preserving_medoids`.
   - Query basins are defined from query anchors and graph metric distance only.
   - Eligible basins satisfy expected sampling support `M * P_b >= tau`.
   - Selection uses lexicographic comparison: covered eligible basin count, covered eligible basin mass, then metric objective.
   - No BM25, dense score, LLM score, answer string, or gold label is used in selection.

4. Basin-aware diagnostic renderer
   - Added diagnostic renderer `basin_path_closure`.
   - Renders selected medoids plus shortest-path bridge chunks from assigned query anchors.
   - Does not add answer-containing chunks or lexical-overlap fallback chunks.

5. Smoke scripts
   - Added HotpotQA and 2Wiki 100-query smoke scripts.
   - Added comparison script with risk gate decisions.

## Oracle Dominance Results

| dataset | method_f1 | gold_support_f1 | answer_containing_f1 | answer_copy_f1 | oracle_dominance_valid | diagnosis |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| hotpotqa | 0.0731 | 0.0927 | 0.3104 | 0.6535 | true | oracle_context_construction_weaker_than_answer_copy |
| 2wikimultihopqa | 0.0814 | 0.0976 | 0.2651 | 0.6375 | true | oracle_context_construction_weaker_than_answer_copy |

Interpretation: answer-copy is much stronger than method F1 on both datasets, so the prompt/generator/evaluator interface is not the primary blocker. Gold-support oracle is weak relative to answer-containing and answer-copy oracles, so gold-support context construction is not a reliable upper bound by itself.

## Required Result Table

| dataset | run | graph_mode | oracle | candidate_recall | projected_recall | post_refine_recall | rendered_recall | context_f1 | avg_context_tokens | retrieval_ms | generation_ms | EM | F1 | oracle_gap | risk_decision |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hotpotqa | current_content | content_graph | false | 0.9600 | 0.9500 | 0.3500 | 0.6700 | 0.2883 | 480.93 | 808.65 | 0.3822 | 0.0000 | 0.0731 | 0.0196 | REFERENCE |
| hotpotqa | basin_preserving_selection | content_graph | false | 0.9600 | 0.9500 | 0.3300 | 0.6600 | 0.2851 | 484.59 | 807.47 | 0.2900 | 0.0000 | 0.0708 | 0.0219 | STOP |
| hotpotqa | basin_preserving_selection_plus_basin_renderer | content_graph | false | 0.9600 | 0.9500 | 0.3300 | 0.5100 | 0.3296 | 305.90 | 816.01 | 0.1834 | 0.0000 | 0.0589 | 0.0338 | STOP |
| 2wikimultihopqa | current_content | content_graph | false | 0.8525 | 0.8325 | 0.3200 | 0.5750 | 0.2646 | 333.39 | 903.82 | 0.2763 | 0.0100 | 0.0814 | 0.0162 | REFERENCE |
| 2wikimultihopqa | basin_preserving_selection | content_graph | false | 0.8525 | 0.8325 | 0.3275 | 0.5775 | 0.2679 | 350.50 | 894.63 | 0.2874 | 0.0100 | 0.0821 | 0.0155 | PASS_SECONDARY_ONLY |
| 2wikimultihopqa | basin_preserving_selection_plus_basin_renderer | content_graph | false | 0.8525 | 0.8325 | 0.3275 | 0.4925 | 0.3547 | 237.98 | 968.68 | 0.1487 | 0.0100 | 0.0850 | 0.0127 | STOP |

## Failure Type Distribution

| dataset | run | projection_miss | selection_miss | rendering_miss | qa_fail | success |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| hotpotqa | current_content | 0 | 43 | 0 | 57 | 0 |
| hotpotqa | basin_preserving_selection | 0 | 0 | 11 | 89 | 0 |
| hotpotqa | basin_preserving_selection_plus_basin_renderer | 0 | 0 | 18 | 82 | 0 |
| 2wikimultihopqa | current_content | 2 | 35 | 1 | 61 | 1 |
| 2wikimultihopqa | basin_preserving_selection | 2 | 0 | 9 | 88 | 1 |
| 2wikimultihopqa | basin_preserving_selection_plus_basin_renderer | 2 | 0 | 13 | 84 | 1 |

## Current Content vs Basin-Preserving Selection

| dataset | metric | current_content | basin_preserving_selection | delta |
| --- | --- | ---: | ---: | ---: |
| hotpotqa | F1 | 0.0731 | 0.0708 | -0.0024 |
| hotpotqa | oracle_gap | 0.0196 | 0.0219 | +0.0024 |
| hotpotqa | local selected survival | 0.3500 | 0.3300 | -0.0200 |
| hotpotqa | rendered_recall | 0.6700 | 0.6600 | -0.0100 |
| hotpotqa | answer_in_context | 0.6500 | 0.6400 | -0.0100 |
| 2wikimultihopqa | F1 | 0.0814 | 0.0821 | +0.0007 |
| 2wikimultihopqa | oracle_gap | 0.0162 | 0.0155 | -0.0007 |
| 2wikimultihopqa | local selected survival | 0.3200 | 0.3275 | +0.0075 |
| 2wikimultihopqa | rendered_recall | 0.5750 | 0.5775 | +0.0025 |
| 2wikimultihopqa | answer_in_context | 0.4300 | 0.4400 | +0.0100 |

Hotpot is the primary decision dataset, and it fails the required adoption gates. 2Wiki selection-only passes the scripted gate, but the gain is too small to override Hotpot failure, and it does not show a decisive selection survival recovery.

## Current Content vs Basin Selection Plus Basin Renderer

| dataset | metric | current_content | plus_basin_renderer | delta |
| --- | --- | ---: | ---: | ---: |
| hotpotqa | F1 | 0.0731 | 0.0589 | -0.0142 |
| hotpotqa | oracle_gap | 0.0196 | 0.0338 | +0.0142 |
| hotpotqa | rendered_recall | 0.6700 | 0.5100 | -0.1600 |
| hotpotqa | context_f1 | 0.2883 | 0.3296 | +0.0412 |
| hotpotqa | answer_in_context | 0.6500 | 0.5500 | -0.1000 |
| 2wikimultihopqa | F1 | 0.0814 | 0.0850 | +0.0035 |
| 2wikimultihopqa | oracle_gap | 0.0162 | 0.0127 | -0.0035 |
| 2wikimultihopqa | rendered_recall | 0.5750 | 0.4925 | -0.0825 |
| 2wikimultihopqa | context_f1 | 0.2646 | 0.3547 | +0.0900 |
| 2wikimultihopqa | answer_in_context | 0.4300 | 0.4200 | -0.0100 |

The basin renderer is diagnostic-only. It increases precision and context F1 but loses recall and answer coverage, so it fails the explicit stop conditions.

## Type B Recovery Analysis

The taxonomy Type B count falls to zero for basin-preserving selection on both datasets. This is expected because selected basins are now explicitly covered. However, this did not translate into robust evidence preservation:

- Hotpot local selected survival falls from 0.3500 to 0.3300.
- Hotpot rendered recall falls from 0.6700 to 0.6600.
- Hotpot answer coverage falls from 0.6500 to 0.6400.
- 2Wiki local selected survival rises only from 0.3200 to 0.3275.

Therefore the current basin diagnostic is too permissive as an adoption signal. It proves that a selected medoid covers an eligible anchor basin, but not that the selected representative, path closure, or rendered context preserves the actual gold-supporting evidence.

## Selected Basin Survival Analysis

The basin objective successfully covers all eligible basins in the logged diagnostics:

- Mean eligible basins: 3.0 on both datasets.
- Mean covered basins: 3.0 on both datasets.
- Mean covered basin mass: 1.0 on both datasets.

The failure is not incomplete basin coverage. The failure is that anchor Voronoi basins induced by top query-mass anchors do not reliably isolate indispensable evidence units. The selected basin can contain gold evidence while the selected medoid and downstream renderer still miss the needed chunk or answer-bearing evidence.

## Rendering and Answer Coverage Analysis

Rendering remains a real secondary bottleneck:

- Hotpot basin renderer reduces tokens from 480.93 to 305.90 but drops rendered recall from 0.6700 to 0.5100 and answer coverage from 0.6500 to 0.5500.
- 2Wiki basin renderer reduces tokens from 333.39 to 237.98 and increases context F1, but drops rendered recall from 0.5750 to 0.4925 and answer coverage from 0.4300 to 0.4200.

This supports keeping the renderer diagnostic-only. It is too sparse for adoption under the current graph path closure rule.

## QA F1 and Oracle Gap

Hotpot:

- current_content F1: 0.0731
- basin_preserving_selection F1: 0.0708
- basin_preserving_selection_plus_basin_renderer F1: 0.0589
- current_content oracle gap: 0.0196
- basin_preserving_selection oracle gap: 0.0219
- plus renderer oracle gap: 0.0338

2Wiki:

- current_content F1: 0.0814
- basin_preserving_selection F1: 0.0821
- basin_preserving_selection_plus_basin_renderer F1: 0.0850
- current_content oracle gap: 0.0162
- basin_preserving_selection oracle gap: 0.0155
- plus renderer oracle gap: 0.0127

2Wiki alone would allow weak diagnostic optimism for selection-only, but Hotpot primary fails and plus-renderer violates recall and answer-coverage stop conditions.

## Risk Gate Decision

Final decision: `STOP`

Reasons:

1. Hotpot primary gate fails for selection-only:
   - F1 regresses.
   - Oracle gap increases.
   - Rendered recall decreases.
   - Context F1 decreases.
   - Answer in context decreases.

2. Hotpot primary gate fails for basin renderer:
   - F1 regresses substantially.
   - Oracle gap increases substantially.
   - Rendered recall decreases substantially.
   - Answer in context decreases.

3. 2Wiki improvements are not enough to adopt:
   - Selection-only gains are very small.
   - Renderer gains are accompanied by rendered recall and answer coverage regressions.
   - Hotpot is the primary decision dataset for this round.

4. No prompt, metric, generator, graph-index construction, score mixing, or dataset-specific tuning change is justified by these diagnostics.

## Heuristic and Dataset-Specific Risk

- Heuristic risk of implemented selector: moderate. The objective is principle-consistent, but the chosen query anchor basins are not yet validated as evidence-preserving partitions.
- Dataset-specific risk of adoption: high. 2Wiki shows tiny gains while Hotpot regresses.
- Local-minimum risk: high for adopting this exact basin definition, because all eligible basins are covered but gold evidence survival does not reliably improve.
- Oracle leakage risk: low. Gold labels and answer strings were used only in diagnostics and oracle runs, not retrieval or scoring.

## Stopped Risky Ideas

The following were not implemented:

- Scalar score mixing of PPR, BM25, dense scores, degree penalties, or LLM scores.
- Dataset-specific thresholds or query-pattern branches.
- Prompt or evaluator changes.
- Graph index construction changes.
- Answer-aware or gold-aware retrieval filters.
- Context budget increases to mask selection or rendering loss.

## Next Principle-Based Improvements

1. Recheck query anchor construction before changing selection again.
   - The current anchor Voronoi basins are fully covered but do not reliably preserve evidence.
   - Next diagnostics should ask whether gold-supporting chunks are assigned to basins whose medoids are path-connected and renderable under the graph metric.

2. Strengthen basin diagnostics before adopting any basin objective.
   - Add per-gold support path realizability: projected gold -> basin medoid -> rendered path.
   - Separate "basin contains gold" from "selected representative preserves gold".

3. Improve oracle context construction.
   - Answer-copy and answer-containing oracles are much stronger than gold-support oracle.
   - The current gold-support oracle may omit answer-bearing phrasing or provide evidence in a form the extractive generator does not exploit.

4. Keep rendering changes diagnostic until recall is protected.
   - Basin path closure is too sparse.
   - Any renderer adoption must preserve rendered recall and answer coverage on Hotpot before considering precision gains.

