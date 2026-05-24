# Minimal d_q Graph Metric Decision Report

## Scope

This run tests whether a minimal graph-aware query distance can give PAMAE medoid coverage a better evidence geometry without changing the PAMAE objective:

`L_q(A)=sum_v rho_q(v) min_a d_q(v,a)+lambda_T T(A)+lambda_k|A|`

Only `d_q` was scaffolded. The graph builder uses query text and node title/text only. It does not use support labels, gold node ids, answer/object fields, or dataset-specific shortcuts.

## Graph Diagnostics Gate

Retrieval ablations were intentionally not launched because the minimal graph failed the pre-registered connectivity gate.

| dataset | avg_edges | largest_component_ratio | disconnected_pair_rate | gold_support_connected_rate | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| HotpotQA | 1209.47 | 0.2228 | 0.9642 | 0.7684 | fail graph gate |
| 2WikiMultiHopQA | 1230.17 | 0.1769 | 0.9625 | 0.4744 | fail graph gate |

Proceed criterion was `avg_disconnected_pair_rate < 0.8`. Both datasets are around `0.96`, so shortest-path distances would mostly collapse to the configured disconnected distance. In that setting, a graph-aware PAMAE run would not be a fair test of evidence-chain geometry.

Detailed diagnostics:

- HotpotQA: `docs/experiments/dq_minimal_graph_metric/hotpotqa_graph_sp.md`
- 2WikiMultiHopQA: `docs/experiments/dq_minimal_graph_metric/2wikimultihopqa_graph_sp.md`

## Interpretation

The minimal edges are not empty; both datasets average roughly 1.2k edges per 500-query-local nodes. The problem is fragmentation: the graph creates many small islands rather than a useful query-conditioned evidence space. This is especially visible on 2WikiMultiHopQA, where only about 47% of gold support pairs are connected by the minimal graph.

This result does not falsify the broader d_q hypothesis. It says the first minimal graph is too sparse or too local to support shortest-path medoid optimization.

## Retrieval Results

No d_q retrieval ablation was run.

Reason: graph diagnostics failed before retrieval. Running `graph_sp` or hybrid distances with a `0.96` disconnected-pair rate would mostly measure the disconnected fallback constant, not evidence geometry.

## Semantic vs Graph-Aware refine_cell

Not evaluated in this run because retrieval was gated off.

## Graph-Aware refine_cell vs top_rho

Not evaluated in this run because retrieval was gated off.

## Subset and Failure Analysis

The failure-analysis script was added for later comparisons, but it was not executed because no d_q retrieval predictions were generated.

## Decision

Do not continue to 100-query d_q retrieval with this exact minimal graph. The current graph construction is not connected enough to serve as a meaningful query-conditioned shortest-path metric.

The d_q direction remains plausible only if the next graph builder improves connectedness without violating the boundary rules.

## Next Steps

1. Keep PAMAE objective unchanged.
2. Improve graph construction before retrieval by adding one or two gold-free bridge signals, then rerun graph diagnostics first.
3. Candidate additions should remain minimal and non-oracular:
   - normalized title mention with aliases from the node title itself
   - shared high-idf query token spans rather than broad capitalized spans
   - paragraph title to sentence/chunk title grouping if already present in metadata and not support-derived
4. Continue to require:
   - nonnegative edge lengths
   - no support/gold/answer/object fields
   - `avg_disconnected_pair_rate < 0.8` before retrieval
5. Defer terminal-conditioned posterior unless a graph-aware d_q shows partial promise.

## Files Produced

- `src/pamae_rag/graph/query_graph.py`
- `src/pamae_rag/graph/graph_distance.py`
- `scripts/analyze_query_graph.py`
- `scripts/analyze_dq_failure_cases.py`
- `configs/ablations_dq/`
- `docs/design/minimal_dq_graph_metric.md`
- `docs/experiments/dq_minimal_graph_metric/hotpotqa_graph_sp.md`
- `docs/experiments/dq_minimal_graph_metric/2wikimultihopqa_graph_sp.md`
