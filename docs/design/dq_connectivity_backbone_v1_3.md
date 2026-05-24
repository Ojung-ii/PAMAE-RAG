# d_q Connectivity Backbone v1.3

## Motivation

The minimal symbolic graph was too fragmented for a fair graph-aware `d_q` retrieval run. On HotpotQA and 2WikiMultiHopQA, the disconnected-pair rate was about `0.96`, so graph shortest-path distances would mostly collapse to the configured fallback distance.

Retrieval should not run when `d_sp` is mostly a disconnected fallback constant. That would test a broken graph, not PAMAE-style medoid coverage over evidence structure.

## Hypothesis

A gold-free semantic kNN backbone can make `d_sp` a meaningful query-local geodesic distance while preserving PAMAE objective consistency.

The PAMAE objective remains unchanged:

`L_q(A)=sum_v rho_q(v) min_a d_q(v,a)+lambda_T T(A)+lambda_k|A|`

Only `d_q` changes. The graph is built once before optimization and then the same distance matrix is used by sample search, full `V_q` validation, refinement, and rendering.

## Allowed Edges

Symbolic edges:

- `same_canonical_title`
- `title_mention`
- `shared_query_span`

Backbone edges:

- `semantic_knn`
- `mutual_semantic_knn`

The backbone uses only node embeddings already present in the query-local universe. Edge lengths are nonnegative semantic/angular distances.

## Forbidden Edges and Signals

The graph must not use:

- support/gold edges
- `gold_node_ids`
- `is_supporting`
- `possible_answers`
- answer/object edges
- `obj`
- `o_wiki_title`
- negative rewards
- dataset shortcuts

Gold labels are allowed only for graph diagnostics and retrieval evaluation.

## Gate Before Retrieval

Run graph diagnostics first. Retrieval should run only if graph connectivity becomes meaningful:

- disconnected-pair rate is not dominated by fallback distances
- largest component ratio is substantial
- gold-support connectivity improves over symbolic-only diagnostics
- average degree is not excessive

This keeps retrieval results interpretable and prevents a disconnected graph from masquerading as a graph-aware metric.

## Why kNN Is Not a PAMAE Heuristic

The semantic kNN backbone does not change the objective and does not add a stage-specific score. It defines a fixed query-local graph metric before PAMAE optimization starts. Once `d_q` is fixed, sample search, full validation, refinement, and rendering all consume the same distance matrix.
