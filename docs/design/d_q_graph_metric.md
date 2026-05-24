# d_q Graph Metric Scaffold

## Current Limitation

Current `d_q` is based on semantic or lexical geometry over evidence nodes. This can cover broad topical neighborhoods, but it does not explicitly represent the support evidence chain needed by multi-hop QA. In compact runs, medoid coverage can reduce the PAMAE objective while moving away from support F1.

## Proposed Future Form

A future branch can evaluate:

```text
d_q(u, v) = lambda_s * d_sem(u, v) + lambda_g * d_sp(u, v)
```

with:

```text
lambda_s >= 0
lambda_g >= 0
```

`d_sp` should be a nonnegative typed-edge shortest-path distance. If constructed as a shortest-path pseudometric over nonnegative edges, it can preserve the interpretation needed for PAMAE-style medoid coverage bounds.

## Candidate Gold-Free Edges

Possible graph edges:

- same canonical title group,
- title mention in text,
- shared query-derived entity mention,
- paragraph or section adjacency when available,
- cross-reference-like title overlap.

## Forbidden Edges

Do not add:

- negative reward edges,
- support/gold edges,
- answer-object edges,
- dataset-specific shortcut rules,
- edges derived from `is_supporting`, `gold_node_ids`, `possible_answers`, `obj`, or `o_wiki_title`.

## PAMAE Connection

For a fixed query `q`, the PAMAE objective depends on a fixed relevance distribution and distance matrix. If `d_q` remains a metric, or at least a nonnegative shortest-path pseudometric, then the medoid coverage interpretation remains intact. The objective itself does not need a bridge bonus or extra stage-specific score.

## Implementation Status

This document is a scaffold only. Do not apply a graph-aware `distance_mode` in this branch. The implementation branch should be:

```text
feature/dq-graph-metric-v1_2
```
