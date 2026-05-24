# Minimal d_q Graph Metric

This experiment should not force the earlier "wide retrieval is bad" framing. The compact v1 result is more specific: compact `top_rho` is strong, while PAMAE medoid optimization is weak when `d_q` is only semantic or lexical geometry.

## Hypothesis

PAMAE-style medoid coverage becomes useful when the query-conditioned distance `d_q` represents evidence structure. A graph-aware `d_q` may align medoid coverage with support evidence chains better than pure semantic/lexical distance.

The PAMAE objective remains:

```text
L_q(A) = sum_v rho_q(v) min_a d_q(v, a) + lambda_T T(A) + lambda_k |A|
```

Only `d_q` changes. Sample search, full-universe validation, refinement, and rendering all receive the same distance matrix for a fixed query.

## Minimal Edge Types

Use only a small set of gold-free edges:

- `same_canonical_title`: nodes with the same normalized title.
- `title_mention`: one node title appears in another node's title/text.
- `shared_query_span`: both nodes contain the same query-derived title/entity span.

Edge lengths must be nonnegative.

## Forbidden Inputs

Do not use:

- gold/support edges,
- `gold_node_ids`,
- `is_supporting`,
- `possible_answers`,
- `obj`,
- `o_wiki_title`,
- answer/object edges,
- negative reward edges,
- dataset shortcuts.

Gold labels may only be used after graph construction for diagnostics and evaluation.

## Success Criteria

The minimal graph metric is promising if any of these hold:

- graph-aware `refine_cell` improves over semantic `refine_cell`,
- graph-aware `refine_cell` narrows the gap to `top_rho`,
- graph-aware `refine_cell` beats `top_rho` on a meaningful subset,
- graph diagnostics show support nodes are structurally connected where semantic medoids fail.

If none hold, PAMAE should not be continued as the main method under this v1 family; it can remain a diagnostic or future-work scaffold.
