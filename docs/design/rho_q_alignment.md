# rho_q Alignment Design

## Purpose

The goal of v1.1 is to make `rho_q(v)` closer to an answer-supporting evidence posterior while keeping the PAMAE core objective unchanged.

`rho_q` is a fixed input distribution for the weighted query-conditioned medoid objective. Improving it changes the evidence mass supplied to PAMAE, not the objective form, sample search, full-universe validation, or monotone refinement rule.

## Allowed Signals

Gold-free relevance modes may use:

- query text,
- node title,
- node text,
- query-derived title or entity spans,
- lexical overlap scores,
- semantic embedding similarity when a query embedding is available.

## Forbidden Signals

Relevance modes must not use:

- `gold_node_ids`,
- `is_supporting`,
- `possible_answers`,
- `obj`,
- `o_wiki_title`,
- answer-derived labels,
- dataset-specific final-method rules.

`diagnostic_subject_title` remains an upper-bound diagnostic only and must not become a default method.

## Relevance Modes

1. `current`: use the existing node relevance field.
2. `title_aware`: combine title-query lexical overlap, normalized title match, and body relevance.
3. `entity_title_aware`: add query-derived entity/title grounding from capitalized, quoted, and possessive spans.
4. `semantic_embedding`: reserved mode for a later branch if query embeddings are consistently available.
5. `hybrid_title_semantic`: combine title-aware and entity-title signals with semantic similarity when a query embedding is available; otherwise fall back to title-aware and record `semantic_component_available=False`.

## Scoring Form

The intended family is:

```text
rho_q(v) = Normalize(
  alpha * r_lex(q, v)
  + beta * r_title(q, v)
  + gamma * r_entity_title(q, v)
  + delta * r_sem(q, v)
)
```

Weights live in config:

```yaml
pamae:
  relevance_weights:
    lexical: 0.25
    title: 0.20
    entity_title: 0.25
    semantic: 0.30
```

Weights are global run settings. They must not be adjusted query-by-query after looking at gold labels.

## Diagnostics

Prediction diagnostics should expose:

- `relevance_mode`,
- `relevance_weights`,
- `semantic_component_available`,
- `query_title_spans`,
- optional `top_relevance_node_ids`.

Separate evaluation diagnostics may use gold labels to measure rank alignment, but those labels must not feed retrieval scoring.
