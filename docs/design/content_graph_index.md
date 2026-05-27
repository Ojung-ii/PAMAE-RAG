# Content-Derived Graph Index

This branch adds a dependency-free content graph index alongside the legacy query-local graph path.

The index is built from chunk text only:

- entity mentions from quoted and capitalized spans in chunk text
- sentence-level facts
- adjacent entity-relation-entity triples when a sentence contains multiple text-derived entities
- chunk-entity edges
- chunk-fact edges
- fact-entity edges
- entity co-fact edges

Title metadata is not used as an entity source or primary backbone. The index records `content_graph_title_metadata_used=false` in diagnostics.

Current status:

- implemented as `pamae_rag.graph.content_graph`
- connected as an optional `pamae.graph.source: content` graph source
- legacy graph construction remains the default `pamae.graph.source: legacy_query`
- no gold labels, possible answers, oracle evidence, dataset-specific rules, or query-pattern branches
- no imported runtime dependency on any external reference implementation

Projection into a chunk-distance graph uses text-derived shared-entity and entity-fact bridge links. Edge lengths are neutral defaults unless explicitly configured; no dataset-specific or answer-aware weights are introduced.
