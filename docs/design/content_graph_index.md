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
- not connected to retrieval or scoring yet
- no gold labels, possible answers, oracle evidence, dataset-specific rules, or query-pattern branches
- no imported runtime dependency on any external reference implementation

Projection into a chunk-distance graph and stage-wise survival diagnostics are intentionally deferred to later commits so the indexing redesign can be reviewed independently from performance changes.
