# Semantic Carrier Adequacy Diagnostic

- Branch: `experiment/semantic-carrier-adequacy`
- Commit: `d08e637`
- Gate outcome: **STOP_BEFORE_100**
- Final decision: **STOP**

Previous hidden-non-tree summary: support-tree diagnostics ended as `DIAGNOSTIC_ONLY` with hidden non-tree recovery on both datasets. This round tests whether existing embeddings can explain or order those carriers inside graph-defined candidates.

PAMAE boundary: the entity-chunk graph, graph-metric medoid selection, local refinement, and `T_q = SPClosure(A_q union Theta_refined)` are unchanged. Semantic information is restricted to diagnostics, lexicographic ordering inside `T_q union S1`, and a fixed `1 + d_ang` diagnostic tree.

Why raw cosine is not used as a PAMAE distance: cosine similarity is not a metric distance and `1 - cosine` is not used as proof-level distance. The implementation uses normalized angular distance `arccos(clamp(dot,-1,1))/pi` for semantic diagnostics.

## 2wikimultihopqa

- Gate decision: **STOP_BEFORE_100**
- Reason: query embeddings are missing; refusing to synthesize semantic query vectors
- Final decision: **STOP**
- Embedding source: `existing_node_embeddings`
- Embedding dim: `128`
- Chunk embedding coverage: 1.0000
- Query embedding available: `False`
- Semantic mode enabled: `False`
- Embedding missing rate: 0.0000

### Semantic Attribution

Query-to-chunk semantic attribution is unavailable unless query embeddings already exist in the examples. No synthetic query vector fallback is permitted.

### Renderer Table

| run | renderer_mode | oracle | diagnostic | answer_in_context | rendered_recall | context_f1 | qa_f1 | avg_context_tokens | retrieval_ms |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Expert Panel Checks

- GraphRAG expert: semantic candidates are graph-constrained to `T_q union S1`; global dense retrieval is not introduced.
- IR expert: semantic attribution cannot be interpreted without existing query embeddings.
- Graph theory expert: angular distance tests pass; semantic-weighted edges are positive in the diagnostic implementation.
- NLP expert: no semantic query ordering is run when query embeddings are absent, avoiding topical non-answer selection artifacts.
- Systems expert: no overnight smoke proceeds when required semantic inputs are unavailable.
- Professor/meta-reviewer: no thresholds, tuned weights, or dataset-specific semantic branches are introduced.

## Final Recommendation

STOP. Existing chunk embeddings are present, but query embeddings are missing from the processed examples. Per the experiment boundary, the run stops after implementation verification and embedding preflight rather than fabricating query vectors or falling back to another retrieval signal.
