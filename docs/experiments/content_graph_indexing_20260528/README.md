# Content Graph Indexing QA Diagnostics

This experiment series redesigns graph indexing toward dependency-free content-derived entity/fact graphs, then adds stage-wise diagnostics before any performance-oriented change.

Fixed QA measurement settings introduced for this series:

- generator: `deterministic_extractive_sentence_v1`
- prompt: `closed_book_free_context_qa_v1`
- metric: `squad_normalized_em_f1_v1`
- sample: first 20 rows from each processed QA file
- seed: config seed `42`
- oracle corpus: matching processed corpus JSON

The default generator is an offline deterministic extractive sentence generator. It is not an LLM upper bound; all comparisons in this series must use the same generator and metric.

Final QA diagnostics now also record `answer_coverage`: whether any normalized gold answer string appears in the rendered context. This is evaluation-only bookkeeping; it is not used by retrieval, scoring, context rendering, or answer generation.

Oracle QA now renders gold support sentences from `support_facts` when all support facts resolve against the gold context, falling back to full gold nodes only when support sentence resolution is incomplete. This is oracle-only context construction and is not used by retrieval, scoring, graph construction, or content rendering.

Stage diagnostics now also record `support_fact_survival`, `support_fact_resolved_survival`, and support-fact counts. These are evaluation-only gold support fact sentence metrics and are not used by retrieval, graph construction, scoring, rendering, or generation.

Current 2Wiki20 QA measurements:

| run | graph_mode | oracle | candidate_recall | projected_recall | post_refine_recall | rendered_recall | context_f1 | avg_context_tokens | retrieval_ms | generation_ms | EM | F1 | oracle_gap | risk_decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline_2wiki20 | legacy_hybrid_sem_graph | false | 0.8625 | n/a | 0.0250 | 0.5000 | 0.2347 | 497.1 | 381.9 | 0.5 | 0.0000 | 0.0355 | 0.0140 | measurement_only |
| oracle_2wiki20 | direct_gold_context | true | n/a | n/a | n/a | 1.0000 | 1.0000 | 47.4 | 0.0 | 0.1 | 0.0000 | 0.0495 | 0.0000 | measurement_only |
| content_graph_2wiki20 | content_hybrid_sem_graph | false | 0.8625 | 0.8000 | 0.3500 | 0.6000 | 0.2843 | 350.1 | 1137.0 | 0.4 | 0.0000 | 0.0580 | -0.0085 | measurement_limited |
| baseline_hotpot20 | legacy_hybrid_sem_graph | false | 0.9750 | n/a | 0.1500 | 0.7250 | 0.3327 | 504.4 | 419.5 | 0.5 | 0.0000 | 0.0672 | 0.0254 | measurement_only |
| oracle_hotpot20 | direct_gold_context | true | n/a | n/a | n/a | 1.0000 | 1.0000 | 56.2 | 0.0 | 0.1 | 0.0000 | 0.0925 | 0.0000 | measurement_only |
| content_graph_hotpot20 | content_hybrid_sem_graph | false | 0.9750 | 0.9250 | 0.3750 | 0.6750 | 0.2871 | 478.8 | 923.5 | 0.5 | 0.0000 | 0.0624 | 0.0301 | no_adoption |

Current stage-wise bottleneck read:

- query/anchor construction survival: `0.8625`
- candidate generation recall: `0.8625`
- content graph projection: `not_configured`
- local refinement survival: `0.0250`
- reranking/scoring survival: `0.0250`
- context rendering recall: `0.5000`
- final QA F1: `0.0355`

Content graph run:

- content graph projection survival: `0.8000`
- local refinement survival: `0.3500`
- reranking/scoring survival: `0.3500`
- context rendering recall: `0.6000`
- final QA F1: `0.0580`
- content graph projection latency: `972.0 ms` average
- latency: `1137.0 ms` retrieval average vs baseline `381.9 ms`

Support-fact survival diagnostics:

| run | query_anchor | candidate_generation | projection | local_refinement | rendered | final_qa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_2wiki20 | 0.8625 | 0.5750 | n/a | 0.0250 | 0.5000 | 0.5000 |
| content_graph_2wiki20 | 0.8625 | 0.7375 | 0.8000 | 0.3500 | 0.6000 | 0.6000 |
| baseline_hotpot20 | 0.9750 | 0.8200 | n/a | 0.1583 | 0.7117 | 0.7117 |
| content_graph_hotpot20 | 0.9750 | 0.9250 | 0.9250 | 0.3833 | 0.6750 | 0.6750 |

Measurement caution:

The support-sentence oracle restores a stronger HotpotQA upper bound (`0.0925` F1), but 2Wiki20 still has `content_graph_2wiki20` above oracle F1 under the fixed offline generator. Do not claim oracle-gap reduction on 2Wiki until that measurement limitation is resolved; use HotpotQA as the valid oracle-gap risk check in this series.

HotpotQA risk check:

The content graph improves refinement survival (`0.3750` vs `0.1500`) and support-fact local refinement survival (`0.3833` vs `0.1583`), but reduces rendered recall (`0.6750` vs `0.7250`), support-fact rendered survival (`0.6750` vs `0.7117`), and QA F1 (`0.0624` vs `0.0672`). With the support-sentence oracle, the Hotpot oracle gap is `0.0301` for content graph versus `0.0254` for baseline, so content graph remains `no_adoption` as a performance improvement.

Automated comparison guard:

- 2Wiki20: [compare_2wiki20.md](compare_2wiki20.md) reports `oracle_dominance_valid=false` because `content_graph_2wiki20` exceeds oracle F1 under the offline deterministic generator. This blocks oracle-gap claims on 2Wiki.
- Hotpot20: [compare_hotpot20.md](compare_hotpot20.md) reports `oracle_dominance_valid=true`, but content graph F1 is lower than baseline F1 and the oracle gap is larger. This blocks content graph adoption as a performance improvement.
- The guard command supports `--require-valid-oracle`; with that flag, the 2Wiki comparison exits nonzero and the Hotpot comparison passes.

Answer selection diagnostic:

| run | answer_coverage | selected_answer_coverage |
| --- | ---: | ---: |
| baseline_2wiki20 | 0.3000 | 0.0500 |
| oracle_2wiki20 | 0.6000 | 0.1000 |
| content_graph_2wiki20 | 0.3000 | 0.0500 |
| baseline_hotpot20 | 0.5000 | 0.2500 |
| oracle_hotpot20 | 0.9000 | 0.3500 |
| content_graph_hotpot20 | 0.5500 | 0.2000 |

The oracle contexts contain answer strings far more often than retrieved contexts, but the fixed extractive sentence generator often selects a non-answer sentence even when the answer appears in the context. This confirms a final-QA selection bottleneck in addition to retrieval/context losses, and keeps performance changes gated on end-to-end QA rather than retrieval-only gains.

Risk-gated ideas stopped:

- fact-scoped projection: reduced Hotpot QA F1 to `0.0505`; not adopted.
- query-overlap sentence rendering: did not improve Hotpot content QA F1 (`0.0624` unchanged); not adopted.
- body-only lexical relevance: reduced Hotpot gold top-8 recall (`0.6250` vs current relevance `0.7750`); not adopted.
- content query-entity relevance: mixed signal, with Hotpot top-8 recall lower than current relevance (`0.7500` vs `0.7750`); not adopted.
- content fact closure context filtering: one-hop content graph Hotpot probe reached only `0.0633` F1, below baseline `0.0672` and with no oracle-gap reduction; not adopted.
