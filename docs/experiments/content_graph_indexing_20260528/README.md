# Content Graph Indexing QA Diagnostics

This experiment series redesigns graph indexing toward dependency-free content-derived entity/fact graphs, then adds stage-wise diagnostics before any performance-oriented change.

Fixed QA measurement settings introduced for this series:

- generator: `deterministic_extractive_sentence_v1`
- prompt: `closed_book_free_context_qa_v1`
- metric: `squad_normalized_em_f1_v1`
- sample: first 20 rows from `data/processed/2wikimultihopqa/examples_100.jsonl`
- seed: config seed `42`
- oracle corpus: `data/processed/2wikimultihopqa_corpus.json`

The default generator is an offline deterministic extractive sentence generator. It is not an LLM upper bound; all comparisons in this series must use the same generator and metric.

Final QA diagnostics now also record `answer_coverage`: whether any normalized gold answer string appears in the rendered context. This is evaluation-only bookkeeping; it is not used by retrieval, scoring, context rendering, or answer generation.

Current 2Wiki20 QA measurements:

| run | graph_mode | oracle | candidate_recall | projected_recall | post_refine_recall | rendered_recall | context_f1 | avg_context_tokens | retrieval_ms | generation_ms | EM | F1 | oracle_gap | risk_decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline_2wiki20 | legacy_hybrid_sem_graph | false | 0.8625 | n/a | 0.0250 | 0.5000 | 0.2347 | 497.1 | 311.8 | 0.4 | 0.0000 | 0.0355 | 0.0140 | measurement_only |
| oracle_2wiki20 | direct_gold_context | true | n/a | n/a | n/a | 1.0000 | 1.0000 | 279.7 | 0.0 | 0.2 | 0.0000 | 0.0495 | 0.0000 | measurement_only |
| content_graph_2wiki20 | content_hybrid_sem_graph | false | 0.8625 | 0.8000 | 0.3500 | 0.6000 | 0.2843 | 350.1 | 967.6 | 0.3 | 0.0000 | 0.0580 | -0.0085 | measurement_limited |
| baseline_hotpot20 | legacy_hybrid_sem_graph | false | 0.9750 | n/a | 0.1500 | 0.7250 | 0.3327 | 504.4 | 318.3 | 0.4 | 0.0000 | 0.0672 | 0.0040 | measurement_only |
| oracle_hotpot20 | direct_gold_context | true | n/a | n/a | n/a | 1.0000 | 1.0000 | 153.0 | 0.0 | 0.1 | 0.0000 | 0.0711 | 0.0000 | measurement_only |
| content_graph_hotpot20 | content_hybrid_sem_graph | false | 0.9750 | 0.9250 | 0.3750 | 0.6750 | 0.2871 | 478.8 | 676.9 | 0.4 | 0.0000 | 0.0624 | 0.0087 | no_adoption |

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
- content graph projection latency: `830.8 ms` average
- latency: `967.6 ms` retrieval average vs baseline `311.8 ms`

Measurement caution:

The fixed offline extractive generator gives content graph F1 above oracle F1. That means the current oracle is not a strict upper bound for this generator. Do not claim successful oracle-gap reduction until the generator/oracle measurement is strengthened or replaced by a real fixed QA generator.

HotpotQA risk check:

The content graph improves refinement survival (`0.3750` vs `0.1500`) but reduces rendered recall (`0.6750` vs `0.7250`) and QA F1 (`0.0624` vs `0.0672`). This prevents adoption as a performance improvement under the QA-gated criterion.

Automated comparison guard:

- 2Wiki20: [compare_2wiki20.md](compare_2wiki20.md) reports `oracle_dominance_valid=false` because `content_graph_2wiki20` exceeds oracle F1 under the offline deterministic generator. This blocks oracle-gap claims on 2Wiki.
- Hotpot20: [compare_hotpot20.md](compare_hotpot20.md) reports `oracle_dominance_valid=true`, but content graph F1 is lower than baseline F1. This blocks content graph adoption as a performance improvement.
- The guard command supports `--require-valid-oracle`; with that flag, the 2Wiki comparison exits nonzero and the Hotpot comparison passes.

Answer selection diagnostic:

| run | answer_coverage | selected_answer_coverage |
| --- | ---: | ---: |
| baseline_2wiki20 | 0.3000 | 0.0500 |
| oracle_2wiki20 | 0.8000 | 0.0500 |
| content_graph_2wiki20 | 0.3000 | 0.0500 |
| baseline_hotpot20 | 0.5000 | 0.2500 |
| oracle_hotpot20 | 0.9000 | 0.3000 |
| content_graph_hotpot20 | 0.5500 | 0.2000 |

The oracle contexts contain answer strings far more often than retrieved contexts, but the fixed extractive sentence generator often selects a non-answer sentence even when the answer appears in the context. This confirms a final-QA selection bottleneck in addition to retrieval/context losses, and keeps performance changes gated on end-to-end QA rather than retrieval-only gains.
