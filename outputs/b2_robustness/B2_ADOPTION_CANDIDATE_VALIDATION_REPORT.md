# B2 Adoption Candidate Validation Report

- Branch: `validate/b2-semantic-carrier-robustness`
- Commit: `14f67c0`
- Fixed common_qa prompt hash: `31e4b446be8b00a4989078fb4a957bc61b19bf4b8014674e2baad4612cc4396d`
- Embedding model: `nvidia/NV-Embed-v2`, dim `4096`, shared query/chunk space, L2 normalized.
- Variants: `current_renderer, metric_path_carrier, tree_shell1_graph_order, tree_shell1_semantic_query_order`
- Primary pass: `False`
- Secondary run: skipped because the primary Hotpot gate failed on retrieval time.
- Final decision: **STOP**
- Next recommendation: Do not adopt B2; inspect the gate blockers and keep the graph-constrained semantic layer experimental.

Verification status: `/home/ojungii/miniconda3/envs/QMRAG/bin/python -m compileall src tests scripts` passed; `/home/ojungii/miniconda3/envs/pamae-rag/bin/python -m pytest -q` passed with 152 tests. Prompt, sample, embedding compatibility, score-mixing, oracle-leakage, and B2 graph-constraint fields were checked from emitted artifacts.

Run size: the requested 1000-query primary run was attempted first, then explicitly fell back to 500 queries after NV-Embed cache expansion exceeded the practical time/server envelope and sandboxed CUDA was unavailable. Both 2Wiki and Hotpot used the same documented 500-query fallback protocol with explicit CUDA outside the sandbox.

Variant definitions: A0 current_renderer is the reference; A1 metric_path_carrier renders strict support-tree chunks; B1 tree_shell1_graph_order renders T_q union S1 by graph order; B2 tree_shell1_semantic_query_order renders the same graph-constrained pool with query angular distance as a lexicographic tie-breaker.

## 2wikimultihopqa

- Decision: **ADOPTION_CANDIDATE_CONFIRMED**
- B2 gate pass: `True` blockers `none`
- Strong confirmation: `False` blockers `strict_time_gate, strict_token_gate`
- Same sample: `True` (identical query IDs and order)
- Prompt: `common_qa` hash `31e4b446be8b00a4989078fb4a957bc61b19bf4b8014674e2baad4612cc4396d` exact `True`
- Embedding: `nvidia/NV-Embed-v2` dim `4096` query coverage 1.0000, chunk coverage 1.0000
- B2 semantic attribution query separation: 0.0227
- B2 semantic attribution tree separation: 0.0172

### Variant Table

| renderer | EM | QA F1 | answer_in_context | rendered_recall | context_f1 | avg_tokens | retrieval_ms | generation_ms | total_ms | support_tree_answer | shell1_answer | shell1_rendered_answer | bridge_retained | bridge_cut | shell1_chunks | rendered_shell1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_renderer | 0.0020 | 0.0790 | 0.4700 | 0.5905 | 0.2668 | 350.0860 | 712.6004 | 0.1604 | 712.7608 | 0.4080 | 0.0000 | 0.0000 | 0.1960 | 0.0160 | 0.0000 | 0.0000 |
| metric_path_carrier | 0.0020 | 0.0794 | 0.3400 | 0.4640 | 0.3449 | 191.7640 | 720.0184 | 0.0931 | 720.1114 | 0.4080 | 0.0000 | 0.0000 | 0.1300 | 0.0120 | 0.0000 | 0.0000 |
| tree_shell1_graph_order | 0.0020 | 0.0792 | 0.4100 | 0.5280 | 0.2440 | 422.4900 | 846.4272 | 0.1883 | 846.6155 | 0.4080 | 0.3260 | 0.0180 | 0.2000 | 0.0120 | 75.3220 | 3.2560 |
| tree_shell1_semantic_query_order | 0.0020 | 0.0811 | 0.5060 | 0.6385 | 0.2906 | 372.0160 | 851.2731 | 0.1661 | 851.4392 | 0.4080 | 0.3260 | 0.1360 | 0.1980 | 0.0140 | 75.3220 | 3.4180 |

### Paired Deltas

| comparison | metric | mean | ci95 | improved | tied | regressed |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| paired_delta_B2_minus_current | qa_f1 | 0.0021 | [-0.0015, 0.0055] | 16 | 476 | 8 |
| paired_delta_B2_minus_current | answer_in_context | 0.0360 | [0.0080, 0.0640] | 35 | 448 | 17 |
| paired_delta_B2_minus_current | rendered_recall | 0.0480 | [0.0300, 0.0665] | 80 | 395 | 25 |
| paired_delta_B2_minus_current | context_f1 | 0.0237 | [0.0153, 0.0320] | 100 | 354 | 46 |
| paired_delta_B2_minus_current | context_tokens | 21.9300 | [14.0300, 29.9640] | 286 | 14 | 200 |
| paired_delta_B2_minus_current | retrieval_ms | 138.6727 | [130.8733, 145.7445] | 478 | 0 | 22 |
| paired_delta_B2_minus_B1 | qa_f1 | 0.0019 | [-0.0029, 0.0065] | 16 | 470 | 14 |
| paired_delta_B2_minus_B1 | answer_in_context | 0.0960 | [0.0700, 0.1240] | 52 | 444 | 4 |
| paired_delta_B2_minus_B1 | rendered_recall | 0.1105 | [0.0930, 0.1300] | 127 | 371 | 2 |
| paired_delta_B2_minus_B1 | context_f1 | 0.0466 | [0.0377, 0.0557] | 143 | 295 | 62 |
| paired_delta_B2_minus_B1 | context_tokens | -50.4740 | [-60.4100, -40.3000] | 160 | 9 | 331 |
| paired_delta_B2_minus_B1 | retrieval_ms | 4.8459 | [-1.5445, 11.2468] | 273 | 0 | 227 |
| paired_delta_B1_minus_A1 | qa_f1 | -0.0003 | [-0.0054, 0.0046] | 11 | 479 | 10 |
| paired_delta_B1_minus_A1 | answer_in_context | 0.0700 | [0.0480, 0.0960] | 35 | 465 | 0 |
| paired_delta_B1_minus_A1 | rendered_recall | 0.0640 | [0.0490, 0.0770] | 78 | 422 | 0 |
| paired_delta_B1_minus_A1 | context_f1 | -0.1009 | [-0.1125, -0.0893] | 67 | 64 | 369 |
| paired_delta_B1_minus_A1 | context_tokens | 230.7260 | [221.6880, 239.7580] | 500 | 0 | 0 |
| paired_delta_B1_minus_A1 | retrieval_ms | 126.4088 | [118.8224, 134.1292] | 465 | 0 | 35 |

## hotpotqa

- Decision: **STOP**
- B2 gate pass: `False` blockers `retrieval_time_gate`
- Strong confirmation: `False` blockers `strict_time_gate`
- Same sample: `True` (identical query IDs and order)
- Prompt: `common_qa` hash `31e4b446be8b00a4989078fb4a957bc61b19bf4b8014674e2baad4612cc4396d` exact `True`
- Embedding: `nvidia/NV-Embed-v2` dim `4096` query coverage 1.0000, chunk coverage 1.0000
- B2 semantic attribution query separation: 0.0480
- B2 semantic attribution tree separation: 0.0474

### Variant Table

| renderer | EM | QA F1 | answer_in_context | rendered_recall | context_f1 | avg_tokens | retrieval_ms | generation_ms | total_ms | support_tree_answer | shell1_answer | shell1_rendered_answer | bridge_retained | bridge_cut | shell1_chunks | rendered_shell1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_renderer | 0.0020 | 0.0719 | 0.6120 | 0.6280 | 0.2685 | 480.6920 | 625.2895 | 0.2048 | 625.4944 | 0.5260 | 0.0000 | 0.0000 | 0.2180 | 0.0220 | 0.0000 | 0.0000 |
| metric_path_carrier | 0.0020 | 0.0666 | 0.4860 | 0.4770 | 0.3235 | 269.8580 | 630.7728 | 0.1257 | 630.8985 | 0.5260 | 0.0000 | 0.0000 | 0.1740 | 0.0060 | 0.0000 | 0.0000 |
| tree_shell1_graph_order | 0.0020 | 0.0685 | 0.5440 | 0.5430 | 0.2406 | 492.3660 | 856.2694 | 0.2091 | 856.4785 | 0.5260 | 0.5240 | 0.0600 | 0.2280 | 0.0100 | 78.2340 | 2.6540 |
| tree_shell1_semantic_query_order | 0.0020 | 0.0743 | 0.7300 | 0.7770 | 0.3366 | 480.4460 | 861.0502 | 0.2084 | 861.2586 | 0.5260 | 0.5240 | 0.3380 | 0.2320 | 0.0060 | 78.2340 | 2.7840 |

### Paired Deltas

| comparison | metric | mean | ci95 | improved | tied | regressed |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| paired_delta_B2_minus_current | qa_f1 | 0.0025 | [-0.0027, 0.0073] | 28 | 456 | 16 |
| paired_delta_B2_minus_current | answer_in_context | 0.1180 | [0.0840, 0.1520] | 72 | 415 | 13 |
| paired_delta_B2_minus_current | rendered_recall | 0.1490 | [0.1230, 0.1760] | 162 | 312 | 26 |
| paired_delta_B2_minus_current | context_f1 | 0.0681 | [0.0567, 0.0799] | 231 | 210 | 59 |
| paired_delta_B2_minus_current | context_tokens | -0.2460 | [-3.6600, 3.5660] | 205 | 38 | 257 |
| paired_delta_B2_minus_current | retrieval_ms | 235.7607 | [227.7377, 244.1578] | 499 | 0 | 1 |
| paired_delta_B2_minus_B1 | qa_f1 | 0.0058 | [0.0006, 0.0113] | 31 | 456 | 13 |
| paired_delta_B2_minus_B1 | answer_in_context | 0.1860 | [0.1500, 0.2220] | 97 | 399 | 4 |
| paired_delta_B2_minus_B1 | rendered_recall | 0.2340 | [0.2100, 0.2600] | 210 | 289 | 1 |
| paired_delta_B2_minus_B1 | context_f1 | 0.0959 | [0.0843, 0.1068] | 250 | 190 | 60 |
| paired_delta_B2_minus_B1 | context_tokens | -11.9200 | [-15.5020, -8.5540] | 181 | 43 | 276 |
| paired_delta_B2_minus_B1 | retrieval_ms | 4.7807 | [-1.7305, 11.4819] | 272 | 0 | 228 |
| paired_delta_B1_minus_A1 | qa_f1 | 0.0019 | [-0.0016, 0.0059] | 10 | 481 | 9 |
| paired_delta_B1_minus_A1 | answer_in_context | 0.0580 | [0.0380, 0.0780] | 29 | 471 | 0 |
| paired_delta_B1_minus_A1 | rendered_recall | 0.0660 | [0.0520, 0.0800] | 66 | 434 | 0 |
| paired_delta_B1_minus_A1 | context_f1 | -0.0829 | [-0.0946, -0.0711] | 61 | 110 | 329 |
| paired_delta_B1_minus_A1 | context_tokens | 222.5080 | [214.1000, 230.5980] | 492 | 8 | 0 |
| paired_delta_B1_minus_A1 | retrieval_ms | 225.4966 | [218.5426, 232.3476] | 498 | 0 | 2 |

## Local-Minimum Guard

- Did we change the PAMAE core retrieval objective? No.
- Did we use global dense retrieval? No.
- Did we use scalar score mixing? No.
- Did semantic ordering improve beyond shell expansion? See paired `B2_minus_B1` deltas above.
- Did B2 preserve answer coverage and rendered recall? See B2 gate checklist above.
- Did B2 remain stable on both 2Wiki and Hotpot? Captured by primary pass.
- Did B2 avoid excessive token/time cost? Captured by token and retrieval gates.
- Is the improvement likely general or dataset-specific? Secondary availability and results determine whether this stays diagnostic-only.
