# Relevance Alignment Diagnostic

| metric | value |
| --- | ---: |
| input | data/processed/hotpotqa/examples_100.jsonl |
| relevance_mode | title_aware |
| num_queries | 100 |
| gold_total | 200 |
| gold_rank_mean | 26.775510204081634 |
| gold_rank_median | 2.0 |
| gold_top1_rate | 0.4 |
| gold_top3_rate | 0.63 |
| gold_top5_rate | 0.7 |
| gold_top10_rate | 0.735 |
| gold_top20_rate | 0.795 |
| gold_top50_rate | 0.865 |
| examples_with_no_gold | 0 |
| examples_with_gold_outside_top50 | 24 |
| mean_gold_relevance | 0.3625280802850005 |
| mean_non_gold_relevance | 0.13875970456852108 |
| relevance_label_auc | 0.9431326005086001 |
| relevance_label_spearman | 0.08761753061056776 |

## Sample Queries Outside Top 50

- `5abe953b5542993f32c2a170`
- `5ab57fc4554299488d4d99c0`
- `5ade9c9c5542997c77adee8c`
- `5a7a3a945542996a35c17147`
- `5a8f5b6f554299458435d5e7`
- `5ab8829d55429934fafe6e08`
- `5a7b1c0a55429931da12c9c9`
- `5adecc7755429975fa854f9d`
- `5a77aa095542995d83181260`
- `5a790d8f554299029c4b5eec`
