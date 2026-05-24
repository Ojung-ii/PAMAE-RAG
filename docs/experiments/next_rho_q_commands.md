# Next rho_q Commands

These commands run 100-query rho alignment ablations only. Do not launch 500-query or full-dataset runs until the 100-query results show improved support F1 or recall-per-token.

## Relevance Alignment Diagnostics

```bash
cd /home/ojungii/PAMAE-RAG
source /home/ojungii/miniconda3/etc/profile.d/conda.sh
conda activate pamae-rag

for ds in hotpotqa 2wikimultihopqa; do
  for mode in current title_aware entity_title_aware hybrid_title_semantic; do
    mkdir -p data/runs/${ds}/rho_alignment
    python scripts/analyze_relevance_alignment.py \
      --input data/processed/${ds}/examples_100.jsonl \
      --relevance-mode ${mode} \
      --output-json data/runs/${ds}/rho_alignment/${mode}.json \
      --output-md data/runs/${ds}/rho_alignment/${mode}.md
  done
done
```

## HotpotQA rho_q 100 Runs

```bash
cd /home/ojungii/PAMAE-RAG
source /home/ojungii/miniconda3/etc/profile.d/conda.sh
conda activate pamae-rag

for cfg in configs/ablations_rho/hotpotqa_rho_*.yaml; do
  variant=$(basename "${cfg}" .yaml)
  rm -rf data/runs/hotpotqa/${variant}
  mkdir -p data/runs/hotpotqa/${variant}
  python scripts/run_retrieval.py \
    --config "${cfg}" \
    --input data/processed/hotpotqa/examples_100.jsonl \
    --output data/runs/hotpotqa/${variant}/results.jsonl
  python scripts/evaluate_retrieval.py \
    --input data/processed/hotpotqa/examples_100.jsonl \
    --predictions data/runs/hotpotqa/${variant}/results.jsonl \
    --output data/runs/hotpotqa/${variant}/metrics.json
done

python scripts/summarize_ablation.py \
  --runs-root data/runs/hotpotqa \
  --output-csv data/runs/hotpotqa/rho_q_summary.csv \
  --output-md data/runs/hotpotqa/rho_q_summary.md
```

## 2Wiki rho_q 100 Runs

```bash
cd /home/ojungii/PAMAE-RAG
source /home/ojungii/miniconda3/etc/profile.d/conda.sh
conda activate pamae-rag

for cfg in configs/ablations_rho/2wikimultihopqa_rho_*.yaml; do
  variant=$(basename "${cfg}" .yaml)
  rm -rf data/runs/2wikimultihopqa/${variant}
  mkdir -p data/runs/2wikimultihopqa/${variant}
  python scripts/run_retrieval.py \
    --config "${cfg}" \
    --input data/processed/2wikimultihopqa/examples_100.jsonl \
    --output data/runs/2wikimultihopqa/${variant}/results.jsonl
  python scripts/evaluate_retrieval.py \
    --input data/processed/2wikimultihopqa/examples_100.jsonl \
    --predictions data/runs/2wikimultihopqa/${variant}/results.jsonl \
    --output data/runs/2wikimultihopqa/${variant}/metrics.json
done

python scripts/summarize_ablation.py \
  --runs-root data/runs/2wikimultihopqa \
  --output-csv data/runs/2wikimultihopqa/rho_q_summary.csv \
  --output-md data/runs/2wikimultihopqa/rho_q_summary.md
```

## Judgment Criteria

1. Does an improved `rho_q` mode beat existing title-aware `top_rho` on support F1 or recall-per-token?
2. Does improved `rho_q` reduce the `refine_cell` vs `top_rho` gap?
3. Does `objective_support_spearman` improve?
4. Do gold rank top10/top20 rates improve in `analyze_relevance_alignment.py`?

If these do not improve on 100-query runs, do not scale to 500 or full datasets. Move to `d_q` graph metric work instead.
