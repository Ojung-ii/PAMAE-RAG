# Next Steps

## Current PAMAE v1 Limitation

The compact scaffold is now working: schema, evaluator metrics, renderer budgets, support labels, and 100-query summaries are all available. However, HotpotQA and 2WikiMultiHopQA both show `top_rho_compact` beating `refine_cell_compact` by a large margin on support F1 and recall-per-node.

This suggests the current PAMAE objective/distance pair is not sufficiently aligned with support evidence quality under compact retrieval. Refinement lowers the PAMAE objective, but it does not improve support F1 on the repaired datasets.

## Support Label State

| dataset | state |
| --- | --- |
| HotpotQA | repaired; 100-query gold universe recall 0.9800 |
| 2WikiMultiHopQA | repaired; 100-query gold universe recall 0.8740, universe-limited |
| MuSiQue | top-rho compact token budget now satisfies 1.0 after renderer fix |

## Recommended Direction

1. Improve `rho_q` first. The compact results are dominated by relevance ranking quality, so a better query-node relevance estimator is the most direct next move.
2. Revisit `d_q` after `rho_q`. A graph-aware shortest-path metric is a plausible v2 direction, but it should be evaluated after the relevance signal is less noisy.
3. Keep terminal-conditioned posterior as v2 work. It should not be mixed into v1 compact ablations.
4. Do not add dataset-specific support heuristics or gold-derived scoring. Gold labels remain evaluation-only.

## Commands For User-Approved Follow-Up

### Optional k-sensitivity on repaired datasets

```bash
cd /home/ojungii/PAMAE-RAG
source /home/ojungii/miniconda3/etc/profile.d/conda.sh
conda activate pamae-rag
export CUDA_VISIBLE_DEVICES=0

for ds in hotpotqa 2wikimultihopqa; do
  mkdir -p data/runs/${ds}/sensitivity_k
  for cfg in sensitivity_k1.yaml sensitivity_k2.yaml sensitivity_k3.yaml sensitivity_k4.yaml; do
    name=$(basename ${cfg} .yaml)
    outdir=data/runs/${ds}/sensitivity_k/${name}
    rm -rf ${outdir}
    mkdir -p ${outdir}

    python scripts/run_retrieval.py \
      --config configs/ablations/${cfg} \
      --input data/processed/${ds}/examples_100.jsonl \
      --output ${outdir}/results.jsonl

    python scripts/evaluate_retrieval.py \
      --input data/processed/${ds}/examples_100.jsonl \
      --predictions ${outdir}/results.jsonl \
      --output ${outdir}/metrics.json
  done

  python scripts/summarize_ablation.py \
    --runs-root data/runs/${ds}/sensitivity_k \
    --output-csv data/runs/${ds}/sensitivity_k_summary.csv \
    --output-md data/runs/${ds}/sensitivity_k_summary.md
done
```

### 500-query compact scale-up, after user approval

```bash
cd /home/ojungii/PAMAE-RAG
source /home/ojungii/miniconda3/etc/profile.d/conda.sh
conda activate pamae-rag
export CUDA_VISIBLE_DEVICES=0

for ds in hotpotqa 2wikimultihopqa musique; do
  python scripts/prepare_dataset.py \
    --dataset ${ds} \
    --qa data/raw/${ds}/${ds}.json \
    --corpus data/raw/${ds}/${ds}_corpus.json \
    --output data/processed/${ds}/examples_500.jsonl \
    --max-nodes-per-query 600 \
    --embedding-dim 128 \
    --limit 500
done

for ds in hotpotqa 2wikimultihopqa musique; do
  for variant in top_rho refine_cell; do
    run_name=component_${variant}_compact_500
    rm -rf data/runs/${ds}/${run_name}
    mkdir -p data/runs/${ds}/${run_name}

    python scripts/run_retrieval.py \
      --config configs/ablations/${ds}_component_${variant}_compact.yaml \
      --input data/processed/${ds}/examples_500.jsonl \
      --output data/runs/${ds}/${run_name}/results.jsonl

    python scripts/evaluate_retrieval.py \
      --input data/processed/${ds}/examples_500.jsonl \
      --predictions data/runs/${ds}/${run_name}/results.jsonl \
      --output data/runs/${ds}/${run_name}/metrics.json
  done

  python scripts/summarize_ablation.py \
    --runs-root data/runs/${ds} \
    --output-csv data/runs/${ds}/component_compact_500_summary.csv \
    --output-md data/runs/${ds}/component_compact_500_summary.md
done
```

### tmux wrapper for longer user-run jobs

```bash
tmux new -s pamae_nightly
cd /home/ojungii/PAMAE-RAG
source /home/ojungii/miniconda3/etc/profile.d/conda.sh
conda activate pamae-rag
export CUDA_VISIBLE_DEVICES=0
```

## Why Full Runs Were Held

Full or 500-query scale-up was intentionally not launched before 09:00 KST. The 100-query component evidence already shows `top_rho_compact` outperforming PAMAE refinement on the repaired datasets, so scaling the current v1 configuration would mostly confirm a known alignment issue rather than test a promising method change.
