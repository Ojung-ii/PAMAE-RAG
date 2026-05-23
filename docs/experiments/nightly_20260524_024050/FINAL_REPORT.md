# Final Report

## Run Identity

- Branch: `experiment/nightly_pamae_compact_20260524_024050`
- Base commit: `1622cf28705cc72c0f811ba8468fc4004b75ff63`
- Final pre-report commit: `6eeb5eb`
- Final report commit: this file is committed immediately after report assembly; use `git rev-parse HEAD` after the final commit for the exact pushed hash.
- Start log: `docs/experiments/nightly_20260524_024050/README.md`
- tmux session name reserved for user-run jobs: `pamae_nightly`

## Environment

- Conda env: `pamae-rag`
- `CUDA_VISIBLE_DEVICES`: `0`
- Python command used for tests: `PYTHONPATH=src /home/ojungii/miniconda3/envs/pamae-rag/bin/python -m pytest -q`
- GPU check: `nvidia-smi` failed with a driver communication error; logs are in `00_nvidia_smi.log` and `00_nvidia_smi_retry.log`.
- CPU/GPU observation: current PAMAE-RAG preprocessing/retrieval is numpy/scikit-learn style CPU code. Low or absent GPU utilization is expected for these retrieval runs and was not treated as a code bug.

## Verification

- Initial pytest: `24 passed in 1.50s`
- Post-adapter pytest: `27 passed in 1.44s`
- Post-budget-fix pytest: `28 passed in 1.53s`
- Final pytest: `28 passed in 1.68s`
- Final `git diff --check`: passed with no output.
- Final `git status --short` before report assembly: clean.

## Code and Adapter Work

HotpotQA and 2Wiki support label extraction were repaired without changing the PAMAE objective and without injecting gold evidence into the retrieval universe.

Key changes:

- Added robust support-like field extraction for HotpotQA/2Wiki raw schemas.
- Added normalized-title matching between support labels and corpus documents.
- Added `scripts/check_gold_universe.py`.
- Removed gold-node oracle injection from candidate universe construction.
- Fixed renderer token budget enforcement so forced anchors cannot exceed `max_context_tokens`.

## Gold Universe Status

| dataset | num_queries | gold_total | gold_in_nodes | gold_universe_recall | all_gold_covered_ratio | status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| HotpotQA | 100 | 200 | 196 | 0.9800 | 0.9600 | repaired and evaluation-capable |
| 2WikiMultiHopQA | 100 | 246 | 215 | 0.8740 | 0.7100 | repaired, but universe-limited |

The previous `gold_total=0` failure is fixed for both datasets.

## MuSiQue Budget Check

MuSiQue `component_top_rho_compact` was rerun after the strict token-budget renderer fix.

| metric | value |
| --- | ---: |
| context_node_budget_satisfied_rate | 1.0 |
| context_token_budget_satisfied_rate | 1.0 |
| avg_context_size | 5.12 |
| avg_context_tokens | 503.85 |
| mean_context_recall | 0.2775 |
| mean_context_f1 | 0.1717 |

This confirms the previous top-rho token-budget violation is repaired.

## HotpotQA Component Results

All HotpotQA compact component variants satisfied both node and token budgets.

| variant | recall | precision | F1 | hit | recall/node | recall/1k tokens | avg nodes | avg tokens | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| top_rho | 0.7150 | 0.2523 | 0.3697 | 0.93 | 0.1262 | 1.4140 | 5.80 | 505.99 | 0.0447 |
| sample_only | 0.6150 | 0.1739 | 0.2702 | 0.86 | 0.0870 | 1.2267 | 7.10 | 502.56 | -0.0934 |
| full_validation | 0.5950 | 0.1698 | 0.2635 | 0.86 | 0.0849 | 1.1848 | 7.00 | 504.05 | -0.1009 |
| refine | 0.5600 | 0.1655 | 0.2547 | 0.82 | 0.0828 | 1.1129 | 6.63 | 504.64 | 0.1208 |
| refine_cell | 0.5600 | 0.1655 | 0.2547 | 0.82 | 0.0828 | 1.1129 | 6.63 | 504.64 | 0.1208 |

Decision: Case C. `top_rho_compact` is clearly better than `refine_cell_compact`.

## 2WikiMultiHopQA Component Results

All 2Wiki compact component variants satisfied both node and token budgets. Interpret results with the universe-limited caveat from gold universe recall 0.8740.

| variant | recall | precision | F1 | hit | recall/node | recall/1k tokens | avg nodes | avg tokens | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| top_rho | 0.6350 | 0.2732 | 0.3692 | 0.97 | 0.1168 | 1.2532 | 5.78 | 507.77 | 0.0817 |
| sample_only | 0.5900 | 0.1836 | 0.2746 | 0.94 | 0.0773 | 1.2420 | 7.50 | 485.55 | 0.2214 |
| full_validation | 0.5800 | 0.1854 | 0.2755 | 0.93 | 0.0776 | 1.2111 | 7.41 | 488.92 | 0.2229 |
| refine | 0.5675 | 0.1849 | 0.2724 | 0.92 | 0.0774 | 1.1673 | 7.29 | 494.85 | 0.1987 |
| refine_cell | 0.5675 | 0.1849 | 0.2724 | 0.92 | 0.0774 | 1.1673 | 7.29 | 494.85 | 0.1987 |

Decision: Case C. `top_rho_compact` is clearly better than `refine_cell_compact`.

## Sensitivity Results

k-sensitivity was not run. The component results finished well before the time cutoff, but both repaired datasets already met Case C. Sensitivity would not address the main observed issue: PAMAE objective reduction does not translate into support F1 gains under the current `rho_q/d_q`.

## Skipped Experiments

- 500-query runs: skipped by instruction before 09:00 KST and because 100-query Case C does not justify scaling this v1 setting automatically.
- Full-dataset runs: skipped by instruction before 09:00 KST.
- sample-size sensitivity and m sensitivity: skipped by instruction and because Case C indicates the signal, not sampling scale, is the limiting factor.
- graph-aware metric and terminal-conditioned posterior: not implemented tonight by boundary rule; reserved for future work.

## Final Decision

Move to `rho_q/d_q` improvement before scaling PAMAE v1.

The compact evaluation scaffold is now usable and budget-compliant, and HotpotQA/2Wiki labels are repaired. But on both repaired datasets, compact top-rho ranking is stronger than the PAMAE refinement variants. That means the immediate bottleneck is relevance/distance alignment, not renderer wiring or sample-search mechanics.

## Remaining Commands

### Optional k-sensitivity

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

### User-approved 500-query top-rho vs refine-cell scale-up

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

### tmux attach pattern

```bash
tmux new -s pamae_nightly
# or later:
tmux attach -t pamae_nightly
```

## Time-Limit Decisions

- New long retrieval grids were completed before 07:30 KST.
- No sensitivity was launched before 08:20 KST because Case C was already decisive.
- No 500-query or full-dataset run was launched before 09:00 KST.
