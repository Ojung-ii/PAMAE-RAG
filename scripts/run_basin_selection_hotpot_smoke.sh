#!/usr/bin/env bash
set -euo pipefail

MAX_QUERIES="${MAX_QUERIES:-100}"
OUT_ROOT="${OUT_ROOT:-outputs/basin_selection_smoke/hotpotqa}"
INPUT="data/processed/hotpotqa/examples_100.jsonl"
CORPUS="data/processed/hotpotqa_corpus.json"

run_variant() {
  local name="$1"
  local config="$2"
  local out_dir="${OUT_ROOT}/${name}"
  mkdir -p "${out_dir}"
  python scripts/run_retrieval.py \
    --config "${config}" \
    --input "${INPUT}" \
    --output "${out_dir}/retrieval.jsonl" \
    --limit "${MAX_QUERIES}"
  python scripts/evaluate_retrieval.py \
    --input "${INPUT}" \
    --predictions "${out_dir}/retrieval.jsonl" \
    --output "${out_dir}/retrieval_metrics.json" \
    --limit "${MAX_QUERIES}"
  python scripts/run_qa.py \
    --input "${INPUT}" \
    --predictions "${out_dir}/retrieval.jsonl" \
    --output "${out_dir}/qa.jsonl" \
    --metrics-output "${out_dir}/qa_metrics.json" \
    --limit "${MAX_QUERIES}"
  python scripts/analyze_failure_taxonomy.py \
    --input "${INPUT}" \
    --qa "${out_dir}/qa.jsonl" \
    --retrieval "${out_dir}/retrieval.jsonl" \
    --output "${out_dir}/failure_taxonomy.json" \
    --limit "${MAX_QUERIES}"
}

run_variant "current_content" "configs/content_graph/hotpotqa_content_graph_refine_cell.yaml"
run_variant "basin_preserving_selection" "configs/content_graph/hotpotqa_content_graph_basin_preserving.yaml"
run_variant "basin_preserving_selection_plus_basin_renderer" "configs/content_graph/hotpotqa_content_graph_basin_preserving_basin_renderer.yaml"

python scripts/run_oracle_diagnostics.py \
  --input "${INPUT}" \
  --corpus "${CORPUS}" \
  --method-metrics "${OUT_ROOT}/current_content/qa_metrics.json" \
  --output "${OUT_ROOT}/oracle_diagnostics.json" \
  --limit "${MAX_QUERIES}"

python scripts/compare_basin_selection_runs.py \
  --runs "${OUT_ROOT}/current_content" \
         "${OUT_ROOT}/basin_preserving_selection" \
         "${OUT_ROOT}/basin_preserving_selection_plus_basin_renderer" \
  --oracle "${OUT_ROOT}/oracle_diagnostics.json" \
  --out "${OUT_ROOT}/basin_selection_comparison.md"
