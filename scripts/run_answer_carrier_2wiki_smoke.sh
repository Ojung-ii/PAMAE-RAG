#!/usr/bin/env bash
set -euo pipefail

MAX_QUERIES=50
OUT_ROOT="outputs/answer_carrier_smoke/2wikimultihopqa"
if [[ -z "${PYTHON_BIN:-}" && -x "/home/ojungii/miniconda3/envs/pamae-rag/bin/python" ]]; then
  PYTHON_BIN="/home/ojungii/miniconda3/envs/pamae-rag/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-queries)
      MAX_QUERIES="$2"
      shift 2
      ;;
    --out-root)
      OUT_ROOT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

INPUT="data/processed/2wikimultihopqa/examples_100.jsonl"
CONFIG="configs/content_graph/2wikimultihopqa_content_graph_refine_cell.yaml"

run_variant() {
  local renderer="$1"
  local out_dir="${OUT_ROOT}/entity_chunk_reference_${renderer}"
  "${PYTHON_BIN}" scripts/run_answer_carrier_variant.py \
    --config "${CONFIG}" \
    --input "${INPUT}" \
    --output-dir "${out_dir}" \
    --renderer-mode "${renderer}" \
    --limit "${MAX_QUERIES}" \
    --max-context-tokens 512
}

run_variant "current_renderer"
run_variant "projected_answer_chunk_oracle"
run_variant "selected_basin_answer_chunk_oracle"
run_variant "current_answer_role_oracle"
run_variant "gold_chunk_role_oracle"
