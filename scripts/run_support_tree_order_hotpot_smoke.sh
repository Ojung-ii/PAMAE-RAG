#!/usr/bin/env bash
set -euo pipefail

MAX_QUERIES=100
OUT_ROOT="outputs/support_tree_order_smoke/hotpotqa"
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

INPUT="data/processed/hotpotqa/examples_100.jsonl"
CURRENT_CONFIG="configs/content_graph/hotpotqa_content_graph_refine_cell.yaml"
METRIC_CONFIG="configs/content_graph/hotpotqa_content_graph_metric_path_carrier.yaml"

run_variant() {
  local renderer="$1"
  local config="$2"
  local override="${3:-}"
  local out_dir="${OUT_ROOT}/entity_chunk_reference_${renderer}"
  local args=(
    scripts/run_answer_carrier_variant.py
    --config "${config}"
    --input "${INPUT}"
    --output-dir "${out_dir}"
    --renderer-mode "${renderer}"
    --limit "${MAX_QUERIES}"
    --max-context-tokens 512
  )
  if [[ -n "${override}" ]]; then
    args+=(--renderer-override "${override}")
  fi
  "${PYTHON_BIN}" "${args[@]}"
}

run_variant "current_renderer" "${CURRENT_CONFIG}"
run_variant "metric_path_carrier" "${METRIC_CONFIG}"
run_variant "tree_all_no_budget" "${CURRENT_CONFIG}"
run_variant "tree_current_budget_order" "${CURRENT_CONFIG}"
run_variant "current_tree_intersection_only" "${CURRENT_CONFIG}"
run_variant "current_only_non_tree" "${CURRENT_CONFIG}"
run_variant "tree_answer_oracle" "${CURRENT_CONFIG}"

"${PYTHON_BIN}" scripts/build_support_tree_order_outputs.py \
  --config "${CURRENT_CONFIG}" \
  --input "${INPUT}" \
  --root "${OUT_ROOT}" \
  --limit "${MAX_QUERIES}"
