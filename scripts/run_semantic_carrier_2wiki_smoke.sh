#!/usr/bin/env bash
set -euo pipefail

MAX_QUERIES=50
OUT_ROOT="outputs/semantic_carrier_smoke/2wikimultihopqa"
if [[ -x "/home/ojungii/miniconda3/envs/effirag/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
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
CURRENT_CONFIG="configs/content_graph/2wikimultihopqa_content_graph_refine_cell.yaml"
METRIC_CONFIG="configs/content_graph/2wikimultihopqa_content_graph_metric_path_carrier.yaml"

"${PYTHON_BIN}" scripts/semantic_embedding_preflight.py \
  --input "${INPUT}" \
  --output-dir "${OUT_ROOT}" \
  --limit "${MAX_QUERIES}"

READY="$("${PYTHON_BIN}" -c 'import json,sys; p=sys.argv[1]; print(json.load(open(p))["semantic_mode_enabled"])' "${OUT_ROOT}/semantic_embedding_preflight.json")"
if [[ "${READY}" != "True" ]]; then
  echo "Semantic carrier smoke stopped by embedding preflight; see ${OUT_ROOT}/semantic_embedding_preflight.json"
  exit 0
fi

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
run_variant "tree_shell1_graph_order" "${CURRENT_CONFIG}" "tree_shell1_graph_order"
run_variant "tree_shell1_semantic_query_order" "${CURRENT_CONFIG}" "tree_shell1_semantic_query_order"
run_variant "tree_shell1_semantic_tree_order" "${CURRENT_CONFIG}" "tree_shell1_semantic_tree_order"
run_variant "semantic_weighted_tree_diagnostic" "${CURRENT_CONFIG}" "semantic_weighted_tree_diagnostic"
run_variant "current_answer_role_oracle" "${CURRENT_CONFIG}"
run_variant "tree_answer_oracle" "${CURRENT_CONFIG}"
run_variant "shell1_answer_oracle" "${CURRENT_CONFIG}" "shell1_answer_oracle"
