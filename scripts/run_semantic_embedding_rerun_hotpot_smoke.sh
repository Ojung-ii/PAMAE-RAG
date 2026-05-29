#!/usr/bin/env bash
set -euo pipefail

MAX_QUERIES=100
OUT_ROOT="outputs/semantic_embedding_rerun/hotpotqa"
PYTHON_BIN="${PYTHON_BIN:-/home/ojungii/miniconda3/envs/QMRAG/bin/python}"
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-/home/dilab/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/home/dilab/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

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
DATASET="hotpotqa"
CACHE_ROOT="outputs/semantic_embedding_cache/nvidia__NV_Embed_v2/${DATASET}"
export PAMAE_SEMANTIC_CACHE_DIR="${CACHE_ROOT}"

"${PYTHON_BIN}" scripts/generate_compatible_embedding_cache.py \
  --config "${METRIC_CONFIG}" \
  --input "${INPUT}" \
  --output-dir "${OUT_ROOT}" \
  --dataset "${DATASET}" \
  --limit "${MAX_QUERIES}" \
  --batch-size "${SEMANTIC_BATCH_SIZE:-8}" \
  --device "${SEMANTIC_DEVICE:-auto}" \
  --max-length "${SEMANTIC_MAX_LENGTH:-1024}"

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
run_variant "tree_shell1_graph_order" "${METRIC_CONFIG}" "tree_shell1_graph_order"
run_variant "tree_shell1_semantic_query_order" "${METRIC_CONFIG}" "tree_shell1_semantic_query_order"
run_variant "tree_shell1_semantic_tree_order" "${METRIC_CONFIG}" "tree_shell1_semantic_tree_order"
run_variant "semantic_weighted_tree_diagnostic" "${METRIC_CONFIG}" "semantic_weighted_tree_diagnostic"
run_variant "current_answer_role_oracle" "${CURRENT_CONFIG}"
run_variant "tree_answer_oracle" "${CURRENT_CONFIG}"
run_variant "shell1_answer_oracle" "${METRIC_CONFIG}" "shell1_answer_oracle"

"${PYTHON_BIN}" scripts/build_semantic_embedding_rerun_outputs.py \
  --config "${METRIC_CONFIG}" \
  --input "${INPUT}" \
  --root "${OUT_ROOT}" \
  --cache-root "${CACHE_ROOT}" \
  --limit "${MAX_QUERIES}"
