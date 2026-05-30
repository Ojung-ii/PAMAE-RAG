#!/usr/bin/env bash
set -euo pipefail

MAX_QUERIES=50
OUT_ROOT="outputs/b2_runtime_validation/hotpotqa"
PYTHON_BIN="${PYTHON_BIN:-/home/ojungii/miniconda3/envs/QMRAG/bin/python}"
INPUT=""
MODES=("production")
VARIANTS=("tree_shell1_semantic_query_order")

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
    --input)
      INPUT="$2"
      shift 2
      ;;
    --mode)
      MODES=("$2")
      shift 2
      ;;
    --modes)
      shift
      MODES=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        MODES+=("$1")
        shift
      done
      ;;
    --variants)
      shift
      VARIANTS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        VARIANTS+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

DATASET="hotpotqa"
CURRENT_CONFIG="configs/content_graph/hotpotqa_content_graph_refine_cell.yaml"
METRIC_CONFIG="configs/content_graph/hotpotqa_content_graph_metric_path_carrier.yaml"
CACHE_ROOT="outputs/semantic_embedding_cache/nvidia__NV_Embed_v2/${DATASET}"
export PAMAE_SEMANTIC_CACHE_DIR="${CACHE_ROOT}"

if [[ -z "${INPUT}" ]]; then
  if (( MAX_QUERIES <= 500 )); then
    INPUT="outputs/b2_robustness/${DATASET}/inputs/examples_500.jsonl"
  else
    echo "No cached ${DATASET} input supports --max-queries ${MAX_QUERIES}; pass --input explicitly." >&2
    exit 2
  fi
fi

if [[ ! -f "${INPUT}" ]]; then
  echo "Missing input file: ${INPUT}" >&2
  exit 2
fi
if [[ ! -f "${CACHE_ROOT}/metadata.json" ]]; then
  echo "Missing NV-Embed-v2 cache metadata: ${CACHE_ROOT}/metadata.json" >&2
  exit 2
fi

config_for_variant() {
  case "$1" in
    current_renderer)
      echo "${CURRENT_CONFIG}"
      ;;
    metric_path_carrier|tree_shell1_graph_order|tree_shell1_semantic_query_order)
      echo "${METRIC_CONFIG}"
      ;;
    *)
      echo "Unsupported B2 runtime variant: $1" >&2
      return 2
      ;;
  esac
}

for mode in "${MODES[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    config="$(config_for_variant "${variant}")"
    out_dir="${OUT_ROOT}/entity_chunk_reference_${variant}_${mode}"
    args=(
      scripts/run_b2_runtime_profile_variant.py
      --config "${config}"
      --input "${INPUT}"
      --output-dir "${out_dir}"
      --renderer-mode "${variant}"
      --runtime-mode "${mode}"
      --limit "${MAX_QUERIES}"
      --max-context-tokens 512
    )
    if [[ "${variant}" != "current_renderer" ]]; then
      args+=(--renderer-override "${variant}")
    fi
    "${PYTHON_BIN}" "${args[@]}"
  done
done
