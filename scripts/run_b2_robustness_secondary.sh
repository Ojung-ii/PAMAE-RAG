#!/usr/bin/env bash
set -euo pipefail

DATASETS=()
MAX_QUERIES=300
OUT_ROOT="outputs/b2_robustness"
PYTHON_BIN="${PYTHON_BIN:-/home/ojungii/miniconda3/envs/QMRAG/bin/python}"
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-/home/dilab/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/home/dilab/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        DATASETS+=("$1")
        shift
      done
      ;;
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

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  DATASETS=(musique popqa)
fi

for dataset in "${DATASETS[@]}"; do
  dataset_root="${OUT_ROOT}/${dataset}"
  mkdir -p "${dataset_root}"
  current_config="configs/content_graph/${dataset}_content_graph_refine_cell.yaml"
  metric_config="configs/content_graph/${dataset}_content_graph_metric_path_carrier.yaml"
  raw_qa="data/processed/${dataset}.json"
  raw_corpus="data/processed/${dataset}_corpus.json"
  if [[ ! -f "${current_config}" || ! -f "${metric_config}" || ! -f "${raw_qa}" || ! -f "${raw_corpus}" ]]; then
    cat > "${dataset_root}/SECONDARY_UNAVAILABLE.md" <<EOF
# Secondary Dataset Unavailable

Dataset: ${dataset}
Reason: missing content-graph config or processed QA/corpus input.
Current config: ${current_config}
Metric config: ${metric_config}
Raw QA: ${raw_qa}
Raw corpus: ${raw_corpus}
EOF
    continue
  fi
  input="${dataset_root}/inputs/examples_${MAX_QUERIES}.jsonl"
  cache_root="outputs/semantic_embedding_cache/nvidia__NV_Embed_v2/${dataset}"
  export PAMAE_SEMANTIC_CACHE_DIR="${cache_root}"
  mkdir -p "${dataset_root}/inputs"
  "${PYTHON_BIN}" scripts/prepare_dataset.py \
    --dataset "${dataset}" \
    --qa "${raw_qa}" \
    --corpus "${raw_corpus}" \
    --output "${input}" \
    --limit "${MAX_QUERIES}"
  "${PYTHON_BIN}" scripts/generate_compatible_embedding_cache.py \
    --config "${metric_config}" \
    --input "${input}" \
    --output-dir "${dataset_root}" \
    --dataset "${dataset}" \
    --limit "${MAX_QUERIES}" \
    --batch-size "${SEMANTIC_BATCH_SIZE:-8}" \
    --device "${SEMANTIC_DEVICE:-auto}" \
    --max-length "${SEMANTIC_MAX_LENGTH:-1024}"
  "${PYTHON_BIN}" scripts/run_answer_carrier_variant.py \
    --config "${current_config}" \
    --input "${input}" \
    --output-dir "${dataset_root}/entity_chunk_reference_current_renderer" \
    --renderer-mode current_renderer \
    --limit "${MAX_QUERIES}" \
    --max-context-tokens 512
  "${PYTHON_BIN}" scripts/run_answer_carrier_variant.py \
    --config "${metric_config}" \
    --input "${input}" \
    --output-dir "${dataset_root}/entity_chunk_reference_tree_shell1_semantic_query_order" \
    --renderer-mode tree_shell1_semantic_query_order \
    --renderer-override tree_shell1_semantic_query_order \
    --limit "${MAX_QUERIES}" \
    --max-context-tokens 512
  "${PYTHON_BIN}" scripts/build_semantic_effect_decomposition_outputs.py \
    --config "${metric_config}" \
    --input "${input}" \
    --root "${dataset_root}" \
    --cache-root "${cache_root}" \
    --limit "${MAX_QUERIES}"
done
