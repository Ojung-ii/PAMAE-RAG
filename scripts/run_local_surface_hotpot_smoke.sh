#!/usr/bin/env bash
set -euo pipefail

MAX_QUERIES=100
OUT_ROOT="outputs/local_surface_smoke/hotpotqa"
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
CONFIG="configs/content_graph/hotpotqa_content_graph_refine_cell.yaml"

run_variant() {
  local renderer="$1"
  local out_dir="${OUT_ROOT}/entity_chunk_reference_${renderer}"
  "${PYTHON_BIN}" scripts/run_local_surface_variant.py \
    --config "${CONFIG}" \
    --input "${INPUT}" \
    --output-dir "${out_dir}" \
    --renderer-mode "${renderer}" \
    --limit "${MAX_QUERIES}" \
    --max-context-tokens 512 \
    --local-sentence-medoids 4
}

run_variant "current_renderer"
run_variant "local_sentence_medoid"
run_variant "fact_mediated_sentence"
run_variant "selected_chunk_answer_sentence_oracle"
run_variant "selected_chunk_gold_sentence_oracle"
