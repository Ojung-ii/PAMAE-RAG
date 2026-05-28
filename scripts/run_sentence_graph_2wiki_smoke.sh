#!/usr/bin/env bash
set -euo pipefail

MAX_QUERIES=50
OUT_ROOT="outputs/sentence_graph_smoke/2wikimultihopqa"
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
REF_CONFIG="configs/content_graph/2wikimultihopqa_content_graph_refine_cell.yaml"
SENT_CONFIG="configs/sentence_graph/2wiki_sentence_smoke.yaml"

run_reference() {
  local out_dir="${OUT_ROOT}/entity_chunk_reference_current_renderer"
  mkdir -p "${out_dir}"
  "${PYTHON_BIN}" scripts/run_retrieval.py \
    --config "${REF_CONFIG}" \
    --input "${INPUT}" \
    --output "${out_dir}/retrieval_trace.jsonl" \
    --limit "${MAX_QUERIES}"
  "${PYTHON_BIN}" scripts/evaluate_retrieval.py \
    --input "${INPUT}" \
    --predictions "${out_dir}/retrieval_trace.jsonl" \
    --output "${out_dir}/retrieval_metrics.json" \
    --limit "${MAX_QUERIES}"
  "${PYTHON_BIN}" scripts/run_qa.py \
    --input "${INPUT}" \
    --predictions "${out_dir}/retrieval_trace.jsonl" \
    --output "${out_dir}/qa.jsonl" \
    --metrics-output "${out_dir}/qa_metrics.json" \
    --limit "${MAX_QUERIES}"
}

run_sentence() {
  local graph_variant="$1"
  local renderer_mode="$2"
  local out_dir="${OUT_ROOT}/${graph_variant}_${renderer_mode}"
  "${PYTHON_BIN}" scripts/run_sentence_graph_variant.py \
    --config "${SENT_CONFIG}" \
    --input "${INPUT}" \
    --output-dir "${out_dir}" \
    --graph-variant "${graph_variant}" \
    --renderer-mode "${renderer_mode}" \
    --limit "${MAX_QUERIES}"
}

run_reference
run_sentence "entity_sentence" "sentence_only"
run_sentence "entity_sentence" "sentence_path"
run_sentence "entity_sentence_chunk_hier" "sentence_only"
run_sentence "entity_sentence_chunk_hier" "sentence_path"
run_sentence "entity_sentence_chunk_hier" "sentence_parent_title"
run_sentence "entity_sentence_chunk_hier" "sentence_local_window"
run_sentence "entity_sentence_chunk_hier" "sentence_parent_chunk"
