#!/usr/bin/env bash
set -euo pipefail

MAX_QUERIES="${MAX_QUERIES:-100}"
OUT_ROOT="${OUT_ROOT:-outputs/path_realizability_smoke/2wikimultihopqa}"
INPUT="data/processed/2wikimultihopqa/examples_100.jsonl"
CORPUS="data/processed/2wikimultihopqa_corpus.json"

run_variant() {
  local name="$1"
  local config="$2"
  local out_dir="${OUT_ROOT}/${name}"
  mkdir -p "${out_dir}"
  python scripts/run_retrieval.py \
    --config "${config}" \
    --input "${INPUT}" \
    --output "${out_dir}/retrieval_trace.jsonl" \
    --limit "${MAX_QUERIES}"
  python scripts/evaluate_retrieval.py \
    --input "${INPUT}" \
    --predictions "${out_dir}/retrieval_trace.jsonl" \
    --output "${out_dir}/retrieval_metrics.json" \
    --limit "${MAX_QUERIES}"
  python scripts/run_qa.py \
    --input "${INPUT}" \
    --predictions "${out_dir}/retrieval_trace.jsonl" \
    --output "${out_dir}/qa.jsonl" \
    --metrics-output "${out_dir}/qa_metrics.json" \
    --limit "${MAX_QUERIES}"
  python scripts/extract_path_realizability_trace.py \
    --retrieval "${out_dir}/retrieval_trace.jsonl" \
    --qa "${out_dir}/qa.jsonl" \
    --output "${out_dir}/path_realizability_trace.jsonl"
  python scripts/analyze_representative_taxonomy.py \
    --retrieval "${out_dir}/retrieval_trace.jsonl" \
    --qa "${out_dir}/qa.jsonl" \
    --output "${out_dir}/representative_taxonomy.json"
  python scripts/run_oracle_diagnostics.py \
    --input "${INPUT}" \
    --corpus "${CORPUS}" \
    --method-metrics "${out_dir}/qa_metrics.json" \
    --output "${out_dir}/oracle_diagnostics.json" \
    --limit "${MAX_QUERIES}"
  python scripts/write_rag_summary.py \
    --retrieval-metrics "${out_dir}/retrieval_metrics.json" \
    --qa-metrics "${out_dir}/qa_metrics.json" \
    --representative-taxonomy "${out_dir}/representative_taxonomy.json" \
    --output "${out_dir}/rag_summary.json"
  python scripts/write_sample_contexts.py \
    --input "${INPUT}" \
    --retrieval "${out_dir}/retrieval_trace.jsonl" \
    --output "${out_dir}/sample_contexts.md" \
    --limit 3
}

run_variant "current_content_current_renderer" "configs/content_graph/2wikimultihopqa_content_graph_refine_cell.yaml"
run_variant "current_content_gold_path_oracle_renderer" "configs/content_graph/2wikimultihopqa_content_graph_gold_path_oracle.yaml"
run_variant "current_content_path_neighborhood_renderer" "configs/content_graph/2wikimultihopqa_content_graph_path_neighborhood.yaml"
run_variant "basin_preserving_selection_current_renderer" "configs/content_graph/2wikimultihopqa_content_graph_basin_preserving.yaml"
run_variant "basin_preserving_selection_gold_path_oracle_renderer" "configs/content_graph/2wikimultihopqa_content_graph_basin_preserving_gold_path_oracle.yaml"
run_variant "basin_preserving_selection_path_neighborhood_renderer" "configs/content_graph/2wikimultihopqa_content_graph_basin_preserving_path_neighborhood.yaml"
