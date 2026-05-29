#!/usr/bin/env bash
set -euo pipefail

VERIFY_PYTHON="${VERIFY_PYTHON:-/home/ojungii/miniconda3/envs/pamae-rag/bin/python}"
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"

"${VERIFY_PYTHON}" -m compileall src tests scripts
"${VERIFY_PYTHON}" -m pytest -q
"${VERIFY_PYTHON}" scripts/audit_embedding_space.py \
  --out outputs/semantic_embedding_rerun/EMBEDDING_SPACE_AUDIT.md

bash scripts/run_semantic_embedding_rerun_2wiki_smoke.sh --max-queries 50
"${VERIFY_PYTHON}" scripts/compare_semantic_embedding_rerun.py \
  --root outputs/semantic_embedding_rerun/2wikimultihopqa \
  --out outputs/semantic_embedding_rerun/2wikimultihopqa/SEMANTIC_EMBEDDING_RERUN_50.md

DECISION="$("${VERIFY_PYTHON}" -c 'import json; print(json.load(open("outputs/semantic_embedding_rerun/2wikimultihopqa/semantic_embedding_rerun_comparison.json"))["decision"])')"
if [[ "${DECISION}" != "GO_TO_100" && "${DECISION}" != "DIAGNOSTIC_ONLY_100" ]]; then
  "${VERIFY_PYTHON}" scripts/compare_semantic_embedding_rerun.py \
    --root outputs/semantic_embedding_rerun \
    --datasets 2wikimultihopqa \
    --out outputs/semantic_embedding_rerun/SEMANTIC_EMBEDDING_RERUN_REPORT.md
  echo "Semantic embedding rerun stopped at 2Wiki 50 gate: ${DECISION}"
  exit 0
fi

bash scripts/run_semantic_embedding_rerun_2wiki_smoke.sh --max-queries 100
bash scripts/run_semantic_embedding_rerun_hotpot_smoke.sh --max-queries 100

"${VERIFY_PYTHON}" scripts/compare_semantic_embedding_rerun.py \
  --root outputs/semantic_embedding_rerun \
  --datasets 2wikimultihopqa hotpotqa \
  --out outputs/semantic_embedding_rerun/SEMANTIC_EMBEDDING_RERUN_REPORT.md
