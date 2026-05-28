#!/usr/bin/env bash
set -euo pipefail

if [[ -x "/home/ojungii/miniconda3/envs/effirag/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-/home/ojungii/miniconda3/envs/effirag/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi
export PYTHON_BIN
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m compileall src tests scripts
"${PYTHON_BIN}" -m pytest -q

bash scripts/run_semantic_carrier_2wiki_smoke.sh --max-queries 50
"${PYTHON_BIN}" scripts/compare_semantic_carrier_runs.py \
  --root outputs/semantic_carrier_smoke/2wikimultihopqa \
  --out outputs/semantic_carrier_smoke/2wikimultihopqa/SEMANTIC_CARRIER_COMPARISON_50.md

DECISION="$("${PYTHON_BIN}" -c 'import json; print(json.load(open("outputs/semantic_carrier_smoke/2wikimultihopqa/semantic_carrier_comparison.json"))["decision"])')"
if [[ "${DECISION}" != "GO_TO_100" && "${DECISION}" != "DIAGNOSTIC_ONLY_100" ]]; then
  "${PYTHON_BIN}" scripts/compare_semantic_carrier_runs.py \
    --root outputs/semantic_carrier_smoke \
    --datasets 2wikimultihopqa \
    --out outputs/semantic_carrier_smoke/SEMANTIC_CARRIER_DIAGNOSTIC_REPORT.md
  echo "Semantic carrier overnight stopped at 2Wiki 50 gate: ${DECISION}"
  exit 0
fi

bash scripts/run_semantic_carrier_2wiki_smoke.sh --max-queries 100
bash scripts/run_semantic_carrier_hotpot_smoke.sh --max-queries 100

"${PYTHON_BIN}" scripts/compare_semantic_carrier_runs.py \
  --root outputs/semantic_carrier_smoke \
  --datasets 2wikimultihopqa hotpotqa \
  --out outputs/semantic_carrier_smoke/SEMANTIC_CARRIER_DIAGNOSTIC_REPORT.md
