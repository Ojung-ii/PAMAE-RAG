#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from pamae_rag.qa.runner import run_qa


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed end-to-end QA over rendered contexts")
    parser.add_argument("--input", required=True, help="Input examples.jsonl")
    parser.add_argument("--predictions", required=True, help="Retrieval predictions JSONL")
    parser.add_argument("--output", required=True, help="Per-query QA output JSONL")
    parser.add_argument("--metrics-output", required=True, help="Aggregate QA metrics JSON")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    metrics = run_qa(
        input_path=args.input,
        prediction_path=args.predictions,
        output_path=args.output,
        metrics_output_path=args.metrics_output,
        limit=args.limit,
    )
    print(json.dumps(metrics.to_json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
