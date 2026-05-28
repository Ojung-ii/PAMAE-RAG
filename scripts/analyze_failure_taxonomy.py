#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pamae_rag.diagnostics.failure_taxonomy import analyze_failure_taxonomy


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify per-query retrieval/QA failure stages.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--qa", required=True)
    parser.add_argument("--retrieval", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    result = analyze_failure_taxonomy(
        args.input,
        args.qa,
        retrieval_path=args.retrieval,
        limit=args.limit,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
