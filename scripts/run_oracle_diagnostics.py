#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pamae_rag.diagnostics.oracle_runs import oracle_diagnostics


def _method_f1(path: str | None) -> float | None:
    if path is None:
        return None
    value = json.loads(Path(path).read_text(encoding="utf-8")).get("mean_f1")
    return float(value) if isinstance(value, (int, float)) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stronger oracle QA diagnostics.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--method-metrics", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    result = oracle_diagnostics(
        args.input,
        corpus_path=args.corpus,
        limit=args.limit,
        method_f1=_method_f1(args.method_metrics),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
