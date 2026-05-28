#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a compact RAG run summary.")
    parser.add_argument("--retrieval-metrics", required=True)
    parser.add_argument("--qa-metrics", required=True)
    parser.add_argument("--representative-taxonomy", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    retrieval = _read(args.retrieval_metrics)
    qa = _read(args.qa_metrics)
    representative = _read(args.representative_taxonomy)
    summary = {
        "retrieval": retrieval,
        "qa": qa,
        "representative_taxonomy": {
            key: representative.get(key)
            for key in [
                "representative_failure_counts",
                "mean_d_medoid_gold",
                "mean_gold_distance_percentile_within_basin",
                "gold_on_support_tree_rate",
                "gold_path_exists_but_not_rendered_rate",
                "answer_chunk_projected_but_not_rendered_rate",
            ]
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

