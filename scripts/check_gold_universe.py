#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def check_gold_universe(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    n = 0
    gold_total = 0
    gold_in = 0
    all_covered = 0
    examples_with_no_gold = 0
    examples_missing_some_gold = 0
    sample_missing_query_ids: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            nodes = {str(x["node_id"]) for x in ex.get("nodes", [])}
            gold = {str(x) for x in ex.get("gold_node_ids", [])}
            qid = str(ex.get("query_id", n))
            n += 1
            gold_total += len(gold)
            present = len(gold & nodes)
            gold_in += present
            if not gold:
                examples_with_no_gold += 1
                if len(sample_missing_query_ids) < 10:
                    sample_missing_query_ids.append(qid)
            elif gold <= nodes:
                all_covered += 1
            else:
                examples_missing_some_gold += 1
                if len(sample_missing_query_ids) < 10:
                    sample_missing_query_ids.append(qid)
    return {
        "num_queries": n,
        "gold_total": gold_total,
        "gold_in_nodes": gold_in,
        "gold_universe_recall": gold_in / max(gold_total, 1),
        "all_gold_covered_ratio": all_covered / max(n, 1),
        "examples_with_no_gold": examples_with_no_gold,
        "examples_missing_some_gold": examples_missing_some_gold,
        "sample_missing_query_ids": sample_missing_query_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check gold labels against each query-local universe")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    print(json.dumps(check_gold_universe(args.input), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
