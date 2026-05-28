#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _by_query(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["query_id"]): row for row in rows if "query_id" in row}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract path-realizability diagnostics from retrieval traces.")
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--qa", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    qa_by_id = _by_query(_read_jsonl(Path(args.qa)))
    out_rows: list[dict[str, Any]] = []
    for retrieval in _read_jsonl(Path(args.retrieval)):
        query_id = str(retrieval.get("query_id"))
        qa = qa_by_id.get(query_id, {})
        diagnostics = retrieval.get("diagnostics") if isinstance(retrieval.get("diagnostics"), dict) else {}
        payload = diagnostics.get("path_realizability") if isinstance(diagnostics, dict) else {}
        if not isinstance(payload, dict):
            continue
        answer_trace = payload.get("answer_trace") if isinstance(payload.get("answer_trace"), dict) else {}
        basin_rows = payload.get("basin_position_rows") if isinstance(payload.get("basin_position_rows"), list) else []
        basin_by_gold = {
            str(row.get("gold_chunk_id")): row
            for row in basin_rows
            if isinstance(row, dict) and row.get("gold_chunk_id") is not None
        }
        for gold_row in payload.get("gold_rows", []):
            if not isinstance(gold_row, dict):
                continue
            row = dict(gold_row)
            row["qa_f1"] = qa.get("f1", row.get("qa_f1"))
            row["exact_match"] = qa.get("exact_match")
            row["prediction"] = qa.get("prediction")
            row["answer_trace"] = answer_trace
            basin = basin_by_gold.get(str(row.get("gold_chunk_id")))
            if basin is not None:
                row["basin_position"] = basin
            out_rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

