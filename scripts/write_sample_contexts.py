#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pamae_rag.data.io import read_jsonl


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a compact markdown sample of rendered contexts.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    examples = {example.query_id: example for example in read_jsonl(args.input, limit=None)}
    rows = _read_jsonl(Path(args.retrieval))[: max(0, int(args.limit))]
    lines = ["# Sample Contexts", ""]
    for row in rows:
        query_id = str(row.get("query_id"))
        example = examples.get(query_id)
        if example is None:
            continue
        node_by_id = {node.node_id: node for node in example.nodes}
        lines.extend([f"## {query_id}", "", f"Query: {example.query}", "", f"Answer: {example.answer}", ""])
        for rank, node_id in enumerate(row.get("context_node_ids", []), start=1):
            node = node_by_id.get(str(node_id))
            if node is None:
                continue
            title = node.metadata.get("title") or node.node_id
            text = str(node.text).replace("\n", " ")
            lines.extend([f"### [{rank}] {title}", "", f"`{node.node_id}`", "", text, ""])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

