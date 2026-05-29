#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pamae_rag.data.io import read_jsonl
from pamae_rag.semantic.embedding_provenance import (
    audit_existing_example_embeddings,
    select_embedding_model,
    write_audit_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit embedding-space compatibility before semantic reruns.")
    parser.add_argument("--input", default="data/processed/2wikimultihopqa/examples_100.jsonl")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    examples = read_jsonl(args.input, limit=args.limit)
    audit = {
        "input": str(args.input),
        "limit": args.limit,
        "existing_embeddings": audit_existing_example_embeddings(examples),
        "model_selection": select_embedding_model(),
    }
    write_audit_markdown(audit, args.out)
    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
