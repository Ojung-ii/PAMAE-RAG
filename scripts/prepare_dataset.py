#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from pamae_rag.data.raw_adapters import prepare_raw_qa_corpus_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare raw QA+corpus JSON files into PAMAE-RAG examples.jsonl")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g., popqa, hotpotqa, 2wikimultihopqa, musique")
    parser.add_argument("--qa", required=True, help="Path to raw QA JSON file")
    parser.add_argument("--corpus", required=True, help="Path to raw corpus JSON file")
    parser.add_argument("--output", required=True, help="Output processed examples.jsonl path")
    parser.add_argument("--max-nodes-per-query", type=int, default=600)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--debug-dir", default=None)
    args = parser.parse_args()

    summary = prepare_raw_qa_corpus_dataset(
        qa_path=args.qa,
        corpus_path=args.corpus,
        output_path=args.output,
        dataset_name=args.dataset,
        max_nodes_per_query=args.max_nodes_per_query,
        embedding_dim=args.embedding_dim,
        max_features=args.max_features,
        limit=args.limit,
        debug_dir=args.debug_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
