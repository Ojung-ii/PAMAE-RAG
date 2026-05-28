#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pamae_rag.data.io import read_jsonl
from pamae_rag.semantic.embedding_store import EmbeddingStore


def run_preflight(input_path: Path, output_dir: Path, limit: int | None) -> dict[str, Any]:
    examples = read_jsonl(input_path, limit=limit)
    rows: list[dict[str, Any]] = []
    chunk_coverages: list[float] = []
    missing_chunks = 0
    chunk_count = 0
    query_available = 0
    embedding_dim = 0
    for example in examples:
        store = EmbeddingStore.from_example(example)
        diagnostics = store.diagnostics().to_json()
        diagnostics["query_id"] = example.query_id
        rows.append(diagnostics)
        chunk_coverages.append(float(diagnostics["chunk_embedding_coverage"]))
        missing_chunks += int(diagnostics["missing_chunk_embedding_count"])
        chunk_count += int(diagnostics["chunk_count"])
        query_available += int(bool(diagnostics["query_embedding_available"]))
        embedding_dim = max(embedding_dim, int(diagnostics["embedding_dim"]))

    chunk_coverage = float(sum(chunk_coverages) / len(chunk_coverages)) if chunk_coverages else 0.0
    query_rate = float(query_available / len(rows)) if rows else 0.0
    semantic_enabled = bool(rows and chunk_coverage > 0.0 and query_rate >= 1.0)
    reason = "semantic inputs available"
    if not rows:
        reason = "no examples were loaded"
    elif chunk_coverage <= 0.0:
        reason = "no existing chunk embeddings were found"
    elif query_rate < 1.0:
        reason = "query embeddings are missing; refusing to synthesize semantic query vectors"

    summary = {
        "input": str(input_path),
        "max_queries": limit,
        "num_examples": len(rows),
        "embedding_source": "existing_node_embeddings" if chunk_coverage > 0.0 else "none",
        "embedding_dim": embedding_dim,
        "chunk_embedding_coverage": chunk_coverage,
        "query_embedding_available_rate": query_rate,
        "query_embedding_available": bool(query_rate >= 1.0),
        "semantic_mode_enabled": semantic_enabled,
        "missing_chunk_embedding_count": missing_chunks,
        "chunk_count": chunk_count,
        "embedding_missing_rate": float(missing_chunks / chunk_count) if chunk_count else 1.0,
        "decision": "READY" if semantic_enabled else "STOP_BEFORE_100",
        "reason": reason,
        "examples": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "semantic_embedding_preflight.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    summary = run_preflight(args.input, args.output_dir, args.limit)
    print(json.dumps({k: summary[k] for k in ("decision", "reason", "semantic_mode_enabled")}, sort_keys=True))


if __name__ == "__main__":
    main()
