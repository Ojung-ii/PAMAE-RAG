#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pamae_rag.config import load_config
from pamae_rag.data.io import read_jsonl, write_jsonl
from pamae_rag.diagnostics.runtime_profile import aggregate_runtime_profiles, context_text_hash
from pamae_rag.pipeline import run_query_pamae
from pamae_rag.qa.runner import run_qa

EXPECTED_PROMPT_HASH = "31e4b446be8b00a4989078fb4a957bc61b19bf4b8014674e2baad4612cc4396d"


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _diag_mean(rows: list[dict[str, Any]], key: str) -> float:
    values: list[float] = []
    for row in rows:
        diagnostics = row.get("diagnostics")
        if not isinstance(diagnostics, dict):
            continue
        value = diagnostics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return _mean(values)


def _runtime_rows(retrieval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in retrieval_rows:
        diagnostics = row.get("diagnostics")
        if isinstance(diagnostics, dict) and isinstance(diagnostics.get("runtime_profile"), dict):
            rows.append(dict(diagnostics["runtime_profile"]))
    return rows


def _metrics(
    *,
    renderer_mode: str,
    runtime_mode: str,
    qa_metrics: dict[str, Any],
    retrieval_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    runtime_rows = _runtime_rows(retrieval_rows)
    runtime_aggregate = aggregate_runtime_profiles(runtime_rows)
    runtime_mean = runtime_aggregate.get("mean", {}) if isinstance(runtime_aggregate, dict) else {}
    diagnostics = [row.get("diagnostics", {}) for row in retrieval_rows if isinstance(row.get("diagnostics"), dict)]
    score_mixing = any(bool(diag.get("score_mixing_detected", False)) for diag in diagnostics)
    oracle_leakage = sum(int(diag.get("oracle_leakage_count", 0) or 0) for diag in diagnostics)
    return {
        "num_queries": int(qa_metrics.get("num_queries", len(retrieval_rows))),
        "graph_variant": "entity_chunk_reference",
        "renderer_mode": renderer_mode,
        "runtime_mode": runtime_mode,
        "qa_prompt_name": qa_metrics.get("qa_prompt_name", qa_metrics.get("prompt_id")),
        "qa_prompt_hash": qa_metrics.get("qa_prompt_hash"),
        "qa_prompt_text_exact_match": bool(qa_metrics.get("qa_prompt_text_exact_match", False)),
        "qa_f1": float(qa_metrics.get("mean_f1", 0.0)),
        "em": float(qa_metrics.get("mean_exact_match", 0.0)),
        "answer_in_context": float(qa_metrics.get("mean_answer_coverage", 0.0)),
        "rendered_recall": float(qa_metrics.get("mean_context_recall", 0.0)),
        "context_f1": float(qa_metrics.get("mean_context_f1", 0.0)),
        "avg_context_tokens": float(qa_metrics.get("avg_context_tokens", 0.0)),
        "retrieval_ms": float(qa_metrics.get("avg_retrieval_ms", 0.0)),
        "generation_ms": float(qa_metrics.get("avg_generation_ms", 0.0)),
        "total_ms": float(qa_metrics.get("avg_retrieval_ms", 0.0)) + float(qa_metrics.get("avg_generation_ms", 0.0)),
        "support_tree_chunk_count": _diag_mean(retrieval_rows, "support_tree_chunk_count"),
        "shell1_chunk_count": _diag_mean(retrieval_rows, "shell1_chunk_count"),
        "candidate_pool_size": _diag_mean(retrieval_rows, "candidate_pool_size"),
        "rendered_shell1_chunk_count": _diag_mean(retrieval_rows, "rendered_shell1_chunk_count"),
        "query_embedding_cache_hit_rate": float(runtime_mean.get("query_embedding_cache_hit_rate", 0.0) or 0.0),
        "time_query_embedding_ms": float(runtime_mean.get("time_query_embedding_ms", 0.0) or 0.0),
        "time_embedding_lookup_ms": float(runtime_mean.get("time_embedding_lookup_ms", 0.0) or 0.0),
        "time_semantic_ordering_ms": float(runtime_mean.get("time_semantic_ordering_ms", 0.0) or 0.0),
        "time_shell1_construction_ms": float(runtime_mean.get("time_shell1_construction_ms", 0.0) or 0.0),
        "time_diagnostics_logging_ms": float(runtime_mean.get("time_diagnostics_logging_ms", 0.0) or 0.0),
        "time_total_retrieval_ms": float(runtime_mean.get("time_total_retrieval_ms", 0.0) or 0.0),
        "runtime_profile": runtime_aggregate,
        "oracle_leakage_count": oracle_leakage,
        "score_mixing_detected": bool(score_mixing),
        "embedding_missing_rate": _diag_mean(retrieval_rows, "embedding_missing_rate"),
    }


def run_variant(
    *,
    config_path: Path,
    input_path: Path,
    output_dir: Path,
    renderer_mode: str,
    runtime_mode: str,
    limit: int | None,
    max_context_tokens: int,
    renderer_override: str | None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    if renderer_override:
        cfg = replace(cfg, pamae=replace(cfg.pamae, renderer=renderer_override))
    cfg = replace(cfg, pamae=replace(cfg.pamae, max_context_tokens=max_context_tokens))
    examples = read_jsonl(input_path, limit=limit)
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieval_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for i, example in enumerate(tqdm(examples, desc=f"{renderer_mode}/{runtime_mode}")):
        started = time.perf_counter()
        result = run_query_pamae(example, cfg, seed_offset=i, runtime_mode=runtime_mode)
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        row = result.to_json()
        row["latency_ms"] = latency_ms
        by_id = {node.node_id: node for node in example.nodes}
        context_nodes = [by_id[node_id] for node_id in row["context_node_ids"] if node_id in by_id]
        row.setdefault("diagnostics", {})["context_text_hash"] = context_text_hash(node.text for node in context_nodes)
        retrieval_rows.append(row)
        profile = dict(row["diagnostics"].get("runtime_profile", {}))
        if profile:
            profile["time_total_retrieval_ms"] = latency_ms
            runtime_rows.append(profile)

    retrieval_path = output_dir / "retrieval_trace.jsonl"
    qa_path = output_dir / "qa.jsonl"
    qa_metrics_path = output_dir / "qa_metrics.json"
    write_jsonl(retrieval_path, retrieval_rows)
    write_jsonl(output_dir / "runtime_profile.jsonl", runtime_rows)
    qa_metrics = run_qa(
        input_path=input_path,
        prediction_path=retrieval_path,
        output_path=qa_path,
        metrics_output_path=qa_metrics_path,
        limit=limit,
    )
    metrics = _metrics(
        renderer_mode=renderer_mode,
        runtime_mode=runtime_mode,
        qa_metrics=qa_metrics.to_json(),
        retrieval_rows=retrieval_rows,
    )
    if metrics["qa_prompt_hash"] != EXPECTED_PROMPT_HASH or not metrics["qa_prompt_text_exact_match"]:
        raise SystemExit(f"common_qa prompt hash mismatch: {metrics['qa_prompt_hash']}")
    (output_dir / "runtime_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B2 runtime profile variant.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--renderer-mode", required=True)
    parser.add_argument("--renderer-override", default=None)
    parser.add_argument("--runtime-mode", choices=("diagnostic", "production"), default="diagnostic")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-context-tokens", type=int, default=512)
    args = parser.parse_args()
    run_variant(
        config_path=args.config,
        input_path=args.input,
        output_dir=args.output_dir,
        renderer_mode=args.renderer_mode,
        runtime_mode=args.runtime_mode,
        limit=args.limit,
        max_context_tokens=args.max_context_tokens,
        renderer_override=args.renderer_override,
    )


if __name__ == "__main__":
    main()
