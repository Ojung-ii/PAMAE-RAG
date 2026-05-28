#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pamae_rag.data.io import read_jsonl, write_jsonl
from pamae_rag.qa.runner import run_qa
from pamae_rag.sentence_graph.graph_variants import validate_sentence_graph_variant
from pamae_rag.sentence_graph.sentence_diagnostics import (
    aggregate_sentence_traces,
    build_sentence_diagnostic_trace,
    sentence_mapping_diagnostics,
)
from pamae_rag.sentence_graph.sentence_graph_builder import build_sentence_graph_index
from pamae_rag.sentence_graph.sentence_renderer import (
    DIAGNOSTIC_RENDERERS,
    SUPPORTED_RENDERERS,
    render_sentence_context,
)
from pamae_rag.sentence_graph.sentence_retriever import (
    SentenceRetrieverConfig,
    retrieve_sentence_medoids,
)


def _read_config(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return raw


def _retriever_config(raw: dict[str, Any]) -> SentenceRetrieverConfig:
    values = dict(raw.get("sentence_graph") or {})
    allowed = set(SentenceRetrieverConfig.__dataclass_fields__)
    return SentenceRetrieverConfig(**{key: values[key] for key in sorted(values) if key in allowed})


def _sentence_graph_values(raw: dict[str, Any]) -> dict[str, Any]:
    values = dict(raw.get("sentence_graph") or {})
    return {
        "max_context_tokens": int(values.get("max_context_tokens", 1200)),
        "max_context_nodes": values.get("max_context_nodes"),
        "sentence_window": int(values.get("sentence_window", 1)),
        "use_chunk_parent_edges_in_metric": bool(values.get("use_chunk_parent_edges_in_metric", False)),
    }


def _qa_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["query_id"])] = row
    return rows


def _augment_with_qa(
    retrieval_rows: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    qa_rows: dict[str, dict[str, Any]],
) -> None:
    for row, trace in zip(retrieval_rows, traces, strict=True):
        qa = qa_rows.get(str(trace["query_id"]), {})
        qa_f1 = qa.get("f1")
        if isinstance(qa_f1, (int, float)) and not isinstance(qa_f1, bool):
            trace["qa_f1"] = float(qa_f1)
        trace["qa_exact_match"] = qa.get("exact_match")
        trace["qa_prediction"] = qa.get("prediction")
        trace["qa_answer_coverage"] = qa.get("answer_coverage")
        diagnostics = row.setdefault("diagnostics", {})
        diagnostics["sentence_trace"] = trace
        diagnostics["qa_f1"] = trace.get("qa_f1")


def run_variant(
    *,
    config_path: Path,
    input_path: Path,
    output_dir: Path,
    graph_variant: str,
    renderer_mode: str,
    limit: int | None,
) -> dict[str, Any]:
    graph_variant = validate_sentence_graph_variant(graph_variant)
    if renderer_mode not in SUPPORTED_RENDERERS:
        raise ValueError(f"Unknown sentence renderer {renderer_mode!r}")
    raw_config = _read_config(config_path)
    retriever_config = _retriever_config(raw_config)
    values = _sentence_graph_values(raw_config)
    seed = int((raw_config.get("experiment") or {}).get("seed", 42))
    examples = read_jsonl(input_path, limit=limit)
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieval_rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for i, example in enumerate(tqdm(examples, desc=f"{graph_variant}/{renderer_mode}")):
        start = time.perf_counter()
        index = build_sentence_graph_index(
            example.nodes,
            graph_variant=graph_variant,
            use_chunk_parent_edges_in_metric=values["use_chunk_parent_edges_in_metric"],
        )
        retrieval = retrieve_sentence_medoids(
            index,
            example.query,
            config=retriever_config,
            seed=seed + i,
        )
        render = render_sentence_context(
            index,
            retrieval,
            renderer_mode=renderer_mode,
            max_context_tokens=values["max_context_tokens"],
            max_context_nodes=values["max_context_nodes"],
            sentence_window=values["sentence_window"],
            use_chunk_parent_edges_in_metric=values["use_chunk_parent_edges_in_metric"],
        )
        trace = build_sentence_diagnostic_trace(
            example=example,
            index=index,
            retrieval=retrieval,
            render=render,
            graph_variant=graph_variant,
            renderer_mode=renderer_mode,
        )
        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        diagnostics = {
            "graph_variant": graph_variant,
            "renderer": renderer_mode,
            "retrieval_variant": "sentence_primary_sample_full_validation_refine",
            "relevance_mode": "ppr_sentence_mass",
            "max_context_tokens": values["max_context_tokens"],
            "max_context_nodes": values["max_context_nodes"],
            "diagnostic_only": renderer_mode in DIAGNOSTIC_RENDERERS,
            "final_context_tokens": render.context_tokens,
            "node_budget_satisfied": (
                values["max_context_nodes"] is None
                or len(render.context_node_ids) <= int(values["max_context_nodes"])
            ),
            "token_budget_satisfied": render.context_tokens <= values["max_context_tokens"],
            "sentence_mapping": sentence_mapping_diagnostics(index, example),
            "sentence_graph": index.diagnostics,
            "sentence_retrieval": retrieval.diagnostics,
            "sentence_rendering": render.diagnostics,
        }
        retrieval_rows.append(
            {
                "query_id": example.query_id,
                "anchor_node_ids": list(retrieval.selected_sentence_ids),
                "anchor_ids": list(retrieval.selected_sentence_ids),
                "context_node_ids": list(render.context_node_ids),
                "context_nodes": list(render.context_nodes),
                "objective_before_refinement": retrieval.phi_before_refine,
                "objective_after_refinement": retrieval.phi_after_refine,
                "support_recall": trace["rendered_recall"],
                "support_hit": 1.0 if trace["gold_sentence_rendered"] else 0.0,
                "exact_phase1": retrieval.exact_phase1,
                "diagnostics": diagnostics,
                "latency_ms": latency_ms,
            }
        )
        traces.append(trace)

    retrieval_path = output_dir / "retrieval_trace.jsonl"
    qa_path = output_dir / "qa.jsonl"
    qa_metrics_path = output_dir / "qa_metrics.json"
    write_jsonl(retrieval_path, retrieval_rows)
    qa_metrics = run_qa(
        input_path=input_path,
        prediction_path=retrieval_path,
        output_path=qa_path,
        metrics_output_path=qa_metrics_path,
        limit=limit,
    )
    _augment_with_qa(retrieval_rows, traces, _qa_by_id(qa_path))
    write_jsonl(retrieval_path, retrieval_rows)
    write_jsonl(output_dir / "sentence_trace.jsonl", traces)
    sentence_metrics = aggregate_sentence_traces(traces)
    sentence_metrics.update(
        {
            "graph_variant": graph_variant,
            "renderer_mode": renderer_mode,
            "diagnostic_only": renderer_mode in DIAGNOSTIC_RENDERERS,
            "retriever_config": asdict(retriever_config),
            "max_context_tokens": values["max_context_tokens"],
            "max_context_nodes": values["max_context_nodes"],
            "qa_metrics": qa_metrics.to_json(),
        }
    )
    (output_dir / "sentence_metrics.json").write_text(
        json.dumps(sentence_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return sentence_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one sentence-primary graph variant.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--graph-variant", required=True)
    parser.add_argument("--renderer-mode", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    metrics = run_variant(
        config_path=Path(args.config),
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        graph_variant=args.graph_variant,
        renderer_mode=args.renderer_mode,
        limit=args.limit,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
