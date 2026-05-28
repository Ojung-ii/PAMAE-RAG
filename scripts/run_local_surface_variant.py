#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pamae_rag.config import load_config
from pamae_rag.data.io import read_jsonl, write_jsonl
from pamae_rag.data.schema import QueryExample
from pamae_rag.diagnostics.selected_chunk_surface import (
    aggregate_selected_chunk_surface_traces,
    selected_chunk_surface_trace,
)
from pamae_rag.local_surface.local_sentence_medoid import LocalMedoidConfig, select_local_sentence_medoids
from pamae_rag.local_surface.local_surface_graph import build_local_surface_graph
from pamae_rag.local_surface.local_surface_renderers import (
    FACT_MEDIATED_SENTENCE,
    LOCAL_SENTENCE_MEDOID,
    ORACLE_RENDERERS,
    SELECTED_CHUNK_ANSWER_SENTENCE_ORACLE,
    SELECTED_CHUNK_GOLD_SENTENCE_ORACLE,
    render_local_surface,
)
from pamae_rag.pipeline import run_query_pamae
from pamae_rag.qa.runner import run_qa

CURRENT_RENDERER = "current_renderer"
SUPPORTED_RENDERERS = {
    CURRENT_RENDERER,
    LOCAL_SENTENCE_MEDOID,
    FACT_MEDIATED_SENTENCE,
    SELECTED_CHUNK_ANSWER_SENTENCE_ORACLE,
    SELECTED_CHUNK_GOLD_SENTENCE_ORACLE,
}


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["query_id"])] = row
    return rows


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return float(sum(1 for row in rows if bool(row.get(key))) / len(rows)) if rows else 0.0


def _conditional_rate(
    rows: list[dict[str, Any]],
    condition_key: str,
    event_key: str,
) -> float:
    subset = [row for row in rows if bool(row.get(condition_key))]
    return _rate(subset, event_key)


def _surface_support_from_trace(trace: dict[str, Any]) -> dict[str, Any]:
    rendered = set(str(x) for x in trace.get("rendered_sentence_ids", []))
    gold = set(str(x) for x in trace.get("gold_sentence_ids", []))
    answer = set(str(x) for x in trace.get("answer_sentence_ids", []))
    gold_rendered = len(rendered & gold)
    rendered_count = len(rendered)
    recall = (gold_rendered / len(gold)) if gold else 0.0
    precision = (gold_rendered / rendered_count) if rendered_count else 0.0
    context_f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "answer_sentence_rendered": bool(rendered & answer),
        "gold_sentence_rendered": bool(rendered & gold),
        "answer_sentence_rendered_count": len(rendered & answer),
        "gold_sentence_rendered_count": gold_rendered,
        "rendered_recall": recall,
        "context_precision": precision,
        "context_f1": context_f1,
    }


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
        answer_coverage = qa.get("answer_coverage")
        if isinstance(answer_coverage, (int, float)) and not isinstance(answer_coverage, bool):
            trace["answer_in_context"] = bool(answer_coverage > 0.0)
        diagnostics = row.setdefault("diagnostics", {})
        diagnostics["local_surface_trace"] = trace
        diagnostics["qa_f1"] = trace.get("qa_f1")
        diagnostics["answer_in_context"] = trace.get("answer_in_context")


def _aggregate_traces(
    *,
    traces: list[dict[str, Any]],
    qa_metrics: dict[str, Any],
    renderer_mode: str,
    retrieval_ms_values: list[float],
) -> dict[str, Any]:
    local_objective_invalid_count = sum(1 for row in traces if bool(row.get("local_objective_invalid")))
    triangle_count = sum(int(row.get("triangle_inequality_violation_count", 0) or 0) for row in traces)
    answer_oracle_gap = 0.0
    return {
        "num_queries": len(traces),
        "renderer_mode": renderer_mode,
        "oracle_renderer": renderer_mode in ORACLE_RENDERERS,
        "answer_sentence_available_in_selected_chunks": _rate(traces, "answer_sentence_available_in_selected_chunks"),
        "gold_sentence_available_in_selected_chunks": _rate(traces, "gold_sentence_available_in_selected_chunks"),
        "answer_sentence_rendered": _rate(traces, "answer_sentence_rendered"),
        "gold_sentence_rendered": _rate(traces, "gold_sentence_rendered"),
        "answer_in_context": _rate(traces, "answer_in_context"),
        "rendered_recall": _mean(float(row.get("rendered_recall", 0.0)) for row in traces),
        "context_f1": _mean(float(row.get("context_f1", 0.0)) for row in traces),
        "qa_f1": float(qa_metrics.get("mean_f1", 0.0)),
        "oracle_gap_answer_containing": answer_oracle_gap,
        "avg_context_tokens": _mean(float(row.get("context_tokens", 0.0)) for row in traces),
        "retrieval_ms": _mean(retrieval_ms_values),
        "triangle_inequality_violation_count": triangle_count,
        "local_objective_invalid_count": local_objective_invalid_count,
        "answer_rendered_given_available": _conditional_rate(
            traces,
            "answer_sentence_available_in_selected_chunks",
            "answer_sentence_rendered",
        ),
        "gold_rendered_given_available": _conditional_rate(
            traces,
            "gold_sentence_available_in_selected_chunks",
            "gold_sentence_rendered",
        ),
        "qa_success_given_answer_rendered": _conditional_rate(
            [
                {
                    **row,
                    "qa_success": float(row.get("qa_f1", 0.0) or 0.0) > 0.0,
                }
                for row in traces
            ],
            "answer_sentence_rendered",
            "qa_success",
        ),
        "selected_chunk_surface": aggregate_selected_chunk_surface_traces(traces),
        "qa_metrics": qa_metrics,
    }


def _trace_for_current(
    *,
    example: QueryExample,
    result_row: dict[str, Any],
    qa_f1: float = 0.0,
) -> dict[str, Any]:
    selected_chunk_ids = tuple(str(x) for x in result_row.get("anchor_node_ids", []))
    rendered_chunk_ids = tuple(str(x) for x in result_row.get("context_node_ids", []))
    trace = selected_chunk_surface_trace(
        example=example,
        selected_chunk_ids=selected_chunk_ids,
        rendered_chunk_ids=rendered_chunk_ids,
        candidate_chunk_ids=[node.node_id for node in example.nodes],
        qa_f1=qa_f1,
    )
    support = _surface_support_from_trace(trace)
    return {
        **trace,
        "renderer_mode": CURRENT_RENDERER,
        "answer_sentence_available_in_selected_chunks": trace["answer_sentence_in_selected_chunks"],
        "gold_sentence_available_in_selected_chunks": trace["gold_support_sentence_in_selected_chunks"],
        "answer_in_context": trace["current_renderer_answer_in_context"],
        "context_tokens": result_row.get("diagnostics", {}).get("final_context_tokens", 0),
        "triangle_inequality_violation_count": 0,
        "local_objective_invalid": False,
        **support,
    }


def _trace_for_local(
    *,
    example: QueryExample,
    result_row: dict[str, Any],
    renderer_mode: str,
    max_context_tokens: int,
    local_sentence_medoids: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_chunk_ids = tuple(str(x) for x in result_row.get("anchor_node_ids", []))
    graph = build_local_surface_graph(example.nodes, selected_chunk_ids)
    medoid_config = LocalMedoidConfig(local_sentence_medoids=local_sentence_medoids)
    medoids = select_local_sentence_medoids(graph, example.query, config=medoid_config)
    render = render_local_surface(
        example=example,
        graph=graph,
        renderer_mode=renderer_mode,
        medoids=medoids,
        medoid_config=medoid_config,
        max_context_tokens=max_context_tokens,
    )
    trace = selected_chunk_surface_trace(
        example=example,
        selected_chunk_ids=selected_chunk_ids,
        rendered_chunk_ids=[],
        candidate_chunk_ids=[node.node_id for node in example.nodes],
        explicit_rendered_sentence_ids=render.rendered_sentence_ids,
    )
    diagnostics = {
        "renderer_mode": renderer_mode,
        "answer_sentence_available_in_selected_chunks": trace["answer_sentence_in_selected_chunks"],
        "gold_sentence_available_in_selected_chunks": trace["gold_support_sentence_in_selected_chunks"],
        "answer_sentence_rendered": render.diagnostics["answer_sentence_rendered"],
        "gold_sentence_rendered": render.diagnostics["gold_sentence_rendered"],
        "answer_in_context": render.diagnostics["answer_in_context"],
        "rendered_recall": render.diagnostics["rendered_recall"],
        "context_f1": render.diagnostics["context_f1"],
        "context_tokens": render.context_tokens,
        "triangle_inequality_violation_count": render.diagnostics.get("triangle_inequality_violation_count", 0),
        "local_objective_invalid": render.diagnostics.get("local_objective_invalid", False),
        "local_sentence_medoids": local_sentence_medoids,
        **graph.diagnostics,
        **medoids.diagnostics,
        **render.diagnostics,
    }
    return {
        **trace,
        **diagnostics,
    }, {
        "context_node_ids": list(render.context_node_ids),
        "context_nodes": list(render.context_nodes),
        "diagnostics": {
            "local_surface_graph": graph.diagnostics,
            "local_sentence_medoid": medoids.diagnostics,
            "local_surface_rendering": render.diagnostics,
        },
    }


def run_variant(
    *,
    config_path: Path,
    input_path: Path,
    output_dir: Path,
    renderer_mode: str,
    limit: int | None,
    max_context_tokens: int,
    local_sentence_medoids: int,
) -> dict[str, Any]:
    if renderer_mode not in SUPPORTED_RENDERERS:
        raise ValueError(f"Unsupported local surface renderer: {renderer_mode}")
    cfg = load_config(config_path)
    examples = read_jsonl(input_path, limit=limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    retrieval_rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    retrieval_ms_values: list[float] = []

    for i, example in enumerate(tqdm(examples, desc=f"entity_chunk_reference/{renderer_mode}")):
        start = time.perf_counter()
        result = run_query_pamae(example, cfg, seed_offset=i)
        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        row = result.to_json()
        row["latency_ms"] = latency_ms
        retrieval_ms_values.append(latency_ms)
        if renderer_mode == CURRENT_RENDERER:
            trace = _trace_for_current(example=example, result_row=row)
        else:
            trace, local_prediction = _trace_for_local(
                example=example,
                result_row=row,
                renderer_mode=renderer_mode,
                max_context_tokens=max_context_tokens,
                local_sentence_medoids=local_sentence_medoids,
            )
            row["context_node_ids"] = local_prediction["context_node_ids"]
            row["context_nodes"] = local_prediction["context_nodes"]
            row.setdefault("diagnostics", {}).update(local_prediction["diagnostics"])
        row.setdefault("diagnostics", {})["renderer"] = renderer_mode
        row["diagnostics"]["local_surface_trace"] = trace
        retrieval_rows.append(row)
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
    _augment_with_qa(retrieval_rows, traces, _read_jsonl(qa_path))
    write_jsonl(retrieval_path, retrieval_rows)
    write_jsonl(output_dir / "local_surface_trace.jsonl", traces)
    metrics = _aggregate_traces(
        traces=traces,
        qa_metrics=qa_metrics.to_json(),
        renderer_mode=renderer_mode,
        retrieval_ms_values=retrieval_ms_values,
    )
    metrics.update(
        {
            "graph_variant": "entity_chunk_reference",
            "global_backbone": "entity_chunk_reference",
            "local_sentence_medoids": local_sentence_medoids,
            "max_context_tokens": max_context_tokens,
        }
    )
    (output_dir / "local_surface_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chunk-backbone local surface renderer variant.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--renderer-mode", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-context-tokens", type=int, default=512)
    parser.add_argument("--local-sentence-medoids", type=int, default=4)
    args = parser.parse_args()

    run_variant(
        config_path=Path(args.config),
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        renderer_mode=args.renderer_mode,
        limit=args.limit,
        max_context_tokens=args.max_context_tokens,
        local_sentence_medoids=args.local_sentence_medoids,
    )


if __name__ == "__main__":
    main()
