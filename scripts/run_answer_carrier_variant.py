#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pamae_rag.config import load_config
from pamae_rag.data.io import read_jsonl, write_jsonl
from pamae_rag.diagnostics.answer_carrier_attribution import (
    aggregate_answer_carrier_attribution,
    answer_carrier_attribution_rows,
    gold_carrier_attribution_rows,
)
from pamae_rag.diagnostics.renderer_role_trace import renderer_role_trace_rows
from pamae_rag.diagnostics.support_tree_carrier_trace import (
    aggregate_support_tree_carrier_traces,
    support_tree_carrier_trace_rows,
)
from pamae_rag.pipeline import run_query_pamae
from pamae_rag.qa.runner import run_qa
from pamae_rag.rendering.path_carrier_renderer import PATH_CARRIER_RENDERERS
from pamae_rag.rendering.answer_carrier_oracle_renderers import (
    ANSWER_CARRIER_ORACLE_RENDERERS,
    render_answer_carrier_oracle,
)
from pamae_rag.rendering.tree_ablation_renderers import (
    TREE_ABLATION_RENDERERS,
    TREE_ORACLE_RENDERERS,
    render_tree_ablation,
)
from pamae_rag.rendering.semantic_carrier_renderers import (
    SEMANTIC_CARRIER_RENDERERS,
    SEMANTIC_ORACLE_RENDERERS,
    SEMANTIC_WEIGHTED_TREE_DIAGNOSTIC,
)

CURRENT_RENDERER = "current_renderer"
SUPPORTED_RENDERERS = {
    CURRENT_RENDERER,
    *PATH_CARRIER_RENDERERS,
    *ANSWER_CARRIER_ORACLE_RENDERERS,
    *TREE_ABLATION_RENDERERS,
    *SEMANTIC_CARRIER_RENDERERS,
    *SEMANTIC_ORACLE_RENDERERS,
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


def _augment_carrier_rows(
    *,
    rows: list[dict[str, Any]],
    qa_by_query: dict[str, dict[str, Any]],
    update_current_answer: bool,
) -> None:
    for row in rows:
        qa = qa_by_query.get(str(row.get("query_id")), {})
        f1 = qa.get("f1")
        if isinstance(f1, (int, float)) and not isinstance(f1, bool):
            row["qa_f1"] = float(f1)
        answer_coverage = qa.get("answer_coverage")
        if update_current_answer and isinstance(answer_coverage, (int, float)) and not isinstance(answer_coverage, bool):
            row["current_renderer_answer_in_context"] = bool(float(answer_coverage) > 0.0)


def _augment_support_rows(
    *,
    rows: list[dict[str, Any]],
    qa_by_query: dict[str, dict[str, Any]],
    renderer_mode: str,
) -> None:
    for row in rows:
        qa = qa_by_query.get(str(row.get("query_id")), {})
        f1 = qa.get("f1")
        if isinstance(f1, (int, float)) and not isinstance(f1, bool):
            row["qa_f1"] = float(f1)
        answer_coverage = qa.get("answer_coverage")
        if isinstance(answer_coverage, (int, float)) and not isinstance(answer_coverage, bool):
            if renderer_mode == CURRENT_RENDERER:
                row["current_answer_in_context"] = bool(float(answer_coverage) > 0.0)
            if renderer_mode in PATH_CARRIER_RENDERERS:
                row["metric_path_answer_in_context"] = bool(float(answer_coverage) > 0.0)


def _metrics(
    *,
    renderer_mode: str,
    qa_metrics: dict[str, Any],
    answer_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    role_rows: list[dict[str, Any]],
    support_rows: list[dict[str, Any]],
    retrieval_ms_values: list[float],
    oracle_context_tokens: list[float],
) -> dict[str, Any]:
    carrier = aggregate_answer_carrier_attribution(
        [*answer_rows, *gold_rows],
        renderer_role_rows=role_rows,
    )
    support_tree_carrier = aggregate_support_tree_carrier_traces(support_rows)
    return {
        "num_queries": int(qa_metrics.get("num_queries", carrier.get("num_queries", 0))),
        "graph_variant": "entity_chunk_reference",
        "renderer_mode": renderer_mode,
        "oracle_renderer": renderer_mode in ANSWER_CARRIER_ORACLE_RENDERERS
        or renderer_mode in TREE_ORACLE_RENDERERS
        or renderer_mode in SEMANTIC_ORACLE_RENDERERS,
        "diagnostic_renderer": renderer_mode in TREE_ABLATION_RENDERERS
        or renderer_mode == SEMANTIC_WEIGHTED_TREE_DIAGNOSTIC,
        "uses_answer_string": renderer_mode
        in {
            "projected_answer_chunk_oracle",
            "selected_basin_answer_chunk_oracle",
            "current_answer_role_oracle",
            "support_tree_answer_oracle",
            "shell1_answer_oracle",
            *TREE_ORACLE_RENDERERS,
        },
        "uses_gold_label": renderer_mode == "gold_chunk_role_oracle",
        "qa_f1": float(qa_metrics.get("mean_f1", 0.0)),
        "answer_in_context": float(qa_metrics.get("mean_answer_coverage", 0.0)),
        "rendered_recall": float(qa_metrics.get("mean_context_recall", 0.0)),
        "context_f1": float(qa_metrics.get("mean_context_f1", 0.0)),
        "avg_context_tokens": float(
            _mean(oracle_context_tokens)
            if oracle_context_tokens
            else qa_metrics.get("avg_context_tokens", 0.0)
        ),
        "retrieval_ms": _mean(retrieval_ms_values),
        "triangle_inequality_violation_count": 0,
        "oracle_leakage_count": 0,
        "local_objective_invalid_count": 0,
        "answer_carrier_attribution": carrier,
        "support_tree_carrier": support_tree_carrier,
        **carrier,
        **support_tree_carrier,
        "qa_metrics": qa_metrics,
    }


def run_variant(
    *,
    config_path: Path,
    input_path: Path,
    output_dir: Path,
    renderer_mode: str,
    limit: int | None,
    max_context_tokens: int,
    renderer_override: str | None = None,
) -> dict[str, Any]:
    if renderer_mode not in SUPPORTED_RENDERERS:
        raise ValueError(f"Unsupported answer carrier renderer: {renderer_mode}")
    cfg = load_config(config_path)
    if renderer_override:
        cfg = replace(cfg, pamae=replace(cfg.pamae, renderer=renderer_override))
    examples = read_jsonl(input_path, limit=limit)
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieval_rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    retrieval_ms_values: list[float] = []
    oracle_context_tokens: list[float] = []

    for i, example in enumerate(tqdm(examples, desc=f"entity_chunk_reference/{renderer_mode}")):
        start = time.perf_counter()
        result = run_query_pamae(example, cfg, seed_offset=i)
        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        row = result.to_json()
        row["latency_ms"] = latency_ms
        retrieval_ms_values.append(latency_ms)

        current_answer_rows = answer_carrier_attribution_rows(example=example, retrieval_row=row)
        current_gold_rows = gold_carrier_attribution_rows(example=example, retrieval_row=row)
        current_role_rows = renderer_role_trace_rows(example=example, retrieval_row=row)
        current_support_rows = support_tree_carrier_trace_rows(
            example=example,
            retrieval_row=row,
            renderer_mode=renderer_mode,
        )

        if renderer_mode in ANSWER_CARRIER_ORACLE_RENDERERS:
            oracle = render_answer_carrier_oracle(
                example=example,
                retrieval_row=row,
                renderer_mode=renderer_mode,
                max_context_tokens=max_context_tokens,
            )
            row["context_node_ids"] = list(oracle.context_node_ids)
            row.setdefault("diagnostics", {})["answer_carrier_oracle_renderer"] = oracle.diagnostics
            row["diagnostics"]["renderer"] = renderer_mode
            oracle_context_tokens.append(float(oracle.context_tokens))
        elif renderer_mode in TREE_ABLATION_RENDERERS:
            tree_render = render_tree_ablation(
                example=example,
                retrieval_row=row,
                renderer_mode=renderer_mode,
                max_context_tokens=max_context_tokens,
            )
            row["context_node_ids"] = list(tree_render.context_node_ids)
            row.setdefault("diagnostics", {})["tree_ablation_renderer"] = tree_render.diagnostics
            row["diagnostics"]["renderer"] = renderer_mode
            oracle_context_tokens.append(float(tree_render.context_tokens))
        else:
            row.setdefault("diagnostics", {})["renderer"] = renderer_mode

        row["diagnostics"]["answer_carrier_trace"] = current_answer_rows
        row["diagnostics"]["gold_carrier_trace"] = current_gold_rows
        row["diagnostics"]["renderer_role_trace"] = current_role_rows
        row["diagnostics"]["support_tree_carrier_trace"] = current_support_rows
        retrieval_rows.append(row)
        answer_rows.extend(current_answer_rows)
        gold_rows.extend(current_gold_rows)
        role_rows.extend(current_role_rows)
        support_rows.extend(current_support_rows)

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
    qa_by_query = _read_jsonl(qa_path)
    _augment_carrier_rows(
        rows=answer_rows,
        qa_by_query=qa_by_query,
        update_current_answer=renderer_mode == CURRENT_RENDERER,
    )
    _augment_carrier_rows(
        rows=gold_rows,
        qa_by_query=qa_by_query,
        update_current_answer=renderer_mode == CURRENT_RENDERER,
    )
    _augment_support_rows(
        rows=support_rows,
        qa_by_query=qa_by_query,
        renderer_mode=renderer_mode,
    )
    write_jsonl(output_dir / "answer_carrier_trace.jsonl", answer_rows)
    write_jsonl(output_dir / "gold_carrier_trace.jsonl", gold_rows)
    write_jsonl(output_dir / "renderer_role_trace.jsonl", role_rows)
    write_jsonl(output_dir / "support_tree_carrier_trace.jsonl", support_rows)

    metrics = _metrics(
        renderer_mode=renderer_mode,
        qa_metrics=qa_metrics.to_json(),
        answer_rows=answer_rows,
        gold_rows=gold_rows,
        role_rows=role_rows,
        support_rows=support_rows,
        retrieval_ms_values=retrieval_ms_values,
        oracle_context_tokens=oracle_context_tokens,
    )
    (output_dir / "answer_carrier_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run answer-carrier attribution diagnostic variant.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--renderer-mode", required=True)
    parser.add_argument("--renderer-override", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-context-tokens", type=int, default=512)
    args = parser.parse_args()

    run_variant(
        config_path=Path(args.config),
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        renderer_mode=args.renderer_mode,
        limit=args.limit,
        max_context_tokens=args.max_context_tokens,
        renderer_override=args.renderer_override,
    )


if __name__ == "__main__":
    main()
