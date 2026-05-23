from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from pamae_rag.config import load_config
from pamae_rag.data.io import read_jsonl, write_jsonl
from pamae_rag.eval.support_recall import evaluate_predictions, write_metrics
from pamae_rag.pipeline import run_query_pamae


def run_retrieval(config_path: str | Path, input_path: str | Path, output_path: str | Path, limit: int | None = None) -> None:
    cfg = load_config(config_path)
    examples = read_jsonl(input_path, limit=limit or cfg.experiment.limit_queries)
    rows = []
    for i, example in enumerate(tqdm(examples, desc="PAMAE-RAG retrieval")):
        start = time.perf_counter()
        result = run_query_pamae(example, cfg, seed_offset=i)
        row = result.to_json()
        row["latency_ms"] = round((time.perf_counter() - start) * 1000.0, 3)
        rows.append(row)
    write_jsonl(output_path, rows)


def evaluate(
    input_path: str | Path,
    prediction_path: str | Path,
    output_path: str | Path,
    limit: int | None = None,
) -> None:
    examples = read_jsonl(input_path, limit=limit)
    predictions = {}
    with Path(prediction_path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            predictions[str(row["query_id"])] = row
    metrics = evaluate_predictions(examples, predictions)
    write_metrics(output_path, metrics)


def validate_data(input_path: str | Path, limit: int | None = None) -> None:
    examples = read_jsonl(input_path, limit=limit)
    if not examples:
        raise SystemExit("No examples found")
    dims = {examples[0].nodes[0].embedding.shape[0]}
    for example in examples:
        if not example.nodes:
            raise SystemExit(f"Example {example.query_id} has no nodes")
        dims |= {node.embedding.shape[0] for node in example.nodes}
    print(json.dumps({"num_examples": len(examples), "embedding_dims": sorted(dims)}, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pamae-rag")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run PAMAE-RAG retrieval")
    run.add_argument("--config", required=True)
    run.add_argument("--input", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--limit", type=int, default=None)

    ev = sub.add_parser("eval", help="Evaluate retrieval predictions")
    ev.add_argument("--input", required=True)
    ev.add_argument("--predictions", required=True)
    ev.add_argument("--output", required=True)
    ev.add_argument("--limit", type=int, default=None)

    vd = sub.add_parser("validate-data", help="Validate examples.jsonl schema")
    vd.add_argument("--input", required=True)
    vd.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        run_retrieval(args.config, args.input, args.output, args.limit)
    elif args.command == "eval":
        evaluate(args.input, args.predictions, args.output, args.limit)
    elif args.command == "validate-data":
        validate_data(args.input, args.limit)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
