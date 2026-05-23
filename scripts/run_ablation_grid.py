from __future__ import annotations

import argparse
import json
from pathlib import Path

from pamae_rag.cli import evaluate, run_retrieval
from pamae_rag.config import load_config


def _config_paths(config_dir: str | None, configs: list[str] | None, dataset: str) -> list[Path]:
    paths: list[Path] = []
    if config_dir is not None:
        all_paths = sorted(Path(config_dir).glob("*.yaml"))
        dataset_key = dataset.lower()
        filtered = [
            path
            for path in all_paths
            if path.stem.lower().startswith(dataset_key) or path.stem.lower().startswith("sensitivity_")
        ]
        paths.extend(filtered or all_paths)
    if configs:
        paths.extend(Path(path) for path in configs)
    if not paths:
        raise ValueError("Provide --config-dir or --configs")
    return list(dict.fromkeys(paths))


def _augment_metrics(metrics_path: Path, *, dataset: str, variant: str, config_path: Path) -> None:
    cfg = load_config(config_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.update(
        {
            "dataset": dataset,
            "variant": variant,
            "relevance_mode": cfg.pamae.relevance_mode,
            "renderer": cfg.pamae.renderer,
            "retrieval_variant": cfg.pamae.retrieval_variant,
            "k": cfg.pamae.k,
            "k_max": cfg.pamae.k_max,
            "max_context_nodes": cfg.pamae.max_context_nodes,
            "max_context_tokens": cfg.pamae.max_context_tokens,
        }
    )
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def run_grid(
    *,
    dataset: str,
    input_path: str,
    config_dir: str | None,
    configs: list[str] | None,
    output_root: str,
    limit: int | None,
) -> None:
    for config_path in _config_paths(config_dir, configs, dataset):
        variant = config_path.stem
        run_dir = Path(output_root) / dataset / variant
        results_path = run_dir / "results.jsonl"
        metrics_path = run_dir / "metrics.json"
        run_retrieval(config_path, input_path, results_path, limit=limit)
        evaluate(input_path, results_path, metrics_path, limit=limit)
        _augment_metrics(metrics_path, dataset=dataset, variant=variant, config_path=config_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a PAMAE-RAG ablation config grid")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--output-root", default="data/runs")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_grid(
        dataset=args.dataset,
        input_path=args.input,
        config_dir=args.config_dir,
        configs=args.configs,
        output_root=args.output_root,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
