#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from pamae_rag.eval.qa_run_compare import QARunSpec, compare_runs, write_summary


def _spec(value: str) -> QARunSpec:
    parts = value.split(":")
    if len(parts) not in {4, 5}:
        raise argparse.ArgumentTypeError(
            "--run values must be name:graph_mode:qa_metrics:retrieval_metrics_or_none[:risk_decision]"
        )
    name, graph_mode, qa_metrics, retrieval_metrics = parts[:4]
    risk_decision = parts[4] if len(parts) == 5 else "measurement"
    return QARunSpec(
        name=name,
        graph_mode=graph_mode,
        qa_metrics_path=Path(qa_metrics),
        retrieval_metrics_path=None if retrieval_metrics == "none" else Path(retrieval_metrics),
        risk_decision=risk_decision,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline/content/oracle QA runs")
    parser.add_argument("--run", action="append", type=_spec, required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument(
        "--require-valid-oracle",
        action="store_true",
        help="Exit nonzero if oracle context is incomplete, QA settings differ, or a non-oracle run beats oracle F1.",
    )
    args = parser.parse_args()

    summary = compare_runs(args.run)
    write_summary(summary, Path(args.output_json), Path(args.output_csv), Path(args.output_md))
    if args.require_valid_oracle and (
        not summary["oracle_context_complete"]
        or not summary["qa_settings_consistent"]
        or not summary["oracle_dominance_valid"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
