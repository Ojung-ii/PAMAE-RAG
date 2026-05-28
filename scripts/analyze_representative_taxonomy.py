#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pamae_rag.diagnostics.representative_taxonomy import analyze_representative_taxonomy


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze representative preservation failures.")
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--qa", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = analyze_representative_taxonomy(
        retrieval_path=args.retrieval,
        qa_path=args.qa,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

