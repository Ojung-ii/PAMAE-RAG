from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from pamae_rag.data.io import read_jsonl
from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.qa.generator import DeterministicExtractiveSentenceGenerator
from pamae_rag.qa.metrics import METRIC_ID, gold_answers, normalize_answer, score_prediction
from pamae_rag.qa.runner import _context_text, _oracle_context


ORACLE_MODES = ("gold_support", "answer_containing", "answer_copy")
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")


@dataclass(frozen=True)
class OracleModeSummary:
    oracle_mode: str
    mean_f1: float
    mean_exact_match: float
    context_found_rate: float
    num_queries: int

    def to_json(self) -> dict[str, Any]:
        return {
            "oracle_mode": self.oracle_mode,
            "mean_f1": self.mean_f1,
            "mean_exact_match": self.mean_exact_match,
            "context_found_rate": self.context_found_rate,
            "num_queries": self.num_queries,
        }


def _read_corpus(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Corpus must be a JSON list: {path}")
    return [dict(item) for item in raw if isinstance(item, dict)]


def _dataset_prefix(example: QueryExample) -> str:
    for node in example.nodes:
        if ":doc:" in node.node_id:
            return node.node_id.split(":doc:", 1)[0]
    return "corpus"


def _corpus_nodes(example: QueryExample, corpus: list[dict[str, Any]]) -> list[EvidenceNode]:
    if not corpus:
        return list(example.nodes)
    prefix = _dataset_prefix(example)
    out: list[EvidenceNode] = []
    for idx, item in enumerate(corpus):
        text = str(item.get("text") or "")
        title = str(item.get("title") or "")
        out.append(
            EvidenceNode(
                node_id=f"{prefix}:doc:{idx}",
                text=text,
                embedding=np.zeros(1, dtype=np.float64),
                token_count=max(1, len(text.split())),
                metadata={"title": title, "corpus_index": idx},
            )
        )
    return out


def _sentences(text: str) -> list[str]:
    return [" ".join(match.group(0).split()) for match in _SENTENCE_RE.finditer(str(text)) if match.group(0).strip()]


def _contains_answer(text: str, answers: Iterable[str]) -> bool:
    text_norm = normalize_answer(text)
    if not text_norm:
        return False
    padded = f" {text_norm} "
    for answer in answers:
        answer_norm = normalize_answer(answer)
        if answer_norm and f" {answer_norm} " in padded:
            return True
    return False


def _answer_containing_context(example: QueryExample, corpus: list[dict[str, Any]]) -> tuple[str, bool]:
    answers = gold_answers(example)
    if not answers:
        return "", False
    sentence_candidates: list[tuple[int, str, str]] = []
    chunk_candidates: list[tuple[int, str, str]] = []
    for node in _corpus_nodes(example, corpus):
        if not _contains_answer(node.text, answers):
            continue
        chunk_candidates.append((max(1, len(node.text.split())), node.node_id, node.text))
        for sentence in _sentences(node.text):
            if _contains_answer(sentence, answers):
                sentence_candidates.append((max(1, len(sentence.split())), node.node_id, sentence))
    candidates = sentence_candidates or chunk_candidates
    if not candidates:
        return "", False
    _length, _node_id, context = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    return context, True


def _answer_copy_context(example: QueryExample) -> tuple[str, bool]:
    answers = gold_answers(example)
    if not answers:
        return "", False
    return f"The answer is: {answers[0]}", True


def _gold_support_context(example: QueryExample, corpus: list[dict[str, Any]]) -> tuple[str, bool]:
    _ids, nodes, missing, _corpus_count, diagnostics = _oracle_context(example, corpus)
    found = not missing and bool(nodes)
    if diagnostics.get("oracle_context_unit") == "support_sentence":
        found = True
    return _context_text(nodes), found


def _mode_context(example: QueryExample, corpus: list[dict[str, Any]], mode: str) -> tuple[str, bool]:
    if mode == "gold_support":
        return _gold_support_context(example, corpus)
    if mode == "answer_containing":
        return _answer_containing_context(example, corpus)
    if mode == "answer_copy":
        return _answer_copy_context(example)
    raise ValueError(f"Unknown oracle mode: {mode}")


def run_oracle_mode(
    examples: Iterable[QueryExample],
    corpus: list[dict[str, Any]],
    mode: str,
) -> OracleModeSummary:
    generator = DeterministicExtractiveSentenceGenerator()
    f1s: list[float] = []
    exact_matches: list[float] = []
    found_values: list[float] = []
    count = 0
    for example in examples:
        count += 1
        context, found = _mode_context(example, corpus, mode)
        found_values.append(1.0 if found else 0.0)
        generated = generator.generate(example.query, context)
        score = score_prediction(generated.answer, gold_answers(example))
        if score is not None:
            f1s.append(float(score.f1))
            exact_matches.append(float(score.exact_match))
    return OracleModeSummary(
        oracle_mode=mode,
        mean_f1=float(sum(f1s) / len(f1s)) if f1s else 0.0,
        mean_exact_match=float(sum(exact_matches) / len(exact_matches)) if exact_matches else 0.0,
        context_found_rate=float(sum(found_values) / len(found_values)) if found_values else 0.0,
        num_queries=count,
    )


def oracle_diagnostics(
    input_path: str | Path,
    *,
    corpus_path: str | Path | None = None,
    limit: int | None = None,
    method_f1: float | None = None,
) -> dict[str, Any]:
    examples = read_jsonl(input_path, limit=limit)
    corpus = _read_corpus(corpus_path)
    summaries = {mode: run_oracle_mode(examples, corpus, mode) for mode in ORACLE_MODES}
    gold_support_f1 = summaries["gold_support"].mean_f1
    answer_containing_f1 = summaries["answer_containing"].mean_f1
    answer_copy_f1 = summaries["answer_copy"].mean_f1
    strongest_context_f1 = max(gold_support_f1, answer_containing_f1)
    oracle_dominance_valid = True
    if method_f1 is not None:
        oracle_dominance_valid = answer_copy_f1 > method_f1 and strongest_context_f1 >= method_f1

    if method_f1 is not None and answer_copy_f1 <= method_f1:
        diagnosis = "answer_copy_oracle_not_above_method_stop_retrieval_work"
    elif answer_copy_f1 > strongest_context_f1:
        diagnosis = "oracle_context_construction_weaker_than_answer_copy"
    elif method_f1 is not None and strongest_context_f1 >= method_f1:
        diagnosis = "retrieval_or_rendering_bottleneck"
    else:
        diagnosis = "oracle_measurement_only"

    return {
        "metric_id": METRIC_ID,
        "method_f1": method_f1,
        "gold_support_f1": gold_support_f1,
        "answer_containing_f1": answer_containing_f1,
        "answer_copy_f1": answer_copy_f1,
        "oracle_dominance_valid": oracle_dominance_valid,
        "diagnosis": diagnosis,
        "oracle_modes": {mode: summary.to_json() for mode, summary in summaries.items()},
    }

