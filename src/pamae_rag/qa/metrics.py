from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any

from pamae_rag.data.schema import QueryExample


METRIC_ID = "squad_normalized_em_f1_v1"

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")


@dataclass(frozen=True)
class AnswerScore:
    exact_match: float
    f1: float
    matched_gold: str


def normalize_answer(text: str) -> str:
    lowered = str(text).lower()
    no_punct = "".join(" " if ch in string.punctuation else ch for ch in lowered)
    no_articles = _ARTICLE_RE.sub(" ", no_punct)
    return " ".join(no_articles.split())


def _tokens(text: str) -> list[str]:
    normalized = normalize_answer(text)
    return normalized.split() if normalized else []


def exact_match_score(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def qa_f1_score(prediction: str, gold: str) -> float:
    pred_tokens = _tokens(prediction)
    gold_tokens = _tokens(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common: dict[str, int] = {}
    for token in gold_tokens:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in pred_tokens:
        count = common.get(token, 0)
        if count > 0:
            overlap += 1
            common[token] = count - 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def gold_answers(example: QueryExample) -> tuple[str, ...]:
    values: list[str] = []
    if example.answer:
        values.append(str(example.answer))
    possible = example.metadata.get("possible_answers")
    if isinstance(possible, list):
        values.extend(str(value) for value in possible if value is not None)
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = normalize_answer(value)
        if key and key not in seen:
            seen.add(key)
            out.append(value)
    return tuple(out)


def score_prediction(prediction: str, answers: tuple[str, ...]) -> AnswerScore | None:
    if not answers:
        return None
    scores = [
        AnswerScore(
            exact_match=exact_match_score(prediction, answer),
            f1=qa_f1_score(prediction, answer),
            matched_gold=answer,
        )
        for answer in answers
    ]
    return max(scores, key=lambda score: (score.f1, score.exact_match, -len(score.matched_gold)))


def score_json(score: AnswerScore | None) -> dict[str, Any]:
    if score is None:
        return {"exact_match": None, "f1": None, "matched_gold": None}
    return {
        "exact_match": score.exact_match,
        "f1": score.f1,
        "matched_gold": score.matched_gold,
    }
