from __future__ import annotations

import re
from dataclasses import dataclass


GENERATOR_ID = "deterministic_extractive_sentence_v1"
PROMPT_ID = "closed_book_free_context_qa_v1"
PROMPT_TEXT = (
    "Use only the provided context to answer the question. "
    "If the context is insufficient, return the most relevant supported sentence."
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
}


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    selected_sentence_index: int | None
    generator_id: str = GENERATOR_ID
    prompt_id: str = PROMPT_ID


def _tokens(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall(str(text).lower()) if token not in _STOPWORDS}


def _sentences(context: str) -> list[str]:
    out: list[str] = []
    for part in _SENTENCE_SPLIT_RE.split(str(context)):
        sentence = " ".join(part.split())
        if sentence:
            out.append(sentence)
    return out


class DeterministicExtractiveSentenceGenerator:
    """A fixed offline generator used only for comparable QA measurement."""

    generator_id = GENERATOR_ID
    prompt_id = PROMPT_ID
    prompt_text = PROMPT_TEXT

    def generate(self, query: str, context: str) -> GeneratedAnswer:
        sentences = _sentences(context)
        if not sentences:
            return GeneratedAnswer(answer="", selected_sentence_index=None)
        query_tokens = _tokens(query)
        if not query_tokens:
            return GeneratedAnswer(answer=sentences[0], selected_sentence_index=0)

        def key(item: tuple[int, str]) -> tuple[int, float, int, int]:
            idx, sentence = item
            sentence_tokens = _tokens(sentence)
            overlap = len(query_tokens & sentence_tokens)
            coverage = overlap / max(len(query_tokens), 1)
            return (overlap, coverage, -len(sentence_tokens), -idx)

        best_idx, best_sentence = max(enumerate(sentences), key=key)
        return GeneratedAnswer(answer=best_sentence, selected_sentence_index=int(best_idx))
