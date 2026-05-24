from __future__ import annotations

import re


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_QUOTED_RE = re.compile(r"['\"]([^'\"]{2,})['\"]")
_POSSESSIVE_RE = re.compile(r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*)'s\b")
_CAPITALIZED_RE = re.compile(r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*\b")

_STOP_PHRASES = {
    "a",
    "an",
    "and",
    "are",
    "did",
    "do",
    "does",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
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


def normalize_text(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(str(text).lower()))


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(str(text).lower()))


def _phrase_is_useful(phrase: str) -> bool:
    norm = normalize_text(phrase)
    if not norm or norm in _STOP_PHRASES:
        return False
    tokens = norm.split()
    if len(tokens) == 1 and tokens[0] in _STOP_PHRASES:
        return False
    return any(tok not in _STOP_PHRASES for tok in tokens)


def extract_query_title_spans(query: str) -> list[str]:
    spans: list[str] = []
    for pattern in (_QUOTED_RE, _POSSESSIVE_RE, _CAPITALIZED_RE):
        for match in pattern.finditer(str(query)):
            text = match.group(1) if match.lastindex else match.group(0)
            pieces = [part.strip() for part in re.split(r"\b(?:and|or|vs\.?|versus)\b", text) if part.strip()]
            spans.extend(piece for piece in pieces if _phrase_is_useful(piece))

    seen: set[str] = set()
    out: list[str] = []
    for span in spans:
        norm = normalize_text(span)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(span.strip())
    return out


def title_overlap_score(query: str, title: str) -> float:
    q_tokens = _tokens(query)
    title_tokens = _tokens(title)
    if not q_tokens or not title_tokens:
        return 0.0
    return len(q_tokens & title_tokens) / len(q_tokens)


def entity_title_grounding_score(query: str, title: str) -> float:
    title_norm = normalize_text(title)
    if not title_norm:
        return 0.0
    title_tokens = set(title_norm.split())
    best = 0.0
    for span in extract_query_title_spans(query):
        span_norm = normalize_text(span)
        if not span_norm:
            continue
        if span_norm == title_norm:
            best = max(best, 1.0)
        elif span_norm in title_norm:
            best = max(best, 0.85)
        elif title_norm in span_norm:
            best = max(best, 0.75)
        else:
            span_tokens = set(span_norm.split())
            if span_tokens:
                best = max(best, len(span_tokens & title_tokens) / len(span_tokens))
    return float(best)
