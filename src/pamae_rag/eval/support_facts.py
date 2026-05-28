from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Sequence

from pamae_rag.data.schema import EvidenceNode


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")


@dataclass(frozen=True)
class ResolvedSupportFact:
    title: str
    sentence_id: int
    node_id: str
    sentence: str


def normalize_support_title(value: Any) -> str:
    return " ".join(_TOKEN_RE.findall(str(value or "").lower()))


def sentence_texts(text: str) -> tuple[str, ...]:
    out: list[str] = []
    for match in _SENTENCE_RE.finditer(str(text)):
        sentence = " ".join(match.group(0).split())
        if sentence:
            out.append(sentence)
    return tuple(out)


def support_facts_from_metadata(metadata: dict[str, Any] | None) -> tuple[dict[str, Any], ...]:
    if not isinstance(metadata, dict):
        return tuple()
    value = metadata.get("support_facts")
    if value is None and isinstance(metadata.get("metadata"), dict):
        value = metadata["metadata"].get("support_facts")
    if not isinstance(value, list):
        return tuple()
    return tuple(dict(item) for item in value if isinstance(item, dict))


def _node_title(node: EvidenceNode) -> str:
    return normalize_support_title(node.metadata.get("title"))


def _support_sentence_id(node: EvidenceNode) -> int | None:
    value = node.metadata.get("support_sentence_id")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sentence_for_fact(node: EvidenceNode, sentence_id: int) -> str | None:
    support_sentence_id = _support_sentence_id(node)
    if support_sentence_id == sentence_id:
        return node.text
    sentences = sentence_texts(node.text)
    if 0 <= sentence_id < len(sentences):
        return sentences[sentence_id]
    return None


def resolve_support_facts(
    nodes: Sequence[EvidenceNode],
    support_facts: Iterable[dict[str, Any]],
) -> tuple[ResolvedSupportFact, ...]:
    by_title: dict[str, list[EvidenceNode]] = {}
    for node in nodes:
        title = _node_title(node)
        if title:
            by_title.setdefault(title, []).append(node)

    resolved: list[ResolvedSupportFact] = []
    for fact in support_facts:
        title = normalize_support_title(fact.get("title"))
        try:
            sentence_id = int(fact.get("sentence_id"))
        except (TypeError, ValueError):
            continue
        for node in by_title.get(title, []):
            sentence = _sentence_for_fact(node, sentence_id)
            if sentence is None:
                continue
            resolved.append(
                ResolvedSupportFact(
                    title=title,
                    sentence_id=sentence_id,
                    node_id=node.node_id,
                    sentence=sentence,
                )
            )
            break
    return tuple(resolved)


def support_fact_stage_metrics(
    *,
    nodes: Sequence[EvidenceNode],
    selected_node_ids: Iterable[str],
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    support_facts = support_facts_from_metadata(metadata)
    resolved = resolve_support_facts(nodes, support_facts)
    selected = {str(node_id) for node_id in selected_node_ids}
    surviving = [fact for fact in resolved if fact.node_id in selected]
    fact_count = len(support_facts)
    resolved_count = len(resolved)
    return {
        "support_fact_count": fact_count,
        "support_fact_resolved_count": resolved_count,
        "support_fact_surviving_count": len(surviving),
        "support_fact_survival": (len(surviving) / fact_count) if fact_count else None,
        "support_fact_resolved_survival": (len(surviving) / resolved_count) if resolved_count else None,
    }
