from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.diagnostics.path_realizability import answer_containing_chunk_ids

PROJECTED_ANSWER_CHUNK_ORACLE = "projected_answer_chunk_oracle"
SELECTED_BASIN_ANSWER_CHUNK_ORACLE = "selected_basin_answer_chunk_oracle"
CURRENT_ANSWER_ROLE_ORACLE = "current_answer_role_oracle"
GOLD_CHUNK_ROLE_ORACLE = "gold_chunk_role_oracle"

ANSWER_CARRIER_ORACLE_RENDERERS = {
    PROJECTED_ANSWER_CHUNK_ORACLE,
    SELECTED_BASIN_ANSWER_CHUNK_ORACLE,
    CURRENT_ANSWER_ROLE_ORACLE,
    GOLD_CHUNK_ROLE_ORACLE,
}


@dataclass(frozen=True)
class AnswerCarrierOracleRenderResult:
    renderer_mode: str
    context_node_ids: tuple[str, ...]
    context_tokens: int
    diagnostics: dict[str, Any]


def _ids(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value is not None))


def _diagnostics(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("diagnostics")
    return value if isinstance(value, dict) else {}


def _node_by_id(nodes: Sequence[EvidenceNode]) -> dict[str, EvidenceNode]:
    return {str(node.node_id): node for node in nodes}


def _ordered_subset(nodes: Sequence[EvidenceNode], ids: Iterable[str]) -> list[str]:
    wanted = set(_ids(ids))
    return [str(node.node_id) for node in nodes if str(node.node_id) in wanted]


def _materialize(
    nodes: Sequence[EvidenceNode],
    chunk_ids: Iterable[str],
    *,
    max_context_tokens: int,
) -> tuple[tuple[str, ...], int]:
    by_id = _node_by_id(nodes)
    out: list[str] = []
    tokens = 0
    for chunk_id in _ids(chunk_ids):
        node = by_id.get(chunk_id)
        if node is None:
            continue
        node_tokens = int(max(1, node.token_count))
        if out and tokens + node_tokens > max_context_tokens:
            break
        out.append(chunk_id)
        tokens += node_tokens
    return tuple(out), tokens


def _oracle_diagnostics(
    *,
    renderer_mode: str,
    context_node_ids: Sequence[str],
    context_tokens: int,
    uses_answer_string: bool,
    uses_gold_label: bool,
) -> dict[str, Any]:
    return {
        "renderer_mode": renderer_mode,
        "oracle_renderer": True,
        "uses_answer_string": uses_answer_string,
        "uses_gold_label": uses_gold_label,
        "context_node_ids": list(context_node_ids),
        "context_tokens": int(context_tokens),
    }


def render_answer_carrier_oracle(
    *,
    example: QueryExample,
    retrieval_row: dict[str, Any],
    renderer_mode: str,
    max_context_tokens: int,
) -> AnswerCarrierOracleRenderResult:
    if renderer_mode not in ANSWER_CARRIER_ORACLE_RENDERERS:
        raise ValueError(f"Unknown answer carrier oracle renderer: {renderer_mode}")

    diagnostics = _diagnostics(retrieval_row)
    answer_ids = set(answer_containing_chunk_ids(example, example.nodes))
    gold_ids = {str(value) for value in example.gold_node_ids}
    projected_ids = set(_ids(diagnostics.get("projected_node_ids", [])))
    if not projected_ids:
        projected_ids = {str(node.node_id) for node in example.nodes}
    selected_basin_ids = set(_ids(diagnostics.get("diagnostic_selected_basin_node_ids", [])))
    current_order = _ids(retrieval_row.get("context_node_ids", []))

    uses_answer = True
    uses_gold = False
    if renderer_mode == PROJECTED_ANSWER_CHUNK_ORACLE:
        selected = _ordered_subset(example.nodes, answer_ids & projected_ids)
    elif renderer_mode == SELECTED_BASIN_ANSWER_CHUNK_ORACLE:
        selected = _ordered_subset(example.nodes, answer_ids & selected_basin_ids)
    elif renderer_mode == CURRENT_ANSWER_ROLE_ORACLE:
        selected = [node_id for node_id in current_order if node_id in answer_ids]
    else:
        selected = [node_id for node_id in current_order if node_id in gold_ids]
        uses_answer = False
        uses_gold = True

    context_ids, tokens = _materialize(example.nodes, selected, max_context_tokens=max_context_tokens)
    return AnswerCarrierOracleRenderResult(
        renderer_mode=renderer_mode,
        context_node_ids=context_ids,
        context_tokens=tokens,
        diagnostics=_oracle_diagnostics(
            renderer_mode=renderer_mode,
            context_node_ids=context_ids,
            context_tokens=tokens,
            uses_answer_string=uses_answer,
            uses_gold_label=uses_gold,
        ),
    )


__all__ = [
    "ANSWER_CARRIER_ORACLE_RENDERERS",
    "CURRENT_ANSWER_ROLE_ORACLE",
    "GOLD_CHUNK_ROLE_ORACLE",
    "PROJECTED_ANSWER_CHUNK_ORACLE",
    "SELECTED_BASIN_ANSWER_CHUNK_ORACLE",
    "AnswerCarrierOracleRenderResult",
    "render_answer_carrier_oracle",
]
