from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.diagnostics.path_realizability import answer_containing_chunk_ids

TREE_ALL_NO_BUDGET = "tree_all_no_budget"
TREE_CURRENT_BUDGET_ORDER = "tree_current_budget_order"
CURRENT_TREE_INTERSECTION_ONLY = "current_tree_intersection_only"
CURRENT_ONLY_NON_TREE = "current_only_non_tree"
TREE_ANSWER_ORACLE = "tree_answer_oracle"

DIAGNOSTIC_TREE_RENDERERS = {
    TREE_ALL_NO_BUDGET,
    TREE_CURRENT_BUDGET_ORDER,
    CURRENT_TREE_INTERSECTION_ONLY,
    CURRENT_ONLY_NON_TREE,
}
TREE_ORACLE_RENDERERS = {TREE_ANSWER_ORACLE}
TREE_ABLATION_RENDERERS = {*DIAGNOSTIC_TREE_RENDERERS, *TREE_ORACLE_RENDERERS}


@dataclass(frozen=True)
class TreeAblationRenderResult:
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


def _chunk_ids(nodes: Sequence[EvidenceNode], values: Iterable[Any]) -> set[str]:
    chunks = {str(node.node_id) for node in nodes if str(getattr(node, "node_type", "chunk")) == "chunk"}
    return {node_id for node_id in _ids(values) if node_id in chunks}


def _path_order(nodes: Sequence[EvidenceNode], retrieval_row: dict[str, Any]) -> list[str]:
    diagnostics = _diagnostics(retrieval_row)
    seen: set[str] = set()
    out: list[str] = []

    def add(values: Iterable[Any]) -> None:
        for node_id in _ids(values):
            if node_id in seen:
                continue
            seen.add(node_id)
            out.append(node_id)

    add(retrieval_row.get("anchor_node_ids", []))
    add(diagnostics.get("refined_anchor_medoid_path_node_ids", []))
    add(diagnostics.get("refined_medoid_medoid_path_node_ids", []))
    add(sorted(_chunk_ids(nodes, diagnostics.get("refined_support_tree_node_ids", []))))
    chunk_set = _chunk_ids(nodes, out)
    return [node_id for node_id in out if node_id in chunk_set]


def _materialize(
    nodes: Sequence[EvidenceNode],
    node_ids: Iterable[str],
    *,
    max_context_tokens: int,
    ignore_budget: bool = False,
) -> tuple[tuple[str, ...], int, tuple[str, ...]]:
    by_id = _node_by_id(nodes)
    out: list[str] = []
    cutoff: list[str] = []
    tokens = 0
    for node_id in _ids(node_ids):
        node = by_id.get(node_id)
        if node is None:
            continue
        node_tokens = max(1, int(node.token_count))
        if not ignore_budget and out and tokens + node_tokens > int(max_context_tokens):
            cutoff.append(node_id)
            continue
        out.append(node_id)
        tokens += node_tokens
    return tuple(out), tokens, tuple(cutoff)


def render_tree_ablation(
    *,
    example: QueryExample,
    retrieval_row: dict[str, Any],
    renderer_mode: str,
    max_context_tokens: int,
) -> TreeAblationRenderResult:
    if renderer_mode not in TREE_ABLATION_RENDERERS:
        raise ValueError(f"Unknown tree ablation renderer: {renderer_mode}")

    diagnostics = _diagnostics(retrieval_row)
    support_tree = _chunk_ids(example.nodes, diagnostics.get("refined_support_tree_node_ids", []))
    current_order = [node_id for node_id in _ids(retrieval_row.get("context_node_ids", [])) if node_id in _chunk_ids(example.nodes, [node_id])]
    current_set = set(current_order)
    path_order = _path_order(example.nodes, retrieval_row)
    uses_answer = False
    ignore_budget = False

    if renderer_mode == TREE_ALL_NO_BUDGET:
        selected = path_order
        ignore_budget = True
    elif renderer_mode == TREE_CURRENT_BUDGET_ORDER:
        selected = [
            *[node_id for node_id in current_order if node_id in support_tree],
            *[node_id for node_id in path_order if node_id not in current_set],
        ]
    elif renderer_mode == CURRENT_TREE_INTERSECTION_ONLY:
        selected = [node_id for node_id in current_order if node_id in support_tree]
    elif renderer_mode == CURRENT_ONLY_NON_TREE:
        selected = [node_id for node_id in current_order if node_id not in support_tree]
    else:
        uses_answer = True
        answer_ids = set(answer_containing_chunk_ids(example, example.nodes))
        selected = [node_id for node_id in path_order if node_id in answer_ids and node_id in support_tree]

    context_ids, tokens, cutoff = _materialize(
        example.nodes,
        selected,
        max_context_tokens=max_context_tokens,
        ignore_budget=ignore_budget,
    )
    return TreeAblationRenderResult(
        renderer_mode=renderer_mode,
        context_node_ids=context_ids,
        context_tokens=tokens,
        diagnostics={
            "renderer_mode": renderer_mode,
            "diagnostic_renderer": renderer_mode in DIAGNOSTIC_TREE_RENDERERS,
            "oracle_renderer": renderer_mode in TREE_ORACLE_RENDERERS,
            "uses_answer_string": uses_answer,
            "uses_gold_label": False,
            "adoption_candidate": False,
            "support_tree_chunk_count": len(support_tree),
            "context_node_ids": list(context_ids),
            "context_tokens": tokens,
            "budget_cutoff_node_ids": list(cutoff),
        },
    )


__all__ = [
    "CURRENT_ONLY_NON_TREE",
    "CURRENT_TREE_INTERSECTION_ONLY",
    "DIAGNOSTIC_TREE_RENDERERS",
    "TREE_ABLATION_RENDERERS",
    "TREE_ALL_NO_BUDGET",
    "TREE_ANSWER_ORACLE",
    "TREE_CURRENT_BUDGET_ORDER",
    "TREE_ORACLE_RENDERERS",
    "TreeAblationRenderResult",
    "render_tree_ablation",
]
