from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.diagnostics.path_realizability import support_tree_nodes
from pamae_rag.eval.support_recall import precision, recall
from pamae_rag.qa.metrics import gold_answers, normalize_answer


@dataclass
class _Budget:
    nodes: Sequence[EvidenceNode]
    max_context_tokens: int
    max_context_nodes: int | None
    selected: list[int]
    used_tokens: int = 0

    def add(self, idx: int, *, force: bool = False) -> bool:
        idx = int(idx)
        if idx in self.selected:
            return False
        tok = max(1, int(self.nodes[idx].token_count))
        if self.used_tokens + tok > self.max_context_tokens:
            return False
        if not force and self.max_context_nodes and len(self.selected) >= self.max_context_nodes:
            return False
        self.selected.append(idx)
        self.used_tokens += tok
        return True


@dataclass(frozen=True)
class PathNeighborhoodRenderResult:
    indices: list[int]
    diagnostics: dict[str, Any]


def _node_id(nodes: Sequence[EvidenceNode], idx: int) -> str:
    return str(nodes[int(idx)].node_id)


def _nearest_anchor_distance(idx: int, query_anchors: Sequence[int], distance_matrix: np.ndarray) -> float:
    if not query_anchors:
        return 0.0
    return min(float(distance_matrix[int(anchor), int(idx)]) for anchor in query_anchors)


def _distance_to_tree(idx: int, tree: set[int], distance_matrix: np.ndarray) -> float:
    if not tree:
        return 0.0
    return min(float(distance_matrix[int(idx), int(node)]) for node in tree)


def _answer_in_context(example: QueryExample | None, nodes: Sequence[EvidenceNode], idxs: Sequence[int]) -> bool | None:
    if example is None:
        return None
    answers = gold_answers(example)
    if not answers:
        return None
    text = " ".join(str(nodes[int(idx)].text) for idx in idxs)
    text_norm = normalize_answer(text)
    if not text_norm:
        return False
    padded = f" {text_norm} "
    return any(
        bool(answer_norm and f" {answer_norm} " in padded)
        for answer_norm in (normalize_answer(answer) for answer in answers)
    )


def _context_f1(example: QueryExample | None, nodes: Sequence[EvidenceNode], idxs: Sequence[int]) -> float | None:
    if example is None:
        return None
    ids = [_node_id(nodes, idx) for idx in idxs]
    rec = recall(ids, example.gold_node_ids)
    prec = precision(ids, example.gold_node_ids)
    if rec is None or prec is None or rec + prec == 0:
        return 0.0
    return float(2 * rec * prec / (rec + prec))


def render_path_neighborhood_indices(
    *,
    nodes: Sequence[EvidenceNode],
    selected_medoids: Sequence[int],
    query_anchors: Sequence[int],
    distance_matrix: np.ndarray,
    rho: np.ndarray,
    max_context_tokens: int,
    max_context_nodes: int | None,
    active_indices: Sequence[int],
    disconnected_distance: float,
    example: QueryExample | None = None,
) -> PathNeighborhoodRenderResult:
    selected = list(dict.fromkeys(int(idx) for idx in selected_medoids))
    anchors = list(dict.fromkeys(int(idx) for idx in query_anchors))
    tree = support_tree_nodes(
        query_anchors=anchors,
        selected_medoids=selected,
        distance_matrix=distance_matrix,
        nodes=nodes,
        disconnected_distance=disconnected_distance,
    )
    budget = _Budget(nodes, max_context_tokens, max_context_nodes, [])
    rendered_support_tree: set[int] = set()
    rendered_neighborhood: set[int] = set()
    for medoid in selected:
        if budget.add(medoid, force=True):
            rendered_support_tree.add(int(medoid))
    for idx in sorted(
        tree,
        key=lambda node: (
            _nearest_anchor_distance(int(node), anchors, distance_matrix),
            _node_id(nodes, int(node)),
            int(node),
        ),
    ):
        if budget.add(int(idx)):
            rendered_support_tree.add(int(idx))

    candidates = sorted(
        (int(idx) for idx in active_indices),
        key=lambda idx: (
            _distance_to_tree(int(idx), tree, distance_matrix),
            -float(rho[int(idx)]),
            _nearest_anchor_distance(int(idx), anchors, distance_matrix),
            _node_id(nodes, int(idx)),
            int(idx),
        ),
    )
    max_tree_distance = 0.0
    for idx in candidates:
        if idx in tree:
            continue
        distance_to_tree = _distance_to_tree(int(idx), tree, distance_matrix)
        if budget.add(int(idx)):
            rendered_neighborhood.add(int(idx))
            max_tree_distance = max(max_tree_distance, float(distance_to_tree))

    ids = [_node_id(nodes, idx) for idx in budget.selected]
    rendered_recall = recall(ids, example.gold_node_ids) if example is not None else None
    diagnostics = {
        "renderer_mode": "path_neighborhood",
        "support_tree_chunk_count": len(rendered_support_tree),
        "neighborhood_chunk_count": len(rendered_neighborhood),
        "max_tree_distance_rendered": max_tree_distance,
        "rendered_recall": rendered_recall,
        "answer_in_context": _answer_in_context(example, nodes, budget.selected),
        "context_f1": _context_f1(example, nodes, budget.selected),
        "context_tokens": int(sum(max(1, int(nodes[idx].token_count)) for idx in budget.selected)),
    }
    return PathNeighborhoodRenderResult(indices=budget.selected, diagnostics=diagnostics)

