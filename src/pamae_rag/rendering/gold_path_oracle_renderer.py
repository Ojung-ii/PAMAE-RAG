from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.diagnostics.path_realizability import path_nodes, support_tree_nodes
from pamae_rag.eval.support_recall import recall
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
class GoldPathOracleRenderResult:
    indices: list[int]
    diagnostics: dict[str, Any]


def _node_id(nodes: Sequence[EvidenceNode], idx: int) -> str:
    return str(nodes[int(idx)].node_id)


def _idx_by_id(nodes: Sequence[EvidenceNode]) -> dict[str, int]:
    return {str(node.node_id): idx for idx, node in enumerate(nodes)}


def _answer_in_context(example: QueryExample, nodes: Sequence[EvidenceNode], idxs: Sequence[int]) -> bool | None:
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


def _selected_basin_ids(selected_medoids: Sequence[int], node_to_basin: dict[int, int]) -> set[int]:
    return {
        int(node_to_basin[int(idx)])
        for idx in selected_medoids
        if int(idx) in node_to_basin
    }


def _medoid_for_gold(
    gold_idx: int,
    selected_medoids: Sequence[int],
    node_to_basin: dict[int, int],
    distance_matrix: np.ndarray,
    nodes: Sequence[EvidenceNode],
) -> int | None:
    basin = node_to_basin.get(int(gold_idx))
    same_basin = [
        int(idx)
        for idx in selected_medoids
        if basin is not None and int(idx) in node_to_basin and int(node_to_basin[int(idx)]) == int(basin)
    ]
    pool = same_basin or [int(idx) for idx in selected_medoids]
    if not pool:
        return None
    return min(
        pool,
        key=lambda idx: (
            float(distance_matrix[int(idx), int(gold_idx)]),
            _node_id(nodes, int(idx)),
            int(idx),
        ),
    )


def render_gold_path_oracle_indices(
    *,
    example: QueryExample,
    nodes: Sequence[EvidenceNode],
    selected_medoids: Sequence[int],
    query_anchors: Sequence[int],
    distance_matrix: np.ndarray,
    max_context_tokens: int,
    max_context_nodes: int | None,
    node_to_basin: dict[int, int],
    disconnected_distance: float,
) -> GoldPathOracleRenderResult:
    selected = list(dict.fromkeys(int(idx) for idx in selected_medoids))
    anchors = list(dict.fromkeys(int(idx) for idx in query_anchors))
    node_to_basin = {int(key): int(value) for key, value in node_to_basin.items()}
    budget = _Budget(nodes, max_context_tokens, max_context_nodes, [])
    for medoid in selected:
        budget.add(medoid, force=True)

    tree = support_tree_nodes(
        query_anchors=anchors,
        selected_medoids=selected,
        distance_matrix=distance_matrix,
        nodes=nodes,
        disconnected_distance=disconnected_distance,
    )
    for idx in sorted(
        tree,
        key=lambda node: (
            min((float(distance_matrix[int(anchor), int(node)]) for anchor in anchors), default=0.0),
            _node_id(nodes, int(node)),
            int(node),
        ),
    ):
        budget.add(int(idx))

    idx_by_id = _idx_by_id(nodes)
    selected_basins = _selected_basin_ids(selected, node_to_basin)
    gold_path_nodes: set[int] = set()
    gold_path_added = False
    for gold_id in sorted(str(value) for value in example.gold_node_ids):
        gold_idx = idx_by_id.get(gold_id)
        if gold_idx is None:
            continue
        basin = node_to_basin.get(int(gold_idx))
        if basin is None or int(basin) not in selected_basins:
            continue
        medoid = _medoid_for_gold(gold_idx, selected, node_to_basin, distance_matrix, nodes)
        path = path_nodes(
            distance_matrix,
            medoid,
            gold_idx,
            nodes,
            disconnected_distance=disconnected_distance,
        )
        if not path:
            continue
        for idx in path:
            gold_path_nodes.add(int(idx))
            if budget.add(int(idx)):
                gold_path_added = True

    context_ids = [_node_id(nodes, idx) for idx in budget.selected]
    diagnostics = {
        "renderer_mode": "gold_path_oracle",
        "oracle_renderer": True,
        "gold_path_added": gold_path_added,
        "gold_path_chunk_count": len(gold_path_nodes),
        "rendered_recall": recall(context_ids, example.gold_node_ids),
        "answer_in_context": _answer_in_context(example, nodes, budget.selected),
        "qa_f1": None,
    }
    return GoldPathOracleRenderResult(indices=budget.selected, diagnostics=diagnostics)

