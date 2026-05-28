from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.qa.metrics import gold_answers, normalize_answer
from pamae_rag.selection.basin_preserving import assign_query_basins, basin_masses, query_anchor_indices


@dataclass(frozen=True)
class PathRealizabilityResult:
    gold_rows: list[dict[str, Any]]
    answer_trace: dict[str, Any]
    basin_position_rows: list[dict[str, Any]]
    support_tree_node_ids: list[str]

    def to_json(self) -> dict[str, Any]:
        return {
            "gold_rows": self.gold_rows,
            "answer_trace": self.answer_trace,
            "basin_position_rows": self.basin_position_rows,
            "support_tree_node_ids": self.support_tree_node_ids,
        }


def _node_ids(nodes: Sequence[EvidenceNode], idxs: Iterable[int]) -> list[str]:
    return [str(nodes[int(idx)].node_id) for idx in idxs]


def _index_by_id(nodes: Sequence[EvidenceNode]) -> dict[str, int]:
    return {str(node.node_id): idx for idx, node in enumerate(nodes)}


def _safe_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _path_exists(distance_matrix: np.ndarray, start: int | None, end: int | None, disconnected_distance: float) -> bool:
    if start is None or end is None:
        return False
    if int(start) == int(end):
        return True
    value = float(distance_matrix[int(start), int(end)])
    return bool(np.isfinite(value) and value < float(disconnected_distance) - 1e-9)


def path_nodes(
    distance_matrix: np.ndarray,
    start: int | None,
    end: int | None,
    nodes: Sequence[EvidenceNode],
    *,
    disconnected_distance: float,
    eps: float = 1e-7,
) -> list[int]:
    if not _path_exists(distance_matrix, start, end, disconnected_distance):
        return []
    start_i = int(start)
    end_i = int(end)
    target = float(distance_matrix[start_i, end_i])
    out: list[int] = []
    for idx in range(distance_matrix.shape[0]):
        via = float(distance_matrix[start_i, idx]) + float(distance_matrix[idx, end_i])
        if abs(via - target) <= eps:
            out.append(int(idx))
    return sorted(out, key=lambda idx: (float(distance_matrix[start_i, idx]), str(nodes[idx].node_id), idx))


def support_tree_nodes(
    *,
    query_anchors: Sequence[int],
    selected_medoids: Sequence[int],
    distance_matrix: np.ndarray,
    nodes: Sequence[EvidenceNode],
    disconnected_distance: float,
) -> set[int]:
    tree: set[int] = set(int(idx) for idx in selected_medoids)
    for anchor in query_anchors:
        tree.add(int(anchor))
        for medoid in selected_medoids:
            tree.update(
                path_nodes(
                    distance_matrix,
                    int(anchor),
                    int(medoid),
                    nodes,
                    disconnected_distance=disconnected_distance,
                )
            )
    medoids = list(dict.fromkeys(int(idx) for idx in selected_medoids))
    for pos, left in enumerate(medoids):
        for right in medoids[pos + 1 :]:
            tree.update(
                path_nodes(
                    distance_matrix,
                    int(left),
                    int(right),
                    nodes,
                    disconnected_distance=disconnected_distance,
                )
            )
    return tree


def answer_containing_chunk_ids(example: QueryExample, nodes: Sequence[EvidenceNode]) -> list[str]:
    answers = gold_answers(example)
    if not answers:
        return []
    normalized_answers = [normalize_answer(answer) for answer in answers]
    normalized_answers = [answer for answer in normalized_answers if answer]
    if not normalized_answers:
        return []
    out: list[str] = []
    for node in nodes:
        text_norm = normalize_answer(str(node.text))
        if not text_norm:
            continue
        padded = f" {text_norm} "
        if any(f" {answer} " in padded for answer in normalized_answers):
            out.append(str(node.node_id))
    return sorted(out)


def _selected_basin_ids(selected_medoids: Sequence[int], node_to_basin: dict[int, int]) -> set[int]:
    return {
        int(node_to_basin[int(idx)])
        for idx in selected_medoids
        if int(idx) in node_to_basin
    }


def _nearest_selected_medoid(
    target: int | None,
    selected_medoids: Sequence[int],
    distance_matrix: np.ndarray,
    nodes: Sequence[EvidenceNode],
) -> int | None:
    if target is None or not selected_medoids:
        return None
    return min(
        (int(idx) for idx in selected_medoids),
        key=lambda idx: (
            float(distance_matrix[int(idx), int(target)]),
            str(nodes[int(idx)].node_id),
            int(idx),
        ),
    )


def _selected_medoid_for_basin(
    basin_id: int | None,
    selected_medoids: Sequence[int],
    node_to_basin: dict[int, int],
    fallback_target: int | None,
    distance_matrix: np.ndarray,
    nodes: Sequence[EvidenceNode],
) -> int | None:
    if basin_id is not None:
        same_basin = [
            int(idx)
            for idx in selected_medoids
            if int(idx) in node_to_basin and int(node_to_basin[int(idx)]) == int(basin_id)
        ]
        if same_basin:
            return min(
                same_basin,
                key=lambda idx: (
                    float(distance_matrix[int(idx), int(fallback_target)]) if fallback_target is not None else 0.0,
                    str(nodes[int(idx)].node_id),
                    int(idx),
                ),
            )
    return _nearest_selected_medoid(fallback_target, selected_medoids, distance_matrix, nodes)


def _weighted_radius_percentile(values: list[tuple[float, float]], q: float) -> float | None:
    if not values:
        return None
    total = sum(max(0.0, float(weight)) for _, weight in values)
    if total <= 0.0:
        ranked = sorted(float(value) for value, _ in values)
        pos = min(len(ranked) - 1, max(0, int(round((len(ranked) - 1) * q))))
        return float(ranked[pos])
    running = 0.0
    for value, weight in sorted(values, key=lambda item: item[0]):
        running += max(0.0, float(weight))
        if running / total >= q:
            return float(value)
    return float(max(value for value, _ in values))


def _basin_position(
    *,
    basin_id: int,
    gold_idx: int,
    selected_medoid: int,
    node_to_basin: dict[int, int],
    distance_matrix: np.ndarray,
    rho: np.ndarray,
) -> dict[str, Any]:
    members = sorted(idx for idx, owner in node_to_basin.items() if int(owner) == int(basin_id))
    distances = [(float(distance_matrix[int(selected_medoid), int(idx)]), float(rho[int(idx)])) for idx in members]
    gold_distance = float(distance_matrix[int(selected_medoid), int(gold_idx)])
    leq = sum(1 for value, _ in distances if value <= gold_distance + 1e-9)
    percentile = leq / max(len(distances), 1)
    return {
        "basin_id": str(basin_id),
        "basin_size": len(members),
        "basin_mass": float(sum(float(rho[int(idx)]) for idx in members)),
        "basin_radius_max": float(max((value for value, _ in distances), default=0.0)),
        "basin_radius_mass_p50": _weighted_radius_percentile(distances, 0.50),
        "basin_radius_mass_p90": _weighted_radius_percentile(distances, 0.90),
        "gold_distance_from_medoid": gold_distance,
        "gold_distance_percentile_within_basin": float(percentile),
    }


def compute_path_realizability(
    *,
    example: QueryExample,
    nodes: Sequence[EvidenceNode],
    candidate_indices: Sequence[int],
    projected_node_ids: Sequence[str],
    selected_medoids: Sequence[int],
    context_indices: Sequence[int],
    distance_matrix: np.ndarray,
    rho: np.ndarray,
    selected_mode: str,
    renderer_mode: str,
    max_context_tokens: int,
    max_context_nodes: int | None,
    disconnected_distance: float,
    node_to_basin: dict[int, int] | None = None,
    query_anchors: Sequence[int] | None = None,
) -> PathRealizabilityResult:
    idx_by_id = _index_by_id(nodes)
    candidate_set = {int(idx) for idx in candidate_indices}
    projected_ids = {str(node_id) for node_id in projected_node_ids}
    if not projected_ids:
        projected_ids = set(idx_by_id)
    selected = list(dict.fromkeys(int(idx) for idx in selected_medoids))
    context = set(int(idx) for idx in context_indices)
    context_ids = {str(nodes[idx].node_id) for idx in context}
    if query_anchors is None:
        query_anchors = query_anchor_indices(candidate_indices, rho, max(1, len(selected) or 1))
    query_anchors = list(dict.fromkeys(int(idx) for idx in query_anchors))
    if node_to_basin is None:
        node_to_basin = assign_query_basins(
            candidate_indices,
            query_anchors,
            distance_matrix,
            [node.node_id for node in nodes],
        )
    else:
        node_to_basin = {int(key): int(value) for key, value in node_to_basin.items()}
    selected_basins = _selected_basin_ids(selected, node_to_basin)
    tree = support_tree_nodes(
        query_anchors=query_anchors,
        selected_medoids=selected,
        distance_matrix=distance_matrix,
        nodes=nodes,
        disconnected_distance=disconnected_distance,
    )
    context_token_count = sum(max(1, int(nodes[idx].token_count)) for idx in context_indices)
    node_budget_saturated = bool(max_context_nodes and len(context_indices) >= int(max_context_nodes))
    token_budget_saturated = bool(context_token_count >= int(max_context_tokens))
    answer_in_context = _answer_in_context_text(example, nodes, context_indices)

    gold_rows: list[dict[str, Any]] = []
    basin_rows: list[dict[str, Any]] = []
    for gold_id in sorted(str(value) for value in example.gold_node_ids):
        gold_idx = idx_by_id.get(gold_id)
        gold_in_active = gold_idx is not None
        gold_in_candidate = bool(gold_idx is not None and gold_idx in candidate_set)
        gold_in_projected = bool(gold_id in projected_ids)
        assigned_anchor = None
        gold_basin = None
        selected_medoid = None
        selected_medoid_basin = None
        if gold_idx is not None and query_anchors:
            assigned_anchor = min(
                query_anchors,
                key=lambda idx: (
                    float(distance_matrix[int(idx), int(gold_idx)]),
                    str(nodes[int(idx)].node_id),
                    int(idx),
                ),
            )
            gold_basin = node_to_basin.get(int(gold_idx), int(assigned_anchor) if gold_in_candidate else None)
            selected_medoid = _selected_medoid_for_basin(
                gold_basin,
                selected,
                node_to_basin,
                gold_idx,
                distance_matrix,
                nodes,
            )
            selected_medoid_basin = (
                node_to_basin.get(int(selected_medoid)) if selected_medoid is not None else None
            )
        nearest_medoid = _nearest_selected_medoid(gold_idx, selected, distance_matrix, nodes)
        anchor_gold_path = path_nodes(
            distance_matrix,
            assigned_anchor,
            gold_idx,
            nodes,
            disconnected_distance=disconnected_distance,
        )
        anchor_medoid_path = path_nodes(
            distance_matrix,
            assigned_anchor,
            selected_medoid,
            nodes,
            disconnected_distance=disconnected_distance,
        )
        medoid_gold_path = path_nodes(
            distance_matrix,
            selected_medoid,
            gold_idx,
            nodes,
            disconnected_distance=disconnected_distance,
        )
        gold_on_medoid_pair_path = False
        for pos, left in enumerate(selected):
            for right in selected[pos + 1 :]:
                if gold_idx is not None and gold_idx in path_nodes(
                    distance_matrix,
                    left,
                    right,
                    nodes,
                    disconnected_distance=disconnected_distance,
                ):
                    gold_on_medoid_pair_path = True
                    break
            if gold_on_medoid_pair_path:
                break
        gold_rendered = bool(gold_idx is not None and gold_idx in context)
        gold_on_tree = bool(gold_idx is not None and gold_idx in tree)
        budget_cutoff = bool(
            gold_idx is not None
            and not gold_rendered
            and gold_on_tree
            and (node_budget_saturated or token_budget_saturated)
        )
        row = {
            "query_id": example.query_id,
            "dataset": _dataset_name(example),
            "gold_chunk_id": gold_id,
            "gold_answer": example.answer,
            "gold_in_candidate": gold_in_candidate,
            "gold_in_projected": gold_in_projected,
            "gold_in_active_universe": gold_in_active,
            "selected_mode": selected_mode,
            "renderer_mode": renderer_mode,
            "assigned_query_anchor_id": _id(nodes, assigned_anchor),
            "assigned_basin_id": _id(nodes, gold_basin),
            "gold_basin_id": _id(nodes, gold_basin),
            "gold_in_selected_basin": bool(gold_basin is not None and int(gold_basin) in selected_basins),
            "selected_medoid_id": _id(nodes, selected_medoid),
            "selected_medoid_basin_id": _id(nodes, selected_medoid_basin),
            "d_anchor_gold": _distance(distance_matrix, assigned_anchor, gold_idx),
            "d_anchor_medoid": _distance(distance_matrix, assigned_anchor, selected_medoid),
            "d_medoid_gold": _distance(distance_matrix, selected_medoid, gold_idx),
            "d_nearest_selected_medoid_gold": _distance(distance_matrix, nearest_medoid, gold_idx),
            "anchor_to_gold_path_exists": bool(anchor_gold_path),
            "anchor_to_medoid_path_exists": bool(anchor_medoid_path),
            "medoid_to_gold_path_exists": bool(medoid_gold_path),
            "gold_on_anchor_medoid_path": bool(gold_idx is not None and gold_idx in anchor_medoid_path),
            "gold_on_medoid_medoid_path": gold_on_medoid_pair_path,
            "gold_on_existing_support_tree": gold_on_tree,
            "medoid_to_gold_path_chunk_count": len(medoid_gold_path),
            "anchor_to_gold_path_chunk_count": len(anchor_gold_path),
            "gold_rendered": gold_rendered,
            "answer_in_context": answer_in_context,
            "render_budget_cutoff_before_gold": budget_cutoff,
            "qa_f1": None,
        }
        gold_rows.append(row)
        if (
            gold_idx is not None
            and gold_basin is not None
            and selected_medoid is not None
            and int(gold_basin) in selected_basins
        ):
            basin_row = _basin_position(
                basin_id=int(gold_basin),
                gold_idx=int(gold_idx),
                selected_medoid=int(selected_medoid),
                node_to_basin=node_to_basin,
                distance_matrix=distance_matrix,
                rho=rho,
            )
            basin_row.update({"query_id": example.query_id, "gold_chunk_id": gold_id})
            basin_rows.append(basin_row)

    answer_trace = _answer_trace(
        example=example,
        nodes=nodes,
        candidate_set=candidate_set,
        projected_ids=projected_ids,
        context=context,
        selected_basins=selected_basins,
        node_to_basin=node_to_basin,
        selected_medoids=selected,
        support_tree=tree,
        distance_matrix=distance_matrix,
    )
    return PathRealizabilityResult(
        gold_rows=gold_rows,
        answer_trace=answer_trace,
        basin_position_rows=basin_rows,
        support_tree_node_ids=_node_ids(nodes, sorted(tree)),
    )


def _dataset_name(example: QueryExample) -> str | None:
    nested = example.metadata.get("metadata")
    if isinstance(nested, dict) and nested.get("dataset") is not None:
        return str(nested.get("dataset"))
    if example.metadata.get("dataset") is not None:
        return str(example.metadata.get("dataset"))
    return None


def _id(nodes: Sequence[EvidenceNode], idx: int | None) -> str | None:
    if idx is None:
        return None
    if 0 <= int(idx) < len(nodes):
        return str(nodes[int(idx)].node_id)
    return str(idx)


def _distance(distance_matrix: np.ndarray, left: int | None, right: int | None) -> float | None:
    if left is None or right is None:
        return None
    return float(distance_matrix[int(left), int(right)])


def _answer_in_context_text(example: QueryExample, nodes: Sequence[EvidenceNode], context_indices: Sequence[int]) -> bool | None:
    answers = gold_answers(example)
    if not answers:
        return None
    context = " ".join(str(nodes[int(idx)].text) for idx in context_indices)
    context_norm = normalize_answer(context)
    if not context_norm:
        return False
    padded = f" {context_norm} "
    return any(
        bool(answer_norm and f" {answer_norm} " in padded)
        for answer_norm in (normalize_answer(answer) for answer in answers)
    )


def _answer_trace(
    *,
    example: QueryExample,
    nodes: Sequence[EvidenceNode],
    candidate_set: set[int],
    projected_ids: set[str],
    context: set[int],
    selected_basins: set[int],
    node_to_basin: dict[int, int],
    selected_medoids: Sequence[int],
    support_tree: set[int],
    distance_matrix: np.ndarray,
) -> dict[str, Any]:
    idx_by_id = _index_by_id(nodes)
    answer_ids = answer_containing_chunk_ids(example, nodes)
    answer_indices = [idx_by_id[node_id] for node_id in answer_ids if node_id in idx_by_id]
    nearest_medoid = None
    nearest_answer_idx = None
    if answer_indices and selected_medoids:
        nearest_answer_idx = min(
            answer_indices,
            key=lambda answer_idx: min(float(distance_matrix[int(m), int(answer_idx)]) for m in selected_medoids),
        )
        nearest_medoid = _nearest_selected_medoid(nearest_answer_idx, selected_medoids, distance_matrix, nodes)
    return {
        "query_id": example.query_id,
        "answer_containing_chunk_found": bool(answer_ids),
        "answer_containing_chunk_ids": answer_ids,
        "answer_chunk_in_candidate": any(idx in candidate_set for idx in answer_indices),
        "answer_chunk_in_projected": any(str(nodes[idx].node_id) in projected_ids for idx in answer_indices),
        "answer_chunk_in_selected_basin": any(
            idx in node_to_basin and int(node_to_basin[idx]) in selected_basins for idx in answer_indices
        ),
        "answer_chunk_rendered": any(idx in context for idx in answer_indices),
        "nearest_selected_medoid_to_answer_chunk": _id(nodes, nearest_medoid),
        "d_medoid_answer_chunk": _distance(distance_matrix, nearest_medoid, nearest_answer_idx),
        "answer_chunk_on_support_tree": any(idx in support_tree for idx in answer_indices),
    }

