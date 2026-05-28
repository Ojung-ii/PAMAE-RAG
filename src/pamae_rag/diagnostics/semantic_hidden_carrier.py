from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

from pamae_rag.data.schema import EvidenceNode, QueryExample
from pamae_rag.diagnostics.path_realizability import answer_containing_chunk_ids
from pamae_rag.semantic.angular_distance import angular_distance, min_angular_distance
from pamae_rag.semantic.embedding_store import EmbeddingStore
from pamae_rag.semantic.semantic_attribution import aggregate_semantic_groups

DistanceLookup = Callable[[str, str], float | None]

SEMANTIC_CARRIER_GROUPS = (
    "current_only_answer",
    "current_only_non_answer",
    "tree_answer",
    "tree_non_answer",
    "projected_nonrendered_answer",
)


def _ids(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value is not None))


def _diagnostics(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("diagnostics")
    return value if isinstance(value, dict) else {}


def _chunk_ids(nodes: Sequence[EvidenceNode], values: Iterable[Any]) -> set[str]:
    chunks = {str(node.node_id) for node in nodes if str(getattr(node, "node_type", "chunk")) == "chunk"}
    return {node_id for node_id in _ids(values) if node_id in chunks}


def _nearest_graph_distance(chunk_id: str, tree_chunks: set[str], distance_lookup: DistanceLookup | None) -> float | None:
    if chunk_id in tree_chunks:
        return 0.0
    if distance_lookup is None or not tree_chunks:
        return None
    values: list[float] = []
    for tree_id in sorted(tree_chunks):
        value = distance_lookup(chunk_id, tree_id)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    if not values:
        return None
    return float(min(values))


def _semantic_distances(
    *,
    chunk_id: str,
    tree_chunks: set[str],
    store: EmbeddingStore,
) -> tuple[float | None, float | None]:
    chunk_embedding = store.node_embedding(chunk_id)
    if chunk_embedding is None:
        return None, None
    query_embedding = store.query_embedding
    query_distance: float | None = None
    if query_embedding is not None:
        query_distance = angular_distance(query_embedding, chunk_embedding)
    tree_embeddings = [store.node_embedding(tree_id) for tree_id in sorted(tree_chunks)]
    tree_embeddings = [embedding for embedding in tree_embeddings if embedding is not None]
    tree_distance = min_angular_distance(chunk_embedding, tree_embeddings) if tree_embeddings else None
    return query_distance, tree_distance


def _row(
    *,
    example: QueryExample,
    chunk_id: str,
    group: str,
    tree_chunks: set[str],
    current_chunks: set[str],
    answer_chunks: set[str],
    gold_chunks: set[str],
    store: EmbeddingStore,
    distance_lookup: DistanceLookup | None,
) -> dict[str, Any]:
    query_distance, tree_distance = _semantic_distances(chunk_id=chunk_id, tree_chunks=tree_chunks, store=store)
    return {
        "query_id": example.query_id,
        "chunk_id": chunk_id,
        "group": group,
        "d_graph_to_tree": _nearest_graph_distance(chunk_id, tree_chunks, distance_lookup),
        "d_ang_query_chunk": query_distance,
        "d_ang_chunk_tree": tree_distance,
        "on_support_tree": chunk_id in tree_chunks,
        "current_rendered": chunk_id in current_chunks,
        "contains_answer": chunk_id in answer_chunks,
        "is_gold": chunk_id in gold_chunks,
    }


def semantic_hidden_carrier_rows(
    *,
    example: QueryExample,
    current_row: dict[str, Any],
    store: EmbeddingStore | None = None,
    distance_lookup: DistanceLookup | None = None,
) -> list[dict[str, Any]]:
    store = store or EmbeddingStore.from_example(example)
    diagnostics = _diagnostics(current_row)
    current_chunks = _chunk_ids(example.nodes, current_row.get("context_node_ids", []))
    tree_chunks = _chunk_ids(example.nodes, diagnostics.get("refined_support_tree_node_ids", []))
    projected_chunks = _chunk_ids(example.nodes, diagnostics.get("projected_node_ids", []))
    answer_chunks = set(answer_containing_chunk_ids(example, example.nodes))
    gold_chunks = {str(node_id) for node_id in example.gold_node_ids}

    current_only = current_chunks - tree_chunks
    groups = {
        "current_only_answer": current_only & answer_chunks,
        "current_only_non_answer": current_only - answer_chunks,
        "tree_answer": tree_chunks & answer_chunks,
        "tree_non_answer": tree_chunks - answer_chunks,
        "projected_nonrendered_answer": (projected_chunks & answer_chunks) - current_chunks,
    }

    rows: list[dict[str, Any]] = []
    for group in SEMANTIC_CARRIER_GROUPS:
        for chunk_id in sorted(groups[group]):
            rows.append(
                _row(
                    example=example,
                    chunk_id=chunk_id,
                    group=group,
                    tree_chunks=tree_chunks,
                    current_chunks=current_chunks,
                    answer_chunks=answer_chunks,
                    gold_chunks=gold_chunks,
                    store=store,
                    distance_lookup=distance_lookup,
                )
            )
    return rows


def aggregate_semantic_hidden_carrier(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    aggregate = aggregate_semantic_groups(rows)
    aggregate["semantic_signal_available"] = bool(
        aggregate.get("semantic_separation_query") is not None
        or aggregate.get("semantic_separation_tree") is not None
    )
    aggregate["semantic_query_signal_available"] = bool(aggregate.get("semantic_separation_query") is not None)
    aggregate["semantic_tree_signal_available"] = bool(aggregate.get("semantic_separation_tree") is not None)
    return aggregate


__all__ = [
    "SEMANTIC_CARRIER_GROUPS",
    "aggregate_semantic_hidden_carrier",
    "semantic_hidden_carrier_rows",
]
