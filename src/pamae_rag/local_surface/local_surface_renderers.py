from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from pamae_rag.data.schema import QueryExample
from pamae_rag.diagnostics.selected_chunk_surface import build_surface_sentence_sets
from pamae_rag.local_surface.local_sentence_medoid import LocalMedoidConfig, LocalMedoidResult, select_local_sentence_medoids
from pamae_rag.local_surface.local_surface_graph import LocalSurfaceGraph, shortest_path
from pamae_rag.qa.metrics import gold_answers, normalize_answer

LOCAL_SENTENCE_MEDOID = "local_sentence_medoid"
FACT_MEDIATED_SENTENCE = "fact_mediated_sentence"
SELECTED_CHUNK_ANSWER_SENTENCE_ORACLE = "selected_chunk_answer_sentence_oracle"
SELECTED_CHUNK_GOLD_SENTENCE_ORACLE = "selected_chunk_gold_sentence_oracle"

ORACLE_RENDERERS = {
    SELECTED_CHUNK_ANSWER_SENTENCE_ORACLE,
    SELECTED_CHUNK_GOLD_SENTENCE_ORACLE,
}


@dataclass(frozen=True)
class LocalSurfaceRenderResult:
    renderer_mode: str
    context_node_ids: tuple[str, ...]
    context_nodes: tuple[dict[str, Any], ...]
    rendered_sentence_ids: tuple[str, ...]
    selected_sentence_ids: tuple[str, ...]
    context_tokens: int
    diagnostics: dict[str, Any]


def _contains_answer(text: str, answers: Iterable[str]) -> bool:
    text_norm = normalize_answer(text)
    if not text_norm:
        return False
    padded = f" {text_norm} "
    for answer in answers:
        answer_norm = normalize_answer(answer)
        if answer_norm and f" {answer_norm} " in padded:
            return True
    return False


def _context_node(graph: LocalSurfaceGraph, sentence_id: str) -> dict[str, Any] | None:
    sentence = graph.sentence_by_id.get(sentence_id)
    if sentence is None:
        return None
    text = f"{sentence.title}: {sentence.sentence_text}" if sentence.title else sentence.sentence_text
    node = sentence.to_context_node(text=text)
    node["metadata"]["context_unit"] = "local_sentence"
    return node


def _materialize(
    graph: LocalSurfaceGraph,
    sentence_ids: Iterable[str],
    *,
    max_context_tokens: int,
) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...], int]:
    ids: list[str] = []
    nodes: list[dict[str, Any]] = []
    tokens = 0
    for sentence_id in dict.fromkeys(str(value) for value in sentence_ids):
        node = _context_node(graph, sentence_id)
        if node is None:
            continue
        node_tokens = int(node["token_count"])
        if ids and tokens + node_tokens > max_context_tokens:
            break
        ids.append(str(node["node_id"]))
        nodes.append(node)
        tokens += node_tokens
    return tuple(ids), tuple(nodes), tokens


def _path_sentence_ids(
    graph: LocalSurfaceGraph,
    anchors: Iterable[str],
    targets: Iterable[str],
) -> tuple[str, ...]:
    sentence_set = set(graph.sentence_ids)
    out: list[str] = []
    for anchor in sorted(dict.fromkeys(str(value) for value in anchors)):
        for target in sorted(dict.fromkeys(str(value) for value in targets)):
            path = shortest_path(graph, anchor, target)
            out.extend(node_id for node_id in path if node_id in sentence_set)
    return tuple(dict.fromkeys(out))


def _window_sentence_ids(graph: LocalSurfaceGraph, sentence_ids: Iterable[str], window: int = 1) -> tuple[str, ...]:
    by_chunk: dict[str, list[str]] = {}
    sentence_by_id = graph.sentence_by_id
    for sentence in graph.sentences:
        by_chunk.setdefault(sentence.chunk_id, []).append(sentence.sentence_id)
    for chunk_id, ids in by_chunk.items():
        by_chunk[chunk_id] = sorted(ids, key=lambda sid: sentence_by_id[sid].sentence_index_in_chunk)
    out: list[str] = []
    for sentence_id in sentence_ids:
        sentence = sentence_by_id.get(str(sentence_id))
        if sentence is None:
            continue
        ids = by_chunk.get(sentence.chunk_id, [])
        try:
            pos = ids.index(sentence.sentence_id)
        except ValueError:
            continue
        for idx in range(max(0, pos - window), min(len(ids), pos + window + 1)):
            out.append(ids[idx])
    return tuple(dict.fromkeys(out))


def _answer_sentence_ids(example: QueryExample, graph: LocalSurfaceGraph) -> tuple[str, ...]:
    answers = gold_answers(example)
    return tuple(
        sentence.sentence_id
        for sentence in graph.sentences
        if answers and _contains_answer(sentence.sentence_text, answers)
    )


def _gold_sentence_ids(example: QueryExample, graph: LocalSurfaceGraph) -> tuple[str, ...]:
    surface = build_surface_sentence_sets(example)
    local = set(graph.sentence_ids)
    return tuple(sentence_id for sentence_id in sorted(surface.gold_sentence_ids) if sentence_id in local)


def _support_metrics(
    *,
    example: QueryExample,
    graph: LocalSurfaceGraph,
    rendered_sentence_ids: Iterable[str],
) -> dict[str, Any]:
    rendered = set(str(sentence_id) for sentence_id in rendered_sentence_ids)
    answer_ids = set(_answer_sentence_ids(example, graph))
    gold_ids = set(_gold_sentence_ids(example, graph))
    all_gold = set(build_surface_sentence_sets(example).gold_sentence_ids)
    rendered_count = len(rendered)
    gold_rendered_count = len(rendered & gold_ids)
    total_gold = len(all_gold)
    recall = (gold_rendered_count / total_gold) if total_gold else 0.0
    precision = (gold_rendered_count / rendered_count) if rendered_count else 0.0
    context_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    text = " ".join(
        graph.sentence_by_id[sentence_id].sentence_text
        for sentence_id in rendered
        if sentence_id in graph.sentence_by_id
    )
    return {
        "answer_sentence_rendered": bool(rendered & answer_ids),
        "gold_sentence_rendered": bool(rendered & gold_ids),
        "answer_sentence_rendered_count": len(rendered & answer_ids),
        "gold_sentence_rendered_count": gold_rendered_count,
        "answer_in_context": _contains_answer(text, gold_answers(example)),
        "rendered_recall": recall,
        "context_precision": precision,
        "context_f1": context_f1,
    }


def render_local_sentence_medoid(
    *,
    example: QueryExample,
    graph: LocalSurfaceGraph,
    medoids: LocalMedoidResult | None = None,
    medoid_config: LocalMedoidConfig | None = None,
    max_context_tokens: int = 512,
) -> LocalSurfaceRenderResult:
    medoids = medoids or select_local_sentence_medoids(graph, example.query, config=medoid_config)
    selected = tuple(medoids.selected_sentence_ids)
    path_sentences = _path_sentence_ids(graph, medoids.query_anchor_entities, selected)
    rendered_sentence_ids = tuple(dict.fromkeys((*selected, *path_sentences)))
    context_ids, context_nodes, context_tokens = _materialize(
        graph,
        rendered_sentence_ids,
        max_context_tokens=max_context_tokens,
    )
    support = _support_metrics(example=example, graph=graph, rendered_sentence_ids=context_ids)
    diagnostics = {
        "renderer_mode": LOCAL_SENTENCE_MEDOID,
        "local_sentence_medoids": (medoid_config or LocalMedoidConfig()).local_sentence_medoids,
        "oracle_renderer": False,
        "uses_answer_string": False,
        "uses_gold_label": False,
        "selected_local_sentence_medoids": list(selected),
        "path_sentence_count": len(path_sentences),
        "context_tokens": context_tokens,
        **support,
        **medoids.diagnostics,
    }
    return LocalSurfaceRenderResult(
        renderer_mode=LOCAL_SENTENCE_MEDOID,
        context_node_ids=context_ids,
        context_nodes=context_nodes,
        rendered_sentence_ids=context_ids,
        selected_sentence_ids=selected,
        context_tokens=context_tokens,
        diagnostics=diagnostics,
    )


def render_answer_sentence_oracle(
    *,
    example: QueryExample,
    graph: LocalSurfaceGraph,
    max_context_tokens: int = 512,
) -> LocalSurfaceRenderResult:
    target_ids = _window_sentence_ids(graph, _answer_sentence_ids(example, graph), window=1)
    context_ids, context_nodes, context_tokens = _materialize(graph, target_ids, max_context_tokens=max_context_tokens)
    support = _support_metrics(example=example, graph=graph, rendered_sentence_ids=context_ids)
    return LocalSurfaceRenderResult(
        renderer_mode=SELECTED_CHUNK_ANSWER_SENTENCE_ORACLE,
        context_node_ids=context_ids,
        context_nodes=context_nodes,
        rendered_sentence_ids=context_ids,
        selected_sentence_ids=tuple(),
        context_tokens=context_tokens,
        diagnostics={
            "renderer_mode": SELECTED_CHUNK_ANSWER_SENTENCE_ORACLE,
            "oracle_renderer": True,
            "uses_answer_string": True,
            "uses_gold_label": False,
            "context_tokens": context_tokens,
            **support,
        },
    )


def render_gold_sentence_oracle(
    *,
    example: QueryExample,
    graph: LocalSurfaceGraph,
    max_context_tokens: int = 512,
) -> LocalSurfaceRenderResult:
    target_ids = _window_sentence_ids(graph, _gold_sentence_ids(example, graph), window=1)
    context_ids, context_nodes, context_tokens = _materialize(graph, target_ids, max_context_tokens=max_context_tokens)
    support = _support_metrics(example=example, graph=graph, rendered_sentence_ids=context_ids)
    return LocalSurfaceRenderResult(
        renderer_mode=SELECTED_CHUNK_GOLD_SENTENCE_ORACLE,
        context_node_ids=context_ids,
        context_nodes=context_nodes,
        rendered_sentence_ids=context_ids,
        selected_sentence_ids=tuple(),
        context_tokens=context_tokens,
        diagnostics={
            "renderer_mode": SELECTED_CHUNK_GOLD_SENTENCE_ORACLE,
            "oracle_renderer": True,
            "uses_answer_string": False,
            "uses_gold_label": True,
            "context_tokens": context_tokens,
            **support,
        },
    )


def render_local_surface(
    *,
    example: QueryExample,
    graph: LocalSurfaceGraph,
    renderer_mode: str,
    medoids: LocalMedoidResult | None = None,
    medoid_config: LocalMedoidConfig | None = None,
    max_context_tokens: int = 512,
) -> LocalSurfaceRenderResult:
    if renderer_mode == LOCAL_SENTENCE_MEDOID:
        return render_local_sentence_medoid(
            example=example,
            graph=graph,
            medoids=medoids,
            medoid_config=medoid_config,
            max_context_tokens=max_context_tokens,
        )
    if renderer_mode == SELECTED_CHUNK_ANSWER_SENTENCE_ORACLE:
        return render_answer_sentence_oracle(example=example, graph=graph, max_context_tokens=max_context_tokens)
    if renderer_mode == SELECTED_CHUNK_GOLD_SENTENCE_ORACLE:
        return render_gold_sentence_oracle(example=example, graph=graph, max_context_tokens=max_context_tokens)
    raise ValueError(f"Unknown local surface renderer: {renderer_mode}")


__all__ = [
    "FACT_MEDIATED_SENTENCE",
    "LOCAL_SENTENCE_MEDOID",
    "ORACLE_RENDERERS",
    "SELECTED_CHUNK_ANSWER_SENTENCE_ORACLE",
    "SELECTED_CHUNK_GOLD_SENTENCE_ORACLE",
    "LocalSurfaceRenderResult",
    "render_local_sentence_medoid",
    "render_local_surface",
]
