from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable

from pamae_rag.sentence_graph.graph_variants import ENTITY_SENTENCE_CHUNK_HIER
from pamae_rag.sentence_graph.sentence_graph_builder import SentenceGraphIndex
from pamae_rag.sentence_graph.sentence_metric_distance import shortest_path_node_ids
from pamae_rag.sentence_graph.sentence_retriever import SentenceRetrievalResult

SENTENCE_ONLY = "sentence_only"
SENTENCE_PATH = "sentence_path"
SENTENCE_PARENT_TITLE = "sentence_parent_title"
SENTENCE_LOCAL_WINDOW = "sentence_local_window"
SENTENCE_PARENT_CHUNK = "sentence_parent_chunk"

DIAGNOSTIC_RENDERERS = {SENTENCE_LOCAL_WINDOW, SENTENCE_PARENT_CHUNK}
SUPPORTED_RENDERERS = {
    SENTENCE_ONLY,
    SENTENCE_PATH,
    SENTENCE_PARENT_TITLE,
    SENTENCE_LOCAL_WINDOW,
    SENTENCE_PARENT_CHUNK,
}


@dataclass(frozen=True)
class SentenceRenderResult:
    renderer_mode: str
    context_node_ids: tuple[str, ...]
    context_nodes: tuple[dict[str, Any], ...]
    rendered_sentence_ids: tuple[str, ...]
    path_sentence_ids: tuple[str, ...]
    context_text: str
    context_tokens: int
    diagnostic_only: bool
    diagnostics: dict[str, Any]


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))


def _sentence_nodes_on_paths(
    index: SentenceGraphIndex,
    retrieval: SentenceRetrievalResult,
    *,
    use_chunk_parent_edges_in_metric: bool,
) -> tuple[str, ...]:
    sentence_ids = set(index.sentence_ids)
    path_sentence_ids: list[str] = []
    for anchor_entity_id in retrieval.query_anchor_entity_ids:
        for selected_id in retrieval.selected_sentence_ids:
            path = shortest_path_node_ids(
                index,
                anchor_entity_id,
                selected_id,
                use_chunk_parent_edges_in_metric=use_chunk_parent_edges_in_metric,
            )
            path_sentence_ids.extend(node_id for node_id in path if node_id in sentence_ids)
    for left, right in combinations(retrieval.selected_sentence_ids, 2):
        path = shortest_path_node_ids(
            index,
            left,
            right,
            use_chunk_parent_edges_in_metric=use_chunk_parent_edges_in_metric,
        )
        path_sentence_ids.extend(node_id for node_id in path if node_id in sentence_ids)
    return _ordered_unique(path_sentence_ids)


def _local_window_sentence_ids(
    index: SentenceGraphIndex,
    base_sentence_ids: Iterable[str],
    *,
    sentence_window: int,
) -> tuple[str, ...]:
    if sentence_window < 0:
        raise ValueError("sentence_window must be nonnegative")
    sentence_by_id = index.sentence_by_id
    by_chunk = index.sentences_by_chunk
    out: list[str] = []
    for sentence_id in base_sentence_ids:
        sentence = sentence_by_id.get(sentence_id)
        if sentence is None:
            continue
        chunk_sentences = by_chunk.get(sentence.chunk_id, tuple())
        positions = {item.sentence_id: pos for pos, item in enumerate(chunk_sentences)}
        pos = positions.get(sentence_id)
        if pos is None:
            continue
        start = max(0, pos - sentence_window)
        end = min(len(chunk_sentences), pos + sentence_window + 1)
        out.extend(item.sentence_id for item in chunk_sentences[start:end])
    return _ordered_unique(out)


def _sentence_context_node(
    index: SentenceGraphIndex,
    sentence_id: str,
    *,
    include_parent_title: bool,
) -> dict[str, Any] | None:
    sentence = index.sentence_by_id.get(sentence_id)
    if sentence is None:
        return None
    text = sentence.sentence_text
    if include_parent_title and sentence.title:
        text = f"{sentence.title}: {sentence.sentence_text}"
    return sentence.to_context_node(text=text)


def _add_with_budget(
    context_nodes: list[dict[str, Any]],
    node: dict[str, Any],
    *,
    seen: set[str],
    max_context_tokens: int,
    max_context_nodes: int | None,
) -> None:
    node_id = str(node.get("node_id"))
    if node_id in seen:
        return
    token_count = max(1, int(node.get("token_count") or len(str(node.get("text") or "").split())))
    used = sum(max(1, int(item.get("token_count") or 1)) for item in context_nodes)
    if max_context_nodes is not None and max_context_nodes > 0 and len(context_nodes) >= max_context_nodes:
        return
    if used + token_count > max_context_tokens:
        return
    item = dict(node)
    item["token_count"] = token_count
    context_nodes.append(item)
    seen.add(node_id)


def render_sentence_context(
    index: SentenceGraphIndex,
    retrieval: SentenceRetrievalResult,
    *,
    renderer_mode: str,
    max_context_tokens: int,
    max_context_nodes: int | None = None,
    sentence_window: int = 1,
    use_chunk_parent_edges_in_metric: bool = False,
) -> SentenceRenderResult:
    if renderer_mode not in SUPPORTED_RENDERERS:
        raise ValueError(f"Unknown sentence renderer: {renderer_mode}")
    if renderer_mode == SENTENCE_PARENT_TITLE and index.graph_variant != ENTITY_SENTENCE_CHUNK_HIER:
        raise ValueError("sentence_parent_title requires entity_sentence_chunk_hier graph metadata")

    selected = retrieval.selected_sentence_ids
    path_sentence_ids: tuple[str, ...] = tuple()
    if renderer_mode in {
        SENTENCE_PATH,
        SENTENCE_PARENT_TITLE,
        SENTENCE_LOCAL_WINDOW,
        SENTENCE_PARENT_CHUNK,
    }:
        path_sentence_ids = _sentence_nodes_on_paths(
            index,
            retrieval,
            use_chunk_parent_edges_in_metric=use_chunk_parent_edges_in_metric,
        )

    base_sentence_ids = _ordered_unique([*selected, *path_sentence_ids])
    rendered_sentence_ids = base_sentence_ids
    if renderer_mode == SENTENCE_LOCAL_WINDOW:
        rendered_sentence_ids = _ordered_unique(
            [*base_sentence_ids, *_local_window_sentence_ids(index, base_sentence_ids, sentence_window=sentence_window)]
        )

    context_nodes: list[dict[str, Any]] = []
    seen: set[str] = set()
    if renderer_mode == SENTENCE_PARENT_CHUNK:
        chunk_ids = _ordered_unique(
            index.sentence_by_id[sentence_id].chunk_id
            for sentence_id in base_sentence_ids
            if sentence_id in index.sentence_by_id
        )
        for chunk_id in chunk_ids:
            chunk = index.chunk_by_id.get(chunk_id)
            if chunk is None:
                continue
            _add_with_budget(
                context_nodes,
                chunk.to_context_node(),
                seen=seen,
                max_context_tokens=max_context_tokens,
                max_context_nodes=max_context_nodes,
            )
        by_chunk = index.sentences_by_chunk
        rendered_sentence_ids = _ordered_unique(
            sentence.sentence_id
            for chunk_id in chunk_ids
            for sentence in by_chunk.get(chunk_id, tuple())
        )
    else:
        include_title = renderer_mode == SENTENCE_PARENT_TITLE
        for sentence_id in rendered_sentence_ids:
            node = _sentence_context_node(index, sentence_id, include_parent_title=include_title)
            if node is None:
                continue
            _add_with_budget(
                context_nodes,
                node,
                seen=seen,
                max_context_tokens=max_context_tokens,
                max_context_nodes=max_context_nodes,
            )

    context_text = "\n\n".join(str(node.get("text") or "") for node in context_nodes)
    context_tokens = int(sum(max(1, int(node.get("token_count") or 1)) for node in context_nodes))
    diagnostics = {
        "renderer_mode": renderer_mode,
        "diagnostic_only": renderer_mode in DIAGNOSTIC_RENDERERS,
        "path_sentence_count": len(path_sentence_ids),
        "rendered_sentence_count": len(rendered_sentence_ids),
        "context_node_count": len(context_nodes),
        "context_tokens": context_tokens,
        "sentence_window": sentence_window if renderer_mode == SENTENCE_LOCAL_WINDOW else None,
        "renders_full_parent_chunk": renderer_mode == SENTENCE_PARENT_CHUNK,
        "renders_parent_title_metadata": renderer_mode == SENTENCE_PARENT_TITLE,
    }
    return SentenceRenderResult(
        renderer_mode=renderer_mode,
        context_node_ids=tuple(str(node["node_id"]) for node in context_nodes),
        context_nodes=tuple(context_nodes),
        rendered_sentence_ids=rendered_sentence_ids,
        path_sentence_ids=path_sentence_ids,
        context_text=context_text,
        context_tokens=context_tokens,
        diagnostic_only=renderer_mode in DIAGNOSTIC_RENDERERS,
        diagnostics=diagnostics,
    )
