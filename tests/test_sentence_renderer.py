import numpy as np

from pamae_rag.data.schema import EvidenceNode
from pamae_rag.sentence_graph.graph_variants import ENTITY_SENTENCE_CHUNK_HIER
from pamae_rag.sentence_graph.sentence_graph_builder import build_sentence_graph_index
from pamae_rag.sentence_graph.sentence_renderer import (
    SENTENCE_LOCAL_WINDOW,
    SENTENCE_PARENT_CHUNK,
    SENTENCE_PARENT_TITLE,
    SENTENCE_PATH,
    render_sentence_context,
)
from pamae_rag.sentence_graph.sentence_retriever import (
    SentenceRetrieverConfig,
    retrieve_sentence_medoids,
)


def _node(node_id: str, text: str, title: str) -> EvidenceNode:
    return EvidenceNode(
        node_id=node_id,
        text=text,
        embedding=np.array([1.0, 0.0]),
        relevance=1.0,
        token_count=max(1, len(text.split())),
        metadata={"title": title, "corpus_index": node_id},
    )


def _retrieval():
    index = build_sentence_graph_index(
        [
            _node(
                "n1",
                "Ada Lovelace met Charles Babbage. Charles Babbage built engines. London hosted Ada.",
                "Ada",
            )
        ],
        graph_variant=ENTITY_SENTENCE_CHUNK_HIER,
    )
    result = retrieve_sentence_medoids(
        index,
        "What did Ada Lovelace discuss with Charles Babbage?",
        config=SentenceRetrieverConfig(k=1, num_samples=2, sample_size_per_k=3),
        seed=5,
    )
    return index, result


def test_sentence_path_renderer_outputs_sentence_text():
    index, result = _retrieval()

    rendered = render_sentence_context(
        index,
        result,
        renderer_mode=SENTENCE_PATH,
        max_context_tokens=128,
    )

    assert rendered.context_nodes
    assert "Charles Babbage" in rendered.context_text
    assert all(node["metadata"]["context_unit"] == "sentence" for node in rendered.context_nodes)


def test_parent_title_renderer_does_not_output_full_chunk():
    index, result = _retrieval()

    rendered = render_sentence_context(
        index,
        result,
        renderer_mode=SENTENCE_PARENT_TITLE,
        max_context_tokens=128,
    )

    assert rendered.context_nodes
    assert "Ada:" in rendered.context_text
    assert "London hosted Ada" not in rendered.context_text
    assert rendered.diagnostics["renders_parent_title_metadata"] is True


def test_parent_chunk_renderer_is_marked_diagnostic_only():
    index, result = _retrieval()

    rendered = render_sentence_context(
        index,
        result,
        renderer_mode=SENTENCE_PARENT_CHUNK,
        max_context_tokens=128,
    )

    assert rendered.diagnostic_only is True
    assert rendered.diagnostics["renders_full_parent_chunk"] is True
    assert "London hosted Ada" in rendered.context_text


def test_local_window_renderer_is_diagnostic_only():
    index, result = _retrieval()

    rendered = render_sentence_context(
        index,
        result,
        renderer_mode=SENTENCE_LOCAL_WINDOW,
        max_context_tokens=128,
        sentence_window=1,
    )

    assert rendered.diagnostic_only is True
    assert rendered.diagnostics["sentence_window"] == 1
