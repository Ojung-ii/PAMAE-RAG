from __future__ import annotations

ENTITY_SENTENCE = "entity_sentence"
ENTITY_SENTENCE_CHUNK_HIER = "entity_sentence_chunk_hier"

SUPPORTED_SENTENCE_GRAPH_VARIANTS = {
    ENTITY_SENTENCE,
    ENTITY_SENTENCE_CHUNK_HIER,
}


def validate_sentence_graph_variant(value: str) -> str:
    variant = str(value)
    if variant not in SUPPORTED_SENTENCE_GRAPH_VARIANTS:
        raise ValueError(
            f"Unsupported sentence graph variant {variant!r}; "
            f"expected one of {sorted(SUPPORTED_SENTENCE_GRAPH_VARIANTS)}"
        )
    return variant
