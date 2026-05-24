from pamae_rag.objective.query_grounding import (
    entity_title_grounding_score,
    extract_query_title_spans,
    normalize_text,
    title_overlap_score,
)


def test_query_title_span_extraction():
    spans = extract_query_title_spans("Who directed 'The Matrix' and what is George Rankin's party?")

    assert "The Matrix" in spans
    assert "George Rankin" in spans
    assert "Who" not in spans
    assert normalize_text("George Rankin's") == "george rankin s"


def test_entity_title_grounding_score_prefers_query_entity_title():
    query = "Who was George Rankin?"

    assert entity_title_grounding_score(query, "George Rankin") > entity_title_grounding_score(query, "Rankin")
    assert title_overlap_score(query, "George Rankin") > 0
