from pamae_rag.qa.metrics import exact_match_score, qa_f1_score


def test_qa_metric_normalizes_articles_and_punctuation():
    assert exact_match_score("The Iron-Man!", "iron man") == 1.0


def test_qa_f1_uses_token_overlap():
    assert qa_f1_score("born in Paris France", "Paris") == 0.4
