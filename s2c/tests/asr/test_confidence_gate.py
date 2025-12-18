from s2c.intent import parse_and_rank

def test_confidence_output_exists():
    nbest = ["rename counts to total_counts in function word_count",
             "random words that likely do not parse"]
    parsed, diag = parse_and_rank(nbest, {"word_count"})
    assert "confidence" in diag
    assert 0.0 <= diag["confidence"] <= 1.0
