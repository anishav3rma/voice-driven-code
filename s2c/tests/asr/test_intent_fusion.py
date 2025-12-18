
from s2c.intent import parse_and_rank

def test_parse_and_rank_prefers_parse_and_hotwords():
    nbest = [
        "please add parameter min count default two to word count",
        "add parameter min count default two to word count",
        "rename counts to total_counts in function word_count",
    ]
    hot = {"word_count", "min_count"}
    parsed, diag = parse_and_rank(nbest, hot)
    assert parsed is not None
    assert parsed["op"] == "add_param"
    assert parsed["function"] == "word_count"
    assert parsed["name"] == "min_count"
    assert parsed.get("default") in (2, "2")
