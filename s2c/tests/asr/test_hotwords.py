
from s2c.hotwords import tokens_from_text

def test_tokens_from_text_extracts_identifiers():
    code = "def word_count(s: str, min_count=2):\n    return {}"
    toks = tokens_from_text(code)
    assert "word_count" in toks
    assert "min_count" in toks
