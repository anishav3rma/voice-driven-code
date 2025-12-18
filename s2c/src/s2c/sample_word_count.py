
import re
from collections import Counter
from typing import Dict

WORD_RE = re.compile(r"[A-Za-z0-9']+")

def word_count(text: str) -> Dict[str, int]:
    """Return lowercase word -> frequency map, ignoring punctuation.
    This implementation is intentionally straightforward for demo/tests.
    """
    tokens = [t.lower() for t in WORD_RE.findall(text)]
    return dict(Counter(tokens))
