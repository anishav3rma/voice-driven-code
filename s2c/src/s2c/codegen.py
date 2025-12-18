
from __future__ import annotations
from typing import List, Dict

def generate(spec: Dict, k: int = 3, strategy: str = "naive") -> List[str]:
    """
    Return up to k candidate Python source strings that implement the task.
    This is a stub baseline you can swap with an LLM later.
    strategies:
      - "naive": intentionally returns 1-2 wrong attempts before a correct one
      - "template": returns a correct template immediately for known tasks
    """
    name = spec.get("name", "")
    sig = spec.get("signature", "")
    out: List[str] = []

    def _reverse_string_wrong() -> str:
        return f"{sig}\n    # WRONG on purpose: returns original string\n    return s\n"

    def _reverse_string_right() -> str:
        return f"{sig}\n    return s[::-1]\n"

    def _is_palindrome_wrong() -> str:
        return f"""{sig}\n    # WRONG: case-sensitive, no punctuation handling\n    t = ''.join(ch for ch in s if ch.isalnum())\n    return t == t[::-1]\n"""

    def _is_palindrome_right() -> str:
        return f"""{sig}\n    import re\n    t = ''.join(ch.lower() for ch in s if ch.isalnum())\n    return t == t[::-1]\n"""

    def _word_count_right() -> str:
        return f"""{sig}\n    import re\n    from collections import Counter\n    tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9']+", s)]\n    counts = dict(Counter(tokens))\n    return counts\n"""

    if strategy == "template":
        if name == "reverse_string":
            out.append(_reverse_string_right())
        elif name == "is_palindrome":
            out.append(_is_palindrome_right())
        elif name == "word_count":
            out.append(_word_count_right())
        return out[:k]

    # default: naive
    if name == "reverse_string":
        out.extend([_reverse_string_wrong(), _reverse_string_right()])
    elif name == "is_palindrome":
        out.extend([_is_palindrome_wrong(), _is_palindrome_right()])
    elif name == "word_count":
        out.extend([_word_count_right()])  # make it easy
    else:
        out.append(f"{sig}\n    raise NotImplementedError\n")

    return out[:k]
