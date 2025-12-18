

def reverse_string(s: str) -> str:
    return s[::-1]

def is_palindrome(s: str) -> bool:
    import re
    t = ''.join(ch.lower() for ch in s if ch.isalnum())
    return t == t[::-1]

def word_count(s: str) -> dict[str,int]:
    import re
    from collections import Counter
    tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9']+", s)]
    counts = dict(Counter(tokens))
    return counts
