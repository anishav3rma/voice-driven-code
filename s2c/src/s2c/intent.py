
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Set, Optional
import re
from .edits.grammar import parse_edit
import math
from dataclasses import dataclass
import string

# _NUM_WORDS = {
#     "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
#     "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
# }

# def _canon_numbers(s: str) -> str:
#     # map number words → digits (safe, domain-specific)
#     def rep(m): return NUM_WORDS[m.group(0)]
#     return re.sub(r"\b(" + "|".join(NUM_WORDS.keys()) + r")\b", rep, s.lower())

# def _repair_default_two(s: str) -> str:
#     # only try this if the utterance looks like add_param and parser later complains
#     # replace homophone near 'default' with the digit 2
#     return re.sub(r"\bdefault\s+(to|too)\b", "default 2", s)

# def _numberize(tok: str) -> str:
#     t = tok.lower().strip()
#     return str(_NUM_WORDS.get(t, tok))

# def _canonicalize_underscores(s: str) -> str:
#     x = s.strip().replace("’", "'")
#     x = re.sub(r"[“”]", '"', x)
#     x = re.sub(r"\s+", " ", x)
#     def join_after(keyword: str, text: str) -> str:
#         pat = re.compile(rf"({keyword}\s+)([a-z]+)\s+([a-z]+)")
#         while True:
#             new = pat.sub(lambda m: f"{m.group(1)}{m.group(2)}_{m.group(3)}", text)
#             if new == text:
#                 return new
#             text = new
#     y = x.lower()
#     for kw in ["parameter", "param", "arg", "function", "func"]:
#         y = join_after(kw, y)
#     y = re.sub(r"\bto\s+([a-z]+)\s+([a-z]+)\b", lambda m: f"to {m.group(1)}_{m.group(2)}", y)
#     y = re.sub(r"(default\s+)([a-z]+)\b", lambda m: f"{m.group(1)}{_numberize(m.group(2))}", y)
#     y = re.sub(r"\bequals\s+([a-z]+|\d+)\b", lambda m: f"= {_numberize(m.group(1))}", y)
#     y = re.sub(r"\bparam\b", "parameter", y)
#     y = re.sub(r"\s*=\s*([0-9]+|['\"][^'\"]*['\"])\b", r" default \1", y)
#     y = re.sub(r"\blines?\s+(\d+)\s*(?:to|through|-)\s*(\d+)\b", r"lines \1 to \2", y)
#     return y

def _op_hint(s: str) -> str:
    s_lo = s.lower()
    if "rename" in s_lo: return "rename"
    if "add param" in s_lo or "add parameter" in s_lo: return "add_param"
    if "wrap" in s_lo and ("try" in s_lo or "except" in s_lo): return "wrap_try_except"
    return "unknown"

# ------------------------- always-on base cleanup -------------------------

# def _base_normalize(s: str) -> str:
#     # straight quotes, collapse whitespace, lowercase
#     x = s.strip().replace("’", "'")
#     x = re.sub(r"[“”]", '"', x)
#     x = re.sub(r"\s+", " ", x)
#     return x.lower()

def _base_normalize(s: str) -> str:
    # 1. Strip leading/trailing whitespace initially
    x = s.strip().replace("’", "'")
    x = re.sub(r"[“”]", '"', x)
    
    # 2. Remove leading and trailing punctuation using regex
    # The pattern r'^[{}]*|[{}]*$'.format(re.escape(string.punctuation), re.escape(string.punctuation))
    # matches any number of punctuation characters at the start (^) or end ($) of the string.
    punctuation_pattern = r'^[{}]*|[{}]*$'.format(re.escape(string.punctuation), re.escape(string.punctuation))
    x = re.sub(punctuation_pattern, '', x)

    # 3. Collapse multiple whitespace occurrences into a single space (from original function intent)
    # The original function had a typo but likely intended this behavior:
    x = re.sub(r'\s+', ' ', x)
    
    # 4. Convert the entire string to lowercase
    return x.lower()

# ------------------------- knobs & helpers -------------------------

_NUM_WORDS: Dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

@dataclass(frozen=True)
class CanonConfig:
    join_after_keyword: bool = True
    number_words: bool = True
    keyword_canon: bool = True
    homophone_repair: bool = True
    punctuation_word_join: bool = True

# Words that begin “windows” and also act as boundaries between windows
# (extend as you add more commands/slots)
_DEFAULT_KEYWORDS: Tuple[str, ...] = (
    "add", "rename", "wrap", "lines", "line",
    "parameter", "param", "argument", "arg",
    "function", "func", "in function", "in",
    "to", "default", "equals", "try", "except", "catch",
)

# ------------------------- individual passes -------------------------

import re

def pass_join_after_keyword(
    s: str,
    keywords: tuple[str, ...] = _DEFAULT_KEYWORDS,
) -> str:
    """
    Join the two tokens immediately following certain keywords with an underscore,
    but ONLY if neither of those two tokens is itself a keyword (single-word),
    and the pair isn't a multi-word keyword (e.g., 'in function').
    Treat 'to' as a keyword so 'to foo bar' -> 'to foo_bar'.
    """
    out = s

    # Normalize keyword lists
    kw_norm = tuple(k.lower() for k in keywords)
    single_kw = {k for k in kw_norm if " " not in k}            # e.g., {'param', 'parameter', 'func', 'to', ...}
    multi_kw_pairs = {tuple(k.split()) for k in kw_norm if " " in k and len(k.split()) == 2}
    # (If you have longer phrases, extend this logic accordingly.)

    # Sort by length so multi-word keywords win in the regex anchor
    kws_sorted = sorted(keywords, key=len, reverse=True)

    for kw in kws_sorted:
        # Anchor the specific keyword; then capture the next two tokens
        # Use IGNORECASE so we match regardless of case
        pat = re.compile(rf"(\b{re.escape(kw)}\s+)([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\b",
                         flags=re.IGNORECASE)

        def repl(m: re.Match) -> str:
            kprefix = m.group(1)
            w1 = m.group(2)
            w2 = m.group(3)
            w1lo, w2lo = w1.lower(), w2.lower()

            # If either following token is a keyword, don't join.
            if w1lo in single_kw or w2lo in single_kw:
                return m.group(0)

            # If the pair equals a multi-word keyword, don't join.
            if (w1lo, w2lo) in multi_kw_pairs:
                return m.group(0)

            # Safe to join.
            return f"{kprefix}{w1}_{w2}"

        # Apply repeatedly until no more changes for this keyword
        while True:
            new = pat.sub(repl, out)
            if new == out:
                break
            out = new

    return out


def pass_number_words(s: str) -> str:
    """Convert number words anywhere to digits (one→1, two→2, ...)."""
    return re.sub(
        r"\b(" + "|".join(_NUM_WORDS.keys()) + r")\b",
        lambda m: str(_NUM_WORDS[m.group(0)]),
        s,
    )

def pass_keyword_canon(s: str) -> str:
    """
    Canonicalize key synonyms and light phrasing:
      - param/argument/func -> parameter/function
      - equals X -> default X
      - lines 3 through 5 / 3-5 -> lines 3 to 5
    """
    x = s
    # synonym canon
    x = re.sub(r"\bparam\b", "parameter", x)
    x = re.sub(r"\bargument\b", "parameter", x)
    x = re.sub(r"\bfunc\b", "function", x)
    x = re.sub(r"\b(?:inside|in)\s+function\b", "in function", x)
    x = re.sub(r"\btry\s*(?:[-/]|and)?\s*catch\b", "try except", x)

    # equals -> default (allow quoted strings or numbers)
    x = re.sub(r"\s*=\s*([0-9]+|['\"][^'\"]*['\"])\b", r" default \1", x)

    # range phrasing canon
    x = re.sub(r"\blines?\s+(\d+)\s*(?:to|through|-)\s*(\d+)\b", r"lines \1 to \2", x)

    return x

def pass_homophone_repair(
    s: str,
    *,
    # map of {anchor_keyword: {homophone: replacement}}, applied in the
    # window after the keyword and before the next anchor keyword
    repair_map: Dict[str, Dict[str, str]] = None,
    anchors: Tuple[str, ...] = _DEFAULT_KEYWORDS,
) -> str:
    """
    Repair homophones only in windows that start right after a keyword and end
    before the next keyword (so we don’t over-correct globally).
    Default behavior: 'default to/too' -> 'default 2'.
    """
    if repair_map is None:
        repair_map = {"default": {"to": "2", "too": "2", "two": "2"}}

    tokens = s.split(" ")
    # Build set for quick boundary checks
    anchor_set = set(anchors)

    i = 0
    while i < len(tokens):
        tk = tokens[i]
        if tk in repair_map:
            # Walk forward until next anchor (exclusive) and repair within
            j = i + 1
            while j < len(tokens) and tokens[j] not in anchor_set:
                rep = repair_map[tk].get(tokens[j])
                if rep is not None:
                    tokens[j] = rep
                j += 1
            i = j
        else:
            i += 1
    return " ".join(tokens)

_PUNCT_SYNONYMS: Dict[str, str] = {
    "underscore": "_",
    "hyphen": "-",
    "dash": "-",
    "dot": ".",
    "period": ".",
    "point": ".",
}

# Helper: treat identifier-ish tokens (lowercase by the time we get here)
_ID = re.compile(r"^[a-z0-9_]+$")

def pass_punctuation_word_join(
    s: str,
    *,
    keywords: Tuple[str, ...] = _DEFAULT_KEYWORDS,
    mapping: Dict[str, str] = _PUNCT_SYNONYMS,
) -> str:
    """
    If transcript contains: <word> (underscore|hyphen|dot|...) <word>
    and the outer words are NOT command keywords, join them using the symbol.
    e.g., 'total underscore count' → 'total_count'
          'user hyphen id'        → 'user-id'
          'file dot name'         → 'file.name'
    Runs iteratively to handle chains like 'min underscore count underscore two'.
    """
    toks = s.split()
    kw = set(keywords)
    i = 0
    while i + 2 < len(toks):
        a, mid, b = toks[i], toks[i + 1], toks[i + 2]
        sym = mapping.get(mid)
        if (
            sym is not None
            and a not in kw and b not in kw
            and _ID.match(a) and _ID.match(b)
        ):
            # collapse the triplet into a single joined token
            toks[i:i + 3] = [f"{a}{sym}{b}"]
            # do not advance i; allow cascading joins (e.g., a _ b _ c)
            continue
        i += 1
    return " ".join(toks)

# ------------------------- main entry -------------------------

def canonicalize(text: str, cfg: CanonConfig = CanonConfig()) -> str:
    """
    Apply always-on base normalization, then optional passes according to cfg.
    Order chosen to match your prior behavior and minimize surprises.
    """
    x = _base_normalize(text)

    if cfg.keyword_canon:
        x = pass_keyword_canon(x)

    if cfg.punctuation_word_join:
        x = pass_punctuation_word_join(x)

    if cfg.join_after_keyword:
        x = pass_join_after_keyword(x)

    if cfg.number_words:
        x = pass_number_words(x)

    if cfg.homophone_repair:
        x = pass_homophone_repair(x)

    return x

def _softmax_conf(scores: list[float], idx_best: int, T: float = 0.5) -> float:
    exps = [math.exp(s / max(T, 1e-6)) for s in scores]
    Z = sum(exps) or 1.0
    return exps[idx_best] / Z

def _quality(parsed: Optional[dict]) -> float:
    if not parsed:
        return 0.0
    op = parsed.get("op")
    if op == "rename":
        ok = bool(parsed.get("target")) and bool(parsed.get("new_name"))
    elif op == "add_param":
        ok = bool(parsed.get("function")) and bool(parsed.get("name"))
    elif op == "wrap":
        ok = (parsed.get("start_line") is not None) and (parsed.get("end_line") is not None)
    else:
        ok = True
    return 1.0 if ok else 0.0

def _semantic_key(parsed: Optional[dict]) -> tuple:
    """Collapse surface variations that mean the same command."""
    if not parsed:
        return ("<unparsed>",)

    op = parsed.get("op")
    N = lambda x: (str(x).strip().lower() if isinstance(x, str) else x)

    if op == "rename":
        # function may be None for module-level renames
        return ("rename",
                N(parsed.get("function")),
                parsed.get("target_kind", "var"),
                N(parsed.get("target")),
                N(parsed.get("new_name")))

    if op == "add_param":
        return ("add_param",
                N(parsed.get("function")),
                N(parsed.get("name")),
                parsed.get("default"))  # already int|None in your parser

    if op == "wrap":
        # you currently always do try/except; if you later add wrapper type, include it here
        s = parsed.get("start_line")
        e = parsed.get("end_line")
        # normalize order just in case
        if s is not None and e is not None and e < s:
            s, e = e, s
        return ("wrap",
                N(parsed.get("function")),
                int(s) if s is not None else None,
                int(e) if e is not None else None)

    # fallback
    return (op,)

def _score(hyp: str, hotwords: set[str], parsed: Optional[dict]) -> float:
    score = 0.0
    if parsed:
        score += 3.0           # parsed bonus
        score += 0.5           # keep your existing bump
    if hotwords:
        hit = sum(1 for w in hotwords if w in hyp)
        score += min(2.0, 0.08 * hit)
    # length prior: lighter penalty
    score += max(0.0, 1.0 - 0.005 * len(hyp))
    return score

# def _score(hyp: str, hotwords: Set[str], parsed: Optional[Dict[str, Any]]) -> float:
#     score = 0.0
#     if parsed:
#         score += 3.0
#         score += 0.5
#     if hotwords:
#         hit = sum(1 for w in hotwords if w in hyp)
#         score += min(2.0, 0.08 * hit)
#     score += max(0.0, 1.0 - 0.01 * len(hyp))
#     return score

# def parse_and_rank(nbest: List[str], hotwords: Set[str], *, disable_canon: bool = False) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
#     cands = []
#     for i, hyp in enumerate(nbest):
#         norm = hyp if disable_canon else _canonicalize_underscores(hyp)
#         parsed = parse_edit(norm)
#         sc = _score(norm, hotwords, parsed)
#         cands.append({"i": i, "raw": hyp, "norm": norm, "parsed": parsed, "score": sc})
#     cands.sort(key=lambda d: (-d["score"], d["i"]))
#     best = cands[0] if cands else None
#     alt = cands[1] if len(cands) > 1 else None
#     confidence = 0.0
#     if best:
#         denom = abs(best["score"]) + 1e-6
#         gap = best["score"] - (alt["score"] if alt else 0.0)
#         confidence = max(0.0, min(1.0, gap / max(denom, 1.0)))
#     diagnostics = {"chosen": best, "candidates": cands, "confidence": confidence}
#     return (best["parsed"] if best and best["parsed"] else None, diagnostics)

def parse_and_rank(nbest: list[str], hotwords: set[str], *, disable_canon: bool = False,
                       canon_config: Optional["CanonConfig"] = None):
    cands = []
    for i, hyp in enumerate(nbest):
        norm = hyp if disable_canon else canonicalize(hyp, cfg=(canon_config or CanonConfig()))
        parsed = parse_edit(norm)
        sc = _score(norm, hotwords, parsed)
        hot_hits = sum(1 for w in hotwords if w in norm)
        cand = {
            "i": i, "raw": hyp, "norm": norm, "parsed": parsed, "score": sc,
            "debug": {
                "len": len(norm),
                "hot_hits": hot_hits,
                "canon_changed": (norm != hyp),
                "op_hint": _op_hint(norm),  # see helper below
            }
        }
        cands.append({"i": i, "raw": hyp, "norm": norm, "parsed": parsed, "score": sc})

    # Group by semantic key so tiny surface differences don’t fight each other
    buckets: dict[tuple, list[dict]] = {}
    for c in cands:
        buckets.setdefault(_semantic_key(c["parsed"]), []).append(c)

    # Score each bucket (max candidate score works well here)
    bucket_list = []
    for key, items in buckets.items():
        bscore = max(c["score"] for c in items)
        rep = max(items, key=lambda c: c["score"])  # representative
        bucket_list.append({"key": key, "items": items, "score": bscore, "rep": rep})
    bucket_list.sort(key=lambda b: -b["score"])

    # Prefer buckets whose representative actually parsed to a command
    parsed_buckets = [b for b in bucket_list if b["rep"]["parsed"]]
    if parsed_buckets:
        bucket_list = parsed_buckets  # drop purely-unparsed buckets from consideration
    else:
        # No parsed candidates at all → return None with low confidence
        diagnostics = {
            "chosen": None,
            "candidates": cands,
            "buckets": {str(b["key"]): [c["i"] for c in b["items"]] for b in bucket_list},
            "confidence": 0.0,
            "consensus": 0.0,
            "softmax": 0.0,
            "reason": "no_parsed_candidate",
            "config": {
                "disable_canon": disable_canon,
                "hotwords_count": len(hotwords),
                "nbest_size": len(nbest),
                "canon": (canon_config.__dict__ if canon_config else None),
            },
        }
        return (None, diagnostics)

    best = bucket_list[0] if bucket_list else None
    if not best:
        diagnostics = {"chosen": None, "candidates": cands, "buckets": {}, "confidence": 0.0}
        return (None, diagnostics)

    # Confidence = softmax over bucket scores + consensus + quality
    scores = [b["score"] for b in bucket_list]
    conf_soft = _softmax_conf(scores, 0, T=0.5)
    consensus = len(best["items"]) / max(1, sum(len(b["items"]) for b in bucket_list))
    quality = _quality(best["rep"]["parsed"])
    confidence = min(1.0, 0.6 * conf_soft + 0.3 * consensus + 0.1 * quality)

    # diagnostics = {
    #     "chosen": best["rep"],                 # top representative candidate
    #     "candidates": cands,                   # raw list for debugging
    #     "buckets": {str(b["key"]): [c["i"] for c in b["items"]] for b in bucket_list},
    #     "confidence": confidence,
    #     "consensus": consensus,
    #     "softmax": conf_soft,
    # }

    diagnostics = {
        "chosen": best["rep"] if best else None,
        "candidates": cands,
        "buckets": {str(b["key"]): [c["i"] for c in b["items"]] for b in bucket_list},
        "bucket_scores": {str(b["key"]): b["score"] for b in bucket_list},
        "bucket_support": {str(b["key"]): len(b["items"]) for b in bucket_list},
        "confidence": confidence,
        "consensus": consensus,
        "softmax": conf_soft,
        "config": {
            "disable_canon": disable_canon,
            "hotwords_count": len(hotwords),
            "nbest_size": len(nbest),
            "canon": (canon_config.__dict__ if canon_config else None),
        },
    }

    diagnostics["chosen_fields"] = sorted((best["rep"]["parsed"] or {}).keys()) if best else []
    return (best["rep"]["parsed"], diagnostics)

def clarify_suggestion(nbest: List[str], diagnostics: Dict[str, Any]) -> str:
    best = diagnostics.get("chosen") or {}
    norm = best.get("norm", "")
    if norm:
        return f'Did you mean: "{norm}" ?'
    if nbest:
        return f'Did you mean: "{nbest[0]}" ?'
    return "Could you rephrase the instruction?"
