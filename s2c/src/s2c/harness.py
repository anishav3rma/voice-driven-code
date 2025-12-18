
from __future__ import annotations
import json, time, re
from pathlib import Path
from typing import Dict, List, Any
from . import verify
from . import codegen

# Minimal task specs (add more over time)
TASKS: List[Dict[str, Any]] = [
    {
        "name": "reverse_string",
        "signature": "def reverse_string(s: str) -> str:",
        "tests": ["tests/tasks/test_reverse_string.py"],
        "prompt": "Write reverse_string(s) that returns the reverse of s."
    },
    {
        "name": "is_palindrome",
        "signature": "def is_palindrome(s: str) -> bool:",
        "tests": ["tests/tasks/test_is_palindrome.py"],
        "prompt": "Return True if s is a palindrome, ignoring punctuation and case."
    },
    {
        "name": "word_count",
        "signature": "def word_count(s: str) -> dict[str,int]:",
        "tests": ["tests/tasks/test_word_count_gen.py"],
        "prompt": "Map lowercase tokens to counts; punctuation should be ignored."
    },
]

USER_SOLUTION = Path(__file__).with_name("user_solution.py")

DEFAULT_STUBS = """\
# Auto-generated stubs (will be replaced by harness)
def reverse_string(s: str) -> str:
    return s

def is_palindrome(s: str) -> bool:
    return False

def word_count(s: str) -> dict[str,int]:
    return {}
"""

def _func_name_from_sig(sig: str) -> str:
    return sig.split("def",1)[1].split("(",1)[0].strip().rstrip(":")

# def _extract_top_level_function(src: str, func_name: str) -> str:
#     """
#     Return the full text of the top-level function `func_name` (including decorators),
#     or "" if not found. Assumes module-level def, not nested defs.
#     """
#     pat = re.compile(
#         rf"(?ms)^"                        # start of a line
#         rf"(?:@[^\n]+\n)*"               # optional decorators
#         rf"def\s+{re.escape(func_name)}\s*\(.*?\):\s*\n"   # def header (multiline signature ok)
#         rf"(?:^[ \t].*\n|^\s*\n)*"       # body: indented or blank lines
#     )
#     m = pat.search(src)
#     return m.group(0) if m else ""

def _upsert_function(func_src: str, func_name: str) -> None:
    text = USER_SOLUTION.read_text() if USER_SOLUTION.exists() else DEFAULT_STUBS
    # pat = re.compile(rf"(def\s+{re.escape(func_name)}\s*\(.*?\):\n)(?:[ \t].*\n)*", re.S)
    pat = re.compile(rf"(def\s+{re.escape(func_name)}\s*\(.*?\)\s*(->\s*.*?)?:\n)(?:[ \t].*\n)*", re.S)
    if re.search(pat, text):
        text = re.sub(pat, func_src if func_src.endswith("\n") else func_src + "\n", text)
    else:
        if not text.endswith("\n"): text += "\n"
        text += "\n" + (func_src if func_src.endswith("\n") else func_src + "\n")
    USER_SOLUTION.write_text(text)

# def _upsert_function(func_src: str, func_name: str) -> None:
#     text = USER_SOLUTION.read_text() if USER_SOLUTION.exists() else DEFAULT_STUBS

#     # robust match for an existing top-level function (decorators + body)
#     pat = re.compile(
#         rf"(?ms)^"
#         rf"(?:@[^\n]+\n)*"
#         rf"def\s+{re.escape(func_name)}\s*\(.*?\):\s*\n"
#         rf"(?:^[ \t].*\n|^\s*\n)*"
#     )

#     new_block = func_src if func_src.endswith("\n") else func_src + "\n"

#     # If we find one, replace it; if not, append (after stripping any stale duplicates).
#     text, n = re.subn(pat, new_block, text, count=1)
#     if n == 0:
#         # purge any stray earlier copies just in case
#         text = re.sub(pat, "", text)
#         if not text.endswith("\n"):
#             text += "\n"
#         text += "\n" + new_block

#     _write_candidate._write_candidate(text)

# def _write_candidate(code: str) -> None:
#     USER_SOLUTION.write_text(code)

# def eval_task(spec: Dict[str, Any], k: int = 3, timeout_s: int = 5, strategy: str = "naive") -> Dict[str, Any]:
#     cands = codegen.generate(spec, k=k, strategy=strategy)
#     attempt_results = []
#     passed = False
#     for i, src in enumerate(cands, 1):
#         #_write_candidate(src)
#         func_name = _func_name_from_sig(spec["signature"])

#         # If strategy returns a full module, extract just the target function.
#         func_text = _extract_top_level_function(src, func_name) if strategy == "template" else src
#         if not func_text:  # fallback if generator already returned a single-function snippet
#             func_text = src

#         _upsert_function(src, func_name)
        
#         res = verify.run_tests(spec["tests"], timeout_s=timeout_s)
#         attempt_results.append({"attempt": i, "pass_frac": res["pass_frac"], "returncode": res["returncode"]})
#         if res["returncode"] == 0 and res["pass_frac"] == 1.0:
#             passed = True
#             break
#     return {"name": spec["name"], "passed": passed, "attempts": attempt_results, "k_used": len(cands)}

def eval_task(spec: Dict[str, Any], k: int = 3, timeout_s: int = 5, strategy: str = "naive") -> Dict[str, Any]:
    cands = codegen.generate(spec, k=k, strategy=strategy)
    attempt_results = []
    passed = False
    for i, src in enumerate(cands, 1):
        func_name = _func_name_from_sig(spec["signature"])
        _upsert_function(src, func_name)
        res = verify.run_tests(spec["tests"], timeout_s=timeout_s)
        attempt_results.append({"attempt": i, "pass_frac": res["pass_frac"], "returncode": res["returncode"]})
        if res["returncode"] == 0 and res["pass_frac"] == 1.0:
            passed = True
            break
    return {"name": spec["name"], "passed": passed, "attempts": attempt_results, "k_used": len(cands)}

def eval_all(k: int = 3, timeout_s: int = 5, strategy: str = "naive") -> Dict[str, Any]:
    per = [eval_task(t, k=k, timeout_s=timeout_s, strategy=strategy) for t in TASKS]
    pass_count = sum(1 for r in per if r["passed"])
    return {"pass_at_k": pass_count / len(per), "results": per}

if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=5)
    ap.add_argument("--strategy", type=str, default="naive", choices=["naive","template"])
    # ap.add_argument("--reset", action="store_true")

    args = ap.parse_args()
    # if args.reset:
    #     USER_SOLUTION.write_text(DEFAULT_STUBS)

    summary = eval_all(k=args.k, timeout_s=args.timeout, strategy=args.strategy)
    print(json.dumps(summary, indent=2))
