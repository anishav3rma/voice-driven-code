from __future__ import annotations
import argparse, json, difflib
from pathlib import Path
from typing import Optional

from .hotwords import extract_from_paths
from .intent import parse_and_rank, clarify_suggestion
from .asr import decode_mock
from .verify import run_tests

from .edit import (
    _apply_edit, _guess_impacted_functions, _select_tests_by_functions,
    _project_root, _print_stdout_block, _massage_verify
)

DEFAULT_FILE = Path(__file__).with_name("user_solution.py")

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="s2c.asr_edit",
        description="ASR n-best -> intent fusion -> AST patch -> impacted tests, with confidence gating.",
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--transcript", type=str, help="Ground-truth utterance text to perturb for n-best (mock backend)")
    g.add_argument("--wav", type=str, help="Path to audio file (requires 'whisper' backend)")

    ap.add_argument("--backend", choices=["mock","whisper"], default="mock", help="ASR backend to use (default: mock)")
    ap.add_argument("--severity", type=float, default=0.4, help="Mock noise severity in [0,1] (default 0.4)")
    ap.add_argument("--n-best", type=int, default=5, help="How many ASR hypotheses to use (default 5)")
    ap.add_argument("--min-conf", type=float, default=0.6, help="Confidence threshold for accepting the parsed edit (default 0.6)")
    ap.add_argument("--auto-accept", dest="auto_accept", action="store_true", help="Ignore confidence gating and apply the top parsed edit")
    ap.add_argument("--file", type=str, default=str(DEFAULT_FILE), help="Target file to patch (default: s2c/user_solution.py)")
    ap.add_argument("--stdout-tail", type=int, default=80, help="How many lines of pytest output to show/save (default 80)")
    ap.add_argument("--stdout-json", choices=["full","tail","lines","none"], default="tail",
                    help="How to include stdout in JSON (default tail)")
    ap.add_argument("--print-diff", action="store_true")
    ap.add_argument("--print-stdout", action="store_true")
    ap.add_argument("--json-out", type=str, help="Write the full run result (JSON) to this path")
    ap.add_argument("--timeout", type=int, default=5)
    args = ap.parse_args(argv)

    target_path = Path(args.file)
    if not target_path.exists():
        out = {"ok": False, "error": f"Target file not found: {target_path}"}
        print(json.dumps(out, indent=2, ensure_ascii=False)); return 2

    root = _project_root()
    # hot = extract_from_paths([target_path] + list((root/"tests").rglob("*.py")))
    hot = extract_from_paths([target_path])
    print(f"Hotwords: {hot}")

    if args.backend == "mock":
        if not args.transcript:
            out = {"ok": False, "error": "Provide --transcript when using --backend mock"}
            print(json.dumps(out, indent=2, ensure_ascii=False)); return 2
        hyp = decode_mock(args.transcript, n_best=args.n_best, hotwords=hot, severity=args.severity)
    else:
        if not args.wav:
            out = {"ok": False, "error": "Provide --wav when using --backend whisper"}
            print(json.dumps(out, indent=2, ensure_ascii=False)); return 2
        try:
            from .asr_whisper import decode_whisper
            hyp = decode_whisper(args.wav, n_best=args.n_best, hotwords=hot)
        except Exception as e:
            out = {"ok": False, "error": f"Whisper backend error: {e.__class__.__name__}: {e}"}
            print(json.dumps(out, indent=2, ensure_ascii=False)); return 2

    nbest = hyp["nbest"]
    parsed, diag = parse_and_rank(nbest, hot)
    conf = diag.get("confidence", 0.0)

    if not parsed:
        out = {"ok": False, "nbest": nbest, "diagnostics": diag, "action": "ask_back",
           "suggested": clarify_suggestion(nbest, diag)}
        print(json.dumps(out, indent=2, ensure_ascii=False)); return 3
    if (conf < args.min_conf) and not args.auto_accept:
        out = {"ok": False, "nbest": nbest, "diagnostics": diag, "action": "ask_back",
               "suggested": clarify_suggestion(nbest, diag), "confidence": conf}
        print(json.dumps(out, indent=2, ensure_ascii=False)); return 3

    before = target_path.read_text()
    after = _apply_edit(before, parsed)
    changed = (after != before)

    if args.print_diff:
        diff = "".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                            fromfile=str(target_path), tofile=str(target_path)))
        print(diff if diff.strip() else "# (no change)")

    if changed:
        target_path.write_text(after)

    impacted_funcs = _guess_impacted_functions(parsed, before)
    tests = _select_tests_by_functions(root, impacted_funcs) or ["tests"]

    res = run_tests(tests, timeout_s=args.timeout)
    raw_stdout = res.get("stdout", "")
    res = _massage_verify(res, args.stdout_json, args.stdout_tail)

    out = {
        "ok": True,
        "backend": args.backend,
        "nbest": nbest,
        "chosen": diag.get("chosen"),
        "confidence": conf,
        "parsed_edit": parsed,
        "changed": changed,
        "selected_tests": tests,
        "verify": res
    }

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.print_stdout:
        _print_stdout_block("PYTEST OUTPUT", raw_stdout, args.stdout_tail)
    return 0 if res.get("returncode", 1) == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
