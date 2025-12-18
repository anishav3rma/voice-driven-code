
from __future__ import annotations
import argparse, difflib, json
from pathlib import Path
from typing import Optional, List, Set
from .hotwords import extract_from_paths
from .intent import parse_and_rank
from .asr import decode_mock
from .verify import run_tests
from .edit import _apply_edit, _guess_impacted_functions, _select_tests_by_functions, _project_root, _print_stdout_block, _massage_verify

def _make_nbest(transcript: str, n_best: int, hotwords: Set[str], severity: float) -> List[str]:
    return decode_mock(transcript, n_best=n_best, hotwords=hotwords, severity=severity)["nbest"]

def _diff(a: str, b: str, name: str) -> str:
    return "".join(difflib.unified_diff(a.splitlines(keepends=True), b.splitlines(keepends=True), fromfile=name, tofile=name))

def _run_curses(nbest: List[str], target: Path, hot: Set[str]):
    import curses  # type: ignore
    def render(stdscr, idx: int, norm: str, parsed, conf: float, diff: str):
        stdscr.clear()
        stdscr.addstr(0, 0, "s2c.tui — n-best review  (↑/↓ select, Enter accept, q quit)")
        stdscr.addstr(1, 0, f"Confidence: {conf:.2f}   Parsed: {bool(parsed)}")
        stdscr.addstr(2, 0, f"Chosen (normalized): {norm}")
        stdscr.addstr(4, 0, "N-BEST:")
        for i, h in enumerate(nbest[:10]):
            mark = "→" if i == idx else " "
            stdscr.addstr(5+i, 0, f"{mark} [{i}] {h[:120]}")
        stdscr.addstr(16, 0, "DIFF PREVIEW (first 30 lines):")
        for j, line in enumerate(diff.splitlines()[:30], start=17):
            try: stdscr.addstr(j, 0, line[:200])
            except Exception: pass
        stdscr.refresh()

    def main(stdscr):
        idx = 0
        parsed, diag = parse_and_rank([nbest[idx]], hot)
        norm = (diag.get("chosen") or {}).get("norm", nbest[idx])
        conf = diag.get("confidence", 0.0)
        before = target.read_text()
        after = _apply_edit(before, parsed) if parsed else before
        diff = _diff(before, after, str(target))
        render(stdscr, idx, norm, parsed, conf, diff)

        while True:
            ch = stdscr.getch()
            if ch in (ord('q'), 27): return None, ""
            if ch in (curses.KEY_DOWN, ord('j')): idx = min(idx+1, len(nbest)-1)
            elif ch in (curses.KEY_UP, ord('k')): idx = max(idx-1, 0)
            elif ch in (10, 13): return parsed, diff  # Enter

            parsed, diag = parse_and_rank([nbest[idx]], hot)
            norm = (diag.get("chosen") or {}).get("norm", nbest[idx])
            conf = diag.get("confidence", 0.0)
            after = _apply_edit(before, parsed) if parsed else before
            diff = _diff(before, after, str(target))
            render(stdscr, idx, norm, parsed, conf, diff)

    import curses  # type: ignore
    return curses.wrapper(main)

def _run_simple(nbest: List[str], target: Path, hot: Set[str]):
    print("\nN-BEST hypotheses:")
    for i, h in enumerate(nbest):
        print(f"  [{i}] {h}")
    try:
        sel = int(input("Pick index to accept (or -1 to cancel): "))
    except Exception:
        return None, ""
    if sel < 0 or sel >= len(nbest): return None, ""
    parsed, diag = parse_and_rank([nbest[sel]], hot)
    before = target.read_text()
    after = _apply_edit(before, parsed) if parsed else before
    return parsed, _diff(before, after, str(target))

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="s2c.tui", description="Interactive n-best review for ASR → intent → edit.")
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--file", type=str, default=str(Path(__file__).with_name("user_solution.py")))
    ap.add_argument("--n-best", type=int, default=5)
    ap.add_argument("--severity", type=float, default=0.5)
    ap.add_argument("--timeout", type=int, default=5)
    ap.add_argument("--print-stdout", action="store_true")
    ap.add_argument("--stdout-json", choices=["full","tail","lines","none"], default="tail")
    ap.add_argument("--stdout-tail", type=int, default=80)
    args = ap.parse_args(argv)

    target = Path(args.file)
    if not target.exists():
        print(json.dumps({"ok": False, "error": f"Target not found: {target}"}, indent=2)); return 2

    root = _project_root()
    hot = extract_from_paths([target] + list((root/"tests").rglob("*.py")))
    nbest = decode_mock(args.transcript, n_best=args.n_best, hotwords=hot, severity=args.severity)["nbest"]

    try:
        parsed, diff = _run_curses(nbest, target, hot)
    except Exception:
        parsed, diff = _run_simple(nbest, target, hot)

    if not parsed:
        print(json.dumps({"ok": False, "action": "cancelled"}, indent=2)); return 3

    before = target.read_text()
    after = _apply_edit(before, parsed)
    changed = (after != before)
    if changed: target.write_text(after)

    impacted_funcs = _guess_impacted_functions(parsed, before)
    tests = _select_tests_by_functions(_project_root(), impacted_funcs) or ["tests"]
    res = run_tests(tests, timeout_s=args.timeout)
    raw_stdout = res.get("stdout", "")
    res = _massage_verify(res, args.stdout_json, args.stdout_tail)

    out = {"ok": True, "parsed_edit": parsed, "changed": changed, "selected_tests": tests, "verify": res}
    print(json.dumps(out, indent=2))
    if args.print_stdout: _print_stdout_block("PYTEST OUTPUT", raw_stdout, args.stdout_tail)
    return 0 if res.get("returncode", 1) == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
