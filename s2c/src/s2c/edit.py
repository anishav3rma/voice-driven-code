
from __future__ import annotations
import argparse, json, difflib, re
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

from .edits.grammar import parse_edit
from .edits.apply_ast import apply_rename_var, apply_add_param, apply_wrap_lines
from .verify import run_tests

import libcst as cst
from libcst import metadata as md

# The file the harness writes functions into
DEFAULT_FILE = Path(__file__).with_name("user_solution.py")

# ---------------- helpers: function spans & impacted tests ----------------

class _FuncSpanCollector(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (md.PositionProvider,)
    def __init__(self) -> None:
        self.func_spans: List[tuple[str, int, int]] = []  # (name, start_line, end_line)
    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        pos = self.get_metadata(md.PositionProvider, node, default=None)
        if pos:
            self.func_spans.append((node.name.value, pos.start.line, pos.end.line))
        return True

def _function_spans(code: str) -> List[tuple[str, int, int]]:
    mod = cst.parse_module(code)
    wrapper = md.MetadataWrapper(mod)
    col = _FuncSpanCollector()
    wrapper.visit(col)
    return col.func_spans

def _functions_overlapping_range(code: str, start_line: int, end_line: int) -> Set[str]:
    out: Set[str] = set()
    for name, s, e in _function_spans(code):
        if not (e < start_line or s > end_line):
            out.add(name)
    return out

def _guess_impacted_functions(edit: Dict[str, Any], before_code: str) -> Set[str]:
    op = edit.get("op")
    if op == "add_param":
        return {edit["function"]}
    if op == "rename":
        func = edit.get("function")
        return {func} if func else set()   # global rename: unknown → full suite
    if op == "wrap":
        func = edit.get("function")
        if func:
            return {func}
        return _functions_overlapping_range(before_code, int(edit["start_line"]), int(edit["end_line"]))
    return set()

def _project_root() -> Path:
    # src/s2c/edit.py -> src -> project root
    return Path(__file__).resolve().parents[2]

def _find_test_files(root_dir: Path) -> List[Path]:
    return [p for p in (root_dir / "tests").rglob("test_*.py")]

def _select_tests_by_functions(root_dir: Path, funcs: Set[str]) -> List[str]:
    """
    Heuristic: pick tests that import user_solution AND mention any impacted function name.
    """
    if not funcs:
        return []  # unknown impact; caller may choose to run full suite
    tests = _find_test_files(root_dir)
    pat = re.compile(r"\b(" + "|".join(re.escape(f) for f in funcs) + r")\b")
    selected: List[str] = []
    for t in tests:
        try:
            txt = t.read_text()
        except Exception:
            continue
        if ("s2c.user_solution" in txt or
            "from s2c import user_solution" in txt or
            "import s2c.user_solution as" in txt) and pat.search(txt):
            selected.append(str(t))
    return selected

# ---------------- stdout helpers ----------------

def _tail(s: str, n: int) -> str:
    if not s:
        return ""
    lines = s.splitlines()
    return "\n".join(lines[-n:])

def _massage_verify(res: dict, mode: str, tail_lines: int) -> dict:
    """
    mode:
      - 'full'  : keep full stdout and add stdout_tail
      - 'tail'  : drop full stdout, keep stdout_tail string
      - 'lines' : drop full stdout, keep stdout_lines (array of lines)
      - 'none'  : drop all stdout fields
    """
    if not isinstance(res, dict):
        return res
    out = dict(res)
    stdout = out.pop("stdout", "") or ""

    if mode == "full":
        out["stdout"] = stdout
        out["stdout_tail"] = _tail(stdout, tail_lines)
    elif mode == "tail":
        out["stdout_tail"] = _tail(stdout, tail_lines)
    elif mode == "lines":
        out["stdout_lines"] = _tail(stdout, tail_lines).splitlines()
    elif mode == "none":
        pass
    else:
        out["stdout_tail"] = _tail(stdout, tail_lines)
    return out

def _print_stdout_block(title: str, text: str, tail_lines: int) -> None:
    if not text:
        return
    shown = _tail(text, tail_lines) if tail_lines > 0 else text
    bar = "=" * len(title)
    print(f"\n{title}\n{bar}\n{shown}")

# ---------------- apply edit ----------------

def _apply_edit(code: str, e: Dict[str, Any]) -> str:
    op = e.get("op")
    if op == "rename":
        return apply_rename_var(code, function=e.get("function"), old=e["target"], new=e["new_name"])
    if op == "add_param":
        return apply_add_param(code, function=e["function"], name=e["name"], default_value=e.get("default"))
    if op == "wrap":
        return apply_wrap_lines(code, start_line=int(e["start_line"]), end_line=int(e["end_line"]), function=e["function"])
    raise ValueError(f"Unsupported op: {op}")

# ---------------- CLI ----------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="s2c.edit",
        description="Parse a text edit command, apply AST-safe patch, then run impacted tests (escalate to full suite on failure).",
    )
    ap.add_argument("command", type=str,
        help="Edit command, e.g. 'rename counts to total_counts in function word_count'")
    ap.add_argument("--file", type=str, default=str(DEFAULT_FILE),
        help="Target file to patch (default: s2c/user_solution.py)")
    ap.add_argument("--tests", nargs="*", default=None,
        help="Override: explicit pytest node ids to run after patch (disables impacted-tests auto-run)")
    ap.add_argument("--timeout", type=int, default=5, help="Verifier timeout (s)")
    ap.add_argument("--print-diff", dest="print_diff", action="store_true", help="Print a unified diff")
    ap.add_argument("--dry-run",  dest="dry_run",  action="store_true", help="Parse + show diff only; don't write or run tests")
    ap.add_argument("--no-escalate", dest="no_escalate", action="store_true",
                    help="Do not run the full test suite if impacted subset fails")
    ap.add_argument("--json-out", dest="json_out", type=str,
                    help="Write the full run result (JSON) to this path")
    ap.add_argument("--print-stdout", dest="print_stdout", action="store_true",
                    help="After JSON, print human-readable pytest stdout tails")
    ap.add_argument("--stdout-tail", dest="stdout_tail", type=int, default=80,
                    help="Max lines of pytest stdout to show/save as tail (default 80)")
    ap.add_argument("--no-stdout-in-json", dest="no_stdout_in_json", action="store_true",
                    help="Drop full stdout from JSON; include only stdout_tail")
    ap.add_argument(
    "--stdout-json",
    choices=["full", "tail", "lines", "none"],
    default="tail",
    help="How pytest stdout is included in the JSON: full, tail (default), lines (array), or none",
)
    args = ap.parse_args(argv)

    # Parse command -> edit JSON
    e = parse_edit(args.command)
    if not e:
        out = {"ok": False, "error": "Could not parse edit command", "command": args.command}
        print(json.dumps(out, indent=2))
        if args.json_out:
            Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json_out).write_text(json.dumps(out, indent=2))
        return 2

    target_path = Path(args.file)
    if not target_path.exists():
        out = {"ok": False, "error": f"Target file not found: {target_path}"}
        print(json.dumps(out, indent=2))
        if args.json_out:
            Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json_out).write_text(json.dumps(out, indent=2))
        return 2

    before = target_path.read_text()
    after  = _apply_edit(before, e)
    changed = (after != before)

    if not args.dry_run and changed:
        target_path.write_text(after)

    # Show diff (optional)
    if args.print_diff:
        diff = "".join(difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=str(target_path),
            tofile=str(target_path),
        ))
        print(diff if diff.strip() else "# (no change)")

    # ----- three execution strategies: explicit, dry-run, impacted+escalation -----
    tests_ok = 0
    if args.tests is not None:
        res = run_tests(args.tests, timeout_s=args.timeout)
        raw_exp_stdout = res.get("stdout", "")        # <--- capture
        res = _massage_verify(res, args.stdout_json, args.stdout_tail)
        out = {
            "ok": True, "parsed_edit": e, "changed": changed,
            "strategy": "explicit",
            "command": args.command,
            "target_file": str(target_path),
            "selected_tests": args.tests,
            "verify": res
        }
        rc = 0 if res.get("returncode", 1) == 0 else 1
        test_ok = rc
    elif args.dry_run:
        out = {
            "ok": True, "parsed_edit": e, "changed": changed,
            "strategy": "dry-run", "command": args.command, "target_file": str(target_path),
            "note": "no tests run"
        }
        rc = 0
    else:
        # Impacted subset → escalate to full suite if subset fails
        root = Path(__file__).resolve().parents[2]
        impacted_funcs = _guess_impacted_functions(e, before)
        impacted = _select_tests_by_functions(root, impacted_funcs)

        stage1_tests = impacted if impacted else ["tests"]
        print("Running Stage-1 Tests")
        stage1_res = run_tests(stage1_tests, timeout_s=args.timeout)
        stage1_raw = stage1_res.get("stdout", "")     # <--- capture
        stage1_res = _massage_verify(stage1_res, args.stdout_json, args.stdout_tail)

        escalated = False
        stage2_tests: Optional[List[str]] = None
        stage2_res: Optional[Dict[str, Any]] = None
        stage2_raw = ""

        if impacted and stage1_res.get("returncode", 1) != 0 and not args.no_escalate:
            escalated = True
            stage2_tests = ["tests"]
            print("Running Stage-2 Tests")
            stage2_res = run_tests(stage2_tests, timeout_s=args.timeout)
            stage2_raw = stage2_res.get("stdout", "")     # <--- capture
            stage2_res = _massage_verify(stage2_res, args.stdout_json, args.stdout_tail)

        final_res = stage2_res if escalated else stage1_res
        rc = 0 if final_res.get("returncode", 1) == 0 else 1
        test_ok = rc

        out = {
            "ok": True,
            "parsed_edit": e,
            "changed": changed,
            "command": args.command,
            "target_file": str(target_path),
            "impacted_functions": sorted(list(impacted_funcs)),
            "stage1": {"selected_tests": stage1_tests, "verify": stage1_res},
            "escalated": escalated,
            "stage2": ({"selected_tests": stage2_tests, "verify": stage2_res} if escalated else None),
            "final": {"returncode": final_res.get("returncode"), "pass_frac": final_res.get("pass_frac")}
        }

    # Write (unless dry-run or tests failed)
    if not args.dry_run and changed and test_ok != 0:
        target_path.write_text(before)
        print(f"REVERTING EDITS, Test Status: {test_ok}")

    # Write JSON report if requested
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(out, indent=2))

    print(json.dumps(out, indent=2))

    # Pretty tails after JSON (optional)
    if args.print_stdout:
        if out.get("strategy") == "explicit":
            _print_stdout_block("PYTEST OUTPUT (EXPLICIT)", raw_exp_stdout, args.stdout_tail)
        elif out.get("strategy") == "dry-run":
            pass
        else:
            _print_stdout_block("PYTEST OUTPUT (STAGE 1)", stage1_raw, args.stdout_tail)
            if out.get("escalated") and out.get("stage2"):
                _print_stdout_block("PYTEST OUTPUT (STAGE 2)", stage2_raw, args.stdout_tail)

    return rc

if __name__ == "__main__":
    raise SystemExit(main())
