
import subprocess, sys, re, json
from typing import List, Optional, Dict, Any

PYTHON_EXE = sys.executable

SUMMARY_RE = re.compile(r"(?P<p>\d+)\s+passed(?:,|\s)(?:(?P<f>\d+)\s+failed)?", re.I)

def _parse_summary(text: str):
    # Look from the bottom up for a line containing counts (passed/failed/etc.)
    passed = failed = 0
    for line in text.splitlines()[::-1]:
        if "passed" in line.lower() or "failed" in line.lower():
            # Extract any "<num> <label>" pairs (passed, failed, skipped, etc.)
            pairs = re.findall(r"(\d+)\s+(passed|failed|skipped|xfailed|xpassed)", line, flags=re.I)
            if pairs:
                for num, label in pairs:
                    n = int(num)
                    lab = label.lower()
                    if lab == "passed":
                        passed = n
                    elif lab == "failed":
                        failed = n
                break
    total = passed + failed
    pass_frac = (passed / total) if total else 0.0
    return {"passed": passed, "failed": failed, "total": total, "pass_frac": pass_frac}

def run_tests(tests: Optional[List[str]] = None, timeout_s: int = 5) -> Dict[str, Any]:
    """Run pytest inside a restricted child process. Returns structured results.

    Args:
        tests: Optional list of pytest node ids (e.g., ['tests/test_word_count.py::test_basic']).
        timeout_s: Wall-time timeout in seconds for the child process.

    Returns:
        dict with keys: pass_frac, passed, failed, total, stdout, returncode.
    """
    args = [PYTHON_EXE, "-m", "s2c._sandbox_entry", "--timeout", str(timeout_s)]
    if tests:
        args += tests

    try:
        cp = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=max(1, timeout_s + 1),
            text=True,
        )
        out = cp.stdout
        rc = cp.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + "\nTIMEOUT: verifier exceeded wall clock."
        rc = 124

    summary = _parse_summary(out)
    summary.update({"stdout": out, "returncode": rc})
    return summary

def _print_cli(res: Dict[str, Any]):
    print(json.dumps(
        {
            "pass_frac": round(res.get("pass_frac", 0.0), 3),
            "passed": res.get("passed"),
            "failed": res.get("failed"),
            "total": res.get("total"),
            "returncode": res.get("returncode"),
        },
        indent=2,
    ))
    tail = "\n".join(res.get("stdout","").splitlines()[-10:])
    print("\n--- pytest tail ---\n" + tail)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", nargs="*", default=None, help="PyTest node ids")
    ap.add_argument("--timeout", type=int, default=5)
    args = ap.parse_args()

    res = run_tests(args.tests, args.timeout)
    _print_cli(res)
    sys.exit(0 if res.get("failed", 1) == 0 and res.get("returncode", 1) == 0 else 1)
