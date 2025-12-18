
from __future__ import annotations
import argparse, json, csv, sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

def _iter_reports(paths: List[str]) -> List[Path]:
    files: List[Path] = []
    for p in paths:
        P = Path(p)
        if P.is_file() and P.suffix.lower() == ".json":
            files.append(P)
        elif P.is_dir():
            files.extend(sorted(P.rglob("*.json")))
        else:
            for q in Path().glob(p):
                if q.is_file() and q.suffix.lower() == ".json":
                    files.append(q)
    # de-dup & sort
    return sorted(set(files), key=lambda x: str(x))

def _getmtime_iso(p: Path) -> str:
    try:
        ts = p.stat().st_mtime
        return datetime.fromtimestamp(ts).isoformat()
    except Exception:
        return ""

def _flatten_report(path: Path, data: Dict[str, Any]) -> Dict[str, Any]:
    final = data.get("final") or data.get("verify") or {}
    stage1 = data.get("stage1", {})
    stage2 = data.get("stage2", {}) or {}
    verify1 = stage1.get("verify") or {}
    verify2 = stage2.get("verify") or {}

    def _list_to_csv(x):
        if isinstance(x, list):
            return ",".join(map(str, x))
        return ""

    return {
        "file": str(path),
        "mtime": _getmtime_iso(path),
        "command": data.get("command", ""),
        "target_file": data.get("target_file", ""),
        "strategy": data.get("strategy", "two-stage"),
        "changed": data.get("changed", False),
        "escalated": data.get("escalated", False),
        "impacted_functions": _list_to_csv(data.get("impacted_functions", [])),
        "stage1_tests": _list_to_csv(stage1.get("selected_tests", [])),
        "stage1_rc": verify1.get("returncode"),
        "stage1_pass_frac": verify1.get("pass_frac"),
        "stage2_tests": _list_to_csv(stage2.get("selected_tests", [])),
        "stage2_rc": verify2.get("returncode"),
        "stage2_pass_frac": verify2.get("pass_frac"),
        "final_rc": final.get("returncode"),
        "final_pass_frac": final.get("pass_frac"),
    }

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="s2c.metrics",
        description="Aggregate s2c.edit JSON reports into a CSV and print a quick summary.",
    )
    ap.add_argument("inputs", nargs="+", help="Report dirs/files/globs (e.g., reports/ or reports/*.json)")
    ap.add_argument("--out-csv", dest="out_csv", type=str, default="reports/aggregate.csv",
                    help="Where to write the aggregated CSV (default reports/aggregate.csv)")
    args = ap.parse_args(argv)

    files = _iter_reports(args.inputs)
    if not files:
        print("No JSON reports found.", file=sys.stderr)
        return 2

    rows: List[Dict[str, Any]] = []
    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception as e:
            print(f"Skipping {f}: {e}", file=sys.stderr)
            continue
        rows.append(_flatten_report(f, data))

    # Write CSV
    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "file","mtime","command","target_file","strategy","changed","escalated","impacted_functions",
        "stage1_tests","stage1_rc","stage1_pass_frac","stage2_tests","stage2_rc","stage2_pass_frac",
        "final_rc","final_pass_frac"
    ]
    with outp.open("w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)

    total = len(rows)
    greens = sum(1 for r in rows if (r.get("final_rc") == 0))
    changed = sum(1 for r in rows if r.get("changed"))
    escal = sum(1 for r in rows if r.get("escalated"))
    print(f"Wrote {total} rows to {outp}")
    print(f"Green runs: {greens}/{total}  |  Changed code: {changed}  |  Escalations: {escal}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
