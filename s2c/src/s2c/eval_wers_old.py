
from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from .hotwords import extract_from_paths
from .intent import parse_and_rank
from .asr import decode_mock
from .edit import _apply_edit, _guess_impacted_functions, _select_tests_by_functions, _project_root
from .verify import run_tests

def _load_prompts(p: Path) -> List[str]:
    lines = [ln.strip() for ln in p.read_text().splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]

def _esr_for_prompts(prompts: List[str], severities: List[float], n_best: int, target_file: Path, timeout: int,
                     use_hotwords: bool, use_canon: bool, count_noop_fail: bool, seed: int) -> List[Dict[str, Any]]:
    root = _project_root()
    hot = extract_from_paths([target_file] + list((root/"tests").rglob("*.py"))) if use_hotwords else set()

    rows: List[Dict[str, Any]] = []
    for sev in severities:
        ok = 0
        total = 0
        for utter in prompts:
            hyp = decode_mock(utter, n_best=n_best, hotwords=hot if use_hotwords else None, severity=sev, seed=seed)
            nbest = hyp["nbest"]
            parsed, diag = parse_and_rank(nbest, hot, disable_canon=(not use_canon))
            if not parsed:
                total += 1
                rows.append({"severity": sev, "utterance": utter, "parsed": False, "changed": False,
                             "pass_frac": None, "returncode": None, "confidence": diag.get("confidence", 0.0),
                             "use_hotwords": use_hotwords, "use_canon": use_canon, "n_best": n_best})
                continue
            before = target_file.read_text()
            after  = _apply_edit(before, parsed)
            changed = (after != before)
            impacted_funcs = _guess_impacted_functions(parsed, before)
            tests = _select_tests_by_functions(root, impacted_funcs) or ["tests"]

            # ✨ NEW: ephemeral apply
            try:
                if changed:
                    target_file.write_text(after)
                res = run_tests(tests, timeout_s=timeout)
            finally:
                if changed:
                    target_file.write_text(before)

            # res = run_tests(tests, timeout_s=timeout)
            success = (case["parsed"] and (case["returncode"] == 0) and (case["changed"] or not count_noop_fail))
            ok += 1 if success else 0
            total += 1
            rows.append({"severity": sev, "utterance": utter, "parsed": True, "changed": changed,
                         "pass_frac": res.get("pass_frac"), "returncode": res.get("returncode"),
                         "confidence": diag.get("confidence", 0.0),
                         "use_hotwords": use_hotwords, "use_canon": use_canon, "n_best": n_best})
        rows.append({"severity": sev, "summary": True, "ESR": (ok/total if total else 0.0), "ok": ok, "total": total,
                     "use_hotwords": use_hotwords, "use_canon": use_canon, "n_best": n_best})
        rows.append({"backend": backend, "severity": sev, "summary": True, "ESR": (ok/total if total else 0.0),
            "ok": ok, "total": total, "parsed_rate": (parsed_ok/total if total else 0.0),
            "changed_rate": (changed_cnt/total if total else 0.0), "use_hotwords": use_hotwords,
            "use_canon": use_canon, "n_best": n_best,})

    return rows

def _write_csv(rows: List[Dict[str, Any]], outp: Path) -> None:
    keys = sorted({k for r in rows for k in r.keys()})
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="s2c.eval_wers",
        description="Evaluate ESR vs mock ASR severity with ablations."
    )
    ap.add_argument("--prompts", type=str, required=True)
    ap.add_argument("--n-best", type=int, default=5)
    ap.add_argument("--severities", type=str, default="0.0,0.3,0.6,0.9")
    ap.add_argument("--file", type=str, default=str(Path(__file__).with_name("user_solution.py")))
    ap.add_argument("--timeout", type=int, default=5)
    ap.add_argument("--out-csv", type=str, default="reports/eval_wers.csv")
    ap.add_argument("--variants", type=str, default="full,no_hotwords,no_canon,nbest1")
    ap.add_argument("--count-noop-fail", action="store_true",
                help="Treat 'no-change' edits as failures")
    ap.add_argument("--seed", type=int, default=42, help="To make runs reproducible with mock")
    args = ap.parse_args(argv)

    target = Path(args.file)
    if not target.exists():
        print(f"Target file not found: {target}", file=sys.stderr); return 2

    severities = [float(x) for x in args.severities.split(",")]
    prompts = _load_prompts(Path(args.prompts))

    all_rows: List[Dict[str, Any]] = []
    requested = [v.strip() for v in args.variants.split(",") if v.strip()]

    if "full" in requested:
        all_rows += _esr_for_prompts(prompts, severities, args.n_best, target, args.timeout, True, True, args.count_noop_fail, args.seed)
    if "no_hotwords" in requested:
        all_rows += _esr_for_prompts(prompts, severities, args.n_best, target, args.timeout, False, True, args.count_noop_fail, args.seed)
    if "no_canon" in requested:
        all_rows += _esr_for_prompts(prompts, severities, args.n_best, target, args.timeout, True, False, args.count_noop_fail, args.seed)
    if "nbest1" in requested:
        all_rows += _esr_for_prompts(prompts, severities, 1, target, args.timeout, True, True, args.count_noop_fail, args.seed)

    _write_csv(all_rows, Path(args.out_csv))

    for s in [r for r in all_rows if r.get("summary")]:
        tag = []
        tag.append("hotwords" if s["use_hotwords"] else "no_hotwords")
        tag.append("canon" if s["use_canon"] else "no_canon")
        tag.append(f"nbest={s['n_best']}")
        
        sev = s.get('severity')
        sev_part = f" severity={sev:.1f}" if isinstance(sev, (int, float)) else ""
        print(f"[{', '.join(tag)}]{sev_part}  ESR={s['ESR']:.3f}  "
             f"parsed={s.get('parsed_rate', 0.0):.2f}  changed={s.get('changed_rate', 0.0):.2f}  "
             f"({s['ok']}/{s['total']})")


    print(f"Wrote {len(all_rows)} rows to {args.out_csv}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
