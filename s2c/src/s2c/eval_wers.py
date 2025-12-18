
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .hotwords import extract_from_paths
from .intent import parse_and_rank
from .asr import decode_mock
from .edit import (
    _apply_edit,
    _guess_impacted_functions,
    _select_tests_by_functions,
    _project_root,
)
from .verify import run_tests


# ---------------------------- helpers ----------------------------

def _load_prompts(p: Path) -> List[str]:
    lines = [ln.strip() for ln in p.read_text().splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def _write_csv(rows: List[Dict[str, Any]], outp: Path) -> None:
    keys = sorted({k for r in rows for k in r.keys()})
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", newline="") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")


def _one_case(parsed: Optional[Dict[str, Any]], target_file: Path, timeout: int, root: Path) -> Dict[str, Any]:
    """Apply edit ephemerally, run targeted tests, and restore file."""
    before = target_file.read_text()
    if not parsed:
        return {"parsed": False, "changed": False, "pass_frac": None, "returncode": None}

    after = _apply_edit(before, parsed)
    changed = (after != before)

    impacted_funcs = _guess_impacted_functions(parsed, before)
    tests = _select_tests_by_functions(root, impacted_funcs) or ["tests"]

    try:
        if changed:
            target_file.write_text(after)
        res = run_tests(tests, timeout_s=timeout)
    finally:
        if changed:
            target_file.write_text(before)

    return {
        "parsed": True,
        "changed": changed,
        "pass_frac": res.get("pass_frac"),
        "returncode": res.get("returncode", 1),
    }


def _decode_asr_mock(utter: str, n_best: int, hotwords: Optional[Set[str]], severity: float, seed: int) -> List[str]:
    hyp = decode_mock(utter, n_best=n_best, hotwords=hotwords, severity=severity, seed=seed)
    return hyp["nbest"]


def _decode_wav_whisper(wav_path: str, n_best: int, hotwords: Optional[Set[str]]) -> List[str]:
    # Import lazily so users without whisper installed can still run mock.
    from .asr_whisper import decode_whisper  # type: ignore
    hyp = decode_whisper(wav_path, n_best=n_best, hotwords=hotwords)
    return hyp["nbest"]


# ----------------------- ESR loops (mock / wav) -----------------------

def _esr_for_prompts(
    prompts: List[str],
    severities: List[float],
    n_best: int,
    target_file: Path,
    timeout: int,
    backend: str,
    use_hotwords: bool,
    use_canon: bool,
    count_noop_fail: bool,
    seed: int,
) -> List[Dict[str, Any]]:
    root = _project_root()
    hot = extract_from_paths([target_file] + list((root / "tests").rglob("*.py"))) if use_hotwords else set()

    rows: List[Dict[str, Any]] = []
    for sev in severities:
        ok = total = parsed_ok = changed_cnt = 0
        for utter in prompts:
            if backend != "mock":
                raise RuntimeError("Text prompts path is only valid with --backend mock")
            nbest = _decode_asr_mock(utter, n_best=n_best, hotwords=hot if use_hotwords else None, severity=sev, seed=seed)

            parsed, diag = parse_and_rank(nbest, hot, disable_canon=(not use_canon))
            case = _one_case(parsed, target_file, timeout, root)

            parsed_ok += 1 if case["parsed"] else 0
            changed_cnt += 1 if case["changed"] else 0
            success = (case["parsed"] and (case["returncode"] == 0) and (case["changed"] or not count_noop_fail))
            ok += 1 if success else 0
            total += 1

            rows.append({
                "backend": backend,
                "severity": sev,
                "utterance": utter,
                "parsed": case["parsed"],
                "changed": case["changed"],
                "pass_frac": case["pass_frac"],
                "returncode": case["returncode"],
                "confidence": diag.get("confidence", 0.0),
                "use_hotwords": use_hotwords,
                "use_canon": use_canon,
                "n_best": n_best,
            })

        rows.append({
            "backend": backend,
            "severity": sev,
            "summary": True,
            "ESR": (ok / total if total else 0.0),
            "ok": ok,
            "total": total,
            "parsed_rate": (parsed_ok / total if total else 0.0),
            "changed_rate": (changed_cnt / total if total else 0.0),
            "use_hotwords": use_hotwords,
            "use_canon": use_canon,
            "n_best": n_best,
        })
    return rows


def _load_wav_rows(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open() as f:
        r = csv.DictReader(f)
        if "wav" not in (r.fieldnames or []):
            raise ValueError("CSV must include a 'wav' column")
        for row in r:
            if row.get("wav"):
                rows.append({"wav": row["wav"], "ref": row.get("ref", "")})
    return rows


def _esr_for_wavs(
    wavs: List[Dict[str, str]],
    n_best: int,
    target_file: Path,
    timeout: int,
    use_hotwords: bool,
    use_canon: bool,
    count_noop_fail: bool,
) -> List[Dict[str, Any]]:
    root = _project_root()
    hot = extract_from_paths([target_file] + list((root / "tests").rglob("*.py"))) if use_hotwords else set()

    rows: List[Dict[str, Any]] = []
    ok = total = parsed_ok = changed_cnt = 0

    for row in wavs:
        nbest = _decode_wav_whisper(row["wav"], n_best=n_best, hotwords=hot if use_hotwords else None)
        parsed, diag = parse_and_rank(nbest, hot, disable_canon=(not use_canon))
        case = _one_case(parsed, target_file, timeout, root)

        parsed_ok += 1 if case["parsed"] else 0
        changed_cnt += 1 if case["changed"] else 0
        success = (case["parsed"] and (case["returncode"] == 0) and (case["changed"] or not count_noop_fail))
        ok += 1 if success else 0
        total += 1

        rows.append({
            "backend": "whisper",
            "wav": row["wav"],
            "ref": row.get("ref", ""),
            "parsed": case["parsed"],
            "changed": case["changed"],
            "pass_frac": case["pass_frac"],
            "returncode": case["returncode"],
            "confidence": diag.get("confidence", 0.0),
            "use_hotwords": use_hotwords,
            "use_canon": use_canon,
            "n_best": n_best,
        })

    rows.append({
        "backend": "whisper",
        "summary": True,
        "ESR": (ok / total if total else 0.0),
        "ok": ok,
        "total": total,
        "parsed_rate": (parsed_ok / total if total else 0.0),
        "changed_rate": (changed_cnt / total if total else 0.0),
        "use_hotwords": use_hotwords,
        "use_canon": use_canon,
        "n_best": n_best,
    })
    return rows


# ------------------------------ main ------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="s2c.eval_wers",
        description="Evaluate ESR with mock (text prompts) or Whisper (wav); supports ablations and CSV output.",
    )
    ap.add_argument("--backend", choices=["mock", "whisper"], default="mock", help="ASR backend")
    ap.add_argument("--prompts", type=str, help="Path to .txt with one utterance per line (mock only)")
    ap.add_argument("--wav-csv", type=str, help="CSV with columns: wav[,ref] (whisper only)")
    ap.add_argument("--n-best", type=int, default=5)
    ap.add_argument("--severities", type=str, default="0.0,0.3,0.6,0.9", help="Mock noise severities (comma-separated)")
    ap.add_argument("--file", type=str, default=str(Path(__file__).with_name("user_solution.py")))
    ap.add_argument("--timeout", type=int, default=5)
    ap.add_argument("--out-csv", type=str, default="reports/eval_wers.csv")
    ap.add_argument("--variants", type=str, default="full,no_hotwords,no_canon,nbest1")
    ap.add_argument("--count-noop-fail", action="store_true", help="Treat 'no-change' edits as failures")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (mock backend)")
    args = ap.parse_args(argv)

    target = Path(args.file)
    if not target.exists():
        print(f"Target file not found: {target}", file=sys.stderr)
        return 2

    all_rows: List[Dict[str, Any]] = []

    if args.backend == "mock":
        if not args.prompts:
            print("With --backend mock you must provide --prompts <file>", file=sys.stderr)
            return 2
        severities = [float(x) for x in args.severities.split(",") if x.strip()]
        prompts = _load_prompts(Path(args.prompts))

        requested = [v.strip() for v in args.variants.split(",") if v.strip()]
        if "full" in requested:
            all_rows += _esr_for_prompts(prompts, severities, args.n_best, target, args.timeout,
                                         "mock", True, True, args.count_noop_fail, args.seed)
        if "no_hotwords" in requested:
            all_rows += _esr_for_prompts(prompts, severities, args.n_best, target, args.timeout,
                                         "mock", False, True, args.count_noop_fail, args.seed)
        if "no_canon" in requested:
            all_rows += _esr_for_prompts(prompts, severities, args.n_best, target, args.timeout,
                                         "mock", True, False, args.count_noop_fail, args.seed)
        if "nbest1" in requested:
            all_rows += _esr_for_prompts(prompts, severities, 1, target, args.timeout,
                                         "mock", True, True, args.count_noop_fail, args.seed)

    else:  # whisper
        if not args.wav_csv:
            print("With --backend whisper you must provide --wav-csv <file.csv>", file=sys.stderr)
            return 2
        wavs = _load_wav_rows(Path(args.wav_csv))
        all_rows += _esr_for_wavs(wavs, args.n_best, target, args.timeout, True, True, args.count_noop_fail)

    # Write CSV
    _write_csv(all_rows, Path(args.out_csv))

    # Print summaries
    summaries = [r for r in all_rows if r.get("summary")]
    for s in summaries:
        tag = [f"backend={s.get('backend','mock')}"]
        if s.get("severity") is not None:
            tag.append(f"severity={s['severity']:.1f}")
        tag.append("hotwords" if s.get("use_hotwords") else "no_hotwords")
        tag.append("canon" if s.get("use_canon") else "no_canon")
        tag.append(f"nbest={s.get('n_best', '?')}")

        sev = s.get("severity")
        sev_part = f" severity={sev:.1f}" if isinstance(sev, (int, float)) else ""
        print(f"[{', '.join(tag)}]{sev_part}  ESR={s['ESR']:.3f}  "
              f"parsed={s.get('parsed_rate', 0.0):.2f}  changed={s.get('changed_rate', 0.0):.2f}  "
              f"({s['ok']}/{s['total']})")

    print(f"Wrote {len(all_rows)} rows to {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
