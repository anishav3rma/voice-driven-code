# app.py — Streamlit UI for the voice→intent→edit demo (Option B)
# Run:  streamlit run app.py

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import streamlit as st
from s2c.intent import parse_and_rank as intent_parse_and_rank, canonicalize, CanonConfig
import tempfile
import os
import time, traceback
from tempfile import NamedTemporaryFile
from dataclasses import asdict, is_dataclass
from s2c.edits.apply_ast import NoWrapTargetFound

# --- Optional mic recorder (simple widget) ---
try:
    from audio_recorder_streamlit import audio_recorder  # pip install audio-recorder-streamlit
except Exception:  # if not installed, we gracefully fall back to file upload
    audio_recorder = None  # type: ignore

# --- s2c imports (your package should be installed in editable mode) ---
from s2c.hotwords import extract_from_paths
from s2c.intent import parse_and_rank
from s2c.asr import decode_mock
try:
    from s2c.asr_whisper import decode_whisper  # available if you installed faster-whisper
except Exception:
    decode_whisper = None  # type: ignore

from s2c.edit import (
    _apply_edit,
    _guess_impacted_functions,
    _select_tests_by_functions,
    _project_root,
)
from s2c.verify import run_tests

# ------------------------- helpers -------------------------

# ---- UI helpers / styles ----
STEPS = [
    "Decode Audio", "Strip Disfluencies", "Canonicalize", "Parse & Rank", "Test & Edit"]

if "step_i" not in st.session_state: st.session_state.step_i = 0
if "step_log" not in st.session_state: st.session_state.step_log = []   # list of dicts
if "step_ctx" not in st.session_state: st.session_state.step_ctx = {}   # carries data to next steps
if "last_wav" not in st.session_state: st.session_state.last_wav = None

def next_step_label():
    i = min(st.session_state.step_i, len(STEPS) - 1)
    return STEPS[i]

def _push_step_result(entry: dict) -> None:
    """Append one step result to the on-page 'Last result' log."""
    st.session_state.step_log.append(entry)

def _save_wav_tmp(wav_bytes: bytes, *, suffix=".wav") -> str:
    f = NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(wav_bytes)
    f.flush(); f.close()
    return f.name

def _estimate_code_panel_height(lines: int, base=140, line_px=22, cap=900):
    # rough match to st.code line height + padding
    # lines = max(1, code.count("\n") + 1)
    return min(cap, max(base, base + lines * line_px))

st.markdown("""
<style>
.lr-panel {
  background: #f6f8ff;            /* light like code area */
  border: 1px solid #e6e9f5;
  border-radius: 10px;
  padding: 12px 14px;
}
</style>
""", unsafe_allow_html=True)

def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _write_text(p: Path, txt: str) -> None:
    p.write_text(txt, encoding="utf-8")


def _unified_diff(a: str, b: str, path: str) -> str:
    return "".join(
        difflib.unified_diff(
            a.splitlines(keepends=True),
            b.splitlines(keepends=True),
            fromfile=f"{path} (before)",
            tofile=f"{path} (after)",
            n=3,
        )
    )

from typing import List  # you already import this at top

def _run_summary(entry: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    cfg = entry.get("config", {})
    diag = entry.get("diag", {})
    action = entry.get("action")

    # 0) Config snapshot
    if cfg:
        lines.append(
            f"ASR={cfg.get('backend')} nbest={cfg.get('n_best')} "
            f"hotwords={'on' if cfg.get('use_hotwords') else 'off'}({cfg.get('hotwords_count',0)}) "
            f"canon={'on' if cfg.get('use_canon') else 'off'} min_conf={cfg.get('min_conf'):.2f}"
        )

    # 1) Chosen candidate & normalization
    ch = (diag.get("chosen") or {})
    raw = ch.get("raw")
    norm = ch.get("norm")
    if raw:
        if norm and norm != raw:
            lines.append(f"Canonicalized: '{raw}' → '{norm}'")
        else:
            lines.append(f"Top hypothesis: '{raw}'")

    # 2) Confidence & consensus
    if "confidence" in diag:
        lines.append(
            f"Confidence={diag['confidence']:.2f}, "
            f"consensus={diag.get('consensus',0.0):.2f}, "
            f"softmax={diag.get('softmax',0.0):.2f}"
        )

    # 3) Parse summary
    p = entry.get("parsed")
    if p is None:
        lines.append("Failed to parse any candidate → ask back")
    else:
        op = p.get("op")
        if op == "add_param":
            lines.append(f"Parsed: add_param name={p.get('name')} default={p.get('default')} in function={p.get('function')}")
        elif op == "rename":
            lines.append(f"Parsed: rename {p.get('target')} → {p.get('new_name')} in function={p.get('function')}")
        elif op == "wrap":
            lines.append(f"Parsed: wrap lines {p.get('start_line')}–{p.get('end_line')} in function={p.get('function')}")
        elif op:
            lines.append(f"Parsed op={op}")

    # 4) Outcome
    if action == "ask_back":
        reason = entry.get("reason", "")
        if reason:
            lines.append(f"Gated: {reason}")
    elif action == "applied":
        if entry.get("changed"):
            diff_lines = len((entry.get("diff", "")).splitlines())
            lines.append(f"Applied edit → changed code (diff {diff_lines} lines)")
        else:
            lines.append("Applied edit → no changes (no-op)")
        ver = entry.get("verify", {})
        if ver:
            lines.append(f"Pytest: {ver.get('passed',0)} passed / {ver.get('failed',0)} failed (rc={ver.get('returncode')})")
        sel = entry.get("tests") or []
        if sel:
            if isinstance(sel, list):
                lines.append(f"Selected tests: {len(sel)} file(s)")
            else:
                lines.append("Selected tests: custom set")

    return lines

def _numbered(text: str) -> str:
    """Return code with left-gutter line numbers for easy reference."""
    lines = text.splitlines()
    width = max(3, len(str(len(lines))))
    out = []
    for i, ln in enumerate(lines, 1):
        out.append(f"{str(i).rjust(width)} | {ln}")
    return "\n".join(out)


def _hotwords_for_repo(target_file: Path) -> Set[str]:
    root = _project_root()
    tests = list((root / "tests").rglob("*.py"))
    return extract_from_paths([target_file] + tests)

# --- Disfluency stripping (regex-free to avoid editor escaping issues) ---
_DISFLUENCY_UNI = {"uh","um","er","ah","eh","hmm","mm","mmm"}
_DISFLUENCY_BI = {("you","know"), ("i","mean"), ("sort","of"), ("kind","of")}

import re

def _strip_disfluencies(
    s: str,
    uni: Optional[Set[str]] = None,
    bi: Optional[Set[Tuple[str, str]]] = None,
) -> str:
    """Remove common speech disfluencies and simple word stutters.
    Matches unigram/bigram fillers ignoring edge punctuation, while preserving punctuation.
    """
    uni_src = uni if uni is not None else _DISFLUENCY_UNI
    bi_src  = bi  if bi  is not None else _DISFLUENCY_BI
    uni_set = {w.lower() for w in uni_src}
    bi_set  = {(a.lower(), b.lower()) for (a, b) in bi_src}

    def core(tok: str) -> str:
        # strip leading/trailing non-word (but keep apostrophes in words)
        return re.sub(r"^[^\w']+|[^\w']+$", "", tok).lower()

    def trail(tok: str) -> str:
        m = re.search(r"[^\w']+$", tok)
        return m.group(0) if m else ""

    words = s.split()
    out: list[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        w_core = core(w)

        # try bigram first (e.g., "you know", "i mean")
        if i + 1 < len(words):
            w2 = words[i + 1]
            if (w_core, core(w2)) in bi_set:
                # preserve trailing punctuation from the second token
                t = trail(w2)
                if t:
                    if out:
                        out[-1] = out[-1] + t
                    else:
                        out.append(t)
                i += 2
                continue

        # unigram filler (e.g., "um", "like")
        if w_core in uni_set:
            # preserve trailing punctuation from this token
            t = trail(w)
            if t:
                if out:
                    out[-1] = out[-1] + t
                else:
                    out.append(t)
            i += 1
            continue

        # collapse exact stutters (case-insensitive, punctuation-agnostic)
        if out and core(out[-1]) == w_core:
            i += 1
            continue

        out.append(w)
        i += 1

    return " ".join(out).strip()

# def _one_run(
#     *,
#     backend: str,
#     text_cmd: Optional[str],
#     wav_bytes: Optional[bytes],
#     n_best: int,
#     min_conf: float,
#     use_hotwords: bool,
#     use_canon: bool,
#     severity: float,
#     target_file: Path,
#     timeout_s: int,
# ) -> Dict[str, Any]:
def _one_run(
    *,
    backend: str,
    text_cmd: Optional[str],
    wav_bytes: Optional[bytes],
    n_best: int,
    min_conf: float,
    use_hotwords: bool,
    use_canon: bool,
    strip_disfluencies: bool,
    severity: float,
    target_file: Path,
    timeout_s: int,
    hotwords_override: Optional[Set[str]] = None,
    use_dec_prompt: bool = False,
    initial_prompt: Optional[str] = None,
    strip_uni: Optional[Set[str]] = None,
    strip_bi: Optional[Set[Tuple[str, str]]] = None,
    vad_ms: int = 400,) -> Dict[str, Any]:
    """Run a single end-to-end command. Returns a rich dict for UI."""
    root = _project_root()
    # hot = _hotwords_for_repo(target_file) if use_hotwords else set()
    if use_hotwords:
        hot = hotwords_override if hotwords_override is not None else _hotwords_for_repo(target_file)
    else:
        hot = set()

    # 1) ASR → nbest
    if backend == "mock":
        nbest_res = decode_mock(text_cmd or "", n_best=n_best, hotwords=hot, severity=severity)
        nbest = nbest_res["nbest"]
    else:
        if decode_whisper is None:
            raise RuntimeError("faster-whisper not available. Install it and ensure asr_whisper is importable.")
        if not wav_bytes:
            raise ValueError("No audio provided. Use the mic recorder or upload a WAV/MP3, or switch to mock text mode.")
        
        suffix = ".wav"
        # Write to a real temp file and close it before decoding
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_wav = Path(tmp.name)
            tmp.write(wav_bytes)

        # # Persist uploaded/recorded audio to a temp file for whisper
        # tmp_wav = Path(".streamlit_tmp.wav")
        # tmp_wav.write_bytes(wav_bytes)
        try:
            # nbest = decode_whisper(str(tmp_wav), n_best=n_best, hotwords=hot)["nbest"]
            nbest = decode_whisper(
                str(tmp_wav),
                n_best=n_best,
                hotwords=hot,
                prompt=(initial_prompt if use_dec_prompt else None,),
                vad_ms=vad_ms,
            )["nbest"]            
        finally:
            try:
                tmp_wav.unlink(missing_ok=True)
            except Exception:
                pass

    # Optional: strip disfluencies from ASR hypotheses
    if strip_disfluencies:
        nbest = [s for s in (_strip_disfluencies(x, strip_uni, strip_bi) for x in nbest) if s]

    canon_cfg = None
    if use_canon:
        canon_cfg = CanonConfig(
            join_after_keyword=canon_knobs["join_after_keyword"],
            number_words=canon_knobs["number_words"],
            keyword_canon=canon_knobs["keyword_canon"],
            homophone_repair=canon_knobs["homophone_repair"],
        )

    # 2) Intent parse/rank (+ canonicalization toggle)
    parsed, diag = parse_and_rank(nbest, hot, disable_canon=(not use_canon), canon_config=canon_cfg)
    if parsed is None:
        return {
            "action": "ask_back",
            "parsed": None,
            "changed": False,
            "nbest": nbest,
            "diag": diag,
            "reason": "no_parse",
            "config": {
                "backend": backend,
                "n_best": n_best,
                "min_conf": min_conf,
                "use_hotwords": use_hotwords,
                "use_canon": use_canon,
                "severity": severity,
                "hotwords_count": len(hot),
            },
        }

    # Confidence gate
    if (diag.get("confidence", 0.0) or 0.0) < min_conf:
        return {
            "action": "ask_back",
            "parsed": parsed,            # keep the parsed dict here
            "changed": False,
            "nbest": nbest,
            "diag": diag,
            "reason": f"confidence {diag.get('confidence',0.0):.2f} < threshold {min_conf:.2f}",
            "config": {
                "backend": backend,
                "n_best": n_best,
                "min_conf": min_conf,
                "use_hotwords": use_hotwords,
                "use_canon": use_canon,
                "severity": severity,
                "hotwords_count": len(hot),
            },
        }
    
    # 3) Apply edit ephemerally → run impacted tests → restore
    before = _read_text(target_file)
    after = _apply_edit(before, parsed)
    changed = (after != before)

    impacted = _guess_impacted_functions(parsed, before)
    tests = _select_tests_by_functions(root, impacted) or ["tests"]

    try:
        if changed:
            _write_text(target_file, after)
        verify = run_tests(tests, timeout_s=timeout_s)
    finally:
        if changed:
            _write_text(target_file, before)

    return {
        "action": "applied",
        "nbest": nbest,
        "diag": diag,
        "parsed": parsed,
        "changed": changed,
        "diff": _unified_diff(before, after, str(target_file)) if changed else "",
        "tests": tests,
        "verify": verify,
        # For code panel preview
        "before_code": before,
        "after_code": after,
        "config": {
            "backend": backend,
            "n_best": n_best,
            "min_conf": min_conf,
            "use_hotwords": use_hotwords,
            "use_canon": use_canon,
            "severity": severity,
            "hotwords_count": len(hot),
        },
    }

def _one_run_pipeline(
    *,
    backend: str,
    text_cmd: str | None,
    wav_bytes: bytes | None,
    n_best: int,
    min_conf: float,
    use_hotwords: bool,
    use_canon: bool,
    use_strip_disfluencies: bool,
    severity: float,                      # if you pass it through elsewhere
    target_file: Path,
    timeout_s: int,
    hotwords_override: set[str] | None,
    use_dec_prompt: bool,
    initial_prompt: str | None,
    strip_uni: set[str],
    strip_bi: set[tuple[str, str]],
    vad_ms: int,
    canon_cfg: CanonConfig,
    project_root: Path | None = None,
) -> dict:
    """
    Full pipeline: Decode → Strip Disfluencies → Canonicalize → Parse & Rank → Test & Edit.
    Logs each step to Last result (step_log) and returns a final run summary.
    """
    # Ensure Last result panel shows step-by-step output for this run
    st.session_state.step_log = []
    st.session_state.step_ctx = {}
    st.session_state.step_i = 0
    st.session_state.run_history = []     # keep modes exclusive

    # --- Step 1: Decode Audio ---
    res1 = step_decode_audio(
        backend=backend,
        wav_bytes=wav_bytes,
        text_cmd=text_cmd,
        n_best=n_best,
        hotwords=(hotwords_override if use_hotwords else None),
        use_dec_prompt=use_dec_prompt,
        initial_prompt=initial_prompt,
        vad_ms=vad_ms,
    )
    _push_step_result(res1)
    if not res1.get("ok"):
        return {"halted_at": "Decode Audio", "result": res1}

    nbest = res1.get("nbest") or []
    st.session_state.step_ctx["nbest"] = nbest

    # --- Step 2: Strip Disfluencies ---
    res2 = step_strip_disfluencies(
        nbest=nbest,
        enable=use_strip_disfluencies,
        uni_set=strip_uni,
        bi_pairs=strip_bi,
    )
    _push_step_result(res2)
    if not res2.get("ok"):
        return {"halted_at": "Strip Disfluencies", "result": res2}

    nbest = res2.get("nbest") or nbest
    st.session_state.step_ctx["nbest"] = nbest

    # --- Step 3: Canonicalize ---
    res3 = step_canonicalize(
        nbest=nbest,
        enable=use_canon,
        cfg=canon_cfg,
    )
    _push_step_result(res3)
    if not res3.get("ok"):
        return {"halted_at": "Canonicalize", "result": res3}

    nbest = res3.get("nbest") or nbest
    st.session_state.step_ctx["nbest"] = nbest

    # --- Step 4: Parse & Rank (no re-canon) ---
    res4 = step_parse_rank(
        nbest=nbest,
        hotwords=(hotwords_override if use_hotwords else set()),
        min_conf=min_conf,
        canon_cfg=canon_cfg,
    )
    _push_step_result(res4)
    if not res4.get("ok"):
        return {"halted_at": "Parse & Rank", "result": res4}

    parsed      = res4.get("parsed")
    confidence  = res4.get("confidence")
    action      = res4.get("action")
    st.session_state.step_ctx.update({"parsed": parsed, "confidence": confidence, "action": action})

    # --- Step 5: Test & Edit ---
    root = _project_root()
    res5 = step_test_edit(
        parsed=parsed,
        action=action,
        confidence=confidence,
        min_conf=min_conf,
        target_file=target_file,
        project_root=root,
        timeout_s=timeout_s,
    )
    _push_step_result(res5)

    # (Optional) stash after_code for Code panel “after” preview in step mode
    det = res5.get("details", {})
    if res5.get("ok") and det.get("after_code"):
        st.session_state.step_ctx["after_code"] = det["after_code"]
        st.session_state.step_ctx["before_code"] = det.get("before_code")
        st.session_state.step_ctx["ok"] = True
    else:
        st.session_state.step_ctx.pop("after_code", None)
        st.session_state.step_ctx["ok"] = False

    # Reset stepper to first step after a full run
    st.session_state.step_i = 0

    # Build a final summary akin to your old _one_run return
    final = {
        "action": action,
        "parsed": parsed,
        "nbest": nbest,
        "changed": res5.get("changed", False),
        "tests": det.get("selected_tests"),
        "verify": det.get("verify"),
        "before_code": det.get("before_code"),
        "after_code": det.get("after_code") if res5.get("ok") else None,
        "diff": det.get("diff") if res5.get("ok") else "",
        "diag": res4.get("details", {}).get("diagnostics"),
        "config": {
            "backend": backend,
            "n_best": n_best,
            "min_conf": min_conf,
            "use_hotwords": use_hotwords,
            "use_canon": use_canon,
            "use_strip_disfluencies": use_strip_disfluencies,
            "severity": severity,
        },
    }
    return final

def _add_run_history(entry: Dict[str, Any]) -> None:
    st.session_state.setdefault("run_history", []).append(entry)


def _last_result() -> Optional[Dict[str, Any]]:
    hist = st.session_state.get("run_history", [])
    return hist[-1] if hist else None

def _stats_from_history(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {"total": 0, "ESR": 0.0, "parsed": 0.0, "changed": 0.0,
                "applied": 0.0, "ask_back": 0.0}

    parsed_ct  = sum(1 for r in rows if r.get("parsed") is not None)
    changed_ct = sum(1 for r in rows if bool(r.get("changed")))
    ok_ct      = sum(1 for r in rows
                     if r.get("verify", {}).get("returncode", 1) == 0
                     and bool(r.get("changed")))
    applied_ct = sum(1 for r in rows if r.get("action") == "applied")
    ask_ct     = sum(1 for r in rows if r.get("action") == "ask_back")

    return {
        "total": total,
        "ESR": ok_ct / total,
        "parsed": parsed_ct / total,
        "changed": changed_ct / total,
        "applied": applied_ct / total,
        "ask_back": ask_ct / total,
    }

# def _stats_from_history(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
#     total = len(rows)
#     if total == 0:
#         return {"total": 0, "ESR": 0.0, "parsed": 0.0, "changed": 0.0}
#     parsed = sum(1 for r in rows if r.get("parsed") or r.get("diag"))  # parsed present implies attempted
#     changed = sum(1 for r in rows if r.get("changed"))
#     ok = sum(1 for r in rows if r.get("verify", {}).get("returncode", 1) == 0 and (r.get("changed", False)))
#     return {
#         "total": total,
#         "ESR": ok / total,
#         "parsed": parsed / total,
#         "changed": changed / total,
#     }

from s2c.asr_whisper import decode_whisper  # uses wav_path on disk

def step_decode_audio(*,
                      backend: str,
                      wav_bytes: bytes | None,
                      text_cmd: str | None,
                      n_best: int,
                      hotwords: set[str] | None,
                      use_dec_prompt: bool,
                      initial_prompt: str | None,
                      vad_ms: int) -> dict:
                      
    """
    Decode audio → return n-best. No canonicalization/parse here.
    """
    t0 = time.time()
    try:
        if backend == "mock":
            if not text_cmd:
                return {"step": "Decode Audio", "ok": False, "reason": "no_text",
                        "message": "No mock text provided."}
            # simple mock n-best
            nbest = [text_cmd.strip()]
            if n_best > 1:
                # tiny perturbations just so user can see a list
                base = text_cmd.strip()
                variants = [
                    base.replace(" default ", " equals "),
                    base.replace(" to ", " 2 "),
                    base.replace(" parameter ", " param "),
                ]
                nbest.extend(variants[:max(0, n_best - 1)])
            return {
                "step": "Decode Audio", "ok": True, "backend": "mock",
                "count": len(nbest), "nbest": nbest,
                "elapsed_ms": int((time.time() - t0) * 1000),
                "details": {                      # NEW: expander payload
                    "nbest": nbest,
                    "backend": "mock",
                    "vad_ms": None,
                },
            }

        # whisper path
        if not wav_bytes:
            return {"step": "Decode Audio", "ok": False, "reason": "no_audio",
                    "message": "No audio captured or uploaded."}

        wav_path = _save_wav_tmp(wav_bytes)
        out = decode_whisper(
            wav_path=wav_path,
            n_best=n_best,
            hotwords=hotwords,
            prompt=(initial_prompt if use_dec_prompt else None,),
            vad_ms=vad_ms,
        )
        nbest = out.get("nbest", [])
        return {
            "step": "Decode Audio", "ok": True, "backend": "whisper",
            "count": len(nbest), "nbest": nbest, "wav_path": wav_path,
            "elapsed_ms": int((time.time() - t0) * 1000),
            "details": {                      # NEW
                "nbest": nbest,
                "backend": "whisper",
                "hotwords_used": bool(hotwords),
                "prompt_used": bool(use_dec_prompt and initial_prompt),
                "vad_ms": vad_ms,
            },
        }
    except Exception as e:
        return {"step": "Decode Audio", "ok": False, "reason": "decode_error",
                "message": str(e), "traceback": traceback.format_exc()}
    
import time

def step_strip_disfluencies(*,
                            nbest: list[str] | None,
                            enable: bool,
                            uni_set: set[str] | None,
                            bi_pairs: set[tuple[str, str]] | None) -> dict:
    """
    Take n-best from ctx and strip disfluencies. If disabled, pass through.
    Returns a dict suitable for Last result step log.
    """
    t0 = time.time()
    if not nbest:
        return {"step": "Strip Disfluencies", "ok": False, "reason": "no_nbest",
                "message": "No decoded hypotheses available. Run 'Decode Audio' first."}

    if not enable:
        # No-op but successful, carry forward unchanged n-best
        return {
            "step": "Strip Disfluencies",
            "ok": True,
            "skipped": True,
            "count": len(nbest),
            "nbest": nbest,
            "elapsed_ms": int((time.time() - t0) * 1000),
            "details": {                      # NEW
                "nbest": nbest,
                "skipped": True,
            },
        }

    before = list(nbest)
    after: list[str] = []
    for s in before:
        out = _strip_disfluencies(s, uni=uni_set, bi=bi_pairs)
        if out:
            after.append(out)

    return {
        "step": "Strip Disfluencies",
        "ok": True,
        "count": len(after),
        "nbest": after,
        "before_nbest": before,
        "dropped": len(before) - len(after),
        "elapsed_ms": int((time.time() - t0) * 1000),
        "details": {                          # NEW
            "before": before,
            "after": after,
            "dropped": len(before) - len(after),
            "rules": {
                "unigrams": sorted(list(uni_set or [])),
                "bigrams": [list(x) for x in (bi_pairs or [])],
            },
        }
    }

def step_canonicalize(
    *,
    nbest: list[str] | None,
    enable: bool,
    cfg: CanonConfig,
) -> dict:
    """
    Apply canonicalization to each hypothesis.
    If disabled, pass-through but still report details.
    """
    t0 = time.time()
    if not nbest:
        return {
            "step": "Canonicalize",
            "ok": False,
            "reason": "no_nbest",
            "message": "No hypotheses available. Run previous steps first.",
            "details": {},
        }

    before = list(nbest)

    if not enable:
        return {
            "step": "Canonicalize",
            "ok": True,
            "skipped": True,
            "count": len(before),
            "nbest": before,
            "elapsed_ms": int((time.time() - t0) * 1000),
            "details": {
                "nbest": before,
                "skipped": True,
                "cfg": asdict(cfg) if is_dataclass(cfg) else vars(cfg),
            },
        }

    after = [canonicalize(s, cfg) for s in before]
    changed = sum(1 for a, b in zip(after, before) if a != b)

    return {
        "step": "Canonicalize",
        "ok": True,
        "count": len(after),
        "nbest": after,
        "before_nbest": before,
        "changed": changed,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "details": {
            "before": before,
            "after": after,
            "changed": changed,
            "cfg": asdict(cfg) if is_dataclass(cfg) else vars(cfg),
        },
    }

def step_parse_rank(
    *,
    nbest: list[str] | None,
    hotwords: set[str],
    min_conf: float,
    canon_cfg: CanonConfig | None = None,
) -> dict:
    """
    Parse & rank current n-best WITHOUT re-canonicalizing (disable_canon=True).
    Returns diagnostics and chosen parse with confidence.
    """
    t0 = time.time()
    if not nbest:
        return {
            "step": "Parse & Rank",
            "ok": False,
            "reason": "no_nbest",
            "message": "No hypotheses available. Run previous steps first.",
            "details": {},
        }

    parsed, diag = intent_parse_and_rank(
        nbest,
        hotwords,
        disable_canon=True,          # <-- do NOT canonicalize again
        canon_config=canon_cfg,      # (not used when disable_canon=True, but passed for traceability)
    )

    if not parsed:
        return {
            "step": "Parse & Rank",
            "ok": False,
            "reason": "no_parse",
            "message": "No candidate parsed into a command.",
            "details": diag,
            "elapsed_ms": int((time.time() - t0) * 1000),
        }

    conf = float(diag.get("confidence", 0.0))
    action = "accept" if conf >= min_conf else "ask_back"
    chosen = diag.get("chosen") or {}

    op = parsed.get("op")
    target = parsed.get("function") or parsed.get("target") or "-"
    msg = f"parsed op={op}, target={target}, conf={conf:.2f} → {action}"

    return {
        "step": "Parse & Rank",
        "ok": True,
        "message": msg,
        "confidence": conf,
        "action": action,
        "parsed": parsed,
        "chosen_index": chosen.get("i"),
        "elapsed_ms": int((time.time() - t0) * 1000),
        "details": {
            "diagnostics": diag,       # includes candidates, buckets, scores, confidence
            "parsed": parsed,
            "action": action,
            "min_conf": min_conf,
        },
    }

import time, traceback
from pathlib import Path

def step_test_edit(
    *,
    parsed: dict | None,
    action: str | None,
    confidence: float | None,
    min_conf: float,
    target_file: Path,
    project_root: Path,
    timeout_s: int,
) -> dict:
    """
    Apply the parsed edit ephemerally, run impacted tests, and restore the file.
    If action == 'ask_back' (low confidence), do not apply; return an explanatory result.
    """
    t0 = time.time()

    if not parsed:
        return {
            "step": "Test & Edit",
            "ok": False,
            "reason": "no_parsed",
            "message": "No parsed command. Run 'Parse & Rank' first.",
            "details": {},
        }

    if (action == "ask_back") or (confidence is not None and confidence < min_conf):
        return {
            "step": "Test & Edit",
            "ok": False,
            "reason": "low_conf",
            "message": f"Confidence {confidence:.2f if confidence is not None else 0.0} < min_conf {min_conf:.2f}. Not applying.",
            "details": {
                "parsed": parsed,
                "confidence": confidence,
                "min_conf": min_conf,
                "action": action,
            },
        }

    try:
        before = _read_text(target_file)
        after = _apply_edit(before, parsed)
    except NoWrapTargetFound as e:
        return {
            "step": "Test & Edit",
            "ok": False,
            "reason": "no_overlap",
            "message": str(e),
            "details": {
                "parsed": parsed,
                "requested_lines": [parsed.get("start_line"), parsed.get("end_line")],
                "function": parsed.get("function"),
            },
        }
    except Exception as e:
        return {
            "step": "Test & Edit",
            "ok": False,
            "reason": "apply_error",
            "message": f"Apply error: {e}",
            "details": {"error": str(e), "traceback": traceback.format_exc()},
        }

    changed = (after != before)
    impacted = _guess_impacted_functions(parsed, before)
    tests = _select_tests_by_functions(project_root, impacted) or ["tests"]

    try:
        if changed:
            _write_text(target_file, after)
        verify = run_tests(tests, timeout_s=timeout_s)
    finally:
        if changed:
            _write_text(target_file, before)

    ok = (verify.get("returncode", 1) == 0)
    passed = verify.get("passed", 0)
    total = verify.get("total", 0)
    frac = verify.get("pass_frac", 0.0)

    msg = f"{'changed & ' if changed else 'no change; '}tests: {passed}/{total} passed ({frac:.3f})"

    details = {
        "parsed": parsed,
        "impacted_functions": sorted(list(impacted)),
        "selected_tests": tests,
        "verify": verify,
        "before_code": before,
    }

    if ok and changed:
        details["after_code"] = after
        details["diff"] = _unified_diff(before, after, str(target_file))
    else:
        # do NOT expose after_code/diff on failure
        details["after_code"] = None
        details["diff"] = ""

    return {
        "step": "Test & Edit",
        "ok": ok,
        "message": msg,
        "changed": changed,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "details": details,
    }

# --------------------------- UI ---------------------------

st.set_page_config(page_title="Voice Coding Study", layout="wide")
# Compact the top whitespace
st.markdown("""
<style>
/* Shrink the top padding of the main block */
.block-container { padding-top: 0.5rem !important; padding-bottom: 1rem !important; }

/* Keep the header but make it minimal height */
[data-testid="stHeader"] { height: 0px; background: transparent; }

/* Optional: keep toolbar but tuck it in */
[data-testid="stToolbar"] { right: 0.5rem; top: 0.25rem; }

/* Optional: also tighten the sidebar's top padding */
[data-testid="stSidebar"] section { padding-top: 0.5rem !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='margin:0'>Voice Coding Study</h1>", unsafe_allow_html=True)

st.markdown("""
<style>
.panel {
  background: #f8f9fb;            /* same feel as your code block */
#   border: 1px solid #e6e9f5;
#   border-radius: 10px;
  padding: 10px 12px;
}
/* make code block transparent inside our panel so colors don't double up */
.panel [data-testid="stCodeBlock"] {
  background: transparent !important;
  border: 0 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Remove horizontal padding entirely */
/* Reduce space under the big title */
[data-testid="stAppViewContainer"] h1 {
  margin-bottom: 0.35rem !important;   /* was ~1+rem */
}
            
[data-testid="stAppViewContainer"] .main { padding-left: 0 !important; padding-right: 0 !important; }
[data-testid="stAppViewContainer"] .main .block-container {
  padding-left: 0 !important; padding-right: 0 !important;
  max-width: 100% !important;
}
/* 2) Tighten per-column inner padding */
div[data-testid="column"] {
  padding-left: 0.5rem !important;   /* tweak: 0 – 0.75rem */
  padding-right: 0.5rem !important;
}

/* Optional: shrink the top padding as well */
.block-container { padding-top: 0.5rem !important; }

</style>
""", unsafe_allow_html=True)



# st.title("🧪 Voice Coding Study")


# with st.sidebar:
#     st.header("Settings")
#     target_path = st.text_input("Target file", value=str(Path("src/s2c/user_solution.py")))
#     timeout_s = st.number_input("Pytest timeout (s)", value=5, min_value=1, step=1)
#     backend = st.selectbox("ASR backend", ["mock", "whisper"], index=0)
#     n_best = st.slider("n-best", 1, 8, 5)
#     min_conf = st.slider("Min confidence to apply", 0.0, 1.0, 0.7, 0.05)
#     use_hot = st.checkbox("Use hotwords", value=True)
#     use_canon = st.checkbox("Use canonicalization", value=True)
#     severity = st.slider("Mock severity", 0.0, 1.0, 0.3, 0.1)
#     if backend == "whisper" and audio_recorder is None:
#         st.caption("Install mic widget: pip install audio-recorder-streamlit")
#     st.caption("Tip: set env S2C_WHISPER_MODEL=base.en or small.en for whisper.")

with st.sidebar:
    st.header("Settings")

    # Basic
    target_path = st.text_input("Target file", value=str(Path("src/s2c/user_solution.py")))
    timeout_s = st.number_input("Pytest timeout (s)", value=5, min_value=1, step=1)

    target_file = Path(target_path)
    code_txt = _numbered(_read_text(target_file))
    panel_h = 380

    # --- ASR backend & backend-specific knobs ---
    backend = st.selectbox("ASR backend", ["mock", "whisper"], index=1)

    # 1) Mock Severity just under backend (visible only for Mock)
    if backend == "mock":
        severity = st.slider("Mock severity", 0.0, 1.0, 0.3, 0.1, help="Noise level for mock ASR")
        use_dec_prompt = False
        dec_prompt_text = None
    else:
        severity = 0.0  # not used by whisper
        # 2) Whisper 'Decoding Prompt' toggle + editable text
        st.markdown("**Whisper decoding**")
        use_dec_prompt = st.checkbox(
            "Use decoding prompt",
            value=True,
            help="Include an initial prompt to bias decoding (keeps numbers/terms).",
        )
        default_prompt = (
            "coding edits. numbers: one two three four five six seven eight nine ten "
            "1 2 3 4 5 6 7 8 9 10. "
            "terms: word_count min_count default rename wrap try except. "
            "examples: add param min_count default 2 to word_count; "
            "rename counts to total_counts."
        )
        dec_prompt_text = None
        if use_dec_prompt:
            dec_prompt_text = st.text_area("Initial prompt", value=default_prompt, height=80)
        if decode_whisper is None:
            st.caption("Install faster-whisper to enable the Whisper backend.")
        
        # NEW: VAD slider (min silence for VAD segmentation)
        vad_ms = st.slider("VAD (ms)", min_value=10, max_value=1000, value=400, step=10,
                   help="Voice Activity Detection: minimum silence duration (ms) between segments.")


    # Shared ASR knobs
    n_best = st.slider("n-best", 1, 8, 5)
    min_conf = st.slider("Min confidence to apply", 0.0, 1.0, 0.7, 0.05)

    use_strip = st.checkbox("Strip disfluencies", value=True)
    strip_uni_set: Optional[Set[str]] = None
    strip_bi_pairs: Optional[Set[Tuple[str, str]]] = None
    if use_strip:
        uni_default = " ".join(sorted(_DISFLUENCY_UNI))
        uni_text = st.text_area(
        "Disfluency unigrams",
        value=uni_default,
        height=60,
        help="Comma or space-separated; case-insensitive. E.g., 'uh, um, hmm'",
        )
        strip_uni_set = {tok.lower() for tok in re.split(r"[\s,]+", uni_text) if tok.strip()}

    bi_default = "\n".join([" ".join(p) for p in _DISFLUENCY_BI])
    bi_text = st.text_area(
        "Disfluency bigrams (one per line)",
        value=bi_default,
        height=80,
        help="Each line: two words, e.g., 'you know'",
    )
    strip_bi_pairs = set()
    for line in bi_text.splitlines():
        parts = [t for t in line.strip().split() if t]
        if len(parts) == 2:
            strip_bi_pairs.add((parts[0].lower(), parts[1].lower()))    

    # 3) Use Hotwords + editable hotword list
    use_hot = st.checkbox("Use hotwords", value=True)
    hotwords_override: Optional[Set[str]] = None
    if use_hot:
        try:
            # pre-fill from repo; user can edit freely
            hot_preview = " ".join(sorted(list(_hotwords_for_repo(Path(target_path)))))[:1000]
        except Exception:
            hot_preview = ""
        hotwords_text = st.text_area(
            "Hotwords (editable)",
            value=hot_preview,
            height=80,
            help="Space or comma-separated terms used for ASR bias and ranking.",
        )
        hotwords_override = set(t for t in re.split(r"[\s,]+", hotwords_text) if t)

    # 4) Canonicalization + sub-knobs
    use_canon = st.checkbox("Use canonicalization", value=True)
    canon_knobs = {
        "keyword_canon": True,
        "canon_punct": True,
        "number_words": True,
        "join_after_keyword": True,
        "homophone_repair": True,
    }
    if use_canon:
        st.caption("Canonicalization knobs")
        canon_knobs["keyword_canon"] = st.checkbox("Keyword canonicalization", value=True, key="canon_keywords")
        canon_knobs["number_words"] = st.checkbox("Number words → digits", value=True, key="canon_numbers")
        canon_knobs["canon_punct"]  = st.checkbox("Punctuation word join", value=True, key="canon_punct")
        canon_knobs["join_after_keyword"] = st.checkbox("Join after keyword", value=True, key="canon_join")
        canon_knobs["homophone_repair"] = st.checkbox("Homophone repair", value=True, key="canon_homophone")

    if backend == "whisper" and audio_recorder is None:
        st.caption("Install mic widget: pip install audio-recorder-streamlit")
    st.caption("Tip: set env S2C_WHISPER_MODEL=base.en or small.en for whisper.")

# Panels layout
# col_code, col_cmd = st.columns([2, 1.4])
# col_status, col_metrics = st.columns([1.4, 1])

row1_code_col, row1_result_col = st.columns([2, 1.4])  # Row 1
row2_command_col, row2_examples_col = st.columns([2, 1.4]) # Row 2
# cmd_row = st.container()                                            # Row 2
metrics_row = st.container()                                        # Row 3

# --- Code Panel (line numbers + before/after toggle; 'after' first & default) ---
with row1_code_col:
    c_label, c_radio, _spacer = st.columns([0.4, 0.4, 0.2], gap="small")

    with c_label:
        st.markdown("<h3 style='margin:0; line-height:1.5'>Code (read-only)</h3>", unsafe_allow_html=True)

    with c_radio:
        # tiny vertical nudge so it lines up nicely with the h3
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        choice = st.radio(
            label="View Mode",
            options=["after", "before"],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
            key="view_mode_code",
        )

    def _is_success(res: dict | None) -> bool:
        if not res:
            return False
        v = res.get("verify") or {}
        return v.get("returncode", 1) == 0

    last = _last_result()
    ctx = st.session_state.get("step_ctx", {})

    # Decide source + caption exactly once
    if choice == "after":
        # Show after only when tests passed.
        if last and _is_success(last) and last.get("after_code"):
            code_src = last["after_code"]
            caption  = "Preview: after-edit code (tests passed)."
        elif ctx.get("ok") and ctx.get("after_code"):
            code_src = ctx["after_code"]
            caption  = "Preview (step mode): after-edit code (tests passed)."
        else:
            # Failure or no after → fall back to unmodified file
            code_src = _read_text(target_file)
            caption  = "Last edit failed (or no edit) — showing unmodified file."
    else:
        code_src = _read_text(target_file)
        caption = None

    # Render once
    st.code(_numbered(code_src), language="python", height=panel_h)
    # if caption:
    #     st.caption(caption)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # st.markdown("</div>", unsafe_allow_html=True)

    # if target_file.exists():
    #     if choice == "after" and last and last.get("after_code"):
    #         st.code(_numbered(last["after_code"]), language="python")
    #         st.caption("Preview: after-edit code (not saved; file on disk was restored after tests).")
    #     else:
    #         st.code(code_txt, language="python")
    #         # if choice == "after":
    #         #     st.caption("No previous edit yet — showing on-disk code.")
    # else:
    #     st.error(f"Target not found: {target_file}")
   

# --- Command Panel ---
with row2_command_col:
    text_cmd: Optional[str] = None
    wav_bytes: Optional[bytes] = None

    # --- Command header row (title + Use Microphone aligned) ---
    cmd_h_left, cmd_h_right = st.columns([0.30, 0.70], gap="small")
    with cmd_h_left:
        st.markdown("<h3 style='margin:0; line-height:1.1'>Command</h3>", unsafe_allow_html=True)
    with cmd_h_right:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)  # small vertical nudge
        use_mic = st.checkbox("Use Microphone", value=(backend != "mock" and audio_recorder is not None), key="use_mic_cb")

    # --- Command body: two columns ---
    # cmd_left, cmd_right = st.columns([0.99, 0.01], gap="large")

    custom_css = """
    .st-key-cmd_container {
        background-color: #ffffff;
        border: none;
    }
    """
    st.html(f"<style>{custom_css}</style>")

    # Single, scrollable panel
    with st.container(key="cmd_container"):
    # with cmd_left:
        if backend == "mock":
            text_cmd = st.text_input(
                "Command (mock ASR)",
                value="add parameter min count default two to word count",
                key="cmd_text_mock",
            )
        else:
            if use_mic and audio_recorder is not None:
                st.caption("Click to record or re-record.  Stay silent or click again to stop recording.")

                # one-time init
                if "last_wav" not in st.session_state:
                    st.session_state.last_wav = None

                rec_col, play_col = st.columns([0.35, 0.65], gap="small")

                with rec_col:
                    audio = audio_recorder(
                        text="🎙️ Record / Stop",
                        icon_size="2x",
                        pause_threshold=1.0,
                        sample_rate=16000,
                        key="rec_btn",
                    )

                with play_col:
                    player = st.empty()  # placeholder lives in the right column
                    # show previous recording (if any) so layout is stable
                    if st.session_state.last_wav:
                        player.audio(st.session_state.last_wav, format="audio/wav")

                # when a new recording arrives, update both state and the right-column player
                if audio:
                    st.session_state.last_wav = audio
                    player.audio(audio, format="audio/wav")


                # audio = audio_recorder(
                #     text="🎙️ Record / Stop",
                #     icon_size="2x",
                #     pause_threshold=1.0,
                #     sample_rate=16000,
                # )
                # if audio:
                #     wav_bytes = audio
                #     st.audio(wav_bytes, format="audio/wav")
            else:
                wav_file = st.file_uploader(
                    "Upload audio (WAV/MP3/M4A) for Whisper",
                    type=["wav", "mp3", "m4a"],
                    key="cmd_audio_upload",
                )
                if wav_file is not None:
                    wav_bytes = wav_file.read()

            # Optional typed command fallback (useful when ASR is off)
            # text_cmd = st.text_input(
            #     "Type a command (optional)",
            #     value="",
            #     key="cmd_text_fallback",
            #     placeholder="e.g., add param min_count default 2 to word_count",
            # )

        # Run button (left column)
        # run = st.button("▶️ Run edit", type="primary", use_container_width=True, key="run_edit_btn")

        # --- ROW 2: Run edit  →  Run step  →  [bold next step] (same line) ---
        # inside your Command left column, on the second row:
        # Buttons row: Run edit | Run step | Reset
        b_run, b_step, b_reset = st.columns([0.18, 0.25, 0.12], gap="small")

        with b_run:
            run = st.button("▶️ Run edit", type="primary", key="run_edit_btn")

        with b_step:
            step_label = STEPS[min(st.session_state.step_i, len(STEPS) - 1)]
            step_clicked = st.button(f"▶️ Run step: {step_label}", type="primary", key="run_step_btn")

        with b_reset:
            reset_clicked = st.button("↩️ Reset", key="reset_btn", help="Reset to first step and clear results")

        canon_cfg = None
        if use_canon:
            canon_cfg = CanonConfig(
                join_after_keyword=canon_knobs["join_after_keyword"],
                number_words=canon_knobs["number_words"],
                keyword_canon=canon_knobs["keyword_canon"],
                homophone_repair=canon_knobs["homophone_repair"],
            )

        def _clear_results():
            st.session_state.step_i = 0
            st.session_state.step_log = []
            st.session_state.step_ctx = {}
            st.session_state.run_history = []
            # Optionally also clear last recorded audio:
            st.session_state.last_wav = None

        if reset_clicked:
            _clear_results()
            st.rerun()

        if run:
            st.session_state.step_log = []
            st.session_state.step_ctx = {}
            st.session_state.step_i = 0
            # also clear any previous run-step view to avoid mixing
            st.session_state.run_history = []

            result: Dict[str, Any] | None = None
            try:
                # result = _one_run(
                #     backend=backend,
                #     text_cmd=text_cmd,
                #     wav_bytes=wav_bytes,
                #     n_best=n_best,
                #     min_conf=min_conf,
                #     use_hotwords=use_hot,
                #     use_canon=use_canon,
                #     severity=severity,
                #     target_file=target_file,
                #     timeout_s=timeout_s,
                #     hotwords_override=hotwords_override,    # NEW
                #     use_dec_prompt=use_dec_prompt,          # NEW
                #     initial_prompt=dec_prompt_text,         # NEW
                #     strip_uni=strip_uni_set,
                #     strip_bi=strip_bi_pairs,
                #     vad_ms=vad_ms,
                # )

                result = _one_run_pipeline(
                    backend=backend,
                    text_cmd=text_cmd,
                    wav_bytes=wav_bytes or st.session_state.last_wav,
                    n_best=n_best,
                    min_conf=min_conf,
                    use_hotwords=use_hot,
                    use_canon=use_canon,
                    use_strip_disfluencies=use_strip,  # <- your toggle
                    severity=severity,
                    target_file=target_file,
                    timeout_s=timeout_s,
                    hotwords_override=hotwords_override,
                    use_dec_prompt=use_dec_prompt,
                    initial_prompt=dec_prompt_text,
                    strip_uni=strip_uni_set,
                    strip_bi=strip_bi_pairs,
                    vad_ms=vad_ms,
                    canon_cfg=canon_cfg,
                    project_root=None,  # or cache st.session_state.root if you want
                )

                _add_run_history(result)
                st.rerun()
            except Exception as e:
                st.exception(e)

        # advance/reset stepper (UI only for now)
        if step_clicked:
            st.session_state.run_history = []

            current = STEPS[st.session_state.step_i]

            if current == "Decode Audio":
                res = step_decode_audio(
                    backend=backend,
                    wav_bytes=wav_bytes or st.session_state.last_wav,  # from recorder or last
                    text_cmd=text_cmd,
                    n_best=n_best,
                    hotwords=(hotwords_override if use_hot else None),
                    use_dec_prompt=use_dec_prompt,
                    initial_prompt=dec_prompt_text,
                    vad_ms=vad_ms,
                )
                _push_step_result(res)
                if res.get("ok"):
                    # carry forward to next step
                    st.session_state.step_ctx["nbest"] = res.get("nbest", [])
                    st.session_state.step_ctx["wav_path"] = res.get("wav_path")
                    st.session_state.step_i = min(st.session_state.step_i + 1, len(STEPS) - 1)
                    st.rerun()
                else:
                    # reset to first step on failure
                    st.session_state.step_ctx.clear()
                    st.session_state.step_i = 0
                    st.rerun()

            elif current == "Strip Disfluencies":
                # Step view is exclusive with full-run
                st.session_state.run_history = []

                src = st.session_state.step_ctx.get("nbest")
                res = step_strip_disfluencies(
                    nbest=src,
                    enable=use_strip,   # ← your Settings toggle
                    uni_set=strip_uni_set,           # ← your Settings set
                    bi_pairs=strip_bi_pairs,         # ← your Settings set
                )
                _push_step_result(res)

                if res.get("ok"):
                    # carry forward the updated n-best (or pass-through if skipped)
                    st.session_state.step_ctx["nbest"] = res.get("nbest", src)
                    st.session_state.step_i = min(st.session_state.step_i + 1, len(STEPS) - 1)
                    st.rerun()   # refresh so the button label advances immediately
                else:
                    st.session_state.step_ctx.clear()
                    st.session_state.step_i = 0
                    st.rerun()

            elif current == "Canonicalize":
                # step mode is exclusive with full-run
                st.session_state.run_history = []

                src = st.session_state.step_ctx.get("nbest")
                res = step_canonicalize(
                    nbest=src,
                    enable=use_canon,     # your “Use Canonicalization” master toggle
                    cfg=canon_cfg,        # from Settings above
                )
                _push_step_result(res)

                if res.get("ok"):
                    st.session_state.step_ctx["nbest"] = res.get("nbest", src)
                    st.session_state.step_i = min(st.session_state.step_i + 1, len(STEPS) - 1)
                    st.rerun()  # immediately update “Run step: …” label
                else:
                    st.session_state.step_ctx.clear()
                    st.session_state.step_i = 0
                    st.rerun()

            elif current == "Parse & Rank":
                st.session_state.run_history = []  # step mode is exclusive with full-run

                src = st.session_state.step_ctx.get("nbest")
                res = step_parse_rank(
                    nbest=src,
                    hotwords=(hotwords_override if use_hot else set()),
                    min_conf=min_conf,          # from your Settings
                    canon_cfg=canon_cfg,        # your CanonConfig from Settings (for tracing)
                )
                _push_step_result(res)

                if res.get("ok"):
                    # carry forward for the next step
                    st.session_state.step_ctx["parsed"] = res.get("parsed")
                    st.session_state.step_ctx["confidence"] = res.get("confidence")
                    st.session_state.step_ctx["diag"] = res.get("details", {}).get("diagnostics")
                    st.session_state.step_ctx["action"] = res.get("action")

                    st.session_state.step_i = min(st.session_state.step_i + 1, len(STEPS) - 1)
                    st.rerun()
                else:
                    st.session_state.step_ctx.clear()
                    st.session_state.step_i = 0
                    st.rerun()

            elif current == "Test & Edit":
                st.session_state.run_history = []  # keep step mode exclusive

                ctx = st.session_state.step_ctx
                res = step_test_edit(
                    parsed=ctx.get("parsed"),
                    action=ctx.get("action"),
                    confidence=ctx.get("confidence"),
                    min_conf=min_conf,                # from Settings
                    target_file=target_file,
                    project_root=_project_root(),
                    timeout_s=timeout_s,
                )
                _push_step_result(res)

                # For code preview in step mode (optional but handy)
                det = res.get("details", {})
                if det.get("after_code"):
                    st.session_state.step_ctx["after_code"] = det["after_code"]
                    st.session_state.step_ctx["before_code"] = det.get("before_code")
                    st.session_state.step_ctx["ok"] = True
                else:
                    st.session_state.step_ctx.pop("after_code", None)
                    st.session_state.step_ctx["ok"] = False

                # Reset to first step after finishing (or on failure)
                st.session_state.step_i = 0
                st.rerun()

            else:
                # placeholder for later steps
                _push_step_result({"step": current, "ok": False, "message": "Step not implemented yet."})
                st.session_state.step_i = 0
                st.rerun()


        # Small help text
        st.caption("Say things like: 'rename total to running_total in function accumulate'.")

# with col_cmd:
#     st.subheader("Command")
#     text_cmd: Optional[str] = None
#     wav_bytes: Optional[bytes] = None

#     if backend == "mock":
#         text_cmd = st.text_input("Type a command (mock ASR)", "add parameter min count default two to word count")
#     else:
#         use_mic = st.checkbox("Use microphone", value=(audio_recorder is not None))
#         if use_mic and audio_recorder is not None:
#             st.caption("Click once to start, again to stop. The captured audio is held locally in the browser.")
#             audio = audio_recorder(pause_threshold=1.0, sample_rate=16000, text="🎙️ Record / Stop")
#             if audio:
#                 wav_bytes = audio
#                 st.audio(wav_bytes, format="audio/wav")
#         else:
#             wav_file = st.file_uploader("Upload WAV/MP3 for whisper", type=["wav", "mp3", "m4a"])
#             if wav_file is not None:
#                 wav_bytes = wav_file.read()

#     run = st.button("▶️ Run edit")

#     # --- Examples panel (scrollable) ---
#     examples_html = """
#     <div style='max-height: 180px; overflow:auto; border:1px solid #ddd; padding:8px; border-radius:6px; background:#fafafa;'>
#       <div style='font-weight:600; margin-bottom:6px;'>Command examples</div>
#       <ul style='margin:0 0 6px 18px;'>
#         <li><em>add param</em> min_count <em>default</em> 2 <em>to</em> word_count</li>
#         <li><em>rename</em> counts <em>to</em> total_counts <em>in function</em> word_count</li>
#         <li><em>wrap</em> <em>lines</em> 3 <em>to</em> 5 <em>in</em> <em>try except</em> <em>in function</em> word_count</li>
#       </ul>
#       <div style='font-size:12px;color:#666;'>Tip: line numbers are shown in the code pane.</div>
#     </div>
#     """
#     st.markdown(examples_html, unsafe_allow_html=True)

# --- Execute & Status Panel ---

with row2_examples_col:
    # st.markdown("#### Command examples")
    st.markdown(
        """
        <div style='max-height: 220px; overflow:auto; border:1px solid #ddd; padding:8px; border-radius:6px; background:#fafafa;'>
            <div style='font-weight:600; margin-bottom:6px;'>Command examples</div>
            <ul style='margin:0 0 6px 18px;'>
            <li><em>add param</em> min_count <em>default</em> 2 <em>to</em> word_count</li>
            <li><em>rename</em> counts <em>to</em> total_counts <em>in function</em> word_count</li>
            <li><em>wrap</em> <em>lines</em> 3 <em>to</em> 5 <em>in</em> <em>try except</em> <em>in function</em> word_count</li>
            </ul>
            <div style='font-size:12px;color:#666;'>Tip: line numbers are shown in the code pane.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metrics_row:
    st.subheader("Metrics (session)")
    hist = st.session_state.get("run_history", [])
    s = _stats_from_history(hist)

    # Assume you already compute these in session state or just-in-time:
    # runs, esr, parsed_rate, changed_rate, askback_rate
    # If some are missing, keep the names but use your actual variables.

    m1, m2, m3, m4, m5 = st.columns([1, 1, 1, 1, 1], gap="large")
    with m1:
        st.metric("Runs", len(hist))
    with m2:
        st.metric("ESR", f"{s['ESR']:.2f}")
    with m3:
        st.metric("Parsed rate", f"{s['parsed']:.2f}")
    with m4:
        st.metric("Changed rate", f"{s['changed']:.2f}")
    with m5:
        st.metric("Ask-back rate", f"{s['ask_back']:.2f}")

    # st.subheader("Metrics (session)")
    # hist = st.session_state.get("run_history", [])
    # st.metric("Runs", len(hist))
    # s = _stats_from_history(hist)
    # st.metric("ESR", f"{s['ESR']:.2f}")
    # st.metric("Parsed rate", f"{s['parsed']:.2f}")
    # st.metric("Changed rate", f"{s['changed']:.2f}")
    # st.metric("Ask-back rate", f"{s['ask_back']:.2f}")

    # if hist:
    #     with st.expander("Raw history JSON"):
    #         st.code(json.dumps(hist, indent=2), language="json")

with row1_result_col:
    custom_css = """
    .st-key-last_results_container {
        background-color: #f8f9fb;
        border: none;
    }
    """
    st.markdown("<h3 style='margin:0; line-height:1.5'>Last result</h3>", unsafe_allow_html=True)
    st.html(f"<style>{custom_css}</style>")

    hist = st.session_state.get("run_history", [])
    last = hist[-1] if hist else None
    step_log = st.session_state.get("step_log", [])

    # Single, scrollable panel
    with st.container(key="last_results_container", height=panel_h):

        if step_log:
            # Show the whole log (oldest to newest)
            for i, entry in enumerate(step_log, start=1):
                ok = bool(entry.get("ok", False))
                icon = "✅" if ok else "❌"
                step = entry.get("step", "?")
                elapsed = entry.get("elapsed_ms")
                elapsed_txt = f" <span style='color:#999'>({elapsed} ms)</span>" if elapsed is not None else ""

                # Friendly message
                # if ok and step == "Decode Audio":
                #     msg = f"{entry.get('count', 0)} hypotheses"
                # else:
                #     msg = entry.get("message") or ("Done" if ok else "Failed")

                # st.markdown(f"{i}. {icon} **{step}** — {msg}{elapsed_txt}", unsafe_allow_html=True)

                # Decode Audio → show nbest expander
                if ok and step == "Decode Audio":
                    msg = f"{entry.get('count', 0)} hypotheses"
                    # with st.expander("Decoded n-best (click to open)", expanded=False):
                    #     st.json({"nbest": entry.get("nbest", [])})
                elif ok and step == "Strip Disfluencies":
                    if entry.get("skipped"):
                        msg = f"skipped (pass-through) — {entry.get('count', 0)} hypotheses"
                    else:
                        msg = f"{entry.get('count',0)} kept, {entry.get('dropped',0)} dropped"
                else:
                    msg = entry.get("message") or ("Done" if ok else "Failed")

                st.markdown(f"{i}. {icon} **{step}** — {msg}{elapsed_txt}", unsafe_allow_html=True)

                # Uniform details expander (closed by default)
                details = entry.get("details")
                if details is None:
                    # fallback details for known steps, so every step shows something
                    if step == "Decode Audio" and ok:
                        details = {"nbest": entry.get("nbest", [])}
                    elif step == "Strip Disfluencies" and ok:
                        details = {
                            "before": entry.get("before_nbest", []),
                            "after": entry.get("nbest", []),
                            "dropped": entry.get("dropped", 0),
                        }
                    else:
                        details = {k: v for k, v in entry.items() if k not in ("traceback",)}

                with st.expander(f"Details — {step}", expanded=False):
                    st.json(details)

            # Thin divider before any full-run output
            # st.markdown("<hr style='border:none;height:1px;background:#e9ecf4;margin:10px 0'/>",
            #             unsafe_allow_html=True)

        # Full "Run edit" summary (if present)
        if last and not step_log:
            st.markdown(_run_summary(last), unsafe_allow_html=True)
            with st.expander("Full JSON"):
                st.json(last)
        elif not step_log and not last:
            st.info("Run Edit or Run Step to see status.")

st.caption("Local, single-user demo UI. All edits are ephemeral and restored after tests.")
