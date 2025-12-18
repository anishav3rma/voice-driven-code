# src/s2c/asr_whisper.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Set

# Single backend: faster-whisper (CTranslate2)
try:
    from faster_whisper import WhisperModel  # type: ignore
except Exception as e:  # fail fast with a clear message
    raise RuntimeError(
        "faster-whisper is not installed. Install with:\n"
        "  pip install faster-whisper\n"
        "Also ensure FFmpeg is available (macOS: `brew install ffmpeg`)."
    ) from e

# Lazily-initialized singleton model
_MODEL: Optional[WhisperModel] = None


def _model_name() -> str:
    # Choose a small English model by default; override via env if needed.
    # Examples: base.en, small.en, medium.en, base, small, medium
    return os.environ.get("S2C_WHISPER_MODEL", "base.en")


def _device() -> str:
    # cpu | cuda | auto  (we default to cpu for portability)
    return os.environ.get("S2C_WHISPER_DEVICE", "cpu")


def _compute_type_for_device(dev: str) -> str:
    # Reasonable defaults; override with S2C_WHISPER_COMPUTE_TYPE if desired.
    override = os.environ.get("S2C_WHISPER_COMPUTE_TYPE")
    if override:
        return override
    if dev == "cpu":
        return "int8"      # fast & good on CPU
    # For GPUs, float16 is typical; bf16 also works on newer GPUs.
    return "float16"


def _temps_from_env(default: List[float]) -> List[float]:
    s = os.environ.get("S2C_WHISPER_TEMPS", "")
    if not s.strip():
        return default
    out: List[float] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out or default


def _maybe_initial_prompt(hotwords: Optional[Set[str]]) -> Optional[str]:
    """
    Basic biasing trick: feed a short list of hotwords as an initial_prompt.
    Set S2C_WHISPER_USE_PROMPT=1 to enable; otherwise return None.
    """
    use = os.environ.get("S2C_WHISPER_USE_PROMPT", "0") in ("1", "true", "True")
    if not use or not hotwords:
        return None
    # keep short to avoid harming decoding; ~50 tokens max
    words = list(sorted(hotwords))[:50]
    return " ".join(words)


def _load_model() -> WhisperModel:
    global _MODEL
    if _MODEL is None:
        name = _model_name()
        dev = _device()
        ct = _compute_type_for_device(dev)
        # Typical CPU-friendly config; adjust via env vars above if needed.
        _MODEL = WhisperModel(
            name,
            device=dev,
            compute_type=ct,
            # You can also set cpu_threads=os.environ.get("S2C_WHISPER_CPU_THREADS", 4)
        )
    return _MODEL


def _dedupe_nbest(nbest: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in nbest:
        key = s.strip()
        if key and key not in seen:
            out.append(s)
            seen.add(key)
    return out

def _prompt_as_str(x):
    """Coerce any UI/pipe value into a plain string prompt (or None)."""
    if x is None:
        return None
    if isinstance(x, str):
        return x
    # If someone accidentally passed a list/tuple, keep only string parts
    if isinstance(x, (list, tuple)):
        parts = [t for t in x if isinstance(t, str)]
        return " ".join(parts) if parts else None
    # Fallback: stringify scalars/objects
    return str(x)

def decode_whisper(
    wav_path: str,
    n_best: int = 5,
    hotwords: Optional[Set[str]] = None,
    prompt: Optional[str] = None,
    vad_ms: int = 400,
) -> Dict[str, List[str]]:
    """
    Decode an audio file with faster-whisper and return {"nbest": [str, ...]}.
    Notes:
      * faster-whisper doesn't expose a general 'n-best list'; we synthesize
        alternatives by re-decoding with higher temperatures.
      * Hotwords aren't directly supported; optionally bias via initial_prompt
        (set env S2C_WHISPER_USE_PROMPT=1 to enable).
    Env knobs:
      - S2C_WHISPER_MODEL           (default: base.en)
      - S2C_WHISPER_DEVICE          (default: cpu)    # cpu | cuda
      - S2C_WHISPER_COMPUTE_TYPE    (default: int8 on cpu, float16 on cuda)
      - S2C_WHISPER_TEMPS           (e.g., "0.0,0.3,0.6,0.8")
      - S2C_WHISPER_USE_PROMPT      (1 to bias with hotwords via initial_prompt)
    """
    wav = str(Path(wav_path))
    n_best = max(1, int(n_best))
    temps = _temps_from_env([0.0, 0.2, 0.4, 0.6, 0.8])[:max(1, n_best)]
    
    # if not prompt:
    #     initial_prompt = _maybe_initial_prompt(hotwords)
    # else:
    #     initial_prompt = prompt

    model = _load_model()
    hyps: List[str] = []
    sanitized_prompt = _prompt_as_str(prompt)

    # Pass 1: deterministic (beam search)
    segments, info = model.transcribe(
        wav,
        language="en",
        task="transcribe",
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=int(vad_ms)),
        initial_prompt=sanitized_prompt,
    )
    txt = "".join(seg.text for seg in segments).strip()
    if txt:
        hyps.append(txt)

    # Additional passes: sampling with higher temperatures (beam_size=1)
    for t in temps[1:]:
        segments, _ = model.transcribe(
            wav,
            beam_size=1,
            temperature=t,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=int(vad_ms)),
            initial_prompt=sanitized_prompt,
        )
        txt2 = "".join(seg.text for seg in segments).strip()
        if txt2:
            hyps.append(txt2)

    # Dedupe, ensure at least one string, pad/truncate to n_best
    hyps = _dedupe_nbest(hyps) or [""]
    # if len(hyps) < n_best:
    #     hyps = hyps + [hyps[0]] * (n_best - len(hyps))
    return {"nbest": hyps[:n_best]}
