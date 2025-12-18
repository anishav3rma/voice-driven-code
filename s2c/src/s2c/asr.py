from __future__ import annotations
from typing import List, Dict, Set, Optional
import random, re

def _noise_tokens(s: str, rng: random.Random, severity: float) -> str:
    x = s
    if "_" in x and rng.random() < 0.5 * severity:
        x = x.replace("_", " ")
    elif " " in x and rng.random() < 0.3 * severity:
        x = re.sub(r"\s+", "_", x)
    if rng.random() < 0.4 * severity:
        x = x.lower()
    if rng.random() < 0.6 * severity:
        x = re.sub(r"[,:;]", "", x)
    return x

_HOMOS = {
    "two": ["too", "to"],
    "for": ["four"],
    "four": ["for"],
    "param": ["pair am", "para"],
    "function": ["funct shun", "funk shun"],
}

def _homophones(s: str, rng: random.Random, severity: float) -> str:
    x = s.lower()
    for k, alts in _HOMOS.items():
        if k in x and rng.random() < 0.3 * severity:
            x = x.replace(k, rng.choice(alts))
    return x

def decode_mock(transcript: str, n_best: int = 5, hotwords: Optional[Set[str]] = None, severity: float = 0.4, seed: Optional[int] = 13) -> Dict[str, List[str]]:
    severity = max(0.0, min(1.0, severity))
    rng = random.Random(seed)
    outs = []
    outs.append(transcript)
    for _ in range(max(0, n_best - 1)):
        t = transcript
        t = _homophones(t, rng, severity)
        t = _noise_tokens(t, rng, severity)
        if not t in outs:
            outs.append(t)
    # NOT SURE PURPOSE OF THIS CODE
    # if hotwords and not any(any(hw in h for hw in hotwords) for h in outs[:3]):
    #     outs.insert(1, transcript)
    return {"nbest": outs[:n_best]}
