# s2c-mvp — Sandbox + Verifier (Milestone M0/M1)

This is the first vertical slice for the Speech-to-Code project: **sandboxed test runner ("verifier")** plus a tiny toy task and tests.

## Prerequisites (macOS/Linux)
- **Python 3.11** (recommended). On macOS: `brew install python@3.11`
- (Later, for ASR) **ffmpeg**: `brew install ffmpeg` — *not needed for this milestone*.

## Setup
```bash
cd s2c-mvp
python3.11 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
```

## Try the verifier
```bash
# Runs tests inside a restricted sandboxed process
python -m s2c.verify --tests tests/test_word_count.py::test_basic
# Or run the whole suite:
python -m s2c.verify
```

You should see a summary like `pass_frac: 1.00` for the included toy task.

## What’s here
- `src/s2c/_sandbox_entry.py` — child process: sets resource limits, disables network, runs pytest.
- `src/s2c/verify.py` — orchestrates a sandboxed test run and returns structured results.
- `tests/test_word_count.py` — toy unit tests for the sample function.
- `src/s2c/sample_word_count.py` — a simple implementation that passes tests (for demo).

## Next steps
- Add more toy tasks + tests under `tests/`.
- Expose `run_tests()` from `s2c.verify` to gate code-gen and edit operations.
