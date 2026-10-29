# ForgeLens

An explainable forgery detector for UPI payment screenshots.

**ForgeLens raises suspicion. It never confirms a payment.**
Only the receiver's bank or UPI app can confirm one.

## Status

| Phase | What | Notes |
|---|---|---|
| 0 | Project setup + PII redaction | [docs/phase-0.md](docs/phase-0.md) |

Planned evidence types:
1. Edits to real screenshots: copy-move, noise inconsistency, ELA (only for recompressed JPEGs)
2. Template mismatch: keypoints + homography against genuine app screenshots
3. OCR field checks

## Quickstart (macOS)

```bash
brew install tesseract        # Ubuntu: sudo apt-get install tesseract-ocr
uv sync                       # Python 3.12 + pinned dependencies from uv.lock
uv run pytest                 # ~15 s on CPU; no GPU needed
```

No uv? `pip install -r requirements.txt && pip install -e .` also works (Python 3.12).

## Redact your screenshots

```bash
mkdir -p data/raw data/private
cp ~/Desktop/Screenshot*.png data/raw/
echo "Your Name" >> data/private/names.txt     # optional, recommended
uv run forgelens-redact                         # data/raw -> data/redacted
open data/private/review/                       # check every sheet by eye
```

Details, numbers and limits: [docs/phase-0.md](docs/phase-0.md).
What lives where in `data/`: [data/README.md](data/README.md).

## Layout

```
src/forgelens/
  ocr.py        tesseract wrapper: word boxes, several image views
  pii.py        detection rules (regex, names list, layout heuristics)
  redact.py     pipeline + CLI: detect -> pad -> fill -> verify -> write
  synthetic.py  fake screenshots with known ground truth (tests, demo)
  config.py     YAML defaults < --config < FORGELENS_* env < CLI flags
  device.py     CPU/CUDA/MPS fallback, seeding, environment report
scripts/        reproducible demos; every number in docs/ comes from one
tests/          CPU-only; CI runs them on macOS + Ubuntu
data/           private and gitignored (see data/README.md)
results/        run logs with env.json, gitignored
```

## Principles

- No number without a reproducible script.
- Per-test precision/recall with confidence intervals.
- Classical methods first; deep learning only as a measured baseline.
- Adversarial testing of my own detector.
- Same code on a CPU-only Mac and on GPUs.
- Real screenshots are redacted; forged samples are never published.
