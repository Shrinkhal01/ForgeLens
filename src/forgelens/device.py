"""Device selection, seeding and the environment report saved next to every result.

GPU is an accelerator, never a requirement: everything falls back to CPU.
Run `uv run forgelens-env` to print the report for this machine.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

log = logging.getLogger(__name__)


def get_device() -> str:
    if os.environ.get("FORCE_CPU") == "1":
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"  # Phase 0 needs no torch; deep-learning baselines come later
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def tesseract_version() -> str | None:
    try:
        import pytesseract

        return str(pytesseract.get_tesseract_version())
    except Exception:  # binary missing, not on PATH, or pytesseract missing
        return None


def _pkg(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def env_report() -> dict:
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "machine": platform.machine(),
        "device": get_device(),
        "tesseract": tesseract_version(),
        "packages": {p: _pkg(p) for p in ("forgelens", "numpy", "pillow", "pytesseract", "torch")},
    }


def save_env_report(run_dir: str | Path) -> Path:
    path = Path(run_dir) / "env.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(env_report(), indent=2) + "\n")
    return path


def main() -> None:
    print(json.dumps(env_report(), indent=2))
