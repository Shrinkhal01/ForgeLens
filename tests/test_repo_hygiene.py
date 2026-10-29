"""Guards against the worst mistake this project can make: committing a real screenshot."""

import shutil
import subprocess
from pathlib import Path

import pytest

from forgelens.device import env_report, get_device

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".heic", ".gif", ".bmp", ".tif", ".tiff")


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def _in_git_checkout() -> bool:
    return shutil.which("git") is not None and _git("rev-parse").returncode == 0


needs_git = pytest.mark.skipif(not _in_git_checkout(), reason="not inside a git checkout")


@needs_git
def test_no_images_tracked_anywhere():
    tracked = _git("ls-files").stdout.splitlines()
    images = [f for f in tracked if f.lower().endswith(IMAGE_EXTS)]
    assert images == [], f"image files are tracked by git: {images}"


@needs_git
@pytest.mark.parametrize(
    "path",
    ["data/raw/x.png", "data/private/names.txt", "data/redacted/x.json", "results/r/env.json"],
)
def test_data_and_results_are_gitignored(path):
    assert _git("check-ignore", "-q", path).returncode == 0, f"{path} is not gitignored"


def test_force_cpu(monkeypatch):
    monkeypatch.setenv("FORCE_CPU", "1")
    assert get_device() == "cpu"


def test_env_report_has_the_basics():
    report = env_report()
    assert {"python", "os", "device", "tesseract", "packages"} <= report.keys()
