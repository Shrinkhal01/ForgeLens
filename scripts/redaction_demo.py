"""Reproduce every number in docs/phase-0.md.

    uv run python scripts/redaction_demo.py

Runs the redaction pipeline on synthetic screenshots (known ground truth, fake PII),
prints a table, and saves results/phase0-demo-<time>/{env.json, metrics.json}.
"""

from __future__ import annotations

import copy
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from forgelens.config import load_config
from forgelens.device import save_env_report, set_seed
from forgelens.ocr import read_lines
from forgelens.redact import load_rgb, redact_image
from forgelens.synthetic import make_screenshot

KEEP = ("500.00", "412345678901", "Completed")
VARIANTS = {
    "light_1080": dict(),
    "dark_1080": dict(dark=True),
    "light_540": dict(width=540),
    "jpeg_q50_720": dict(width=720),
}


def covered_fraction(fields, boxes, size) -> list[float]:
    mask = np.zeros((size[1], size[0]), bool)
    for b in boxes:
        x, y, w, h = b.box
        mask[y : y + h, x : x + w] = True
    return [float(mask[f.box[1] : f.box[1] + f.box[3], f.box[0] : f.box[0] + f.box[2]].mean())
            for f in fields]  # fmt: skip


def run(name, img, fields, cfg) -> dict:
    t0 = time.perf_counter()
    out, boxes, residual, fill_ok = redact_image(img, cfg, names=[])
    seconds = time.perf_counter() - t0
    pii = [f for f in fields if f.kind != "keep"]
    cov = covered_fraction(pii, boxes, img.size)
    text = " ".join(ln.text for ln in read_lines(out))
    area = sum(b.box[2] * b.box[3] for b in boxes)
    return {
        "variant": name,
        "size": f"{img.width}x{img.height}",
        "pad": f"{cfg['redact']['pad_px']}px/{cfg['redact']['pad_frac']:.0%}",
        "pii_fields": len(pii),
        "fields_fully_covered": sum(c == 1.0 for c in cov),
        "worst_field_coverage": round(min(cov), 4),
        "boxes": len(boxes),
        "redacted_area_pct": round(100 * area / (img.width * img.height), 2),
        "kept_fields_readable": f"{sum(k in text for k in KEEP)}/{len(KEEP)}",
        "residual": len(residual),
        "fill_check": fill_ok,
        "seconds": round(seconds, 2),
    }


def main() -> None:
    set_seed(0)
    cfg = load_config(environ={})
    demo_dir = Path("data/demo")
    demo_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, kwargs in VARIANTS.items():
        img, fields = make_screenshot(**kwargs)
        if name.startswith("jpeg"):  # a screenshot forwarded on WhatsApp gets re-encoded
            path = demo_dir / f"{name}.jpg"
            img.save(path, quality=50)
            img = load_rgb(path)
        rows.append(run(name, img, fields, cfg))

    # Ablation: why padding exists. Same image, no padding at all.
    no_pad = copy.deepcopy(cfg)
    no_pad["redact"].update(pad_px=0, pad_frac=0.0)
    img, fields = make_screenshot()
    rows.append(run("light_1080 (no pad)", img, fields, no_pad))

    cols = list(rows[0])
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")

    run_dir = Path("results") / f"phase0-demo-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    env = save_env_report(run_dir)
    (run_dir / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nsaved {run_dir}/metrics.json and {env.name} (tesseract "
          f"{json.loads(env.read_text())['tesseract']})")  # fmt: skip


if __name__ == "__main__":
    main()
