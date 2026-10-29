"""Redact names, UPI IDs, phone numbers and emails from payment screenshots.

    uv run forgelens-redact                 # data/raw/ -> data/redacted/
    uv run forgelens-redact path/to/img.png --out somewhere/

Per image: OCR -> find PII -> pad boxes -> solid fill -> VERIFY -> write.
Verify = (1) every pixel inside every box is the fill colour, and (2) a fresh OCR of
the redacted image finds no UPI ID, phone, email or listed name. An image that fails
is not written. Detected text is never printed or saved anywhere.

Verify re-reads with the same OCR, so it catches mechanical mistakes (a box in the
wrong place, padding too small), not text the OCR could not read in the first place.
That second kind of miss is what the review sheets in data/private/review/ are for.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from . import __version__
from .config import load_config
from .device import save_env_report
from .ocr import Box, OCRUnavailable, check_tesseract, read_lines
from .pii import Finding, find_pii, load_names

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
EXIT_OK, EXIT_NOT_ALL_PASSED, EXIT_SETUP_ERROR = 0, 1, 2


@dataclass
class Result:
    source: Path
    output_id: str
    status: str  # PASS, FAIL or NEEDS_REVIEW
    boxes: list[Finding]
    residual: list[Finding] = field(default_factory=list)
    fill_ok: bool = True
    seconds: float = 0.0
    size: tuple[int, int] = (0, 0)


# --- the four steps --------------------------------------------------------------------


def load_rgb(path: Path) -> Image.Image:
    """Open read-only, apply EXIF rotation, drop alpha. The raw file is never written."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def pad_box(box: Box, pad_px: int, pad_frac: float, size: tuple[int, int]) -> Box:
    x, y, w, h = box
    p = max(pad_px, round(pad_frac * h))
    x0, y0 = max(0, x - p), max(0, y - p)
    x1, y1 = min(size[0], x + w + p), min(size[1], y + h + p)
    return (x0, y0, x1 - x0, y1 - y0)


def apply_fill(img: Image.Image, boxes: list[Box], fill: tuple[int, int, int]) -> Image.Image:
    arr = np.array(img)  # a copy; the input image is untouched
    for x, y, w, h in boxes:
        arr[y : y + h, x : x + w] = fill
    return Image.fromarray(arr)  # fresh image: no EXIF, ICC or text metadata carried over


def fill_check(img: Image.Image, boxes: list[Box], fill: tuple[int, int, int]) -> bool:
    arr = np.asarray(img)
    return all(bool((arr[y : y + h, x : x + w] == fill).all()) for x, y, w, h in boxes)


def output_id(path: Path) -> str:
    """First 12 hex chars of the file's SHA-256: anonymous, stable, de-duplicates copies."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def redact_image(
    img: Image.Image, cfg: dict, names: list[str], manual: list[Box] = ()
) -> tuple[Image.Image, list[Finding], list[Finding], bool]:
    """Pure pipeline on an in-memory image. Returns (redacted, boxes, residual, fill_ok)."""
    r = cfg["redact"]
    fill = tuple(r["fill"])
    found = find_pii(
        read_lines(img, r["ocr_psm"], r["upscale_below_px"]), names, r["name_similarity"]
    )
    found += [Finding("manual", "manual", tuple(b)) for b in manual]
    boxes = [
        Finding(f.kind, f.source, pad_box(f.box, r["pad_px"], r["pad_frac"], img.size))
        for f in found
    ]
    redacted = apply_fill(img, [b.box for b in boxes], fill)
    fill_ok = fill_check(redacted, [b.box for b in boxes], fill)
    residual = find_pii(
        read_lines(redacted, r["ocr_psm"], r["upscale_below_px"]),
        names,
        r["name_similarity"],
        heuristics=False,
    )
    return redacted, boxes, residual, fill_ok


# --- review sheet -----------------------------------------------------------------------

COLOURS = {
    "name": (230, 40, 40), "upi_id": (40, 110, 230), "phone": (20, 160, 60),
    "email": (200, 120, 0), "manual": (150, 60, 200), "residual": (255, 0, 255),
}  # fmt: skip


def review_sheet(
    original: Image.Image, redacted: Image.Image, res: Result, grid: int = 100
) -> Image.Image:
    """Original with boxes + pixel grid (left) next to the redacted output (right)."""
    w, h = original.size
    fs = max(14, w // 45)
    font = ImageFont.load_default(size=fs)
    banner = max(40, w // 18)
    sheet = Image.new("RGB", (w * 2 + 20, h + banner), (255, 255, 255))
    sheet.paste(original, (0, banner))
    sheet.paste(redacted, (w + 20, banner))
    d = ImageDraw.Draw(sheet)

    for gx in range(grid, w, grid):  # grid with pixel coordinates, for manual boxes
        d.line([(gx, banner), (gx, banner + h)], fill=(0, 200, 255), width=1)
        d.text((gx + 2, banner + 2), str(gx), fill=(0, 140, 200), font=font)
    for gy in range(grid, h, grid):
        d.line([(0, banner + gy), (w, banner + gy)], fill=(0, 200, 255), width=1)
        d.text((2, banner + gy + 2), str(gy), fill=(0, 140, 200), font=font)

    marks = [(b.kind, b.box) for b in res.boxes] + [("residual", f.box) for f in res.residual]
    for kind, (x, y, bw, bh) in marks:
        colour = COLOURS.get(kind, (0, 0, 0))
        d.rectangle([x, banner + y, x + bw - 1, banner + y + bh - 1], outline=colour, width=3)
        d.text((x, banner + y - fs - 2), kind, fill=colour, font=font)
    for kind, (x, y, bw, bh) in [m for m in marks if m[0] == "residual"]:
        d.rectangle(
            [w + 20 + x, banner + y, w + 20 + x + bw, banner + y + bh],
            outline=COLOURS[kind],
            width=3,
        )

    colour = {"PASS": (20, 140, 60), "FAIL": (200, 30, 30)}.get(res.status, (200, 120, 0))
    d.rectangle([0, 0, sheet.width, banner - 4], fill=colour)
    d.text(
        (10, 6), f"{res.status}  {res.output_id}  {summary_counts(res)}", fill="white", font=font
    )
    return sheet


def summary_counts(res: Result) -> str:
    counts: dict[str, int] = {}
    for b in res.boxes:
        counts[b.kind] = counts.get(b.kind, 0) + 1
    boxes = " ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"
    return f"boxes: {boxes} | residual: {len(res.residual)} | fill_check: {res.fill_ok}"


# --- files --------------------------------------------------------------------------------


def collect_inputs(paths: list[Path], skip_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        candidates = sorted(p.rglob("*")) if p.is_dir() else [p]
        for c in candidates:
            if any(c.resolve().is_relative_to(s.resolve()) for s in skip_dirs):
                continue  # never re-read our own outputs
            if c.suffix.lower() == ".heic":
                print(
                    f"skip {c.name}: HEIC not supported; convert first: sips -s format png "
                    f"'{c.name}' --out '{c.stem}.png'"
                )
            elif c.is_file() and c.suffix.lower() in IMAGE_EXTS:
                files.append(c)
    return files


def update_map(map_path: Path, results: list[Result]) -> None:
    """Private source-filename -> output-id table. Filenames can contain names, so it is
    kept in data/private/, never next to the shareable outputs."""
    rows: dict[str, dict] = {}
    if map_path.exists():
        with map_path.open(newline="") as fh:
            rows = {r["source"]: r for r in csv.DictReader(fh)}
    for res in results:
        rows[res.source.name] = {
            "source": res.source.name,
            "output_id": res.output_id,
            "status": res.status,
        }
    map_path.parent.mkdir(parents=True, exist_ok=True)
    with map_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["source", "output_id", "status"])
        writer.writeheader()
        writer.writerows(rows[k] for k in sorted(rows))


def manifest(res: Result, cfg: dict, tesseract: str) -> dict:
    """Shareable sidecar. Kinds and boxes only: no detected text, no source filename.
    Later phases use these boxes as a 'known edit' mask so redaction is not scored as forgery."""
    return {
        "id": res.output_id,
        "forgelens_version": __version__,
        "tesseract": tesseract,
        "image_size": list(res.size),
        "fill": cfg["redact"]["fill"],
        "boxes": [
            {"kind": b.kind, "source": b.source, **dict(zip("xywh", b.box, strict=True))}
            for b in res.boxes
        ],
        "verify": {"status": res.status, "residual": len(res.residual), "fill_check": res.fill_ok},
        "config": cfg["redact"],
    }


def process_file(
    path: Path,
    out_dir: Path,
    private_dir: Path,
    cfg: dict,
    names: list[str],
    manual: dict[str, list[Box]],
    tesseract: str,
) -> Result:
    t0 = time.perf_counter()
    original = load_rgb(path)
    oid = output_id(path)
    redacted, boxes, residual, fill_ok = redact_image(
        original, cfg, names, manual.get(path.name, [])
    )

    if residual or not fill_ok:
        status = "FAIL"
    elif not boxes and not cfg["redact"]["allow_empty"]:
        status = "NEEDS_REVIEW"  # every UPI screenshot has a name or ID; zero hits is suspicious
    else:
        status = "PASS"
    res = Result(path, oid, status, boxes, residual, fill_ok, 0.0, original.size)

    review_dir = private_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    review_sheet(original, redacted, res).save(review_dir / f"{oid}.png")
    if status == "PASS":
        out_dir.mkdir(parents=True, exist_ok=True)
        redacted.save(out_dir / f"{oid}.png", format="PNG")  # lossless: no JPEG re-encode
        (out_dir / f"{oid}.json").write_text(
            json.dumps(manifest(res, cfg, tesseract), indent=2) + "\n"
        )
    res.seconds = time.perf_counter() - t0
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="forgelens-redact", description=__doc__.split("\n")[0])
    ap.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[Path("data/raw")],
        help="image files or folders (default: data/raw)",
    )
    ap.add_argument("--out", type=Path, default=Path("data/redacted"), help="shareable outputs")
    ap.add_argument(
        "--private",
        type=Path,
        default=Path("data/private"),
        help="never-share folder: names.txt, manual_boxes.json, review sheets, map",
    )
    ap.add_argument("--config", type=Path, help="extra YAML config layered over the defaults")
    ap.add_argument("--pad-px", type=int, help="override redact.pad_px")
    ap.add_argument(
        "--allow-empty",
        action="store_true",
        default=None,
        help="let images with zero detections PASS",
    )
    args = ap.parse_args(argv)

    flags = {"pad_px": args.pad_px, "allow_empty": args.allow_empty}
    cfg = load_config(
        args.config, cli={"redact": {k: v for k, v in flags.items() if v is not None}}
    )

    try:
        tesseract = check_tesseract()
    except OCRUnavailable as exc:
        print(exc, file=sys.stderr)
        return EXIT_SETUP_ERROR

    names_file = args.private / "names.txt"
    names = load_names(names_file.read_text()) if names_file.exists() else []
    manual_file = args.private / "manual_boxes.json"
    manual = json.loads(manual_file.read_text()) if manual_file.exists() else {}

    missing = [p for p in args.inputs if not p.exists()]
    if missing:
        print(f"not found: {', '.join(map(str, missing))}", file=sys.stderr)
        print(
            "Put your screenshots in data/raw/ (mkdir -p data/raw) and run again.", file=sys.stderr
        )
        return EXIT_SETUP_ERROR
    files = collect_inputs(args.inputs, skip_dirs=[args.out, args.private])
    if not files:
        print("no .png/.jpg/.jpeg/.webp images found in: " + ", ".join(map(str, args.inputs)))
        return EXIT_SETUP_ERROR

    print(
        f"tesseract {tesseract} | {len(names)} names in names.txt | "
        f"{sum(map(len, manual.values()))} manual boxes | {len(files)} images"
    )
    results = []
    for path in files:
        res = process_file(path, args.out, args.private, cfg, names, manual, tesseract)
        results.append(res)
        print(
            f"{res.status:<12} {path.name} -> {res.output_id}  {summary_counts(res)}  "
            f"{res.seconds:.1f}s"
        )

    update_map(args.private / "redaction_map.csv", results)
    run_dir = Path("results") / f"redact-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    save_env_report(run_dir)
    (run_dir / "summary.json").write_text(
        json.dumps(
            [
                {
                    "id": r.output_id,
                    "status": r.status,
                    "boxes": len(r.boxes),
                    "residual": len(r.residual),
                    "fill_check": r.fill_ok,
                    "seconds": round(r.seconds, 2),
                }
                for r in results
            ],
            indent=2,
        )
        + "\n"
    )

    passed = sum(r.status == "PASS" for r in results)
    print(f"\n{passed}/{len(results)} PASS. Outputs: {args.out}/  Run log: {run_dir}/")
    print(
        f"Before sharing anything, open every sheet in {args.private / 'review'}/ and check "
        "for names the OCR missed. Add misses to names.txt or manual_boxes.json and re-run."
    )
    return EXIT_OK if passed == len(results) else EXIT_NOT_ALL_PASSED


if __name__ == "__main__":
    sys.exit(main())
