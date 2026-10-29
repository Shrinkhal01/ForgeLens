"""Redaction mechanics (no OCR) and end-to-end runs on synthetic screenshots (OCR)."""

import hashlib
import json

import numpy as np
import pytest
from conftest import requires_tesseract
from PIL import Image

from forgelens import redact
from forgelens.config import load_config
from forgelens.ocr import OCRUnavailable, read_lines
from forgelens.synthetic import make_screenshot

BLACK = (0, 0, 0)


# --- mechanics, no OCR -------------------------------------------------------------------


def test_pad_box_grows_and_clips_to_image():
    assert redact.pad_box((10, 10, 50, 20), 6, 0.15, (100, 100)) == (4, 4, 62, 32)
    assert redact.pad_box((0, 90, 50, 40), 6, 0.15, (100, 100)) == (0, 84, 56, 16)  # clipped


def test_fill_changes_only_pixels_inside_boxes():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (200, 300, 3), dtype=np.uint8))
    boxes = [(10, 20, 50, 30), (100, 150, 80, 40)]
    out = redact.apply_fill(img, boxes, BLACK)

    before, after = np.asarray(img), np.asarray(out)
    inside = np.zeros(before.shape[:2], bool)
    for x, y, w, h in boxes:
        inside[y : y + h, x : x + w] = True
    assert (after[inside] == 0).all()
    assert (after[~inside] == before[~inside]).all()  # bit-identical everywhere else
    assert redact.fill_check(out, boxes, BLACK)


def test_fill_check_catches_one_bad_pixel():
    img = redact.apply_fill(Image.new("RGB", (50, 50), "white"), [(5, 5, 20, 20)], BLACK)
    arr = np.array(img)
    arr[10, 10] = (1, 0, 0)
    assert not redact.fill_check(Image.fromarray(arr), [(5, 5, 20, 20)], BLACK)


def test_config_precedence():
    assert load_config(environ={})["redact"]["pad_px"] == 6
    env = {"FORGELENS_REDACT_PAD_PX": "10"}
    assert load_config(environ=env)["redact"]["pad_px"] == 10
    assert load_config(environ=env, cli={"redact": {"pad_px": 12}})["redact"]["pad_px"] == 12


def test_missing_tesseract_is_a_clear_exit_code(monkeypatch, tmp_path, capsys):
    def boom():
        raise OCRUnavailable("Tesseract OCR is not installed")

    monkeypatch.setattr(redact, "check_tesseract", boom)
    assert redact.main([str(tmp_path)]) == redact.EXIT_SETUP_ERROR
    assert "not installed" in capsys.readouterr().err


# --- end to end with OCR ----------------------------------------------------------------


def coverage(fields, boxes, size) -> list[float]:
    """Fraction of each ground-truth field's pixels that fall inside some redaction box."""
    mask = np.zeros((size[1], size[0]), bool)
    for f in boxes:
        x, y, w, h = f.box
        mask[y : y + h, x : x + w] = True
    out = []
    for f in fields:
        x, y, w, h = f.box
        out.append(float(mask[y : y + h, x : x + w].mean()))
    return out


def jpeg_roundtrip(img: Image.Image, quality: int, tmp_path) -> Image.Image:
    path = tmp_path / "forwarded.jpg"
    img.save(path, quality=quality)
    return redact.load_rgb(path)


VARIANTS = {
    "light_1080": dict(),
    "dark_1080": dict(dark=True),
    "light_540": dict(width=540),  # small enough to trigger the 2x OCR view
    "jpeg_q50_720": dict(width=720),
}


@requires_tesseract
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_synthetic_pii_fully_covered_and_rest_readable(variant, cfg, tmp_path):
    img, fields = make_screenshot(**VARIANTS[variant])
    if variant.startswith("jpeg"):
        img = jpeg_roundtrip(img, 50, tmp_path)
    out, boxes, residual, fill_ok = redact.redact_image(img, cfg, names=[])

    pii = [f for f in fields if f.kind != "keep"]
    assert min(coverage(pii, boxes, img.size)) == 1.0
    assert residual == [] and fill_ok
    text = " ".join(ln.text for ln in read_lines(out))
    for keep in ("500.00", "412345678901", "Completed"):
        assert keep in text, f"{keep!r} should survive redaction"


@requires_tesseract
def test_names_list_adds_its_own_evidence(cfg):
    img, _ = make_screenshot()
    _, boxes, _, _ = redact.redact_image(img, cfg, names=["ASHA VERMA", "ROHAN MEHTA"])
    sources = [b.source for b in boxes if b.kind == "name"]
    assert len(sources) == 2 and all("names_list" in s for s in sources)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A fake project folder with data/raw holding one synthetic screenshot with EXIF."""
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    img, fields = make_screenshot()
    exif = Image.Exif()
    exif[0x010F] = "TestPhoneMaker"  # EXIF "Make"
    src = raw / "Screenshot_asha_payment.png"
    img.save(src, exif=exif)
    return tmp_path, src, fields


@requires_tesseract
def test_cli_writes_clean_outputs_and_leaves_raw_untouched(workspace):
    root, src, fields = workspace
    raw_hash = hashlib.sha256(src.read_bytes()).hexdigest()

    assert redact.main([]) == redact.EXIT_OK

    assert hashlib.sha256(src.read_bytes()).hexdigest() == raw_hash
    oid = raw_hash[:12]
    out_png, out_json = root / f"data/redacted/{oid}.png", root / f"data/redacted/{oid}.json"
    assert out_png.exists() and out_json.exists()
    assert (root / f"data/private/review/{oid}.png").exists()
    assert "Screenshot_asha_payment.png" in (root / "data/private/redaction_map.csv").read_text()

    with Image.open(out_png) as im:
        assert not im.getexif() and "exif" not in im.info

    shared = out_json.read_text().lower()
    for f in fields:
        if f.kind != "keep":
            for token in f.text.lower().replace("@", " ").replace(".", " ").split():
                if len(token) > 3:
                    assert token not in shared, f"manifest leaks {token!r}"
    assert "screenshot_asha" not in shared
    assert json.loads(shared)["verify"]["status"] == "pass"


@requires_tesseract
def test_broken_fill_is_caught_and_nothing_is_written(workspace, monkeypatch):
    root, _, _ = workspace
    monkeypatch.setattr(redact, "apply_fill", lambda img, boxes, fill: img.copy())
    assert redact.main([]) == redact.EXIT_NOT_ALL_PASSED
    assert not list((root / "data").glob("redacted/*.png"))
    assert "FAIL" in (root / "data/private/redaction_map.csv").read_text()


@requires_tesseract
def test_blank_image_needs_review_not_pass(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data/raw").mkdir(parents=True)
    Image.new("RGB", (400, 700), "white").save(tmp_path / "data/raw/blank.png")
    assert redact.main([]) == redact.EXIT_NOT_ALL_PASSED
    assert "NEEDS_REVIEW" in (tmp_path / "data/private/redaction_map.csv").read_text()
    assert redact.main(["--allow-empty"]) == redact.EXIT_OK
