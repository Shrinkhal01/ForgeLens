"""Thin layer over tesseract: word boxes grouped into lines, read from several views.

Why several views: tesseract misses light-on-dark text more often than dark-on-light,
and small text more often than large. Reading a grayscale view, an inverted view and
(for small images) a 2x upscaled view, then taking the union, buys recall for ~1 s.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image, ImageOps

Box = tuple[int, int, int, int]  # x, y, w, h in original-image pixels

INSTALL_HINT = (
    "Tesseract OCR is not installed or not on PATH.\n"
    "  macOS:   brew install tesseract\n"
    "  Ubuntu:  sudo apt-get install tesseract-ocr\n"
    "  Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
    "Then check with: tesseract --version"
)


class OCRUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class Word:
    text: str
    box: Box
    conf: float


@dataclass(frozen=True)
class Line:
    view: str
    words: tuple[Word, ...]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def box(self) -> Box:
        return union([w.box for w in self.words])


def union(boxes: list[Box]) -> Box:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return (x0, y0, x1 - x0, y1 - y0)


def check_tesseract() -> str:
    """Return the tesseract version, or raise OCRUnavailable with install instructions."""
    try:
        import pytesseract

        return str(pytesseract.get_tesseract_version())
    except Exception as exc:  # ImportError, TesseractNotFoundError, permission errors
        raise OCRUnavailable(INSTALL_HINT) from exc


def make_views(img: Image.Image, upscale_below_px: int) -> list[tuple[str, Image.Image, float]]:
    gray = ImageOps.grayscale(img)
    views = [("gray", gray, 1.0), ("inverted", ImageOps.invert(gray), 1.0)]
    if img.width < upscale_below_px:
        big = gray.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS)
        views.append(("gray_2x", big, 2.0))
    return views


def read_lines(img: Image.Image, psm: int = 11, upscale_below_px: int = 1000) -> list[Line]:
    """OCR every view; return lines (top-to-bottom per view) with boxes in original pixels."""
    import pytesseract

    lines: list[Line] = []
    for name, view, scale in make_views(img, upscale_below_px):
        data = pytesseract.image_to_data(
            view, config=f"--psm {psm}", output_type=pytesseract.Output.DICT
        )
        grouped: dict[tuple[int, int, int], list[Word]] = {}
        for i, text in enumerate(data["text"]):
            text = text.strip()
            if not text:
                continue
            # Scale back conservatively: floor the top-left, ceil the bottom-right.
            x0 = math.floor(data["left"][i] / scale)
            y0 = math.floor(data["top"][i] / scale)
            x1 = math.ceil((data["left"][i] + data["width"][i]) / scale)
            y1 = math.ceil((data["top"][i] + data["height"][i]) / scale)
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            word = Word(text, (x0, y0, x1 - x0, y1 - y0), float(data["conf"][i]))
            grouped.setdefault(key, []).append(word)
        view_lines = [Line(name, tuple(ws)) for ws in grouped.values()]
        lines.extend(sorted(view_lines, key=lambda ln: (ln.box[1], ln.box[0])))
    return lines
