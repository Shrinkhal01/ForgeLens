import os

import pytest

from forgelens.config import load_config
from forgelens.ocr import Line, OCRUnavailable, Word, check_tesseract

try:
    check_tesseract()
    HAS_TESSERACT = True
except OCRUnavailable:
    HAS_TESSERACT = False
    if os.environ.get("FORGELENS_REQUIRE_TESSERACT") == "1":  # set in CI
        raise

requires_tesseract = pytest.mark.skipif(
    not HAS_TESSERACT, reason="tesseract not installed (macOS: brew install tesseract)"
)


@pytest.fixture
def cfg():
    return load_config(environ={})


def make_line(text: str, y: int = 0, x: int = 0, view: str = "gray", h: int = 20) -> Line:
    """A fake OCR line: each character is 10 px wide, words separated by one space."""
    words, pos = [], x
    for token in text.split():
        words.append(Word(token, (pos, y, 10 * len(token), h), 95.0))
        pos += 10 * (len(token) + 1)
    return Line(view, tuple(words))
