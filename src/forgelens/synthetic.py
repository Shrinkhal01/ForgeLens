"""Synthetic test screenshots with fake PII, for tests and the Phase 0 demo.

These are deliberately plain: no app branding, no real layout, and a
"SYNTHETIC TEST IMAGE" header. They exist so the redaction pipeline can be
measured against known ground truth without using anyone's real data.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Field:
    kind: str  # "name", "upi_id", "phone", "email" = PII; "keep" = must stay readable
    text: str
    box: tuple[int, int, int, int]  # x, y, w, h in pixels


# (label drawn before the value, value, kind). An empty label puts the value on its own line.
_LINES: list[tuple[str, str, str]] = [
    ("", "SYNTHETIC TEST IMAGE - NOT A RECEIPT", "keep"),
    ("", "Paid to", "keep"),
    ("", "ASHA VERMA", "name"),
    ("UPI ID: ", "asha.verma@testbank", "upi_id"),
    ("Phone: ", "+91 98765 43210", "phone"),
    ("Amount: ", "Rs 500.00", "keep"),
    ("Transaction ID: ", "412345678901", "keep"),
    ("", "29 Oct 2026, 3:41 pm", "keep"),
    ("From: ", "ROHAN MEHTA", "name"),
    ("Email: ", "rohan.m@example.com", "email"),
    ("", "Status: Completed", "keep"),
]


def make_screenshot(
    width: int = 1080, dark: bool = False, font_px: int | None = None
) -> tuple[Image.Image, list[Field]]:
    """Draw the synthetic screenshot. Returns the image and ground-truth field boxes."""
    font_px = font_px or max(12, round(width * 0.04))  # 43 px text at 1080 wide
    line_h = round(font_px * 2.2)
    height = round(width * 16 / 9)
    bg, fg = ((18, 18, 18), (235, 235, 235)) if dark else ((255, 255, 255), (20, 20, 20))

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=font_px)

    fields: list[Field] = []
    x0, y = round(width * 0.08), round(height * 0.08)
    for label, value, kind in _LINES:
        x = x0
        if label:
            draw.text((x, y), label, font=font, fill=fg)
            x += round(draw.textlength(label, font=font))
        draw.text((x, y), value, font=font, fill=fg)
        left, top, right, bottom = draw.textbbox((x, y), value, font=font)
        fields.append(Field(kind, value, (left, top, right - left, bottom - top)))
        y += line_h
    return img, fields
