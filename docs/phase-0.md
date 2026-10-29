# Phase 0 — setup + redaction

Goal: turn real UPI screenshots into data I can safely work with.
Names, UPI IDs, phone numbers and emails get covered with solid black boxes.

## Run it

```bash
uv run forgelens-redact                   # data/raw/ -> data/redacted/
uv run python scripts/redaction_demo.py   # re-makes every number on this page
```

## The pipeline, on one field

Synthetic 1080×1920 screenshot, field `Phone: +91 98765 43210`.

1. **OCR.** Tesseract reads 2 views (grayscale + inverted for dark mode).
   Images narrower than 1000 px get a 3rd, 2× upscaled view.
   22 lines read in total (11 per view).
2. **Detect.** The phone regex matches `+91 98765 43210` across 3 OCR words.
   Box = union of those words: x=232, y=545, w=337, h=32.
3. **Pad.** pad = max(6 px, 15% × 32 px = 5 px) = 6 px.
   Box becomes x=226, y=539, w=349, h=44.
4. **Fill.** Every pixel in the box set to (0, 0, 0).
   Pixels outside boxes stay bit-identical (tested).
5. **Verify.** (a) all pixels in all boxes are exactly black;
   (b) fresh OCR of the output finds 0 UPI IDs, phones, emails or listed names.
6. **Write** only on PASS: `data/redacted/<id>.png` + `<id>.json`.

Why step 3 exists: the true ink box of the phone text is x=229, w=342.
OCR's box starts 3 px late and ends 2 px early.
Without padding, 160 of 10,944 ink pixels (1.5%) stay visible.
With 6 px padding: 0.

## Results on synthetic screenshots

5 PII fields per image (2 names, UPI ID, phone, email) + 3 fields that must stay readable
(amount, transaction ID, status). From `scripts/redaction_demo.py`, tesseract 5.3.4:

| Variant | Size | Fields fully covered | Worst field coverage | Area redacted | Kept fields readable | Verify | Time |
|---|---|---|---|---|---|---|---|
| light | 1080×1920 | 5/5 | 100% | 4.17% | 3/3 | PASS | 1.4 s |
| dark mode | 1080×1920 | 5/5 | 100% | 4.18% | 3/3 | PASS | 1.2 s |
| low-res | 540×960 | 5/5 | 100% | 5.57% | 3/3 | PASS | 1.6 s |
| JPEG q=50 (WhatsApp-like) | 720×1280 | 5/5 | 100% | 4.80% | 3/3 | PASS | 2.0 s |
| light, **no padding** | 1080×1920 | **0/5** | **93.0%** | 3.01% | 3/3 | **PASS** | 1.3 s |

Read the last row carefully: verify still says PASS while 584 ink pixels of
`ASHA VERMA` (7%) are visible. Slivers of letters are not readable by OCR,
so verify cannot see them. Padding is what fixes it; verify is not.

## What these numbers do NOT prove

- Synthetic text is clean and evenly spaced. Real app screenshots have icons,
  colored cards, small grey text. Real-world recall is unmeasured until I label
  real screenshots.
- Verify uses the same OCR as detection. It catches mechanical bugs
  (box in the wrong place, broken fill), not text OCR never read.
  The human review sheets are the real safety net.
- Name detection without `names.txt` relies on layout: the line after "Paid to"/"From",
  or the line above a UPI ID. A name anywhere else is missed unless listed.
- Box width leaks text length (a 9-letter name gives a ~9-letter box).
  Acceptable for a private dataset; not anonymisation in the strict sense.

## Decisions

| Choice | Why | Rejected |
|---|---|---|
| Solid fill | Blur and pixelation can be partly reversed | Gaussian blur, mosaic |
| PNG output | No extra JPEG pass; later ELA tests stay meaningful | Re-saving as JPEG |
| Output name = SHA-256[:12] | Filenames can contain names; also de-duplicates | Keep original name |
| No detected text stored or printed | The manifest is meant to be shareable | Logging matches |
| Tesseract | CPU-only, deterministic, no torch on an 8 GB Mac | EasyOCR (needs torch) |
| `--psm 11` (sparse text) | Payment UIs scatter labels around icons | psm 3 (page layout) |
| 0 detections = NEEDS_REVIEW | Every real UPI screenshot has a name or ID | Silent PASS |
| Manifest keeps boxes | Later phases mask them so redaction isn't scored as forgery | — |

## Known gaps

- Transaction IDs (UTR) are not redacted; they're needed for OCR field checks.
- `919876543210` (91 glued on, no + or space) is not treated as a phone,
  so 12-digit transaction IDs starting with 91 aren't blacked out.
- HEIC isn't read. Convert first: `sips -s format png in.heic --out in.png`.

## If the review sheet shows a miss

- A name: add it to `data/private/names.txt` and re-run.
- Anything else: read x, y off the sheet's 100 px grid and add to
  `data/private/manual_boxes.json`: `{"IMG_0012.png": [[80, 340, 300, 60]]}`.
