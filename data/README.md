# data/ — private, gitignored

Everything in this folder except this file is ignored by git.
`tests/test_repo_hygiene.py` fails if any image is ever tracked.

| Folder | What goes here | Share? |
|---|---|---|
| `raw/` | Original screenshots (mine, friends' with permission). Never modified. | Never |
| `private/names.txt` | Names to always redact, one per line (`#` = comment) | Never |
| `private/manual_boxes.json` | Boxes the OCR missed: `{"file.png": [[x, y, w, h]]}` | Never |
| `private/review/` | Review sheets: original + boxes + grid, next to the output | Never |
| `private/redaction_map.csv` | Original filename → output id | Never |
| `redacted/` | Verified outputs `<id>.png` + manifest `<id>.json` | Only after review |
| `forged/` | Forged samples (later phases) | Never published |
| `demo/` | Synthetic images from `scripts/redaction_demo.py` | Fine |

Rules
- Redact first, forge later: forgeries are made from redacted images.
- Read every review sheet before anything leaves `redacted/`.
- Output ids are the first 12 hex chars of the original file's SHA-256.
