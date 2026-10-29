"""Find PII in OCR lines: UPI IDs, phone numbers, emails and names.

Pure functions over `Line` objects, so every rule is unit-testable without tesseract.

Two strengths of detector:
  * strict  - regexes and your names list. Used for detection AND for the verify pass.
  * heuristic - name guesses from layout (the line after "Paid to", the line above a
    UPI ID). Used for detection only: they over-redact on purpose, and would raise
    false alarms in the verify pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from .ocr import Box, Line, Word, union

KINDS = ("name", "upi_id", "phone", "email")


@dataclass(frozen=True)
class Finding:
    kind: str  # one of KINDS
    source: str  # "regex", "names_list", "label_anchor", "near_id", "manual"
    box: Box


# --- regexes ---------------------------------------------------------------------------

# OCR often reads "@" as a copyright or registered sign. Same length, so offsets survive.
_AT_LOOKALIKES = str.maketrans({"©": "@", "®": "@"})

EMAIL_RE = re.compile(
    r"(?<![\w.+\-])[A-Za-z0-9][\w.+\-]*\s?@\s?[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}"
)
# UPI ID (VPA) = username@handle, e.g. asha.verma@okaxis, 9876543210@ybl. Handle has no dot.
UPI_RE = re.compile(
    r"(?<![\w.\-])[A-Za-z0-9][\w.\-]{1,255}\s?@\s?[A-Za-z][A-Za-z0-9]{1,63}(?![\w@.])"
)
# Indian mobile: 10 digits starting 6-9, grouped 10 / 5+5 / 3+3+4, optional +91, "91 " or 0.
# A bare "91" glued to the number (919876543210) is NOT treated as a prefix, so 12-digit
# transaction IDs starting with 91 are not redacted as phones. Trade-off documented in docs/.
PHONE_RE = re.compile(
    r"(?<![\d+])(?:\+\s?91[\s\-]?|\b91[\s\-]|\b0)?"
    r"[6-9](?:\d{9}|\d{4}[\s\-]\d{5}|\d{2}[\s\-]\d{3}[\s\-]\d{4})(?!\d)"
)

# Inside mostly-numeric tokens, OCR confuses O/o with 0 and I/l/| with 1.
_DIGIT_LOOKALIKES = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1"})


def _normalise_digits(token: str) -> str:
    digits = sum(c.isdigit() for c in token)
    if len(token) >= 2 and digits / len(token) >= 0.6:
        return token.translate(_DIGIT_LOOKALIKES)
    return token


def _word_spans(words: tuple[Word, ...]) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for w in words:
        spans.append((pos, pos + len(w.text)))
        pos += len(w.text) + 1  # the joining space
    return spans


def _box_of(line: Line, start: int, end: int) -> Box:
    """Union box of every word that overlaps characters [start, end) of line.text."""
    spans = _word_spans(line.words)
    hit = [w.box for (a, b), w in zip(spans, line.words, strict=True) if a < end and b > start]
    return union(hit)


def regex_findings(line: Line) -> list[Finding]:
    found: list[Finding] = []
    text = line.text.translate(_AT_LOOKALIKES)

    for m in EMAIL_RE.finditer(text):
        found.append(Finding("email", "regex", _box_of(line, *m.span())))
    # Blank out emails (same length) so the UPI regex doesn't report them a second time.
    text = EMAIL_RE.sub(lambda m: " " * len(m.group()), text)
    for m in UPI_RE.finditer(text):
        found.append(Finding("upi_id", "regex", _box_of(line, *m.span())))

    digits = " ".join(_normalise_digits(w.text) for w in line.words)
    for m in PHONE_RE.finditer(digits):
        found.append(Finding("phone", "regex", _box_of(line, *m.span())))
    return found


# --- names -------------------------------------------------------------------------------

_LETTERS = re.compile(r"[^A-Z ]")


def _norm_name(text: str) -> str:
    return " ".join(_LETTERS.sub("", text.upper()).split())


def load_names(text: str) -> list[str]:
    """names.txt format: one name per line, '#' starts a comment."""
    names = [_norm_name(ln.split("#", 1)[0]) for ln in text.splitlines()]
    return [n for n in names if n]


def names_list_findings(line: Line, names: list[str], threshold: float) -> list[Finding]:
    """Fuzzy-match each listed name (and each of its tokens) against the line's words."""
    tokens = [_norm_name(w.text) for w in line.words]
    found: list[Finding] = []
    for name in names:
        parts = name.split()
        k = len(parts)
        for i in range(len(tokens) - k + 1):  # the full name as a window of k words
            window = " ".join(tokens[i : i + k])
            if window and SequenceMatcher(None, window, name).ratio() >= threshold:
                found.append(
                    Finding("name", "names_list", union([w.box for w in line.words[i : i + k]]))
                )
        for part in parts:  # first or last name on its own ("Hi Asha")
            if len(part) < 3:
                continue
            for tok, word in zip(tokens, line.words, strict=True):
                if tok and SequenceMatcher(None, tok, part).ratio() >= threshold:
                    found.append(Finding("name", "names_list", word.box))
    return found


# Words that appear next to names in payment UIs but are never names themselves.
UI_WORDS = {
    "paid", "to", "from", "received", "sent", "successful", "success", "completed",
    "pending", "failed", "upi", "id", "bank", "banking", "name", "transaction", "txn",
    "amount", "rs", "inr", "share", "done", "view", "details", "phone", "email",
    "status", "payment", "account", "holder", "debited", "credited", "split", "history",
    "powered", "by", "help", "support", "ref", "no", "number", "mobile", "receipt",
    "test", "image", "synthetic", "not", "a", "screenshot", "you",
}  # fmt: skip
LABELS = [  # longest first, so "paid to" wins over "to"
    "received from", "account holder", "banking name", "paid to", "sent to",
    "payee", "payer", "name", "from", "to",
]  # fmt: skip
_NAME_TOKEN = re.compile(r"^[A-Za-z][A-Za-z.'\-]*$")
# Max blank space between a label and the line below it, in text heights. Payment UIs
# leave 1-2 text heights between "Paid to" and the name; 3 errs towards over-redacting.
MAX_GAP_LINES = 3.0


def looks_like_name(words: tuple[Word, ...] | list[Word]) -> bool:
    if not 1 <= len(words) <= 5:
        return False
    tokens = [w.text.strip(":,") for w in words]
    if not all(_NAME_TOKEN.match(t) for t in tokens):
        return False
    if any(t.lower().strip(".") in UI_WORDS for t in tokens):
        return False
    return sum(len(t) for t in tokens) >= 3


def _line_below(lines: list[Line], i: int) -> Line | None:
    """Nearest line in the same view that starts below line i and overlaps it horizontally."""
    cur = lines[i]
    cx, cy, cw, ch = cur.box
    for nxt in lines[i + 1 :]:
        if nxt.view != cur.view:
            break
        nx, ny, nw, nh = nxt.box
        if ny < cy + ch / 2:
            continue  # same row (another column)
        if ny - (cy + ch) > MAX_GAP_LINES * max(ch, nh):
            return None  # too far down
        if nx < cx + cw and cx < nx + nw:
            return nxt
    return None


def heuristic_name_findings(lines: list[Line], id_lines: set[int]) -> list[Finding]:
    """Layout guesses. `lines` must be sorted per view, as `ocr.read_lines` returns them."""
    found: list[Finding] = []
    for i, line in enumerate(lines):
        lowered = [w.text.lower().strip(":") for w in line.words]
        for label in LABELS:
            k = len(label.split())
            if lowered[:k] == label.split():
                rest = line.words[k:]
                if rest and looks_like_name(rest):
                    found.append(Finding("name", "label_anchor", union([w.box for w in rest])))
                elif not rest:
                    below = _line_below(lines, i)
                    if below and looks_like_name(below.words):
                        found.append(Finding("name", "label_anchor", below.box))
                break
    for i in id_lines:  # the line just above a UPI ID / phone / email is often the name
        if i > 0 and lines[i - 1].view == lines[i].view and _line_below(lines, i - 1) is lines[i]:
            if looks_like_name(lines[i - 1].words):
                found.append(Finding("name", "near_id", lines[i - 1].box))
    return found


# --- putting it together -------------------------------------------------------------------


def find_pii(
    lines: list[Line], names: list[str], threshold: float, heuristics: bool = True
) -> list[Finding]:
    found: list[Finding] = []
    id_lines: set[int] = set()
    for i, line in enumerate(lines):
        hits = regex_findings(line)
        if hits:
            id_lines.add(i)
        found += hits
        found += names_list_findings(line, names, threshold)
    if heuristics:
        found += heuristic_name_findings(lines, id_lines)
    return merge(found)


def _iou(a: Box, b: Box) -> float:
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    smaller = min(a[2] * a[3], b[2] * b[3]) or 1
    return inter / smaller  # overlap relative to the smaller box


def merge(findings: list[Finding], overlap: float = 0.5) -> list[Finding]:
    """Merge same-kind boxes that overlap (the same text read from several OCR views)."""
    merged: list[Finding] = []
    for f in findings:
        for j, m in enumerate(merged):
            if m.kind == f.kind and _iou(m.box, f.box) >= overlap:
                sources = "+".join(sorted(set(m.source.split("+")) | {f.source}))
                merged[j] = Finding(m.kind, sources, union([m.box, f.box]))
                break
        else:
            merged.append(f)
    return merged
