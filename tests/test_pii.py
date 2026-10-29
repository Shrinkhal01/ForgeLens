"""Detection rules on fake OCR lines. No tesseract needed."""

import pytest
from conftest import make_line

from forgelens.pii import Finding, find_pii, load_names, merge, regex_findings


def kinds(text: str) -> list[str]:
    return sorted(f.kind for f in regex_findings(make_line(text)))


@pytest.mark.parametrize(
    "text",
    ["asha.verma@okaxis", "9876543210@ybl", "UPI ID: rahul_k-1@paytm", "asha.verma©okaxis"],
)
def test_upi_ids_found(text):
    assert "upi_id" in kinds(text)


def test_email_is_email_not_upi():
    assert kinds("Email: rohan.m@example.com") == ["email"]


@pytest.mark.parametrize(
    "text",
    [
        "9876543210",
        "98765 43210",
        "+91 98765 43210",
        "+91-98765-43210",
        "987 654 3210",
        "09876543210",
        "98765 4321O",  # OCR read the last 0 as the letter O
        "98765 432l0",  # ...and a 1 as a lower-case l
    ],
)
def test_phones_found(text):
    assert kinds(text) == ["phone"]


@pytest.mark.parametrize(
    "text",
    [
        "Transaction ID: 412345678901",  # 12-digit UTR
        "919876543210",  # 91 glued on with no + or space: treated as an ID (documented)
        "5876543210",  # Indian mobiles start with 6-9
        "98765432101",  # 11 digits
        "Rs 1,500.00",
        "29 Oct 2026, 3:41 pm",
    ],
)
def test_not_phones(text):
    assert kinds(text) == []


def test_split_phone_box_covers_both_words():
    (f,) = regex_findings(make_line("Phone: 98765 43210"))
    # "Phone:" is 60 px + 10 px space; the number's two words span x = 70..180
    assert f.box == (70, 0, 110, 20)


def test_load_names_normalises_and_skips_comments():
    text = "# friends\nAsha Verma\n  rohan   mehta  # college\n\n"
    assert load_names(text) == ["ASHA VERMA", "ROHAN MEHTA"]


def test_names_list_fuzzy_full_and_partial():
    names = ["ASHA VERMA"]
    full = find_pii([make_line("Paid to ASHA VERNA")], names, 0.8, heuristics=False)
    assert [f.source for f in full] == ["names_list"]
    first = find_pii([make_line("Thanks Asha, received")], names, 0.8, heuristics=False)
    assert len(first) == 1
    other = find_pii([make_line("ASHOK KUMAR")], names, 0.8, heuristics=False)
    assert other == []


def test_label_then_name_on_next_line():
    lines = [make_line("Paid to", y=0), make_line("ASHA VERMA", y=40)]
    (f,) = find_pii(lines, [], 0.8)
    assert (f.kind, f.source, f.box[1]) == ("name", "label_anchor", 40)


def test_label_and_name_on_same_line():
    (f,) = find_pii([make_line("From: ROHAN MEHTA")], [], 0.8)
    assert f.kind == "name" and f.box[0] == 60  # starts after "From: "


@pytest.mark.parametrize("below", ["Completed", "Bank account", "Rs 500.00"])
def test_label_followed_by_non_name(below):
    assert find_pii([make_line("Paid to", y=0), make_line(below, y=40)], [], 0.8) == []


def test_name_above_upi_id():
    lines = [make_line("Asha Verma", y=0), make_line("asha.verma@okaxis", y=40)]
    found = {f.source for f in find_pii(lines, [], 0.8)}
    assert found == {"regex", "near_id"}


def test_line_far_below_label_is_not_a_name():
    assert find_pii([make_line("Paid to", y=0), make_line("ASHA VERMA", y=200)], [], 0.8) == []


def test_heuristics_off_for_verify_pass():
    lines = [make_line("Paid to", y=0), make_line("ASHA VERMA", y=40)]
    assert find_pii(lines, [], 0.8, heuristics=False) == []


def test_merge_joins_same_text_from_two_views():
    a = Finding("phone", "regex", (100, 50, 200, 30))
    b = Finding("phone", "regex", (102, 51, 199, 29))
    c = Finding("name", "near_id", (100, 50, 200, 30))
    # union: x 100..301, y 50..80 -> one phone box; the name box stays separate
    assert merge([a, b, c]) == [Finding("phone", "regex", (100, 50, 201, 30)), c]
