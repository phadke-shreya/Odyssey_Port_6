"""Tests for the chunking strategy.

The two load-bearing ones are test_table_is_never_split and
test_every_child_resolves_to_its_parent: they pin the two decisions the whole
retrieval design rests on.
"""

import pytest

from app import config
from app.chunker import (
    Block,
    ContentType,
    chunk_document,
    heading_depth,
    looks_like_heading,
)

TABLE_MARKDOWN = (
    "Table from HR_Policy.pdf p4, section 3.1 Leave: entitlement by role\n"
    "| Role | Vacation Days | Sick Days |\n"
    "|---|---|---|\n"
    "| Junior | 15 | 10 |\n"
    "| Senior | 20 | 12 |\n"
    "| Manager | 25 | 15 |\n"
)

SECTIONED_PROSE = """1. Introduction
This handbook describes company policy for all employees.

5. Working Arrangements
General rules about how and where work happens.

5.2 Remote Work Policy
Fridays are designated as optional remote days. Remote work is subject to
written manager approval and a minimum of three office days per week.

5.3 Overtime
Overtime must be approved in advance by a line manager.
"""

UNSTRUCTURED_PROSE = (
    "This document has no numbering at all. It is simply several paragraphs "
    "of running text with nothing that resembles a heading anywhere in it.\n\n"
    "Here is a second paragraph which also carries no heading, so the "
    "chunker must fall back to splitting purely by size.\n\n"
    "And a third paragraph for good measure, still with no headings."
)


def test_table_is_never_split() -> None:
    """A table must survive as exactly one chunk, with its headers intact."""
    blocks = [Block(text=TABLE_MARKDOWN, page=4, content_type=ContentType.TABLE)]

    chunks = chunk_document(blocks, "HR_Policy.pdf")

    table_chunks = [chunk for chunk in chunks if chunk.content_type == "table"]
    assert len(table_chunks) == 1, "table was split into multiple chunks"

    chunk = table_chunks[0]
    # Every row must still be present, and still sit alongside the header row.
    assert "| Role | Vacation Days | Sick Days |" in chunk.text
    for role in ("Junior", "Senior", "Manager"):
        assert role in chunk.text
    # A table is its own parent: the model receives the whole table.
    assert chunk.parent_text == chunk.text


def test_every_child_resolves_to_its_parent() -> None:
    """Small-to-big retrieval only works if each child carries its parent."""
    blocks = [Block(text=SECTIONED_PROSE, page=1, content_type=ContentType.PROSE)]

    chunks = chunk_document(blocks, "handbook.pdf")

    assert chunks, "no chunks produced"
    for chunk in chunks:
        assert chunk.parent_id, "child has no parent_id"
        assert chunk.parent_text, "child has no parent text to expand into"
        # The parent must genuinely contain the child, else expansion is a lie.
        assert chunk.text.strip()[:60] in chunk.parent_text
        # Parents must be big enough to be worth expanding to.
        assert len(chunk.parent_text) >= len(chunk.text)


def test_section_headings_become_breadcrumbs() -> None:
    """Detected headings must reach the chunk metadata, for citations."""
    blocks = [Block(text=SECTIONED_PROSE, page=1, content_type=ContentType.PROSE)]

    chunks = chunk_document(blocks, "handbook.pdf")

    sections = {chunk.section for chunk in chunks if chunk.section}
    assert any("Remote Work Policy" in section for section in sections)
    # Nesting must produce a trail, not just the leaf heading.
    assert any(">" in section for section in sections), sections


def test_unstructured_document_falls_back_to_size_parents() -> None:
    """A document with no headings must still chunk, not crash or return zero."""
    blocks = [Block(text=UNSTRUCTURED_PROSE, page=1, content_type=ContentType.PROSE)]

    chunks = chunk_document(blocks, "messy.pdf")

    assert chunks, "fallback produced no chunks"
    assert all(chunk.section == "" for chunk in chunks)


def test_image_only_page_is_flagged_not_dropped() -> None:
    """A page with no text must leave a visible trace, never vanish silently."""
    placeholder = (
        "Page 7 of manual.pdf appears to be an image or scanned page "
        "(no extractable text)."
    )
    blocks = [Block(text=placeholder, page=7, content_type=ContentType.IMAGE_ONLY)]

    chunks = chunk_document(blocks, "manual.pdf")

    assert len(chunks) == 1
    assert chunks[0].content_type == "image_only"
    assert chunks[0].page == 7


def test_oversized_section_is_split_and_labelled() -> None:
    """A huge section must be capped, and say which part it is."""
    huge = "5.2 Remote Work Policy\n" + ("Filler sentence here. " * 400)
    blocks = [Block(text=huge, page=2, content_type=ContentType.PROSE)]

    chunks = chunk_document(blocks, "big.pdf")

    parent_ids = {chunk.parent_id for chunk in chunks}
    assert len(parent_ids) > 1, "oversized section was not split"
    for chunk in chunks:
        assert len(chunk.parent_text) <= config.PARENT_MAX_CHARS + 100


@pytest.mark.parametrize(
    "line",
    [
        "1. Introduction",
        "5.2 Remote Work Policy",
        "5.2.1 Equipment",
        "SECTION 4",
        "Article 3",
        "WORKPLACE SAFETY",
    ],
)
def test_recognises_real_headings(line: str) -> None:
    assert looks_like_heading(line) is True


@pytest.mark.parametrize(
    "line",
    [
        "3.5 kg of flour is required for the recipe.",
        "Employees may carry forward unused leave up to a maximum of 5 days.",
        "",
        "   ",
        "This is a normal sentence that ends with a period.",
    ],
)
def test_rejects_things_that_only_look_like_headings(line: str) -> None:
    assert looks_like_heading(line) is False


def test_heading_depth_tracks_nesting() -> None:
    assert heading_depth("5. Working Arrangements") == 1
    assert heading_depth("5.2 Remote Work") == 2
    assert heading_depth("5.2.1 Equipment") == 3


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError):
        chunk_document([], "empty.pdf")


# --- regression tests: real false positives seen in the IRS documents ------
# Each string below was wrongly classified as a heading before the detector was
# tightened, and each produced a misleading citation.


@pytest.mark.parametrize(
    "line",
    [
        # Page footers: a bare number with no dot after it.
        "2 Publication 501 (2025)",
        "8 Publication 15 (2026)",
        # Numbered LIST items: sentences, not titles.
        "1. Innocent spouse relief. How to file",
        "11. If your spouse itemizes deductions, you Return",
        "3. You have a child or stepchild (not a foster",
        "4. The exclusion from income for dependent",
        # Hyphen-wrapped fragments split across two lines.
        "2. You paid more than half the cost of keep-",
        # Print artifacts and form numbers.
        "AH XSL/XML",
        "8332 Release/Revocation of Release of",
        # Table-of-contents entries with dot leaders.
        "2. Who Are Employees? . . . . . . . . . . . . . 14",
        "11. Depositing Taxes . . . . . . . . . . . . . . 31",
        # A sentence that merely begins with the word "Section".
        "Section 3509 rates aren't available if you intentionally",
    ],
)
def test_real_world_false_positives_are_rejected(line: str) -> None:
    assert looks_like_heading(line) is False, line


@pytest.mark.parametrize(
    "line",
    [
        "3. Family Employees",
        "11. Depositing Taxes",
        "14. Federal Unemployment",
        "SECTION 4",
    ],
)
def test_real_world_headings_still_detected(line: str) -> None:
    assert looks_like_heading(line) is True, line


def test_title_case_separates_titles_from_sentences() -> None:
    from app.chunker import is_title_case

    assert is_title_case("Remote Work Policy") is True
    assert is_title_case("Depositing Taxes") is True
    assert is_title_case("If your spouse itemizes deductions") is False
    assert is_title_case("you paid more than half the cost") is False


# --- generic ALL-CAPS part-labels are not topic headings -------------------


@pytest.mark.parametrize(
    "line",
    ["DISCUSSION", "REFERENCES", "EXAMPLES", "NOTE", "SCOPE", "GENERAL"],
)
def test_generic_labels_are_not_headings(line: str) -> None:
    """ "page 44 | DISCUSSION" tells the reader nothing -- reject these."""
    assert looks_like_heading(line) is False, line


@pytest.mark.parametrize(
    "line",
    ["WORKPLACE SAFETY", "REMOTE WORK POLICY", "EMPLOYEE BENEFITS"],
)
def test_meaningful_all_caps_headings_survive(line: str) -> None:
    assert looks_like_heading(line) is True, line


def test_numbered_heading_is_not_nested_under_an_unnumbered_one() -> None:
    """A numbered section owns its own hierarchy.

    Regression: NIST documents produced breadcrumbs like
    "REFERENCES > 03.05.03 Multi-Factor Authentication", which wrongly implies
    the requirement lives inside a References section.
    """
    text = "\n".join(
        [
            "SECURITY REQUIREMENTS",
            "Some introductory text about the requirements here.",
            "03.05.03 Multi-Factor Authentication",
            "Implement multi-factor authentication for access to accounts.",
            "03.06.04 Incident Response Training",
            "Provide incident response training to users.",
        ]
    )
    blocks = [Block(text=text, page=44, content_type=ContentType.PROSE)]

    chunks = chunk_document(blocks, "standard.pdf")

    sections = {chunk.section for chunk in chunks if chunk.section}
    assert any(section.startswith("03.05.03") for section in sections), sections
    for section in sections:
        assert not section.startswith("SECURITY REQUIREMENTS >"), section


def test_sibling_headings_do_not_nest_under_each_other() -> None:
    """Regression: "3.1 Access Control > 3.5 Identification" was wrong.

    Both are depth 2, so they are siblings. The trail nested them because it
    truncated by list position instead of by actual heading depth.
    """
    text = "\n".join(
        [
            "3.1. Access Control",
            "Rules about who may access what in the system.",
            "3.5. Identification and Authentication",
            "Rules about proving who a user claims to be.",
            "3.5.3. Multi-Factor Authentication",
            "Require more than one factor for privileged accounts.",
        ]
    )
    blocks = [Block(text=text, page=44, content_type=ContentType.PROSE)]

    sections = {item.section for item in chunk_document(blocks, "standard.pdf")}

    # A sibling replaces, it does not nest.
    assert not any(
        section.startswith("3.1. Access Control > 3.5.") for section in sections
    ), sections
    # A genuine child still nests under its parent.
    assert any(
        "3.5. Identification" in section and "3.5.3." in section for section in sections
    ), sections


@pytest.mark.parametrize("line", ["UTC", "AH", "PDF", "IRS"])
def test_short_all_caps_acronyms_are_not_headings(line: str) -> None:
    """Running headers and acronyms are not section titles."""
    assert looks_like_heading(line) is False, line
