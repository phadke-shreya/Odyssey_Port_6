"""Tests for PDF parsing.

The behaviour-critical ones are test_image_only_page_is_reported (a scan must
never vanish silently) and test_table_text_is_not_duplicated_in_prose (a table
must be stored once, as a table).
"""

from pathlib import Path

import pytest

from app.chunker import ContentType
from app.pdf_parser import is_real_table, parse_pdf, table_to_markdown

DOCS = Path(__file__).resolve().parent.parent / "documents"
SCAN = DOCS / "scanned_notice_IMAGE_ONLY.pdf"
WITH_TABLES = DOCS / "Product_Manual_ComputeModule4.pdf"

needs_scan = pytest.mark.skipif(not SCAN.exists(), reason="scan fixture missing")
needs_tables = pytest.mark.skipif(
    not WITH_TABLES.exists(), reason="table fixture missing"
)


# --- table rendering -----------------------------------------------------


def test_table_to_markdown_keeps_headers_with_rows() -> None:
    rows = [
        ["Role", "Vacation Days"],
        ["Junior", "15"],
        ["Manager", "25"],
    ]

    markdown = table_to_markdown(rows)

    lines = markdown.split("\n")
    assert lines[0] == "| Role | Vacation Days |"
    assert lines[1] == "|---|---|"
    assert "| Manager | 25 |" in markdown


def test_table_to_markdown_handles_empty_cells() -> None:
    rows = [["Role", None], ["Junior", "15"]]

    markdown = table_to_markdown(rows)

    assert "| Role |  |" in markdown


def test_ragged_rows_are_padded_not_dropped() -> None:
    rows = [["A", "B", "C"], ["1"], ["2", "3"]]

    markdown = table_to_markdown(rows)

    # Every row must have the same number of pipes, or Markdown misaligns.
    counts = {line.count("|") for line in markdown.split("\n")}
    assert len(counts) == 1


# --- layout-box rejection ------------------------------------------------


def test_single_column_box_is_not_a_real_table() -> None:
    """PDFs use borders for callouts; those are not data tables."""
    assert is_real_table([["Get forms faster at IRS.gov"]]) is False
    assert is_real_table([["A note"], ["spanning two lines"]]) is False


def test_two_by_two_grid_is_a_real_table() -> None:
    assert is_real_table([["Role", "Days"], ["Junior", "15"]]) is True


# --- input validation ----------------------------------------------------


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        parse_pdf(Path("documents/does_not_exist.pdf"))


def test_non_pdf_is_rejected(tmp_path: Path) -> None:
    fake = tmp_path / "notes.txt"
    fake.write_text("this is not a pdf")

    with pytest.raises(ValueError):
        parse_pdf(fake)


# --- real documents ------------------------------------------------------


@needs_scan
def test_scanned_page_is_never_silently_dropped() -> None:
    """A page with no text layer must always leave a trace.

    The invariant is representation, not a particular kind: with OCR available the
    page becomes OCR text, and without it a placeholder saying so. What must never
    happen is the page vanishing, because a reader would then have no way to know
    part of the document was not indexed.

    The two branches are tested individually in test_ocr.py.
    """
    blocks = parse_pdf(SCAN)

    assert blocks, "the scanned page produced nothing at all"
    kinds = {b.content_type for b in blocks}
    assert kinds & {ContentType.OCR, ContentType.IMAGE_ONLY}, kinds

    block = blocks[0]
    assert block.page == 1
    if block.content_type is ContentType.IMAGE_ONLY:
        assert "image or scanned page" in block.text
    else:
        # OCR is imperfect, so check for recovered content, not a transcript.
        assert "bangalore" in block.text.lower()


@needs_tables
def test_real_pdf_yields_prose_and_tables() -> None:
    blocks = parse_pdf(WITH_TABLES)

    prose = [b for b in blocks if b.content_type is ContentType.PROSE]
    tables = [b for b in blocks if b.content_type is ContentType.TABLE]
    assert len(prose) > 5
    assert len(tables) >= 1


@needs_tables
def test_every_table_chunk_starts_with_a_caption() -> None:
    """The caption must be line 1, so it is always inside the embedding window."""
    blocks = parse_pdf(WITH_TABLES)

    tables = [b for b in blocks if b.content_type is ContentType.TABLE]
    for block in tables:
        first_line = block.text.split("\n")[0]
        assert first_line.startswith("Table from ")
        assert WITH_TABLES.name in first_line


@needs_tables
def test_table_text_is_not_duplicated_in_prose() -> None:
    """A table is stored once, as a table.

    The check is deliberately PER PAGE. Comparing against every page's prose
    gives false alarms: a real document legitimately repeats phrases, so a cell
    value on page 2 may also appear in body text on page 10. Duplication caused
    by the parser would show up on the SAME page the table came from.
    """
    blocks = parse_pdf(WITH_TABLES)

    prose_by_page = {
        b.page: b.text for b in blocks if b.content_type is ContentType.PROSE
    }
    tables = [b for b in blocks if b.content_type is ContentType.TABLE]
    assert tables, "no tables found to check"

    leaks = []
    for block in tables:
        same_page_prose = prose_by_page.get(block.page, "")
        rows = [r for r in block.text.split("\n") if r.startswith("|")]
        for row in rows[2:]:
            cells = [c.strip() for c in row.strip("|").split("|")]
            for cell in cells:
                if len(cell) > 25 and cell in same_page_prose:
                    leaks.append((block.page, cell[:40]))
                    break

    assert not leaks, "table rows still present as prose: {}".format(leaks)
