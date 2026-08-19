"""Read a PDF into clean, classified blocks ready for chunking.

Four jobs, in order:

1. Extract tables properly (pdfplumber finds them; guessing from whitespace
   does not) and render them as Markdown so the row/column relationship
   survives.
2. Remove that table text from the page's plain text, so a table is not stored
   twice -- once as a proper table and once as leftover spaghetti.
3. Notice pages with no extractable text: those are scans or screenshots, and
   embeddings cannot see pictures. They become a visible placeholder, never a
   silent gap.
4. Strip lines that repeat on most pages (running headers, footers, page
   numbers) because that noise otherwise pollutes every single chunk.

See plan.md section 4 for the reasoning behind each.
"""

import logging
import re
from collections import Counter
from pathlib import Path

import pdfplumber

from app import config, ocr
from app.chunker import Block, ContentType

logger = logging.getLogger(__name__)

# How much of a page's top and bottom to inspect for running headers/footers.
_EDGE_LINES = 3

# A line must appear on at least this fraction of pages to count as furniture.
_REPEAT_FRACTION = 0.5

# Collapse runs of 3+ blank lines, and stray runs of spaces.
_MANY_BLANKS = re.compile(r"\n{3,}")
_MANY_SPACES = re.compile(r"[ \t]{2,}")


def _cell(value: str | None) -> str:
    """Normalise one table cell: never None, never multi-line, never padded."""
    if value is None:
        return ""
    return " ".join(value.split())


def is_real_table(rows: list[list[str | None]]) -> bool:
    """Reject layout boxes that pdfplumber reports as tables.

    PDFs use table borders for visual callouts and sidebars all the time. Those
    come back from find_tables() looking like a one-column, one-row table. Real
    data tables have at least two columns and at least two rows, so requiring
    that filters out the decoration without losing anything useful.
    """
    cleaned = [[_cell(c) for c in row] for row in rows if row]
    cleaned = [row for row in cleaned if any(row)]
    if len(cleaned) < 2:
        return False
    width = max(len(row) for row in cleaned)
    if width < 2:
        return False
    # At least two rows must actually populate two or more columns, otherwise
    # it is a single column of text that happens to sit inside a border.
    populated = sum(1 for row in cleaned if sum(1 for c in row if c) >= 2)
    return populated >= 2


def table_to_markdown(rows: list[list[str | None]]) -> str:
    """Render extracted table rows as a Markdown table.

    Markdown is used because it keeps each value visibly attached to its column
    header. Raw PDF table text collapses into ambiguous whitespace where you can
    no longer tell which number belongs to which column.
    """
    cleaned = [[_cell(c) for c in row] for row in rows if row]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return ""

    width = max(len(row) for row in cleaned)
    padded = [row + [""] * (width - len(row)) for row in cleaned]

    header = padded[0]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * width) + "|")
    for row in padded[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _caption_for(page: pdfplumber.page.Page, table, source: str) -> str:
    """Build a one-line description of what a table is about.

    A bare grid of numbers has almost nothing for an embedding model to latch
    onto -- nothing in "| Junior | 15 | 10 |" says "vacation". The caption is
    what makes a table findable, so it always goes on the FIRST line of the
    chunk. Where possible it borrows the line of text sitting just above the
    table, which is usually the table's own title.
    """
    hint = ""
    try:
        above = page.crop((0, 0, page.width, max(table.bbox[1] - 2, 1)))
        text = above.extract_text() or ""
        candidates = [ln.strip() for ln in text.split("\n") if ln.strip()]
        # Walk upwards and take the first line that reads like a title. A line
        # starting lower-case is almost always the tail of a wrapped sentence,
        # which makes a confusing caption ("...ing requirements for most").
        for line in reversed(candidates[-4:]):
            if len(line) >= 3 and (line[0].isupper() or line[0].isdigit()):
                hint = line[: config.MAX_HEADING_CHARS]
                break
    except Exception:  # noqa: BLE001 - a missing caption must never fail a parse
        logger.debug(
            "Could not read caption text above table on page %s", page.page_number
        )

    base = "Table from {}, page {}".format(source, page.page_number)
    if hint:
        return "{}: {}".format(base, hint)
    return base


def _text_outside_tables(page: pdfplumber.page.Page, tables: list) -> str:
    """Return the page's text with every table's characters removed.

    Without this, each table is indexed twice: once as a clean Markdown table
    and once as the whitespace-mangled version that extract_text() returns.
    """
    if not tables:
        return page.extract_text() or ""

    boxes = [t.bbox for t in tables]

    def keep(obj: dict) -> bool:
        if obj.get("object_type") != "char":
            return True
        for x0, top, x1, bottom in boxes:
            inside_x = obj["x0"] >= x0 - 1 and obj["x1"] <= x1 + 1
            inside_y = obj["top"] >= top - 1 and obj["bottom"] <= bottom + 1
            if inside_x and inside_y:
                return False
        return True

    try:
        return page.filter(keep).extract_text() or ""
    except Exception:  # noqa: BLE001 - fall back to unfiltered text
        logger.warning(
            "Could not filter table text on page %s; text may duplicate",
            page.page_number,
        )
        return page.extract_text() or ""


def _repeated_lines(page_texts: list[str]) -> set[str]:
    """Find running headers and footers by looking for lines that repeat.

    Only the top and bottom few lines of each page are considered, so a genuinely
    repeated sentence in the body is not mistaken for furniture.
    """
    if len(page_texts) < 3:
        return set()

    counts: Counter[str] = Counter()
    for text in page_texts:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        edges = lines[:_EDGE_LINES] + lines[-_EDGE_LINES:]
        for line in set(edges):
            counts[line] += 1

    threshold = max(2, int(len(page_texts) * _REPEAT_FRACTION))
    repeated = {line for line, n in counts.items() if n >= threshold}
    if repeated:
        logger.info("Stripping %s repeated header/footer lines", len(repeated))
    return repeated


def _clean(text: str, repeated: set[str]) -> str:
    """Drop furniture lines and normalise whitespace."""
    kept = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and stripped in repeated:
            continue
        if stripped.isdigit():  # a bare page number
            continue
        kept.append(_MANY_SPACES.sub(" ", line.rstrip()))
    joined = "\n".join(kept)
    return _MANY_BLANKS.sub("\n\n", joined).strip()


def parse_pdf(path: Path) -> list[Block]:
    """Read one PDF into classified blocks, in document order.

    Args:
        path: Path to a .pdf file.

    Returns:
        Blocks of type PROSE, TABLE, or IMAGE_ONLY, ready for chunk_document.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file is not a PDF or yields nothing usable.
    """
    if not path.exists():
        raise FileNotFoundError("No such file: {}".format(path))
    if path.suffix.lower() != ".pdf":
        raise ValueError("Not a PDF: {}".format(path.name))

    source = path.name
    blocks: list[Block] = []
    raw_texts: list[str] = []
    table_rows: list[tuple[int, str]] = []
    image_pages: list[int] = []
    ocr_pages: list[tuple[int, str, float]] = []

    try:
        opened = pdfplumber.open(path)
    except Exception as error:  # noqa: BLE001 - any PDF library failure
        raise ValueError(
            "Could not open {}. It may be corrupt or password protected.".format(
                path.name
            )
        ) from error

    with opened as pdf:
        for page in pdf.pages:
            found = page.find_tables()

            for table in found:
                try:
                    rows = table.extract()
                    if not is_real_table(rows):
                        continue
                    markdown = table_to_markdown(rows)
                except Exception:  # noqa: BLE001 - skip a broken table, keep going
                    logger.warning(
                        "Could not extract a table on page %s of %s",
                        page.page_number,
                        source,
                    )
                    continue
                if not markdown:
                    continue
                caption = _caption_for(page, table, source)
                table_rows.append(
                    (page.page_number, "{}\n{}".format(caption, markdown))
                )

            try:
                body = _text_outside_tables(page, found)
            except Exception:  # noqa: BLE001 - a damaged page must not kill the file
                logger.warning(
                    "Could not read text on page %s of %s", page.page_number, source
                )
                body = ""
            raw_texts.append(body)

            # An image-only page is one with almost no text AND no table.
            if len(body.strip()) < config.MIN_CHARS_FOR_TEXT_PAGE and not found:
                # Try OCR before declaring it unreadable. Only these pages are
                # OCR'd, so a long document with one scan pays for one page.
                result = ocr.ocr_page(page)
                if result.usable:
                    ocr_pages.append((page.page_number, result.text, result.confidence))
                else:
                    image_pages.append(page.page_number)

    repeated = _repeated_lines(raw_texts)

    for index, body in enumerate(raw_texts, start=1):
        cleaned = _clean(body, repeated)
        if cleaned:
            blocks.append(
                Block(text=cleaned, page=index, content_type=ContentType.PROSE)
            )

    for page_number, markdown in table_rows:
        blocks.append(
            Block(text=markdown, page=page_number, content_type=ContentType.TABLE)
        )

    for page_number, text, confidence in ocr_pages:
        logger.info(
            "Page %s of %s had no text layer; OCR recovered %s characters "
            "at %.0f%% confidence",
            page_number,
            source,
            len(text),
            confidence,
        )
        blocks.append(Block(text=text, page=page_number, content_type=ContentType.OCR))

    for page_number in image_pages:
        logger.warning(
            "Page %s of %s has no extractable text (likely a scan); "
            "it cannot be searched",
            page_number,
            source,
        )
        blocks.append(
            Block(
                text=(
                    "Page {} of {} appears to be an image or scanned page, so "
                    "its contents could not be indexed. Please open the "
                    "original document to read this page."
                ).format(page_number, source),
                page=page_number,
                content_type=ContentType.IMAGE_ONLY,
            )
        )

    if not blocks:
        raise ValueError("No usable content extracted from {}".format(source))

    logger.info(
        "%s: %s prose, %s tables, %s OCR pages, %s unreadable pages",
        source,
        sum(1 for b in blocks if b.content_type is ContentType.PROSE),
        len(table_rows),
        len(ocr_pages),
        len(image_pages),
    )
    return blocks
