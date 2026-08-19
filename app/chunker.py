"""Turn parsed document blocks into parent/child chunks ready for embedding.

The strategy, in one paragraph: parents are what the model reads (big, full
context), children are what gets embedded and searched (small, precise). Parent
boundaries come from the document's own numbered headings where they exist,
because the author already decided where one topic ends -- far better than any
character count. Tables are the deliberate exception: a table is never split, so
it is its own parent AND its own only child, because slicing it would orphan the
rows from their column headers.

See plan.md section 4 for the full reasoning.
"""

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config
from app.headings import (
    build_breadcrumb,
    heading_depth,
    is_numbered_heading,
    looks_like_heading,
)
from app.models import Block, Chunk, ContentType

logger = logging.getLogger(__name__)


def _explode_long_lines(
    lines: list[tuple[str, int]],
) -> list[tuple[str, int]]:
    """Break any single line that is longer than a whole parent may be.

    Line-based grouping cannot split a line, so one enormous paragraph with no
    newlines in it would otherwise sail past the parent size cap. Each piece
    keeps the page number of the line it came from.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.PARENT_MAX_CHARS,
        chunk_overlap=0,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    exploded: list[tuple[str, int]] = []
    for text, page in lines:
        if len(text) <= config.PARENT_MAX_CHARS:
            exploded.append((text, page))
            continue
        for piece in splitter.split_text(text):
            if piece.strip():
                exploded.append((piece, page))
    return exploded


def _emit_parent(
    lines: list[tuple[str, int]], section: str
) -> list[tuple[str, int, str]]:
    """Turn accumulated (line, page) pairs into one or more parents.

    Page numbers are tracked per line rather than per section, so a parent that
    is split into parts reports the page each PART actually starts on. Taking the
    heading's page instead would cite page 1 for text that lives on page 30 --
    a citation that sends the reader to the wrong place is worse than no
    citation at all.
    """
    lines = _explode_long_lines(lines)

    parents: list[tuple[str, int, str]] = []
    batch: list[str] = []
    batch_page = lines[0][1] if lines else 1
    length = 0

    for text, page in lines:
        if batch and length + len(text) + 1 > config.PARENT_MAX_CHARS:
            parents.append(("\n".join(batch).strip(), batch_page, section))
            batch = []
            batch_page = page
            length = 0
        if not batch:
            batch_page = page
        batch.append(text)
        length += len(text) + 1

    if batch:
        parents.append(("\n".join(batch).strip(), batch_page, section))

    parents = [parent for parent in parents if parent[0]]

    # Label the pieces only when there is more than one, so a normal section
    # keeps a clean breadcrumb.
    if len(parents) > 1 and section:
        total = len(parents)
        parents = [
            (text, page, "{} (part {} of {})".format(section, index, total))
            for index, (text, page, _) in enumerate(parents, start=1)
        ]
    return parents


def _prose_parents(blocks: list[Block]) -> list[tuple[str, int, str]]:
    """Group prose blocks into parents, returning (text, page, section) triples.

    Prefers real section headings. Falls back to size-based splitting when the
    document has fewer than MIN_HEADINGS_FOR_SECTIONS of them -- plenty of real
    PDFs have no numbering at all, and a nice feature must never be the thing
    that breaks on an unseen document.
    """
    lines: list[tuple[str, int]] = []
    for block in blocks:
        for line in block.text.split("\n"):
            lines.append((line, block.page))

    heading_count = sum(1 for line, _ in lines if looks_like_heading(line))
    if heading_count < config.MIN_HEADINGS_FOR_SECTIONS:
        logger.info(
            "Only %s headings found; using size-based parents instead",
            heading_count,
        )
        return _emit_parent(lines, "")

    parents: list[tuple[str, int, str]] = []
    # Each entry is (depth, text). Nesting is decided by depth, not by position:
    # "3.1 Access Control" and "3.5 Identification" are siblings, so the second
    # must REPLACE the first rather than appear beneath it.
    trail: list[tuple[int, str]] = []
    current: list[tuple[str, int]] = []
    current_section = ""

    for line, page in lines:
        if looks_like_heading(line):
            if current:
                parents.extend(_emit_parent(current, current_section))
                current = []
            stripped_line = line.strip()
            numbered = is_numbered_heading(stripped_line) is not None
            # A numbered heading owns its own numbering, so it must not be
            # nested under an unnumbered one: "REFERENCES > 03.05.03 Multi-Factor
            # Authentication" wrongly implies the section lives in References.
            if numbered and trail and is_numbered_heading(trail[0][1]) is None:
                trail.clear()
            if not numbered:
                trail.clear()
            depth = heading_depth(line)
            # Keep only entries shallower than this heading: anything at this
            # depth or deeper is a sibling or a child of what came before, not an
            # ancestor. A filter rather than a pop-loop, per CLAUDE.md section 7 --
            # equivalent here because the trail is always ordered by depth.
            trail = [entry for entry in trail if entry[0] < depth]
            trail.append((depth, stripped_line))
            current_section = build_breadcrumb(trail)
            current.append((line.strip(), page))
        else:
            current.append((line, page))

    if current:
        parents.extend(_emit_parent(current, current_section))

    return parents


def _children_for(text: str) -> list[str]:
    """Slice a parent into small, overlapping children for embedding.

    Cuts at the nicest available seam: paragraph, then sentence, then space,
    and only mid-word as a last resort. Overlap protects a fact that happens to
    straddle a cut line.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHILD_CHUNK_SIZE,
        chunk_overlap=config.CHILD_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [piece for piece in splitter.split_text(text) if piece.strip()]


def _children_for_ocr(text: str) -> list[str]:
    """Slice OCR text into children, one recovered paragraph at a time.

    A scanned page is often a list of unrelated single lines -- a heading, a
    closure notice, a laptop deadline, a contact address -- rather than flowing
    prose. Packing those into one 400-character child blurs its embedding across
    every fact in it, and then no single fact retrieves well.

    Measured on one such page: the query "when must employees collect their
    laptops?" scored 0.582 against the whole page (refused) and 0.229 against just
    the sentence that answers it (retrieved). That is a dilution cost of 0.353,
    which is more than the entire margin of the relevance threshold.

    So the page's own paragraph breaks decide the cuts -- the same principle as
    using headings for prose. The parent is still the whole page, so an answer
    keeps full context.
    """
    paragraphs = [block.strip() for block in text.split("\n\n") if block.strip()]
    children: list[str] = []
    pending = ""

    for paragraph in paragraphs:
        # A fragment on its own carries no meaning, so join it to the next one.
        candidate = "{}\n{}".format(pending, paragraph) if pending else paragraph
        if len(candidate) < config.OCR_MIN_CHILD_CHARS:
            pending = candidate
            continue
        if len(candidate) <= config.CHILD_CHUNK_SIZE:
            children.append(candidate)
        else:
            children.extend(_children_for(candidate))
        pending = ""

    if pending:
        # Whatever is left is too short to stand alone: attach it to the previous
        # child rather than creating a meaningless one.
        if children:
            children[-1] = "{}\n{}".format(children[-1], pending)
        else:
            children.append(pending)

    return children


class _ParentIds:
    """Hands out a unique id per parent within one document.

    A tiny class rather than a passed-around counter, so the three chunk builders
    below cannot accidentally reuse an id -- which would silently merge two
    unrelated sections during retrieval's de-duplication step.
    """

    def __init__(self, filename: str) -> None:
        self._filename = filename
        self._issued = 0

    def next(self) -> str:
        """Return the next parent id for this document."""
        self._issued += 1
        return "{}::p{}".format(self._filename, self._issued)


def _chunks_from_prose(
    blocks: list[Block], filename: str, ids: _ParentIds
) -> list[Chunk]:
    """Group prose into section parents, then slice each into children.

    Grouped across blocks, not per page, so a section that spans a page break
    stays one parent.
    """
    chunks: list[Chunk] = []
    for text, page, section in _prose_parents(blocks):
        parent_id = ids.next()
        for child in _children_for(text):
            chunks.append(
                Chunk(
                    text=child,
                    parent_id=parent_id,
                    parent_text=text,
                    source=filename,
                    page=page,
                    section=section,
                    content_type=ContentType.PROSE.value,
                )
            )
    return chunks


def _chunks_from_ocr(
    blocks: list[Block], filename: str, ids: _ParentIds
) -> list[Chunk]:
    """Slice each OCR'd page into children, keeping the page as the parent.

    Per page rather than grouped, because a scan carries no section headings to
    group by.
    """
    chunks: list[Chunk] = []
    for block in blocks:
        parent_id = ids.next()
        for child in _children_for_ocr(block.text):
            chunks.append(
                Chunk(
                    text=child,
                    parent_id=parent_id,
                    parent_text=block.text,
                    source=filename,
                    page=block.page,
                    section="",
                    content_type=ContentType.OCR.value,
                )
            )
    return chunks


def _chunks_from_whole_blocks(
    blocks: list[Block], filename: str, ids: _ParentIds
) -> list[Chunk]:
    """Turn each table or placeholder into exactly one unsliced chunk.

    A table is its own parent and its own only child. Slicing it would separate
    the rows from their column headers, which is what makes the numbers mean
    anything.
    """
    chunks: list[Chunk] = []
    for block in blocks:
        if len(block.text) > config.CHUNK_WARN_CHARS:
            logger.warning(
                "Chunk on page %s of %s is %s chars, over the %s-char input "
                "window of the configured embedding model; the remainder will "
                "not be searchable",
                block.page,
                filename,
                len(block.text),
                config.CHUNK_WARN_CHARS,
            )
        chunks.append(
            Chunk(
                text=block.text,
                parent_id=ids.next(),
                parent_text=block.text,
                source=filename,
                page=block.page,
                section="",
                content_type=block.content_type.value,
            )
        )
    return chunks


def _of_type(blocks: list[Block], *kinds: ContentType) -> list[Block]:
    """Blocks matching any of the given content types, in document order."""
    return [block for block in blocks if block.content_type in kinds]


def chunk_document(blocks: list[Block], filename: str) -> list[Chunk]:
    """Split parsed blocks into child chunks ready for embedding.

    Each content type gets the strategy that suits it, which is the whole point of
    classifying them in the parser:

    - prose: section parents, sliced into small children
    - OCR text: one parent per page, sliced on the page's own paragraph breaks
    - tables and placeholders: one chunk each, never sliced

    Args:
        blocks: Parsed blocks from pdf_parser, in document order.
        filename: Source document name, stored on every chunk for citations.

    Returns:
        Child chunks, each carrying its parent's full text in metadata.

    Raises:
        ValueError: If blocks is empty.
    """
    if not blocks:
        raise ValueError("chunk_document called with no blocks")

    ids = _ParentIds(filename)
    chunks: list[Chunk] = []
    chunks.extend(
        _chunks_from_prose(_of_type(blocks, ContentType.PROSE), filename, ids)
    )
    chunks.extend(_chunks_from_ocr(_of_type(blocks, ContentType.OCR), filename, ids))
    chunks.extend(
        _chunks_from_whole_blocks(
            _of_type(blocks, ContentType.TABLE, ContentType.IMAGE_ONLY),
            filename,
            ids,
        )
    )

    logger.info("%s produced %s child chunks", filename, len(chunks))
    return chunks
