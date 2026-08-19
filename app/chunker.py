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
import re
from dataclasses import dataclass
from enum import StrEnum

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config

logger = logging.getLogger(__name__)


class ContentType(StrEnum):
    """What kind of content a block holds. Drives which strategy applies."""

    PROSE = "prose"
    TABLE = "table"
    IMAGE_ONLY = "image_only"


@dataclass
class Block:
    """One piece of a parsed page, before any chunking has happened."""

    text: str
    page: int
    content_type: ContentType


@dataclass
class Chunk:
    """A child chunk: the unit that gets embedded and stored in ChromaDB.

    It carries its parent's full text so that retrieval can expand a matched
    child back up to the whole section without a second lookup -- and so the
    parent survives a restart, since Chroma persists metadata to disk.
    """

    text: str
    parent_id: str
    parent_text: str
    source: str
    page: int
    section: str
    content_type: str


# Heading detection is deliberately HIGH PRECISION. Real documents are full of
# things that look like headings but are not: numbered list items, page footers,
# wrapped sentence fragments, print artifacts. A WRONG section label in a
# citation is worse than no label -- and missing a heading is safe, because the
# caller falls back to size-based parents. So every rule below errs toward "no".

# "5.2 Remote Work Policy" / "5.2.1 Equipment" -- multi-level, dot optional.
_MULTI_LEVEL_HEADING = re.compile(r"^(\d+(?:\.\d+)+)\.?\s+([A-Z].*)$")

# "1. Introduction" -- single level, the trailing dot is REQUIRED. Without that
# requirement, page footers like "8 Publication 15 (2026)" match.
_SINGLE_LEVEL_HEADING = re.compile(r"^(\d+)\.\s+([A-Z].*)$")

# "SECTION 4", "Article 3:", "Chapter 12".
_KEYWORD_HEADING = re.compile(
    r"^(SECTION|ARTICLE|CHAPTER|PART|APPENDIX)\s+([0-9IVXLC]+)\b(.*)$",
    re.IGNORECASE,
)

# Sentence-ending punctuation. A real heading almost never ends this way.
_SENTENCE_END = (".", "!", "?", ",", ";", ":")

# Words that do not count when judging Title Case.
_SMALL_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "and",
        "or",
        "for",
        "to",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "as",
        "is",
        "are",
        "not",
        "but",
        "if",
        "you",
    }
)

_WORDS = re.compile(r"[A-Za-z][A-Za-z'\u2019-]*")

# Dot leaders ". . . . 14" mark a table-of-contents entry. The same heading
# appears again in the body, so indexing the TOC version just adds a duplicate
# with a page number stuck on the end.
_DOT_LEADER = re.compile(r"\.\s*\.\s*\.")

# Roman numerals, so "STEP I" and "PART IV" are judged on their word alone.
_ROMAN_NUMERAL = re.compile(r"[IVXLCDM]+")

# ALL-CAPS words that label a PART of a section rather than naming a topic.
# Many standards documents repeat these under every requirement, and using them
# as breadcrumbs produces citations like "page 44 | DISCUSSION", which tells the
# reader nothing about where they are.
_GENERIC_LABELS = frozenset(
    {
        "DISCUSSION",
        "REFERENCES",
        "EXAMPLES",
        "EXAMPLE",
        "NOTE",
        "NOTES",
        "PURPOSE",
        "SCOPE",
        "SUMMARY",
        "OVERVIEW",
        "BACKGROUND",
        "GENERAL",
        "INTRODUCTION",
        "CONTENTS",
        "APPENDIX",
        "GLOSSARY",
        "INDEX",
        "ACKNOWLEDGEMENTS",
        "ACKNOWLEDGMENTS",
        "ABSTRACT",
        "KEYWORDS",
        "AUDIENCE",
        "DEFINITIONS",
        "REQUIREMENTS",
        "DISCLAIMER",
        # Seen repeating on every page of a real HR policy manual and a lab
        # safety SOP. As breadcrumbs they are worse than nothing: every
        # citation would read "page 7 | POLICY".
        "STATEMENT",
        "POLICY",
        "PROCEDURE",
        "TABLE",
        "OF",
        "FORWARD",
        "FOREWORD",
        "STEP",
        "RESPONSIBILITIES",
        "APPLICABILITY",
        "REVISION",
        "REVISIONS",
        "APPROVAL",
    }
)


def is_title_case(text: str) -> bool:
    """Whether text reads like a title rather than a sentence.

    This is the rule that separates "5.2 Remote Work Policy" (a heading) from
    "11. If your spouse itemizes deductions, you" (a numbered list item). Titles
    capitalise their significant words; sentences do not.
    """
    words = _WORDS.findall(text)
    significant = [w for w in words if len(w) >= 3 and w.lower() not in _SMALL_WORDS]
    if not significant:
        return False
    capitalised = sum(1 for w in significant if w[0].isupper())
    return capitalised / len(significant) >= 0.6


def _numbered_match(line: str):
    """Return a regex match if the line is a numbered heading, else None."""
    return _MULTI_LEVEL_HEADING.match(line) or _SINGLE_LEVEL_HEADING.match(line)


def looks_like_heading(line: str) -> bool:
    """Decide whether a single line is a section heading.

    Over-detection is as damaging as under-detection: a document yielding
    hundreds of "headings" has really yielded none, and every citation built from
    them is misleading. When in doubt this returns False, and the caller falls
    back to size-based parents with no section label at all.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > config.MAX_HEADING_CHARS:
        return False
    if stripped.endswith(_SENTENCE_END):
        return False
    # A trailing hyphen means a word was split across lines: a fragment, not a
    # heading (e.g. "2. You paid more than half the cost of keep-").
    if stripped.endswith(("-", "\u2013", "\u2014")):
        return False
    # Headings rarely contain a comma; sentences and list items often do.
    if "," in stripped:
        return False
    # Slashes signal print artifacts and form names ("AH XSL/XML").
    if "/" in stripped:
        return False
    # A table-of-contents entry, not the section itself.
    if _DOT_LEADER.search(stripped):
        return False

    numbered = _numbered_match(stripped)
    if numbered:
        return is_title_case(numbered.group(2))

    keyword = _KEYWORD_HEADING.match(stripped)
    if keyword:
        # "SECTION 4" alone is a heading. "Section 3509 rates aren't available
        # if you..." is a sentence that happens to start with the word.
        remainder = keyword.group(3).strip(" :-\u2013\u2014")
        return not remainder or is_title_case(remainder)

    letters = [c for c in stripped if c.isalpha()]
    if not (len(letters) >= 3 and all(c.isupper() for c in letters)):
        return False
    # Reject generic part-labels like "DISCUSSION" that name no topic. Single
    # letters and Roman numerals are ignored, so "STEP I" is judged on the word
    # "STEP" alone and correctly rejected.
    found = [
        w
        for w in _WORDS.findall(stripped.upper())
        if len(w) > 1 and not _ROMAN_NUMERAL.fullmatch(w)
    ]
    if not found or set(found).issubset(_GENERIC_LABELS):
        return False
    # A short ALL-CAPS fragment is usually a running header or an acronym
    # ("UTC"), not a section title. Require either several words or one
    # substantial word.
    return len(found) >= 2 or len(found[0]) >= 6


def heading_depth(line: str) -> int:
    """How deeply nested a heading is: "5" -> 1, "5.2" -> 2, "5.2.1" -> 3.

    Used to maintain the breadcrumb trail. Non-numbered headings are treated as
    top level, since there is no reliable depth signal in them.
    """
    match = _numbered_match(line.strip())
    if not match:
        return 1
    return len(match.group(1).split("."))


def build_breadcrumb(trail: list[tuple[int, str]]) -> str:
    """Join a heading stack into a citation-ready string.

    [(1, "5. Working Arrangements"), (2, "5.2 Remote Work")] becomes
    "5. Working Arrangements > 5.2 Remote Work".
    """
    return " > ".join(text for _, text in trail)


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

    parents = [p for p in parents if p[0]]

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
            numbered = _numbered_match(stripped_line) is not None
            # A numbered heading owns its own numbering, so it must not be
            # nested under an unnumbered one: "REFERENCES > 03.05.03 Multi-Factor
            # Authentication" wrongly implies the section lives in References.
            if numbered and trail and _numbered_match(trail[0][1]) is None:
                trail.clear()
            if not numbered:
                trail.clear()
            depth = heading_depth(line)
            # Drop every entry at this depth or deeper: those are siblings or
            # children of what came before, not ancestors of this heading.
            while trail and trail[-1][0] >= depth:
                trail.pop()
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


def chunk_document(blocks: list[Block], filename: str) -> list[Chunk]:
    """Split parsed blocks into child chunks ready for embedding.

    Prose is split into section-based parents, then into small children. Tables
    are never split: each becomes a single chunk that is its own parent and its
    own only child. Image-only pages become a visible placeholder chunk rather
    than being silently dropped.

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

    chunks: list[Chunk] = []
    counter = 0

    # Tables and image placeholders are handled per block; prose is grouped
    # across blocks so that a section can span a page break.
    prose_blocks = [b for b in blocks if b.content_type is ContentType.PROSE]
    other_blocks = [b for b in blocks if b.content_type is not ContentType.PROSE]

    if prose_blocks:
        for text, page, section in _prose_parents(prose_blocks):
            counter += 1
            parent_id = "{}::p{}".format(filename, counter)
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

    for block in other_blocks:
        counter += 1
        parent_id = "{}::p{}".format(filename, counter)
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
        # A table is its own parent and its own only child -- deliberately not
        # sliced, because rows are meaningless without their column headers.
        chunks.append(
            Chunk(
                text=block.text,
                parent_id=parent_id,
                parent_text=block.text,
                source=filename,
                page=block.page,
                section="",
                content_type=block.content_type.value,
            )
        )

    logger.info("%s produced %s child chunks", filename, len(chunks))
    return chunks
