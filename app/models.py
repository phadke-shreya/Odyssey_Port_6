"""The data types passed between modules.

These live here, rather than inside whichever module happened to need them first,
so that dependencies point in one direction: every module depends on this one, and
this one depends on nothing.

That matters concretely. When Retrieved lived in vector_store, importing
rag_chain pulled in chromadb -- even though rag_chain never touches the database
and CLAUDE.md section 12 says it must not know Chroma exists. Likewise pdf_parser
imported Block from chunker, so the producer depended on its own consumer.
"""

from dataclasses import dataclass, field
from enum import StrEnum


class ContentType(StrEnum):
    """What kind of content a block holds. Drives which strategy applies."""

    PROSE = "prose"
    TABLE = "table"
    IMAGE_ONLY = "image_only"
    # Text recovered from a picture by OCR. Chunked like prose, but labelled
    # separately so citations can warn that it may contain errors.
    OCR = "ocr"


@dataclass
class Block:
    """One piece of a parsed page, before any chunking has happened."""

    text: str
    page: int
    content_type: ContentType


@dataclass
class Chunk:
    """A child chunk: the unit that gets embedded and stored in the database.

    It carries its parent's full text so that retrieval can expand a matched child
    back up to the whole section without a second lookup -- and so the parent
    survives a restart, since the metadata is persisted alongside the vectors.
    """

    text: str
    parent_id: str
    parent_text: str
    source: str
    page: int
    section: str
    content_type: str


@dataclass
class Retrieved:
    """One parent section retrieved for a question, ready to cite."""

    text: str
    source: str
    page: int
    section: str
    content_type: str
    distance: float
    match_type: str = "semantic"

    def citation(self) -> str:
        """Format this source the way it is shown to the user."""
        parts = [self.source, "page {}".format(self.page)]
        if self.section:
            parts.append(self.section)
        # OCR text is a machine's reading of a picture, not the document's own
        # text layer. Saying so is the difference between a citation the reader
        # can trust and one they cannot.
        if self.content_type == ContentType.OCR.value:
            parts.append("OCR - may contain errors")
        return " | ".join(parts)


@dataclass
class Answer:
    """An answer and the sources it came from. These are never separated."""

    text: str
    sources: list[Retrieved] = field(default_factory=list)
    generated: bool = True
    notice: str = ""

    @property
    def is_dont_know(self) -> bool:
        """Whether this is a refusal rather than an answer.

        A property so callers ask a question instead of string-matching the text
        themselves, which would drift the moment the wording changes.
        """
        return self.text.strip().startswith("I don't know")
