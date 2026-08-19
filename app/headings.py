"""Decide whether a line of text is a section heading, and how deep it is.

Split out of the chunker because it answers a different question. The chunker asks
"where should this document be cut"; this module only ever answers "is this one
line a heading". It knows nothing about chunks, parents, blocks or embeddings, and
that is what makes it testable on strings alone -- most of the regression tests in
the suite are single lines pulled out of real PDFs.

It is also the fiddliest logic in the project. The first version found 120
"headings" in two documents and essentially every one was wrong: page footers,
numbered list items, wrapped sentence fragments, print artifacts. Hence the
governing principle below.
"""

import re

from app import config

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
    significant = [
        word for word in words if len(word) >= 3 and word.lower() not in _SMALL_WORDS
    ]
    if not significant:
        return False
    capitalised = sum(1 for word in significant if word[0].isupper())
    return capitalised / len(significant) >= 0.6


def is_numbered_heading(line: str) -> re.Match[str] | None:
    """Match a numbered heading, returning the match so callers can read its parts.

    Public because the chunker needs to distinguish a numbered heading from an
    unnumbered one: a numbered section owns its own hierarchy and must not be
    nested under an ALL-CAPS label that happens to precede it.
    """
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

    numbered = is_numbered_heading(stripped)
    if numbered:
        return is_title_case(numbered.group(2))

    keyword = _KEYWORD_HEADING.match(stripped)
    if keyword:
        # "SECTION 4" alone is a heading. "Section 3509 rates aren't available
        # if you..." is a sentence that happens to start with the word.
        remainder = keyword.group(3).strip(" :-\u2013\u2014")
        return not remainder or is_title_case(remainder)

    letters = [character for character in stripped if character.isalpha()]
    if not (len(letters) >= 3 and all(character.isupper() for character in letters)):
        return False
    # Reject generic part-labels like "DISCUSSION" that name no topic. Single
    # letters and Roman numerals are ignored, so "STEP I" is judged on the word
    # "STEP" alone and correctly rejected.
    found = [
        word
        for word in _WORDS.findall(stripped.upper())
        if len(word) > 1 and not _ROMAN_NUMERAL.fullmatch(word)
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
    match = is_numbered_heading(line.strip())
    if not match:
        return 1
    return len(match.group(1).split("."))


def build_breadcrumb(trail: list[tuple[int, str]]) -> str:
    """Join a heading stack into a citation-ready string.

    [(1, "5. Working Arrangements"), (2, "5.2 Remote Work")] becomes
    "5. Working Arrangements > 5.2 Remote Work".
    """
    return " > ".join(text for _, text in trail)
