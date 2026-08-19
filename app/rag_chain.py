"""Turn a question plus retrieved sections into a cited answer.

This module owns the prompt and the call to the language model. It does not know
that ChromaDB exists -- it receives already-retrieved sections, so swapping the
vector database out touches vector_store.py alone.

Two rules are enforced here rather than hoped for:

1. An answer can never come back without its sources. They travel together in a
   single Answer object, so there is no code path that returns text alone.
2. If nothing was retrieved, the model is never called at all. "I don't know" is
   decided by the retrieval distance, not left to the model's goodwill.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app import config
from app.vector_store import Retrieved

if TYPE_CHECKING:  # imported lazily at runtime, see _build_llm
    from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# The exact words used when the documents do not contain the answer. Kept as a
# constant so the API, the UI, and the tests all agree on it.
DONT_KNOW = (
    "I don't know. I could not find anything about that in the uploaded " "documents."
)

NO_DOCUMENTS = (
    "There are no documents indexed yet. Upload a PDF first, or run "
    "'python ingest.py'."
)

# A string LITERAL with {named} placeholders. Nothing is ever interpolated into
# this: LangChain fills the slots. Mixing .format() into a prompt template makes
# the two brace systems collide -- see CLAUDE.md section 5.
PROMPT_TEMPLATE = """You are a careful assistant answering questions about a \
company's internal documents.

Answer using ONLY the numbered sources below.

How to write the answer:
- Write it as normal prose, in your own words. Two or three sentences.
- Put the citation in brackets at the END of a sentence, like this: (Source 2)
- Do NOT begin with, or copy, the "Source N: filename | page | section" heading
  line. That line is for you, not for the reader.
- Do NOT paste the source text verbatim. Summarise it.

Example of a GOOD answer:
  Employees must get written manager approval before working remotely, and must
  still spend at least three days a week in the office (Source 1).

Example of a BAD answer (never do this):
  Source 1: handbook.pdf | page 12 | 5.2 Remote Work Policy
  Fridays are designated as optional remote days.

Other rules:
- The ONLY valid labels are the "Source N" headings below. The document text
  contains its own bracketed reference numbers such as [50] or [41]; those
  belong to the document and are NOT citations. Never write them.
- If the sources answer the question, answer it. Do not add any sentence saying
  you do not know.
- If the sources genuinely do not answer the question, your ENTIRE reply must be
  exactly this and nothing else: {dont_know}
- Never mix the two: either answer, or say you do not know.
- Never use outside knowledge, and never guess.
- If two sources disagree, say so and cite both. Do not silently pick one.

Sources:
{context}

Question: {question}

Answer:"""


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
        themselves, which would drift from DONT_KNOW the moment it is reworded.
        """
        return self.text.strip().startswith("I don't know")


class GenerationUnavailable(RuntimeError):
    """Raised when no working language model is configured."""


def format_context(sections: list[Retrieved]) -> str:
    """Number the retrieved sections so the model can cite them.

    Labelled "Source 1", not "[1]", on purpose: real documents contain their own
    bracketed reference markers (NIST text is full of "[50]"), and a model asked
    to cite with [n] will happily copy those straight out of the source text.
    """
    blocks = []
    for index, section in enumerate(sections, start=1):
        blocks.append(
            "Source {}: {}\n{}".format(index, section.citation(), section.text)
        )
    return "\n\n".join(blocks)


def _build_llm() -> "ChatOpenAI":
    """Create the chat model, honouring a company gateway if one is set.

    Raises:
        GenerationUnavailable: If no API key is configured, or the LangChain
            OpenAI package cannot be loaded.
    """
    if not config.CHAT_API_KEY:
        raise GenerationUnavailable(
            "No chat API key is set, so answers cannot be written. "
            "Retrieval and citations still work."
        )
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as error:  # pragma: no cover - dependency issue
        raise GenerationUnavailable("langchain-openai is not installed.") from error

    return ChatOpenAI(
        model=config.CHAT_MODEL,
        temperature=config.CHAT_TEMPERATURE,
        api_key=config.CHAT_API_KEY,
        base_url=config.CHAT_BASE_URL,
        timeout=60,
        max_retries=1,
    )


def _friendly_error(error: Exception) -> str:
    """Translate a provider exception into something a user can act on.

    The user must never see a stack trace. The full traceback goes to the log.
    """
    name = type(error).__name__
    text = str(error).lower()

    if "authenticationerror" in name.lower() or "invalid_api_key" in text:
        return (
            "The chat API key was rejected. Check CHAT_API_KEY in your .env "
            "file -- and if you use a gateway, check CHAT_BASE_URL too."
        )
    if "notfound" in name.lower() or "model_not_found" in text:
        return (
            "The model '{}' was not found. If you use a company gateway it may "
            "expose different model names.".format(config.CHAT_MODEL)
        )
    if "ratelimit" in name.lower():
        return "The service is rate limited right now. Please try again shortly."
    if "connection" in name.lower() or "timeout" in name.lower():
        return (
            "Could not reach the AI service. Check your internet connection "
            "and CHAT_BASE_URL."
        )
    return "The AI service returned an error ({}).".format(name)


def answer_question(question: str, sections: list[Retrieved]) -> Answer:
    """Write an answer to a question using only the given sections.

    Never calls the model when there is nothing to answer from, and always
    returns the sources alongside the text.

    Args:
        question: The user's question, already validated.
        sections: Parent sections from vector_store.search, nearest first.

    Returns:
        An Answer. When generation is unavailable the sources are still
        returned, with `generated` False and a `notice` explaining why.
    """
    if not sections:
        # Decided by retrieval distance, not by the model.
        return Answer(text=DONT_KNOW, sources=[], generated=False)

    try:
        llm = _build_llm()
    except GenerationUnavailable as error:
        logger.warning("Generation unavailable: %s", error)
        return Answer(
            text="",
            sources=sections,
            generated=False,
            notice=str(error),
        )

    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm

    try:
        response = chain.invoke(
            {
                "context": format_context(sections),
                "question": question,
                "dont_know": DONT_KNOW,
            }
        )
    except Exception as error:  # noqa: BLE001 - boundary: never leak a traceback
        logger.exception("Model call failed")
        return Answer(
            text="",
            sources=sections,
            generated=False,
            notice=_friendly_error(error),
        )

    raw = str(getattr(response, "content", "") or "")
    cleaned = _normalise_citations(
        _strip_echoed_source_header(
            _strip_document_markers(_strip_contradictory_hedge(raw))
        )
    )
    return Answer(text=cleaned, sources=sections, generated=True)


def _strip_contradictory_hedge(text: str) -> str:
    """Remove a trailing "I don't know" that follows a real answer.

    Smaller models sometimes answer the question and then append the fallback
    sentence anyway, which reads as the app contradicting itself. If something
    substantial came first, that is the answer -- drop the hedge.
    """
    cleaned = text.strip()
    position = cleaned.find("I don't know.")
    if position <= 0:
        return cleaned
    before = cleaned[:position].strip()
    if len(before) < 40:
        return cleaned
    logger.info("Dropped a contradictory trailing hedge from the answer")
    return before


# Bracketed numbers belong to the source documents, never to our citations,
# which are written as "(Source 1)". Anything matching this in an answer was
# copied out of the document text and points at nothing the reader can see.
_DOCUMENT_MARKER = re.compile(r"\s*\[\d{1,3}\]")


def _strip_document_markers(text: str) -> str:
    """Remove reference markers the model copied out of the source text.

    NIST-style documents are full of "[50]". A model that echoes one has
    produced a citation the reader cannot follow, so it is removed rather than
    shown. Valid citations use the "(Source N)" form and are untouched.
    """
    cleaned = _DOCUMENT_MARKER.sub("", text)
    if cleaned != text:
        logger.info("Removed document reference markers from the answer")
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


# "Source 2: report.pdf | page 55 | 3.8 Media Protection" at the start of an
# answer is the context heading echoed back, not prose the reader wants.
_ECHOED_HEADER = re.compile(r"^Source\s*\d+\s*:.*(?:\n|$)", re.MULTILINE)


def _strip_echoed_source_header(text: str) -> str:
    """Drop context heading lines the model copied into its answer.

    Small models tend to reproduce the shape of their input. The heading is
    routing information for the model; the reader already sees the same details,
    formatted properly, in the Sources list below the answer.
    """
    cleaned = _ECHOED_HEADER.sub("", text).strip()
    if not cleaned:  # the whole answer was headings; keep the original
        return text.strip()
    if cleaned != text.strip():
        logger.info("Removed an echoed source heading from the answer")
    return cleaned


# Models write "[Source 2]", "Source 2", or "(source 2)" interchangeably. The
# meaning is identical, so normalise rather than fight it in the prompt.
# Two alternatives on purpose: a bracketed form may pad inside the brackets,
# but the bare form must not swallow the spaces around it.
_CITATION_VARIANTS = re.compile(
    r"[\[(]\s*Source\s*(\d+)\s*[\])]|\bSource\s+(\d+)\b", re.IGNORECASE
)


def _normalise_citations(text: str) -> str:
    """Render every citation the model wrote in one consistent form.

    A small model will not follow a bracket style reliably, and the style
    carries no meaning -- so it is corrected here instead of being demanded ever
    more loudly in the prompt.
    """

    def replace(match: re.Match[str]) -> str:
        """Rewrite one matched citation into the canonical form."""
        number = match.group(1) or match.group(2)
        return "(Source {})".format(number)

    return _CITATION_VARIANTS.sub(replace, text)
