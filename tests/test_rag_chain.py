"""Tests for prompt construction and answer cleanup.

Both behaviours here were real defects seen in live output, not hypotheticals.
"""

from app.rag_chain import (
    DONT_KNOW,
    PROMPT_TEMPLATE,
    Answer,
    _strip_contradictory_hedge,
    format_context,
)
from app.vector_store import Retrieved


def _section(text: str, page: int = 55) -> Retrieved:
    return Retrieved(
        text=text,
        source="NIST.SP.800-171r3.pdf",
        page=page,
        section="3.8. Media Protection",
        content_type="prose",
        distance=0.21,
    )


def test_sources_are_labelled_source_n_not_bracket_n():
    """Regression: the model copied "[50]" out of the document text.

    NIST documents carry their own bracketed reference markers. Labelling our
    sources "[1]" made them indistinguishable, so the model cited [50] -- a
    number that pointed at nothing.
    """
    context = format_context([_section("See SP 800-88 [50] for details.")])

    assert context.startswith("Source 1: ")
    assert "[1]" not in context
    # The document's own marker is left alone; the prompt tells the model to
    # ignore it rather than us mangling the source text.
    assert "[50]" in context


def test_prompt_forbids_the_documents_own_reference_numbers():
    assert "Source N" in PROMPT_TEMPLATE
    assert "(Source 1)" in PROMPT_TEMPLATE
    assert "NOT citations" in PROMPT_TEMPLATE


def test_prompt_forbids_mixing_an_answer_with_i_dont_know():
    assert "ENTIRE reply" in PROMPT_TEMPLATE
    assert "Never mix the two" in PROMPT_TEMPLATE


def test_trailing_hedge_is_stripped_from_a_real_answer():
    """Regression: the app answered, then contradicted itself in the next line."""
    answered = (
        "Media sanitization requires clearing, purging, and destroying media "
        "so that information cannot be reconstructed (Source 2).\n\n"
        "I don't know. I could not find anything about that in the documents."
    )

    cleaned = _strip_contradictory_hedge(answered)

    assert cleaned.startswith("Media sanitization requires")
    assert "I don't know" not in cleaned


def test_a_pure_dont_know_answer_is_left_alone():
    assert _strip_contradictory_hedge(DONT_KNOW) == DONT_KNOW
    assert Answer(text=DONT_KNOW).is_dont_know is True


def test_short_preamble_before_a_hedge_is_not_treated_as_an_answer():
    """ "Hmm. I don't know..." is a refusal, not an answer plus a hedge."""
    text = "Hmm. " + DONT_KNOW

    assert _strip_contradictory_hedge(text) == text


def test_document_reference_markers_are_removed_from_answers():
    """Regression: the model wrote "[50]", a marker copied from NIST text.

    That number points at the document's own bibliography, not at anything the
    reader can see in the Sources list, so showing it is misleading.
    """
    from app.rag_chain import _strip_document_markers

    text = (
        "(Source 1) [50] also references SP 800-88, which covers clearing, "
        "purging [41] and destroying media."
    )

    cleaned = _strip_document_markers(text)

    assert "[50]" not in cleaned
    assert "[41]" not in cleaned
    # Our own citation format must survive untouched.
    assert "(Source 1)" in cleaned
    assert "SP 800-88" in cleaned


def test_echoed_source_heading_is_removed():
    """Regression: the answer began by copying the context heading line."""
    from app.rag_chain import _strip_echoed_source_header

    text = (
        "Source 2: NIST.SP.800-171r3.pdf | page 55 | 03.08.03 Media Sanitization\n"
        "\n"
        "Sanitize system media that contain CUI prior to disposal (Source 2)."
    )

    cleaned = _strip_echoed_source_header(text)

    assert cleaned.startswith("Sanitize system media")
    assert "page 55" not in cleaned
    # The inline citation must survive; only the heading line goes.
    assert "(Source 2)" in cleaned


def test_an_answer_that_is_only_a_heading_is_left_alone():
    """Never return an empty answer just because it was badly formatted."""
    from app.rag_chain import _strip_echoed_source_header

    text = "Source 1: handbook.pdf | page 12"

    assert _strip_echoed_source_header(text) == text


def test_prompt_shows_a_worked_example():
    from app.rag_chain import PROMPT_TEMPLATE

    assert "Example of a GOOD answer" in PROMPT_TEMPLATE
    assert "Example of a BAD answer" in PROMPT_TEMPLATE


def test_citation_styles_are_normalised_to_one_form():
    """Small models write [Source 2], Source 2, and (source 2) interchangeably."""
    from app.rag_chain import _normalise_citations

    assert _normalise_citations("Wages are capped [Source 2].") == (
        "Wages are capped (Source 2)."
    )
    assert _normalise_citations("see source 3 for detail") == (
        "see (Source 3) for detail"
    )
    assert _normalise_citations("already (Source 1) fine") == (
        "already (Source 1) fine"
    )
