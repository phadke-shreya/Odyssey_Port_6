"""Tests for exact-identifier retrieval alongside vector search.

Pure vector search is weak on identifiers: an embedding of "Table 2-2" carries
almost no meaning. A literal-match pass fixes that -- but it must not become a
back door that lets out-of-scope questions through, which is what most of these
tests are actually guarding.
"""

import pytest

from app import config, vector_store
from app.vector_store import identifier_terms

needs_index = pytest.mark.skipif(
    not (config.CHROMA_DIR / "chroma.sqlite3").exists(),
    reason="run 'python ingest.py' first",
)


# --- what counts as an identifier ----------------------------------------


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What does Table 2-2 show?", "Table 2-2"),
        ("What is in Policy #14?", "Policy #14"),
        ("What does control 03.05.03 require?", "control 03.05.03"),
        ("Explain 03.08.03.", "03.08.03"),
        ("What is GPIO14 used for?", "GPIO14"),
    ],
)
def test_identifiers_are_extracted(question: str, expected: str) -> None:
    assert expected in identifier_terms(question)


@pytest.mark.parametrize(
    "question",
    [
        # A bare year must NOT count as an identifier. If it did, this question
        # would trigger a literal lookup and could defeat the refusal.
        "Who won the football World Cup in 2022?",
        "What happened in 1999?",
        "How many GPIO pins are available?",
        "What is the capital of Peru?",
        "How much sick leave do employees get?",
        "Give me 3 examples.",
    ],
)
def test_plain_questions_yield_no_identifiers(question: str) -> None:
    """This is the safety property the refusal rate depends on."""
    assert identifier_terms(question) == []


def test_bare_numbers_are_never_identifiers() -> None:
    """Structure is required: punctuation, or letters mixed with digits."""
    assert identifier_terms("section 5") == []
    assert identifier_terms("chapter 12") == []
    # But structured forms are.
    assert identifier_terms("clause 5.2.1") != []
    assert identifier_terms("part A-7") != []


# --- behaviour against the real index ------------------------------------


@needs_index
@pytest.mark.parametrize(
    "question",
    [
        "What does Table 2-2 show?",
        "What does control 03.05.03 require?",
        "What is GPIO14 used for?",
    ],
)
def test_identifier_questions_retrieve_the_identifier(question: str) -> None:
    """The chunk containing the literal identifier must be retrieved.

    Before the literal pass existed, these returned either nothing or the wrong
    document -- measured at 25% grounded across the hard question set.
    """
    hits = vector_store.search(question)

    assert hits, "no results for {}".format(question)
    term = identifier_terms(question)[-1]
    assert any(
        term.lower() in hit.text.lower() for hit in hits
    ), "retrieved nothing containing {}".format(term)


@needs_index
def test_exact_matches_are_labelled_and_ranked_first() -> None:
    """An exact hit is not a guess, so it outranks semantic matches."""
    hits = vector_store.search("What does control 03.05.03 require?")

    assert hits
    assert hits[0].match_type == "exact"
    # Semantic hits, if any, come after.
    kinds = [hit.match_type for hit in hits]
    assert kinds == sorted(kinds, key=lambda k: 0 if k == "exact" else 1)


@needs_index
@pytest.mark.parametrize(
    "question",
    [
        "Who won the football World Cup in 2022?",
        "What is the capital of Peru?",
        "What is the weather today?",
    ],
)
def test_literal_pass_does_not_leak_out_of_scope_questions(question: str) -> None:
    """Adding keyword matching must not weaken the anti-hallucination guard."""
    assert vector_store.search(question) == []


# --- the named-entity guard ----------------------------------------------
# Distance alone cannot tell "topically close" from "actually answers". A
# question naming GDPR scored 0.36 against a US compliance document that never
# mentions GDPR -- close enough to pass the threshold, but not an answer.


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What are the GDPR penalties for a data breach?", ["GDPR"]),
        ("Does this meet HIPAA requirements?", ["HIPAA"]),
        ("What does the EAP cover?", ["EAP"]),
    ],
)
def test_named_entities_are_detected(question: str, expected: list[str]) -> None:
    from app.vector_store import named_entities

    assert named_entities(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        # Ordinary words must NOT be treated as named entities. Requiring them
        # to appear would break synonym matching, which is the whole point of
        # embeddings -- "vacation days" should still find "annual leave".
        "How much sick leave do employees get?",
        "What are the vacation entitlements?",
        "What does control 03.05.03 require?",
        "Who should I contact in an emergency?",
    ],
)
def test_ordinary_questions_name_no_entities(question: str) -> None:
    from app.vector_store import named_entities

    assert named_entities(question) == []


@needs_index
@pytest.mark.parametrize(
    "question",
    [
        # These DO name acronyms (GPIO, HDMI), and the guard must not block them
        # because the corpus genuinely discusses both. This is the case that
        # would break if the guard were made any stricter.
        "How many GPIO pins are available?",
        "Does the module support dual HDMI output?",
    ],
)
def test_acronyms_present_in_the_corpus_are_not_blocked(question: str) -> None:
    from app.vector_store import named_entities

    assert named_entities(question), "expected this question to name an acronym"
    assert vector_store.search(question), "the guard blocked a valid question"


@needs_index
def test_question_naming_an_absent_entity_is_refused() -> None:
    """A near-miss: close in topic, but the corpus never mentions GDPR."""
    assert vector_store.search("What are the GDPR penalties for a data breach?") == []


@needs_index
def test_question_naming_a_present_entity_still_works() -> None:
    """The guard must not block entities the documents genuinely discuss."""
    hits = vector_store.search("What does the EAP provide?")

    assert hits, "the guard blocked an entity that is in the corpus"
    assert any("eap" in hit.text.lower() for hit in hits)


# --- an identifier question must not carry a near-miss identifier --------


def test_exact_matches_crowd_out_semantic_ones() -> None:
    """Once the literal pass hits, semantic near-misses are dropped.

    A unit test on the rule itself, so it holds without an index.
    """
    from app.models import Retrieved
    from app.vector_store import _drop_semantic_padding

    def result(match_type: str, distance: float) -> Retrieved:
        """One retrieved section, with only the fields this rule reads."""
        return Retrieved(
            text="text",
            source="doc.pdf",
            page=1,
            section="",
            content_type="prose",
            distance=distance,
            match_type=match_type,
        )

    mixed = [result("exact", 0.0), result("semantic", 0.43)]
    kept = _drop_semantic_padding(mixed)

    assert [hit.match_type for hit in kept] == ["exact"]

    # With nothing exact, every semantic hit survives untouched -- this is the
    # path all 23 ordinary questions take.
    only_semantic = [result("semantic", 0.2), result("semantic", 0.43)]
    assert _drop_semantic_padding(only_semantic) == only_semantic


@needs_index
def test_a_control_question_never_returns_a_different_control() -> None:
    """The reason the rule exists: 03.04.03 must not ride along with 03.05.03.

    Measured before the fix: "What does control 03.05.03 require?" returned
    03.04.03 Configuration Change Control at distance 0.4331, inside the
    threshold, because NIST controls are near-identical in shape. A model given
    both may blend them into an answer that is wrong and correctly cited.
    """
    hits = vector_store.search("What does control 03.05.03 require?")

    assert hits, "the identifier question retrieved nothing"
    assert any("03.05.03" in hit.text for hit in hits)
    for hit in hits:
        assert "03.04.03" not in hit.text, "a different control rode along"


# --- a literal match must be a whole identifier, not any substring -------


@pytest.mark.parametrize(
    ("term", "text", "expected"),
    [
        # The real heading being looked for: a trailing dot is part of it.
        ("3.5", "3.5. Identification and Authentication", True),
        ("03.05.03", "03.05.03 Multi-Factor Authentication", True),
        ("Table 2-2", "Table 2-2 shows the exposure limits", True),
        ("#37", "Policy #37 covers safety", True),
        ("GPIO14", "GPIO14 is used for I2C", True),
        # All of these were real false positives found by asking
        # "Tell me about section 3.5", each from a different document.
        ("3.5", "4 x M2.5 mounting holes (inset 3.5mm from module edge)", False),
        ("3.5", "Campus South West Zone (includes HFS) 206.543.5677", False),
        ("3.5", "version 10.3.5 released", False),
        ("3.5", "see 3.5.7 below", False),
        ("GPIO14", "GPIO145 is not a pin", False),
        # A bare number is only a section number where section numbers live:
        # at the start of its own line. These two are a table cell and a
        # cross-reference in prose, and boundaries alone let both through.
        ("3.5", "| I O(gpio) | Output current | 4mA default | 3.5 | 5.3 |", False),
        ("3.5", "see section 3.5 below for detail", False),
        ("3.5", "Chapter 3\n3.5. Identification and Authentication", True),
        # Terms carrying their own evidence stay matchable mid-line.
        ("Table 2-2", "see Table 2-2 for the exposure limits", True),
        ("03.06.04", "  03.06.04 Incident Response Training", True),
    ],
)
def test_a_literal_match_must_be_a_whole_identifier(
    term: str, text: str, expected: bool
) -> None:
    """Chroma's $contains is a substring match, which is too blunt on its own."""
    from app.vector_store import _is_whole_identifier

    assert _is_whole_identifier(term, text) is expected


@needs_index
def test_a_section_number_does_not_match_a_phone_number() -> None:
    """End to end version of the bug above, across documents.

    "section 3.5" used to return a screw-hole dimension, a current rating and a
    phone number, all from documents that have no section 3.5 at all.
    """
    hits = vector_store.search("Tell me about section 3.5")

    assert hits, "the identifier question retrieved nothing"
    for hit in hits:
        assert "206.543.5677" not in hit.text
        assert "3.5mm" not in hit.text
