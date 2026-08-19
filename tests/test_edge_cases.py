"""Edge-case and robustness tests -- the "break it on purpose" pass.

Everything here is something a real user (or a mentor at a demo) will actually
do: paste nothing, paste an essay, type in another language, ask about something
the documents have never heard of, or point the app at a broken file.
"""

from pathlib import Path

import pytest

from app import config, vector_store
from app.rag_chain import DONT_KNOW, answer_question

DOCS = Path(__file__).resolve().parent.parent / "documents"

# These tests read the real database, so they only run once it has been built.
needs_index = pytest.mark.skipif(
    not (config.CHROMA_DIR / "chroma.sqlite3").exists(),
    reason="run 'python ingest.py' first",
)


# --- input the user should not be able to break things with ---------------


@pytest.mark.parametrize(
    ("label", "question"),
    [
        ("empty", ""),
        ("spaces only", "     "),
        ("tabs and newlines", "\t\n  \n"),
        ("too long", "x" * 5000),
    ],
)
def test_bad_input_is_refused_with_a_useful_message(label: str, question: str) -> None:
    """Never a stack trace, always a sentence telling the user what to do."""
    with pytest.raises(ValueError) as caught:
        vector_store.search(question)

    message = str(caught.value)
    assert message, "{}: empty error message".format(label)
    # The message must be a readable sentence, not a type name or a repr.
    assert message[0].isupper()
    assert message.endswith((".", ")"))


@needs_index
@pytest.mark.parametrize(
    ("label", "question"),
    [
        ("hindi", "मुझे कितने छुट्टी के दिन मिलते हैं?"),
        ("emoji only", "🤔📄❓"),
        ("single char", "a"),
        ("punctuation", "???!!!"),
        ("sql-ish", "'; DROP TABLE chunks; --"),
        ("very long word", "supercalifragilistic" * 20),
    ],
)
def test_odd_but_valid_input_does_not_crash(label: str, question: str) -> None:
    """Strange input may find nothing, but must never raise."""
    hits = vector_store.search(question)
    assert isinstance(hits, list), label
    # Whatever comes back must be answerable without exploding.
    result = answer_question(question, hits)
    assert isinstance(result.text, str)


# --- out-of-scope questions must not be answered from thin air -----------


@needs_index
@pytest.mark.parametrize(
    "question",
    [
        "Who won the football World Cup in 2022?",
        "What is the capital of Peru?",
        "How do I bake sourdough bread?",
        "What is the tallest mountain in the world?",
        "Who is the current prime minister of Japan?",
        "What time does the moon rise tomorrow?",
        "Write me a poem about penguins.",
        "What is my bank account balance?",
    ],
)
def test_out_of_scope_questions_get_i_dont_know(question: str) -> None:
    """Nothing in these documents answers these, so nothing may be invented."""
    hits = vector_store.search(question)
    result = answer_question(question, hits)

    assert result.is_dont_know, "answered an out-of-scope question: {}".format(question)
    assert result.text == DONT_KNOW
    assert result.sources == []


# --- consistency: the same question must behave the same way -------------


@needs_index
def test_same_question_returns_the_same_sources() -> None:
    """The same question must cite the same sources, in the same order.

    This is what M6B1 (consistent output) actually requires, and it is asserted
    strictly.

    Distances are compared with a tolerance rather than exactly. A hosted
    embedding API is not bit-reproducible: the same text embedded twice can differ
    in the fifth decimal place, which moved distances without changing the
    ranking. Asserting exact float equality was a valid assumption with a local
    model and became a false failure with a hosted one -- the ordering is the
    contract, not the float.
    """
    question = "How much sick leave do employees get?"

    first = vector_store.search(question)
    second = vector_store.search(question)

    assert [s.citation() for s in first] == [s.citation() for s in second]
    assert [s.content_type for s in first] == [s.content_type for s in second]
    for one, two in zip(first, second, strict=True):
        assert one.distance == pytest.approx(two.distance, abs=1e-3)


# --- an empty or missing database ----------------------------------------


def test_search_on_an_empty_database_returns_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh install with no documents must answer, not crash."""
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "empty_db")

    hits = vector_store.search("anything at all")

    assert hits == []
    assert answer_question("anything at all", hits).is_dont_know


def test_stats_survives_a_missing_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """/health must answer even before anything has been ingested."""
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "not_created_yet")

    summary = vector_store.stats()

    assert summary["chunks"] == 0
    assert summary["sources"] == []


# --- broken files --------------------------------------------------------


def test_corrupt_pdf_raises_a_clean_error(tmp_path: Path) -> None:
    """A file that claims to be a PDF but is not must fail readably."""
    from app.pdf_parser import parse_pdf

    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4\nthis is not really a pdf at all\n")

    with pytest.raises(ValueError) as caught:
        parse_pdf(broken)

    assert "corrupt" in str(caught.value).lower()


def test_empty_file_raises_a_clean_error(tmp_path: Path) -> None:
    from app.pdf_parser import parse_pdf

    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")

    with pytest.raises(ValueError):
        parse_pdf(empty)
