"""Tests for the guard on CLAUDE.md section 11 invariant 1.

Ingest and query MUST embed with the same model. Vectors from two different
models occupy different, incomparable spaces, and querying across them does not
raise -- it silently returns confident nonsense, which is the nastiest failure
mode a RAG system has. The fingerprint stamped into the collection is what turns
that silent failure into a loud one.

That guard was the only invariant in the project with nothing verifying it, which
is exactly backwards: it protects the invariant whose breakage is hardest to see.

The embedding function is stubbed out on purpose. What is being tested is the
fingerprint comparison, not Chroma's embedder, and a stub keeps the test offline
and fast -- no model download, no API key.
"""

from pathlib import Path

import pytest

from app import config, vector_store


class _StubEmbeddings:
    """A fixed-length embedding function, so no model or network is needed."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        """Return one small constant vector per input document."""
        return [[0.1, 0.2, 0.3] for _ in input]

    def name(self) -> str:
        """Chroma identifies a stored embedding function by this."""
        return self._name


@pytest.fixture
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the store at an empty temporary database with a stub embedder."""
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma_db")
    monkeypatch.setattr(
        vector_store, "_embedding_function", lambda: _StubEmbeddings("stub")
    )
    return tmp_path


def _build_with_fingerprint(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Open the collection while the config reports a given embedding model."""
    monkeypatch.setattr(config, "embedding_fingerprint", lambda: value)
    vector_store.get_collection()


def test_a_matching_fingerprint_opens_normally(
    isolated_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ordinary case: same model at ingest and at query."""
    _build_with_fingerprint(monkeypatch, "openai:text-embedding-3-small")
    _build_with_fingerprint(monkeypatch, "openai:text-embedding-3-small")


def test_a_changed_embedding_model_is_refused(
    isolated_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Querying vectors built by another model must raise, not return nonsense."""
    _build_with_fingerprint(monkeypatch, "openai:text-embedding-3-small")

    monkeypatch.setattr(config, "embedding_fingerprint", lambda: "local:other-model")
    with pytest.raises(vector_store.EmbeddingModelMismatch) as raised:
        vector_store.get_collection()

    message = str(raised.value)
    # An error the user cannot act on is barely better than silence (section 9).
    assert "text-embedding-3-small" in message, "does not say what built the data"
    assert "local:other-model" in message, "does not say what is configured now"
    assert "re-ingest" in message.lower() or "reset" in message.lower()


def test_the_check_can_be_skipped_deliberately(
    isolated_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ingest.py --reset needs to open a stale collection in order to drop it."""
    _build_with_fingerprint(monkeypatch, "openai:text-embedding-3-small")

    monkeypatch.setattr(config, "embedding_fingerprint", lambda: "local:other-model")
    vector_store.get_collection(check_fingerprint=False)
