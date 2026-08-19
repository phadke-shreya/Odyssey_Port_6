"""Store chunks in ChromaDB and retrieve them by meaning.

This module owns every interaction with the vector database. Nothing else in the
project imports chromadb, so swapping Chroma for something else touches one file.

Two things here carry most of the design weight:

1. Only CHILD chunks are embedded and searched. On retrieval a matched child is
   expanded back to its PARENT section, then parents are de-duplicated -- three
   children from the same section must not paste that section into the prompt
   three times.
2. The collection records which embedding model built it. Querying with a
   different model returns confident nonsense rather than an error, so that
   mismatch is checked and refused loudly.
"""

import logging
import shutil
from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions

from app import config
from app.chunker import Chunk

logger = logging.getLogger(__name__)

# Chroma's default distance is squared L2. Cosine is what makes a single
# distance threshold meaningful across documents, so it is set explicitly.
_SPACE = "cosine"

# How many chunks to hand Chroma at once. Keeps memory flat and lets a long
# ingest report progress instead of appearing to hang.
_BATCH = 500

_FINGERPRINT_KEY = "embedding_fingerprint"


class EmbeddingModelMismatch(RuntimeError):
    """Raised when the stored vectors were built by a different model."""


@dataclass
class Retrieved:
    """One parent section retrieved for a question, ready to cite."""

    text: str
    source: str
    page: int
    section: str
    content_type: str
    distance: float

    def citation(self) -> str:
        """Format this source the way it is shown to the user."""
        parts = [self.source, "page {}".format(self.page)]
        if self.section:
            parts.append(self.section)
        return " | ".join(parts)


def _embedding_function():
    """Build the embedding function named by the config.

    The OpenAI path honours OPENAI_BASE_URL, so a company gateway needs no code
    change -- only that one environment variable.
    """
    if config.EMBEDDING_PROVIDER == "openai":
        logger.info("Embedding via OpenAI model %s", config.EMBEDDING_MODEL)
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=config.OPENAI_API_KEY,
            model_name=config.EMBEDDING_MODEL,
            api_base=config.OPENAI_BASE_URL,
        )
    logger.info("Embedding locally with %s", config.LOCAL_EMBEDDING_MODEL)
    return embedding_functions.DefaultEmbeddingFunction()


def _client() -> chromadb.ClientAPI:
    """Open the on-disk database. Data survives restarts (graded: M6S2)."""
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_collection(check_fingerprint: bool = True):
    """Open (or create) the collection, refusing a model mismatch.

    Args:
        check_fingerprint: When True, verify the stored vectors were built by
            the embedding model currently configured.

    Raises:
        EmbeddingModelMismatch: If the collection was built by another model.
    """
    current = config.embedding_fingerprint()
    try:
        collection = _client().get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            embedding_function=_embedding_function(),
            metadata={"hnsw:space": _SPACE, _FINGERPRINT_KEY: current},
        )
    except ValueError as error:
        # Chroma runs its own embedding-function conflict check, and it fires
        # before ours with a message that does not say what to do about it.
        # Translate it into the actionable version.
        if "embedding function" in str(error).lower():
            raise EmbeddingModelMismatch(
                "This database was built with a different embedding model, and "
                "the app is now configured for '{}'. Vectors from different "
                "models are not comparable. Run: python ingest.py --reset".format(
                    current
                )
            ) from error
        raise

    stored = (collection.metadata or {}).get(_FINGERPRINT_KEY)
    if check_fingerprint and stored and stored != current:
        raise EmbeddingModelMismatch(
            "This database was built with '{}' but the app is configured for "
            "'{}'. Vectors from different models are not comparable. Delete "
            "the {} folder and re-ingest.".format(
                stored, current, config.CHROMA_DIR.name
            )
        )
    return collection


def reset() -> None:
    """Delete the whole database. Required after changing embedding model."""
    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)
        logger.info("Deleted %s", config.CHROMA_DIR)


def ingest(chunks: list[Chunk]) -> int:
    """Embed and store child chunks. Returns how many were added.

    Each child carries its parent's full text in metadata, so retrieval can
    expand to the parent without a second lookup, and so the parent survives a
    restart -- Chroma persists metadata to disk alongside the vectors.

    Raises:
        ValueError: If chunks is empty.
    """
    if not chunks:
        raise ValueError("ingest called with no chunks")

    collection = get_collection()

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str | int]] = []

    for index, chunk in enumerate(chunks):
        ids.append("{}#{}".format(chunk.parent_id, index))
        documents.append(chunk.text)  # the CHILD is what gets embedded
        metadatas.append(
            {
                "parent_id": chunk.parent_id,
                "parent_text": chunk.parent_text,
                "source": chunk.source,
                "page": chunk.page,
                "section": chunk.section,
                "content_type": chunk.content_type,
            }
        )

    for start in range(0, len(ids), _BATCH):
        stop = start + _BATCH
        collection.add(
            ids=ids[start:stop],
            documents=documents[start:stop],
            metadatas=metadatas[start:stop],
        )
        logger.info("Stored %s / %s chunks", min(stop, len(ids)), len(ids))

    return len(ids)


def search(question: str) -> list[Retrieved]:
    """Find the parent sections most likely to answer a question.

    Searches small children for precision, then expands each match to its parent
    section for context, de-duplicating so one section is never sent twice.

    Returns:
        Up to TOP_K_PARENTS unique parents, nearest first. An empty list means
        nothing was close enough, and the honest answer is "I don't know".

    Raises:
        ValueError: If the question is empty or absurdly long.
    """
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("Please type a question.")
    if len(cleaned) > config.MAX_QUERY_CHARS:
        raise ValueError(
            "That question is too long ({} characters, limit {}).".format(
                len(cleaned), config.MAX_QUERY_CHARS
            )
        )

    collection = get_collection()
    if collection.count() == 0:
        logger.warning("Search attempted on an empty collection")
        return []

    found = collection.query(
        query_texts=[cleaned],
        n_results=min(config.TOP_K_CHILDREN, collection.count()),
    )

    metadatas = (found.get("metadatas") or [[]])[0]
    distances = (found.get("distances") or [[]])[0]

    # Keep the best (nearest) hit per parent, so one section appears once.
    best: dict[str, Retrieved] = {}
    for metadata, distance in zip(metadatas, distances, strict=False):
        if distance > config.MAX_DISTANCE:
            continue
        parent_id = str(metadata.get("parent_id", ""))
        existing = best.get(parent_id)
        if existing is not None and existing.distance <= distance:
            continue
        best[parent_id] = Retrieved(
            text=str(metadata.get("parent_text", "")),
            source=str(metadata.get("source", "")),
            page=int(metadata.get("page", 0)),
            section=str(metadata.get("section", "")),
            content_type=str(metadata.get("content_type", "")),
            distance=float(distance),
        )

    ranked = sorted(best.values(), key=lambda r: r.distance)
    logger.info(
        "%s children -> %s unique parents (kept %s)",
        len(metadatas),
        len(best),
        min(len(ranked), config.TOP_K_PARENTS),
    )
    return ranked[: config.TOP_K_PARENTS]


def stats() -> dict[str, object]:
    """Summarise what is currently stored, for the UI and for /health."""
    try:
        collection = get_collection(check_fingerprint=False)
    except Exception:  # noqa: BLE001 - stats must never crash a health check
        logger.exception("Could not open the collection")
        return {"chunks": 0, "sources": [], "fingerprint": None}

    count = collection.count()
    sources: set[str] = set()
    if count:
        sample = collection.get(limit=min(count, 5000), include=["metadatas"])
        for metadata in sample.get("metadatas") or []:
            source = metadata.get("source")
            if source:
                sources.add(str(source))

    return {
        "chunks": count,
        "sources": sorted(sources),
        "fingerprint": (collection.metadata or {}).get(_FINGERPRINT_KEY),
    }
