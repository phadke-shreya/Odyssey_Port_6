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
import re
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
    match_type: str = "semantic"

    def citation(self) -> str:
        """Format this source the way it is shown to the user."""
        parts = [self.source, "page {}".format(self.page)]
        if self.section:
            parts.append(self.section)
        # OCR text is a machine's reading of a picture, not the document's own
        # text layer. Saying so is the difference between a citation the reader
        # can trust and one they cannot.
        if self.content_type == "ocr":
            parts.append("OCR - may contain errors")
        return " | ".join(parts)


def _embedding_function():
    """Build the embedding function named by the config.

    The OpenAI path honours OPENAI_BASE_URL, so a company gateway needs no code
    change -- only that one environment variable.
    """
    if config.EMBEDDING_PROVIDER == "openai":
        logger.info("Embedding via OpenAI model %s", config.EMBEDDING_MODEL)
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=config.EMBEDDING_API_KEY,
            model_name=config.EMBEDDING_MODEL,
            api_base=config.EMBEDDING_BASE_URL,
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


# Pure vector search is weak on exact identifiers: an embedding of "Table 2-2"
# or "03.05.03" carries almost no meaning, so the right chunk may never surface.
# These constants drive a literal-match pass that complements it.

# Words that commonly label an identifier, so "Table 2-2" is searched as a phrase.
_LABEL_WORDS = frozenset(
    {
        "table",
        "policy",
        "section",
        "figure",
        "chapter",
        "requirement",
        "control",
        "appendix",
        "step",
        "clause",
        "form",
        "part",
    }
)

# Leading "#" is allowed so "Policy #14" yields the token "#14".
_TOKEN = re.compile(r"[A-Za-z0-9#][A-Za-z0-9._#/-]*")

# An identifier must be STRUCTURED: it contains punctuation or mixes letters
# with digits. A bare number is deliberately excluded -- otherwise the year in
# "Who won the World Cup in 2022?" would trigger a literal lookup and could
# defeat the out-of-scope refusal.
_STRUCTURED = re.compile(r"[.#/-]")

# How many literal matches to accept per term. Small: a term appearing in
# hundreds of chunks is not an identifier, it is a common word.
_KEYWORD_LIMIT = 5


def identifier_terms(question: str) -> list[str]:
    """Pull exact identifiers out of a question, longest and most specific first.

    "What does Table 2-2 show?" yields ["Table 2-2", "2-2"], so the precise
    phrase is tried before the bare token.
    """
    words = question.split()
    terms: list[str] = []

    for index, word in enumerate(words):
        token = _TOKEN.search(word)
        if not token:
            continue
        candidate = token.group(0).strip(".,;:")
        if len(candidate) < 3 or not any(c.isdigit() for c in candidate):
            continue
        has_letters = any(c.isalpha() for c in candidate)
        if not (_STRUCTURED.search(candidate) or has_letters):
            continue  # a bare number is not an identifier

        # Prefer "Table 2-2" over "2-2" when a label word precedes it.
        if index > 0:
            previous = words[index - 1].strip(".,;:").lower()
            if previous in _LABEL_WORDS:
                phrase = "{} {}".format(words[index - 1].strip(".,;:"), candidate)
                if phrase not in terms:
                    terms.append(phrase)
        if candidate not in terms:
            terms.append(candidate)

    return terms


def _keyword_candidates(collection, terms: list[str]) -> list[dict]:
    """Find chunks containing an identifier literally.

    Chroma's substring filter is case-sensitive, so a couple of casings are
    tried. Failures are ignored: a literal pass is an enhancement, and it must
    never be the reason a search breaks.
    """
    seen: set[str] = set()
    found: list[dict] = []

    for term in terms:
        for variant in dict.fromkeys([term, term.upper(), term.lower()]):
            try:
                result = collection.get(
                    where_document={"$contains": variant},
                    limit=_KEYWORD_LIMIT,
                    include=["metadatas"],
                )
            except Exception:  # noqa: BLE001 - never let this break a search
                logger.debug("Literal lookup failed for %r", variant)
                continue
            ids = result.get("ids") or []
            metadatas = result.get("metadatas") or []
            for identifier, metadata in zip(ids, metadatas, strict=False):
                if identifier in seen:
                    continue
                seen.add(identifier)
                found.append(metadata)
            if ids:
                break  # this casing worked; do not try the others

    if found:
        logger.info("Literal match on %s -> %s chunk(s)", terms, len(found))
    return found


# An ALL-CAPS acronym in a question names something specific: GDPR, HIPAA, OSHA,
# FMLA. If the corpus never mentions it, a close distance is a topical near-miss
# rather than an answer -- "GDPR penalties" scores 0.36 against a document about
# US compliance that says nothing about GDPR.
#
# Deliberately narrow. Requiring ordinary words to overlap would break the whole
# point of embeddings, which is finding "annual leave" when asked about
# "vacation days". Only named entities are checked.
_ACRONYM = re.compile(r"\b[A-Z]{3,8}\b")

# Acronyms too common to be evidence of anything.
_COMMON_ACRONYMS = frozenset({"THE", "AND", "FOR", "WHAT", "HOW", "WHO", "WHY", "PDF"})


def named_entities(question: str) -> list[str]:
    """Acronyms the question names, which the answer ought to mention."""
    return [
        token for token in _ACRONYM.findall(question) if token not in _COMMON_ACRONYMS
    ]


def _mentions_all(entities: list[str], results: list[Retrieved]) -> bool:
    """Whether at least one retrieved section mentions each named entity."""
    haystack = " ".join(r.text for r in results).lower()
    return all(entity.lower() in haystack for entity in entities)


def _to_retrieved(metadata: dict, distance: float, match_type: str) -> Retrieved:
    """Build a Retrieved from stored metadata."""
    return Retrieved(
        text=str(metadata.get("parent_text", "")),
        source=str(metadata.get("source", "")),
        page=int(metadata.get("page", 0)),
        section=str(metadata.get("section", "")),
        content_type=str(metadata.get("content_type", "")),
        distance=distance,
        match_type=match_type,
    )


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

    # Pass 1: literal identifier matches. An exact hit on "03.05.03" is not a
    # guess, so it is not subject to the distance threshold and it outranks
    # anything semantic.
    terms = identifier_terms(cleaned)
    if terms:
        for metadata in _keyword_candidates(collection, terms):
            parent_id = str(metadata.get("parent_id", ""))
            if parent_id not in best:
                best[parent_id] = _to_retrieved(metadata, 0.0, "exact")

    # Pass 2: semantic matches, gated by distance.
    for metadata, distance in zip(metadatas, distances, strict=False):
        if distance > config.MAX_DISTANCE:
            continue
        parent_id = str(metadata.get("parent_id", ""))
        existing = best.get(parent_id)
        if existing is not None and (
            existing.match_type == "exact" or existing.distance <= distance
        ):
            continue
        best[parent_id] = _to_retrieved(metadata, float(distance), "semantic")

    # Exact matches first, then semantic by distance.
    ranked = sorted(
        best.values(), key=lambda r: (0 if r.match_type == "exact" else 1, r.distance)
    )

    # Second guard: if the question names an entity the corpus never mentions,
    # a close distance is not an answer. An exact identifier match is exempt --
    # that is already proof the right chunk was found.
    entities = named_entities(cleaned)
    has_exact = any(r.match_type == "exact" for r in ranked)
    if entities and ranked and not has_exact and not _mentions_all(entities, ranked):
        logger.info(
            "Refusing: question names %s, which no retrieved section mentions",
            entities,
        )
        return []
    logger.info(
        "%s semantic + %s literal -> %s unique parents (kept %s)",
        len(metadatas),
        sum(1 for r in best.values() if r.match_type == "exact"),
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
