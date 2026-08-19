"""All tunable settings for SmartDoc, in one place.

Nothing else in the project reads os.environ or hardcodes a model name, a
chunk size, or a path. If a value might ever need changing, it lives here.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
CHROMA_COLLECTION = "smartdoc"

# --- Credentials ---------------------------------------------------------
# Embedding and chat are separate jobs and may use DIFFERENT providers: paid
# embeddings from OpenAI alongside a free local chat model, for instance. They
# therefore get their own credentials. A single shared pair is still honoured as
# a fallback, so an existing .env with only OPENAI_* keeps working.
#
# Sharing one variable between both jobs was a real bug: configuring a local
# chat model meant overwriting OPENAI_API_KEY, which silently broke embeddings.
_SHARED_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
_SHARED_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip() or None

# Blank base URL means talk to api.openai.com directly. Set it to route through
# a company gateway or a local server -- no code change needed.
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "").strip() or _SHARED_API_KEY
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "").strip() or _SHARED_BASE_URL

CHAT_API_KEY = os.getenv("CHAT_API_KEY", "").strip() or _SHARED_API_KEY
CHAT_BASE_URL = os.getenv("CHAT_BASE_URL", "").strip() or _SHARED_BASE_URL

# Kept for anything that still refers to the old single-pair names.
OPENAI_API_KEY = _SHARED_API_KEY
OPENAI_BASE_URL = _SHARED_BASE_URL

# --- Models --------------------------------------------------------------
# Two DIFFERENT models doing two different jobs. Never swap them.
# The embedding model must be identical at ingest and at query time, or
# retrieval silently returns garbage: vectors from different models are not
# comparable. Changing it means deleting CHROMA_DIR and re-ingesting.
# Both are overridable from .env, because a company gateway or a local server
# (Ollama, LM Studio) exposes different model names for the same jobs.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Which service does the embedding. "openai" needs a working key; "local" runs
# ChromaDB's built-in MiniLM model on this machine and needs no key at all.
# Switching this invalidates every stored vector -- see embedding_fingerprint().
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()

# The local fallback. Its input window is only ~256 tokens (~1000 chars), so
# large table chunks get silently truncated by it. That is exactly why the
# hosted model is preferred once a working key exists.
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Deterministic answers: the same question must produce the same answer.
CHAT_TEMPERATURE = 0

# --- Chunking ------------------------------------------------------------
# Parents are what the model READS (big, full context).
# Children are what gets EMBEDDED and searched (small, precise).
PARENT_MAX_CHARS = 3000
CHILD_CHUNK_SIZE = 400
CHILD_OVERLAP = 80

# Below this many detected headings, a document is treated as unstructured
# and parents fall back to size-based splitting on paragraph boundaries.
MIN_HEADINGS_FOR_SECTIONS = 3

# A heading line is short. Anything longer is prose that merely looks like one.
MAX_HEADING_CHARS = 80

# Pages yielding less text than this are probably scans or screenshots.
MIN_CHARS_FOR_TEXT_PAGE = 100

# Warn above this: an embedding model silently truncates anything past its
# input window, which would quietly gut a large table chunk. The limit depends
# on which model is actually in use, so the warning follows the config rather
# than being hardcoded to the smallest one.
_LOCAL_LIMIT_CHARS = 1000  # all-MiniLM-L6-v2: ~256 tokens
_HOSTED_LIMIT_CHARS = 30000  # text-embedding-3-small / nomic-embed-text: ~8k tokens
CHUNK_WARN_CHARS = (
    _HOSTED_LIMIT_CHARS if EMBEDDING_PROVIDER == "openai" else _LOCAL_LIMIT_CHARS
)

# --- OCR (reading pages that have no text layer) -------------------------
# Only pages with no extractable text are OCR'd, so a long document with one
# scanned page pays for one page. Set OCR_ENABLED=false to skip it entirely.
OCR_ENABLED = os.getenv("OCR_ENABLED", "true").strip().lower() not in {
    "false",
    "0",
    "no",
}

# 300dpi is the usual sweet spot: 150 loses small print, 600 is slower with
# little gain. Roughly 0.6s per page.
OCR_DPI = int(os.getenv("OCR_DPI", "300"))

# Below these, the OCR output is discarded and the page is reported as
# unreadable. Garbled text is worse than no text: it pollutes search results and
# can be quoted back to the user as though the document said it.
OCR_MIN_CONFIDENCE = float(os.getenv("OCR_MIN_CONFIDENCE", "55"))
OCR_MIN_CHARS = int(os.getenv("OCR_MIN_CHARS", "40"))

# Separately: below this an individual OCR paragraph is a fragment rather than a
# fact, so it is joined to its neighbour instead of becoming a chunk nobody can
# retrieve. Same value as OCR_MIN_CHARS today but a different question -- one is
# "is this page worth indexing", this is "is this line worth its own chunk" -- so
# they are deliberately not shared.
OCR_MIN_CHILD_CHARS = int(os.getenv("OCR_MIN_CHILD_CHARS", "40"))

# --- Retrieval -----------------------------------------------------------
TOP_K_CHILDREN = 8  # fetched, then de-duplicated by parent
TOP_K_PARENTS = 4  # unique parents actually sent to the model

# Cosine distance above which a chunk is treated as irrelevant. If every hit is
# further away than this, the honest answer is "I don't know".
#
# THIS IS A PROPERTY OF THE EMBEDDING MODEL, NOT OF THE PROVIDER. Different
# models spread their distances completely differently, so it is keyed by model
# name. Getting this wrong is not a subtle degradation: running
# text-embedding-3-small at the value tuned for nomic-embed-text made the app
# answer "I don't know" to 10 of 23 questions the documents genuinely answer.
#
# Measured on this corpus with `python eval/run_eval.py --sweep`:
#
#   nomic-embed-text        in-scope <=0.245  nearest excluded 0.435  -> 0.375
#   text-embedding-3-small  in-scope <=0.479  nearest excluded 0.562  -> 0.520
#
# Note how different those are. The same threshold cannot serve both.
#
# Each value below sits inside its measured window, not at an edge. Re-measure
# after changing EMBEDDING_MODEL -- and if a model is missing here it falls back
# to a deliberately permissive default, which favours answering over refusing.
# Only measured models are listed: a guessed number that looks measured is worse
# than no entry, because the fallback at least announces itself as a default.
_MAX_DISTANCE_BY_MODEL = {
    "nomic-embed-text": 0.375,
    "text-embedding-3-small": 0.52,
    # ChromaDB's local default.
    "all-MiniLM-L6-v2": 0.55,
}
_MAX_DISTANCE_DEFAULT = 0.55

_active_embedding_model = (
    EMBEDDING_MODEL if EMBEDDING_PROVIDER == "openai" else LOCAL_EMBEDDING_MODEL
)
MAX_DISTANCE = _MAX_DISTANCE_BY_MODEL.get(
    _active_embedding_model, _MAX_DISTANCE_DEFAULT
)

# --- Input limits --------------------------------------------------------
MAX_QUERY_CHARS = 2000
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# --- Where the UI finds the API ------------------------------------------
API_URL = os.getenv("API_URL", "http://127.0.0.1:8006").rstrip("/")


def embedding_fingerprint() -> str:
    """Identify exactly which embedding model built a stored collection.

    Vectors from different models live in different, incomparable spaces.
    Querying a collection built by one model using another does not raise -- it
    silently returns confident nonsense, which is the nastiest failure mode in a
    RAG system. Stamping this string into the collection lets us detect the
    mismatch and refuse, instead of lying to the user.
    """
    if EMBEDDING_PROVIDER == "openai":
        return "openai:{}".format(EMBEDDING_MODEL)
    return "local:{}".format(LOCAL_EMBEDDING_MODEL)


def missing_credentials() -> str:
    """Return a human-readable problem with the config, or an empty string.

    Called at startup so a missing key is reported once, clearly, instead of
    surfacing as a confusing failure in the middle of a user's question.
    """
    if EMBEDDING_PROVIDER == "openai" and not EMBEDDING_API_KEY:
        return (
            "EMBEDDING_PROVIDER is 'openai' but no key is set. Add "
            "EMBEDDING_API_KEY (or OPENAI_API_KEY) to .env, or set "
            "EMBEDDING_PROVIDER=local to embed on this machine instead."
        )
    return ""
