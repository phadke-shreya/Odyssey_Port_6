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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Blank means talk to api.openai.com directly. Set it to route through a
# company gateway instead -- no code change needed, only this variable.
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip() or None

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

# --- Retrieval -----------------------------------------------------------
TOP_K_CHILDREN = 8  # fetched, then de-duplicated by parent
TOP_K_PARENTS = 4  # unique parents actually sent to the model

# Cosine distance above which a chunk is treated as irrelevant. If every hit is
# further away than this, the honest answer is "I don't know".
#
# This number is NOT model-independent. Different embedding models spread their
# distances differently, so a threshold tuned for one lets out-of-scope
# questions through on another. Measured on this document set:
#
#   all-MiniLM-L6-v2   in-scope 0.22-0.34   out-of-scope 0.6+
#   nomic-embed-text   in-scope 0.19-0.25   out-of-scope 0.43+
#
# Re-measure after changing EMBEDDING_MODEL with: python eval/run_eval.py --sweep
# It prints in-scope recall against out-of-scope refusal for a range of values.
#
# Measured for nomic-embed-text on this corpus (23 in-scope, 10 out-of-scope):
#
#   threshold   in-scope kept   out-of-scope refused
#     0.30            91%              100%
#     0.35           100%              100%   <- window opens
#     0.40           100%              100%   <- window closes
#     0.45           100%               80%
#     0.55           100%               10%
#
# The safe window is [0.35, 0.40], so 0.375 sits in the middle of it with margin
# on both sides. Picking an edge value would leave no room for a document set
# slightly harder than this one.
_MAX_DISTANCE_LOCAL = 0.55
_MAX_DISTANCE_HOSTED = 0.375
MAX_DISTANCE = (
    _MAX_DISTANCE_HOSTED if EMBEDDING_PROVIDER == "openai" else _MAX_DISTANCE_LOCAL
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
    if EMBEDDING_PROVIDER == "openai" and not OPENAI_API_KEY:
        return (
            "OPENAI_API_KEY is not set, but EMBEDDING_PROVIDER is 'openai'. "
            "Add the key to .env, or set EMBEDDING_PROVIDER=local to embed on "
            "this machine instead."
        )
    return ""
