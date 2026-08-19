"""Load every PDF in documents/ into the vector database.

Usage:
    python ingest.py            # add documents to the existing database
    python ingest.py --reset    # wipe the database first, then load everything

Run --reset after changing the embedding model: vectors built by a different
model are not comparable, and the app will refuse to query them.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from app import config, vector_store
from app.chunker import ContentType, chunk_document
from app.pdf_parser import parse_pdf

logger = logging.getLogger("ingest")


def main() -> int:
    """Parse arguments, run the job, and return a process exit code."""
    parser = argparse.ArgumentParser(description="Load PDFs into ChromaDB.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="delete the existing database before loading",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s"
    )

    problem = config.missing_credentials()
    if problem:
        logger.error(problem)
        return 1

    pdfs = sorted(config.DOCUMENTS_DIR.glob("*.pdf"))
    if not pdfs:
        logger.error("No PDFs found in %s", config.DOCUMENTS_DIR)
        return 1

    if args.reset:
        vector_store.reset()

    logger.info("Embedding with: %s", config.embedding_fingerprint())

    started = time.time()
    total_chunks = 0
    failures: list[tuple[Path, str]] = []

    for path in pdfs:
        try:
            blocks = parse_pdf(path)
            chunks = chunk_document(blocks, path.name)
            stored = vector_store.ingest(chunks)
            total_chunks += stored
            scans = sum(
                1 for block in blocks if block.content_type is ContentType.IMAGE_ONLY
            )
            note = " ({} page(s) unreadable)".format(scans) if scans else ""
            logger.info("%s: stored %s chunks%s", path.name, stored, note)
        except Exception as error:  # noqa: BLE001 - one bad PDF must not stop the rest
            logger.exception("Failed on %s", path.name)
            failures.append((path, str(error)))

    elapsed = time.time() - started
    summary = vector_store.stats()

    print()
    print("Ingest finished in {:.1f}s".format(elapsed))
    print("  chunks stored this run : {}".format(total_chunks))
    print("  chunks in database     : {}".format(summary["chunks"]))
    print("  documents in database  : {}".format(len(summary["sources"])))
    for source in summary["sources"]:
        print("      - {}".format(source))
    print("  embedding model        : {}".format(summary["fingerprint"]))
    print("  database location      : {}".format(config.CHROMA_DIR))

    if failures:
        print()
        print("{} document(s) failed:".format(len(failures)))
        for path, message in failures:
            print("  - {}: {}".format(path.name, message))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
