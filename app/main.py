"""FastAPI endpoints for SmartDoc.

This layer only does HTTP: validate input, call the modules that hold the logic,
and turn any failure into a clean message. No business logic lives here.
"""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field

from app import config, vector_store
from app.chunker import ContentType, chunk_document
from app.pdf_parser import parse_pdf
from app.rag_chain import NO_DOCUMENTS, answer_question

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SmartDoc",
    description="Ask questions about a library of PDFs and get cited answers.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., description="A question in plain English.")


class SourceOut(BaseModel):
    citation: str
    source: str
    page: int
    section: str
    content_type: str
    distance: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    generated: bool
    notice: str = ""


@app.get("/health")
def health() -> dict[str, object]:
    """Report whether the app is usable, and what is currently indexed."""
    summary = vector_store.stats()
    return {
        "status": "ok",
        "chunks": summary["chunks"],
        "documents": summary["sources"],
        "embedding_model": summary["fingerprint"],
        "api_key_configured": bool(config.OPENAI_API_KEY),
        "config_problem": config.missing_credentials(),
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer a question from the indexed documents, with citations."""
    try:
        sections = vector_store.search(request.question)
    except ValueError as error:
        # Bad input from the user: their fault, so tell them plainly.
        return AskResponse(answer=str(error), sources=[], generated=False)
    except Exception as error:  # noqa: BLE001 - boundary
        logger.exception("Retrieval failed")
        return AskResponse(
            answer="",
            sources=[],
            generated=False,
            notice="Could not search the documents. {}".format(error),
        )

    summary = vector_store.stats()
    if not summary["chunks"]:
        return AskResponse(answer=NO_DOCUMENTS, sources=[], generated=False)

    result = answer_question(request.question, sections)
    return AskResponse(
        answer=result.text,
        sources=[
            SourceOut(
                citation=s.citation(),
                source=s.source,
                page=s.page,
                section=s.section,
                content_type=s.content_type,
                distance=round(s.distance, 4),
                text=s.text,
            )
            for s in result.sources
        ],
        generated=result.generated,
        notice=result.notice,
    )


@app.post("/upload")
def upload(file: Annotated[UploadFile, File()]) -> dict[str, object]:
    """Add one PDF to the library.

    The uploaded filename is never trusted as a path: only its basename is used,
    and the file is written to a temporary directory.
    """
    safe_name = Path(file.filename or "upload.pdf").name
    if not safe_name.lower().endswith(".pdf"):
        return {"ok": False, "error": "Only PDF files are supported."}

    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / safe_name
        with target.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)

        size = target.stat().st_size
        if size > config.MAX_UPLOAD_BYTES:
            return {
                "ok": False,
                "error": "That file is {} MB; the limit is {} MB.".format(
                    size // (1024 * 1024),
                    config.MAX_UPLOAD_BYTES // (1024 * 1024),
                ),
            }

        try:
            blocks = parse_pdf(target)
            chunks = chunk_document(blocks, safe_name)
            stored = vector_store.ingest(chunks)
        except ValueError as error:
            return {"ok": False, "error": str(error)}
        except Exception:  # noqa: BLE001 - boundary
            logger.exception("Upload failed for %s", safe_name)
            return {
                "ok": False,
                "error": "Could not process that PDF. It may be corrupt or "
                "password protected.",
            }

    unreadable = sum(1 for b in blocks if b.content_type is ContentType.IMAGE_ONLY)
    return {
        "ok": True,
        "document": safe_name,
        "chunks_added": stored,
        "unreadable_pages": unreadable,
    }
