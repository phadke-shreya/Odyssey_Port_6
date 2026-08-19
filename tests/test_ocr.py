"""Tests for reading pages that have no text layer.

The behaviour that matters most is the refusal: low-confidence OCR must be
discarded rather than indexed. Garbled text is worse than absent text, because it
gets quoted back to the user as though the document said it.
"""

from pathlib import Path

import pytest

from app import config, ocr
from app.chunker import ContentType, chunk_document
from app.ocr import OcrResult
from app.pdf_parser import parse_pdf

DOCS = Path(__file__).resolve().parent.parent / "documents"
SCAN = DOCS / "scanned_notice_IMAGE_ONLY.pdf"

needs_scan = pytest.mark.skipif(not SCAN.exists(), reason="scan fixture missing")
needs_ocr = pytest.mark.skipif(not ocr.available(), reason="tesseract is not installed")


# --- the usability rule --------------------------------------------------


def test_good_ocr_is_usable():
    result = OcrResult(text="A" * 200, confidence=85.0)

    assert result.usable is True


def test_low_confidence_ocr_is_rejected():
    """Garbled text must not be indexed. It would be cited as fact."""
    result = OcrResult(text="A" * 200, confidence=20.0)

    assert result.usable is False


def test_too_little_text_is_rejected():
    """A handful of characters is not worth a chunk."""
    result = OcrResult(text="abc", confidence=99.0)

    assert result.usable is False


def test_empty_ocr_is_rejected():
    assert OcrResult(text="", confidence=0.0).usable is False


def test_confidence_ignores_tesseract_empty_boxes():
    """image_to_data returns blank boxes with conf -1; they must not count.

    Including them would drag the average down and make good OCR look unusable.
    """
    from app.ocr import _mean_confidence

    data = {
        "conf": ["-1", "90", "-1", "80", "not-a-number"],
        "text": ["", "Hello", "  ", "world", "x"],
    }

    assert _mean_confidence(data) == pytest.approx(85.0)


def test_confidence_of_nothing_is_zero():
    from app.ocr import _mean_confidence

    assert _mean_confidence({"conf": [], "text": []}) == 0.0


# --- graceful degradation ------------------------------------------------


def test_ocr_is_skipped_when_disabled(monkeypatch):
    """Turning OCR off must not error -- it just yields nothing."""
    monkeypatch.setattr(config, "OCR_ENABLED", False)

    assert ocr.available() is False


@needs_scan
def test_page_is_reported_unreadable_when_ocr_is_off(monkeypatch):
    """With OCR unavailable, the old flagging behaviour must still work.

    This is the fallback that keeps a scanned page visible rather than silent.
    """
    monkeypatch.setattr(config, "OCR_ENABLED", False)

    blocks = parse_pdf(SCAN)

    kinds = {b.content_type for b in blocks}
    assert ContentType.IMAGE_ONLY in kinds
    assert ContentType.OCR not in kinds


# --- the real thing ------------------------------------------------------


@needs_scan
@needs_ocr
def test_scanned_page_becomes_searchable_text():
    """The whole point: a page with no text layer becomes readable content."""
    blocks = parse_pdf(SCAN)

    ocr_blocks = [b for b in blocks if b.content_type is ContentType.OCR]
    assert ocr_blocks, "OCR produced nothing for a scanned page"

    text = ocr_blocks[0].text.lower()
    # Distinctive words from the page. OCR is imperfect, so this checks for
    # content rather than an exact transcription.
    assert "bangalore" in text
    assert "closure" in text or "closed" in text


@needs_scan
@needs_ocr
def test_ocr_text_is_chunked_and_labelled():
    """OCR text is sliced like prose, and every chunk is marked as OCR."""
    blocks = parse_pdf(SCAN)

    chunks = chunk_document(blocks, SCAN.name)

    assert chunks
    assert all(c.content_type == "ocr" for c in chunks)
    assert all(c.parent_text for c in chunks)


def test_citation_warns_when_text_came_from_ocr():
    """A reader must be able to tell OCR from the document's own text."""
    from app.vector_store import Retrieved

    ocr_source = Retrieved(
        text="...",
        source="notice.pdf",
        page=1,
        section="",
        content_type="ocr",
        distance=0.2,
    )
    normal_source = Retrieved(
        text="...",
        source="handbook.pdf",
        page=4,
        section="5.2 Remote Work",
        content_type="prose",
        distance=0.2,
    )

    assert "OCR" in ocr_source.citation()
    assert "may contain errors" in ocr_source.citation()
    assert "OCR" not in normal_source.citation()


# --- how OCR text is split -----------------------------------------------
# A scanned page is often a list of unrelated single lines, not flowing prose.
# Measured on one such page: "when must employees collect their laptops?" scored
# 0.582 against the whole page (refused) but 0.229 against just the sentence that
# answers it. Packing unrelated facts into one child blurs its embedding.


def test_ocr_text_is_split_on_its_own_paragraph_breaks():
    from app.chunker import _children_for_ocr

    text = (
        "ACME CORP - INTERNAL NOTICE\n\n"
        "The Bangalore office will be closed on the third Friday of each month.\n\n"
        "Employees must collect laptops before 6 pm on Thursday.\n\n"
        "Contact facilities@acme.example for access requests."
    )

    children = _children_for_ocr(text)

    assert len(children) > 1, "unrelated facts were packed into one child"
    # The laptop fact must be retrievable on its own, not diluted.
    assert any("laptops" in c and "Bangalore" not in c for c in children), children


def test_short_ocr_fragments_are_joined_not_left_alone():
    """A three-word fragment is not a fact and cannot be retrieved usefully."""
    from app.chunker import _children_for_ocr

    text = "Page 1\n\nof 4\n\nEmployees must collect laptops before 6 pm on Thursday."

    children = _children_for_ocr(text)

    assert all(len(c) >= 20 for c in children), children


def test_a_long_ocr_paragraph_is_still_size_limited():
    from app.chunker import _children_for_ocr

    text = "word " * 400  # one enormous paragraph, no breaks

    children = _children_for_ocr(text)

    assert len(children) > 1
    assert all(len(c) <= config.CHILD_CHUNK_SIZE + 100 for c in children)


def test_ocr_children_keep_the_whole_page_as_parent():
    """Split for search, whole for answering -- the parent/child rule."""
    from app.chunker import Block, chunk_document

    text = (
        "ACME CORP - INTERNAL NOTICE\n\n"
        "The office closes on the third Friday of each month for maintenance.\n\n"
        "Employees must collect laptops before 6 pm on Thursday."
    )
    blocks = [Block(text=text, page=1, content_type=ContentType.OCR)]

    chunks = chunk_document(blocks, "notice.pdf")

    assert len(chunks) > 1
    assert len({c.parent_id for c in chunks}) == 1, "should share one parent"
    assert all(c.parent_text == text for c in chunks)


def test_ocr_layout_reconstruction_preserves_lines():
    """Regression: joining every word with a space destroyed the structure.

    Without line breaks the chunker has nothing to cut on, so a whole page
    collapses into one chunk regardless of how many separate facts it holds.
    """
    from app.ocr import reconstruct_layout

    data = {
        "text": ["ACME", "NOTICE", "The", "office", "closes", "Collect", "laptops"],
        "block_num": [1, 1, 1, 1, 1, 1, 1],
        "par_num": [1, 1, 2, 2, 2, 3, 3],
        "line_num": [1, 1, 1, 1, 1, 1, 1],
    }

    text = reconstruct_layout(data)

    assert text.startswith("ACME NOTICE")
    assert "\n\n" in text, "paragraph breaks were lost"
    assert "The office closes" in text
    assert "Collect laptops" in text
