"""Read text out of pages that have none, using OCR.

Embeddings cannot see pictures, so a scanned page is invisible to search. OCR
turns the picture back into text, which then behaves like any other page: it can
be chunked, embedded, retrieved, cited and quoted.

Two principles here:

1. **OCR is a fallback, never the first choice.** A page with a real text layer is
   always read directly. Only pages with no extractable text are OCR'd, which also
   keeps ingest fast: a 205-page manual with one scanned page pays for one page.
2. **Low-confidence OCR is worse than none.** Garbled text pollutes search results
   and can be quoted back as if it were the document. Below a confidence floor the
   result is discarded and the page is reported as unreadable instead.
"""

import logging
from dataclasses import dataclass

from pdfplumber.page import Page

from app import config

logger = logging.getLogger(__name__)

# Set once, the first time OCR is attempted, so a missing binary is reported one
# time rather than on every page of every document.
_availability_logged = False


@dataclass
class OcrResult:
    """Text recovered from an image, with a confidence score out of 100."""

    text: str
    confidence: float

    @property
    def usable(self) -> bool:
        """Whether this is good enough to index.

        Both tests matter: a handful of characters is not worth a chunk, and
        low-confidence output is actively harmful because it would be quoted back
        to the user as though the document said it.
        """
        return (
            len(self.text.strip()) >= config.OCR_MIN_CHARS
            and self.confidence >= config.OCR_MIN_CONFIDENCE
        )


def available() -> bool:
    """Whether OCR can actually run here.

    Checked rather than assumed: pytesseract is a thin wrapper around the
    tesseract binary, so the import succeeding proves nothing.
    """
    global _availability_logged

    if not config.OCR_ENABLED:
        return False
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
    except Exception:  # noqa: BLE001 - any failure means OCR is unavailable
        if not _availability_logged:
            logger.warning(
                "OCR is enabled but tesseract is not usable; scanned pages will "
                "be reported as unreadable instead. Install it with: "
                "brew install tesseract  (or apt-get install tesseract-ocr)"
            )
            _availability_logged = True
        return False
    return True


def reconstruct_layout(data: dict) -> str:
    """Rebuild lines and paragraphs from Tesseract's per-word output.

    Joining every word with a space loses the page's structure, and that has a
    real cost: the chunker cuts on paragraph and line breaks first, so a page
    flattened to one line cannot be split sensibly. Six unrelated facts then share
    one chunk and its embedding becomes a blur of all of them, so no single fact
    retrieves well.

    Tesseract reports a block, paragraph and line number per word, so the
    structure can be put back.
    """
    lines: dict[tuple[int, int, int], list[str]] = {}
    order: list[tuple[int, int, int]] = []

    words = data.get("text", [])
    for index, word in enumerate(words):
        if not str(word).strip():
            continue
        key = (
            int(data.get("block_num", [0] * len(words))[index] or 0),
            int(data.get("par_num", [0] * len(words))[index] or 0),
            int(data.get("line_num", [0] * len(words))[index] or 0),
        )
        if key not in lines:
            lines[key] = []
            order.append(key)
        lines[key].append(str(word))

    # A blank line between paragraphs, a single newline between lines within one.
    pieces: list[str] = []
    previous: tuple[int, int, int] | None = None
    for key in order:
        if previous is not None:
            same_paragraph = key[0] == previous[0] and key[1] == previous[1]
            pieces.append("\n" if same_paragraph else "\n\n")
        pieces.append(" ".join(lines[key]))
        previous = key
    return "".join(pieces).strip()


def _mean_confidence(data: dict) -> float:
    """Average Tesseract's per-word confidence, ignoring its empty boxes.

    image_to_data returns a row per detected box, many of which are blank with a
    confidence of -1. Including those would drag the average down and make good
    OCR look bad.
    """
    scores = []
    for raw_conf, word in zip(data.get("conf", []), data.get("text", []), strict=False):
        if not str(word).strip():
            continue
        try:
            value = float(raw_conf)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            scores.append(value)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def ocr_page(page: Page) -> OcrResult:
    """Render one PDF page to an image and read the text out of it.

    Args:
        page: A pdfplumber page.

    Returns:
        An OcrResult. On any failure this returns empty text with zero
        confidence rather than raising: a page that cannot be OCR'd should fall
        back to being reported as unreadable, not break the whole document.
    """
    if not available():
        return OcrResult(text="", confidence=0.0)

    try:
        import pytesseract

        image = page.to_image(resolution=config.OCR_DPI).original
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        text = reconstruct_layout(data)
        confidence = _mean_confidence(data)
    except Exception:  # noqa: BLE001 - never let OCR break a parse
        logger.warning("OCR failed on page %s", getattr(page, "page_number", "?"))
        return OcrResult(text="", confidence=0.0)

    logger.info(
        "OCR read %s characters from page %s at %.0f%% confidence",
        len(text),
        getattr(page, "page_number", "?"),
        confidence,
    )
    return OcrResult(text=text, confidence=confidence)
