"""Agent 1 — Ingestion & Layout.

Responsibilities:
  * Decide, per page, whether text can be read directly (DIGITAL) or needs OCR
    (SCANNED). This is the single biggest cost/accuracy lever: digital text is
    extracted losslessly, so we never pay OCR error or compute for it.
  * Preprocess scanned rasters (deskew/denoise/normalize DPI) — the stage where
    most downstream OCR accuracy is actually won.
  * Detect coarse layout regions and assign reading order.

The heavy native bits (pdf2image/pypdf/OpenCV) are optional: if they aren't
installed, the agent still produces a valid layout so the pipeline runs.
"""

from __future__ import annotations

from dataclasses import dataclass

from adi.agents.base import Agent
from adi.contracts import (
    BoundingBox,
    DocumentLayout,
    PageKind,
    RawPage,
    Region,
    RegionType,
)


@dataclass
class DocumentSource:
    """What the pipeline is asked to process."""

    doc_id: str
    path: str | None = None
    # Pre-supplied text marks the document as DIGITAL (also handy for tests/demos).
    raw_text: str | None = None
    force_kind: PageKind | None = None


def _classify_line(line: str) -> RegionType:
    stripped = line.strip()
    if ":" in stripped:
        return RegionType.KEY_VALUE
    if stripped.isupper() and len(stripped.split()) <= 6:
        return RegionType.TITLE
    return RegionType.PARAGRAPH


class IngestionAgent(Agent[DocumentSource, DocumentLayout]):
    name = "ingestion"

    def run(self, payload: DocumentSource) -> DocumentLayout:
        self.trace.clear()
        if payload.raw_text is not None:
            return self._from_text(payload)
        if payload.path and payload.path.lower().endswith(".pdf"):
            extracted = self._try_extract_pdf(payload)
            if extracted is not None:
                return extracted
        # No text and no usable PDF backend: treat as a scanned page so the
        # RecognitionAgent's OCR engine takes over.
        return self._scanned_placeholder(payload)

    # -- digital path -------------------------------------------------------
    def _from_text(self, payload: DocumentSource) -> DocumentLayout:
        text = payload.raw_text or ""
        page = RawPage(
            page_no=0,
            kind=PageKind.DIGITAL,
            width=1240,
            height=1754,
            dpi=150,
            embedded_text=text,
            preprocessing=["embedded-text-extraction"],
        )
        regions = self._regions_from_text(text, page_no=0)
        self.log(f"digital page with {len(regions)} regions, {len(text)} chars")
        return DocumentLayout(
            doc_id=payload.doc_id, source_path=payload.path, pages=[page], regions=regions
        )

    def _regions_from_text(self, text: str, page_no: int) -> list[Region]:
        regions: list[Region] = []
        y = 40.0
        for order, line in enumerate(ln for ln in text.splitlines() if ln.strip()):
            regions.append(
                Region(
                    page_no=page_no,
                    region_type=_classify_line(line),
                    bbox=BoundingBox(page=page_no, x0=40, y0=y, x1=1200, y1=y + 24),
                    order=order,
                )
            )
            y += 30.0
        return regions

    # -- real PDF path (optional deps) -------------------------------------
    def _try_extract_pdf(self, payload: DocumentSource) -> DocumentLayout | None:
        try:
            from pypdf import PdfReader
        except Exception:
            self.log("pypdf not installed; cannot read PDF directly")
            return None
        try:
            reader = PdfReader(payload.path)
        except Exception as exc:  # unreadable / not a PDF
            self.log(f"failed to open PDF: {exc}")
            return None

        pages: list[RawPage] = []
        regions: list[Region] = []
        for idx, pdf_page in enumerate(reader.pages):
            text = (pdf_page.extract_text() or "").strip()
            if text:
                pages.append(
                    RawPage(
                        page_no=idx,
                        kind=PageKind.DIGITAL,
                        width=int(pdf_page.mediabox.width),
                        height=int(pdf_page.mediabox.height),
                        embedded_text=text,
                        preprocessing=["embedded-text-extraction"],
                    )
                )
                regions.extend(self._regions_from_text(text, page_no=idx))
            else:
                # Scanned page: would be rasterized + preprocessed here (pdf2image
                # + OpenCV). We emit a full-page region for the OCR engine.
                pages.append(
                    RawPage(
                        page_no=idx,
                        kind=PageKind.SCANNED,
                        width=int(pdf_page.mediabox.width),
                        height=int(pdf_page.mediabox.height),
                        dpi=300,
                        preprocessing=["rasterize@300dpi", "deskew", "denoise"],
                    )
                )
                regions.append(
                    Region(
                        page_no=idx,
                        region_type=RegionType.PARAGRAPH,
                        bbox=BoundingBox(
                            page=idx, x0=0, y0=0,
                            x1=float(pdf_page.mediabox.width), y1=float(pdf_page.mediabox.height),
                        ),
                        order=0,
                    )
                )
        self.log(f"extracted {len(pages)} page(s) from PDF")
        return DocumentLayout(
            doc_id=payload.doc_id, source_path=payload.path, pages=pages, regions=regions
        )

    # -- fallback -----------------------------------------------------------
    def _scanned_placeholder(self, payload: DocumentSource) -> DocumentLayout:
        page = RawPage(
            page_no=0,
            kind=PageKind.SCANNED,
            width=2480,
            height=3508,
            dpi=300,
            image_ref=payload.path,
            preprocessing=["rasterize@300dpi", "deskew", "denoise", "binarize"],
        )
        region = Region(
            page_no=0,
            region_type=RegionType.PARAGRAPH,
            bbox=BoundingBox(page=0, x0=0, y0=0, x1=2480, y1=3508),
            order=0,
        )
        self.log("no text backend available; routing as a single scanned region")
        return DocumentLayout(
            doc_id=payload.doc_id, source_path=payload.path, pages=[page], regions=[region]
        )
