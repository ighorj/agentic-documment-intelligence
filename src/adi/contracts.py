"""Shared, typed data contracts that flow between the three agents.

Every agent consumes and produces one of these models. Keeping the contracts in
a single module (rather than letting each agent invent its own dict shape) is
what makes the agents independently testable and swappable: as long as an
implementation honours the contract, it can be replaced without touching the
rest of the pipeline.

Pipeline data flow
------------------
    PDF bytes
        -> IngestionAgent   -> DocumentLayout (RawPage + Region[] per page)
        -> RecognitionAgent -> RecognizedDocument (TextBlock[] with confidence)
        -> ValidationAgent  -> DocumentResult (validated, structured fields)
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class PageKind(str, Enum):
    """How a page's text should be obtained."""

    DIGITAL = "digital"  # vectorized text already present; extract directly (near-perfect)
    SCANNED = "scanned"  # image only; must go through OCR


class RegionType(str, Enum):
    """Coarse layout role of a region. Drives downstream handling
    (e.g. tables are parsed differently from prose)."""

    TITLE = "title"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    KEY_VALUE = "key_value"
    FIGURE = "figure"
    OTHER = "other"


class BoundingBox(BaseModel):
    """Pixel coordinates on a page, origin top-left. Carried end-to-end so every
    extracted value keeps provenance back to where it was seen on the page."""

    page: int = Field(ge=0)
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)


# ---------------------------------------------------------------------------
# Agent 1 output: layout
# ---------------------------------------------------------------------------


class RawPage(BaseModel):
    """A single page after ingestion/preprocessing."""

    page_no: int = Field(ge=0)
    kind: PageKind
    width: int = Field(ge=0)
    height: int = Field(ge=0)
    dpi: int = Field(default=300, ge=1)
    # Reference to the (pre-processed) raster, kept as a path/id rather than raw
    # bytes so the contract stays lightweight and serializable.
    image_ref: str | None = None
    # Text lifted directly from a DIGITAL page (no OCR needed).
    embedded_text: str | None = None
    # Preprocessing operations actually applied (deskew, denoise, ...).
    preprocessing: list[str] = Field(default_factory=list)


class Region(BaseModel):
    """A detected layout region on a page that recognition should process."""

    page_no: int = Field(ge=0)
    region_type: RegionType = RegionType.PARAGRAPH
    bbox: BoundingBox
    # Reading order index within the page (lower = earlier).
    order: int = 0


class DocumentLayout(BaseModel):
    """Output of the IngestionAgent: per-page structure + regions to recognize."""

    doc_id: str
    source_path: str | None = None
    pages: list[RawPage] = Field(default_factory=list)
    regions: list[Region] = Field(default_factory=list)

    def regions_for(self, page_no: int) -> list[Region]:
        return sorted(
            (r for r in self.regions if r.page_no == page_no),
            key=lambda r: r.order,
        )


# ---------------------------------------------------------------------------
# Agent 2 output: recognized text
# ---------------------------------------------------------------------------


class Token(BaseModel):
    """A single recognized token with its own confidence and location."""

    text: str
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    bbox: BoundingBox | None = None


class TextBlock(BaseModel):
    """Recognized text for one region: the joined text plus per-token detail."""

    page_no: int = Field(ge=0)
    region_type: RegionType = RegionType.PARAGRAPH
    text: str = ""
    tokens: list[Token] = Field(default_factory=list)
    bbox: BoundingBox | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    engine: str = "unknown"

    @property
    def low_confidence(self) -> bool:
        return self.confidence < 0.75


class RecognizedDocument(BaseModel):
    """Output of the RecognitionAgent."""

    doc_id: str
    blocks: list[TextBlock] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n".join(b.text for b in self.blocks if b.text)

    @property
    def mean_confidence(self) -> float:
        scored = [b.confidence for b in self.blocks if b.text]
        return sum(scored) / len(scored) if scored else 0.0


# ---------------------------------------------------------------------------
# Agent 3 output: validated, structured result
# ---------------------------------------------------------------------------


class ExtractedField(BaseModel):
    """A single structured field the system was asked to find."""

    name: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    # Whether the value passed its domain validator (e.g. valid ICD-10 / IBAN).
    valid: bool = True
    validation_error: str | None = None
    # The original OCR text before any correction was applied.
    raw_value: str | None = None
    # A short verbatim quote from the document supporting this value (AI extractor).
    evidence: str | None = None
    source_bbox: BoundingBox | None = None


class DocumentResult(BaseModel):
    """Final pipeline output: clean, structured, validated data with provenance."""

    doc_id: str
    domain: str = "generic"
    fields: list[ExtractedField] = Field(default_factory=list)
    full_text: str = ""
    mean_confidence: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)

    def field(self, name: str) -> ExtractedField | None:
        return next((f for f in self.fields if f.name == name), None)


# ---------------------------------------------------------------------------
# Agent 4 output: confidence / quality report
# ---------------------------------------------------------------------------


class WeakRegion(BaseModel):
    """A region whose recognition confidence fell below the report threshold."""

    page_no: int = Field(ge=0)
    region_type: RegionType = RegionType.PARAGRAPH
    confidence: float = Field(ge=0.0, le=1.0)
    text_preview: str = ""
    engine: str = "unknown"
    bbox: BoundingBox | None = None
    # The individual tokens dragging the block's confidence down.
    weak_tokens: list[str] = Field(default_factory=list)


class PageConfidence(BaseModel):
    """Per-page roll-up of recognition confidence."""

    page_no: int = Field(ge=0)
    mean_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    block_count: int = Field(ge=0, default=0)
    weak_block_count: int = Field(ge=0, default=0)


class FieldIssue(BaseModel):
    """A structured field flagged for review (low confidence and/or invalid)."""

    name: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    valid: bool = True
    reason: str = ""


class ConfidenceReport(BaseModel):
    """Output of the ReportingAgent: where OCR is confident and where it is not."""

    doc_id: str
    overall_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    grade: str = "F"  # A/B/C/D/F quality grade derived from confidence + coverage
    total_blocks: int = Field(ge=0, default=0)
    weak_block_count: int = Field(ge=0, default=0)
    threshold: float = Field(ge=0.0, le=1.0, default=0.75)
    pages: list[PageConfidence] = Field(default_factory=list)
    weak_regions: list[WeakRegion] = Field(default_factory=list)
    field_issues: list[FieldIssue] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)

    @property
    def weak_block_ratio(self) -> float:
        return self.weak_block_count / self.total_blocks if self.total_blocks else 0.0
