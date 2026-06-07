"""Agent 4 — Confidence & Quality Reporting.

A terminal agent that turns the per-token/per-block confidence the Recognition
agent already produced into an actionable quality report: an overall grade,
per-page roll-ups, the specific weak regions (with page, bbox and a text preview
so a human can find them fast), any structured fields that warrant review, and
concrete recommendations for *how* to improve the result.

It optionally takes the ValidationAgent's DocumentResult so it can also surface
field-level problems (low confidence or failed validation), but it works on a
RecognizedDocument alone — confidence is fundamentally an OCR-stage property.
"""

from __future__ import annotations

from adi.agents.base import Agent
from adi.contracts import (
    ConfidenceReport,
    DocumentResult,
    FieldIssue,
    PageConfidence,
    RecognizedDocument,
    WeakRegion,
)

# Confidence at or above this is considered reliable. Anything below is "weak"
# and shows up in the report's drill-downs.
_DEFAULT_THRESHOLD = 0.75
# Tokens below this confidence are called out individually as the likely culprits.
_TOKEN_THRESHOLD = 0.60
_PREVIEW_CHARS = 80


def _grade(confidence: float, weak_ratio: float) -> str:
    """Letter grade from mean confidence, penalised by how much of the document
    is weak (a high average hides little if 40% of blocks are shaky)."""
    score = confidence * (1.0 - 0.5 * weak_ratio)
    if score >= 0.95:
        return "A"
    if score >= 0.85:
        return "B"
    if score >= 0.75:
        return "C"
    if score >= 0.60:
        return "D"
    return "F"


class ReportingAgent(Agent[RecognizedDocument, ConfidenceReport]):
    name = "reporting"

    def __init__(
        self,
        threshold: float = _DEFAULT_THRESHOLD,
        token_threshold: float = _TOKEN_THRESHOLD,
        max_weak_regions: int = 25,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.token_threshold = token_threshold
        self.max_weak_regions = max_weak_regions

    def run(self, payload: RecognizedDocument, result: DocumentResult | None = None) -> ConfidenceReport:
        self.trace.clear()
        blocks = [b for b in payload.blocks if b.text]

        pages = self._page_rollups(blocks)
        weak_regions = self._weak_regions(blocks)
        field_issues = self._field_issues(result)

        overall = payload.mean_confidence
        weak_count = sum(1 for b in blocks if b.confidence < self.threshold)
        weak_ratio = weak_count / len(blocks) if blocks else 0.0
        grade = _grade(overall, weak_ratio)

        report = ConfidenceReport(
            doc_id=payload.doc_id,
            overall_confidence=round(overall, 4),
            grade=grade,
            total_blocks=len(blocks),
            weak_block_count=weak_count,
            threshold=self.threshold,
            pages=pages,
            weak_regions=weak_regions,
            field_issues=field_issues,
        )
        report.recommendations = self._recommendations(report, blocks)
        self.log(
            f"graded '{report.doc_id}' {grade} "
            f"({weak_count}/{len(blocks)} weak blocks, conf={overall:.2f})"
        )
        return report

    # -- per-page roll-ups --------------------------------------------------
    def _page_rollups(self, blocks) -> list[PageConfidence]:
        by_page: dict[int, list] = {}
        for b in blocks:
            by_page.setdefault(b.page_no, []).append(b)
        rollups = []
        for page_no in sorted(by_page):
            page_blocks = by_page[page_no]
            mean = sum(b.confidence for b in page_blocks) / len(page_blocks)
            weak = sum(1 for b in page_blocks if b.confidence < self.threshold)
            rollups.append(
                PageConfidence(
                    page_no=page_no,
                    mean_confidence=round(mean, 4),
                    block_count=len(page_blocks),
                    weak_block_count=weak,
                )
            )
        return rollups

    # -- weak regions (sorted worst-first) ---------------------------------
    def _weak_regions(self, blocks) -> list[WeakRegion]:
        weak = [b for b in blocks if b.confidence < self.threshold]
        weak.sort(key=lambda b: b.confidence)
        regions = []
        for b in weak[: self.max_weak_regions]:
            preview = b.text if len(b.text) <= _PREVIEW_CHARS else b.text[:_PREVIEW_CHARS] + "…"
            weak_tokens = [t.text for t in b.tokens if t.confidence < self.token_threshold]
            regions.append(
                WeakRegion(
                    page_no=b.page_no,
                    region_type=b.region_type,
                    confidence=round(b.confidence, 4),
                    text_preview=preview,
                    engine=b.engine,
                    bbox=b.bbox,
                    weak_tokens=weak_tokens,
                )
            )
        return regions

    # -- field-level issues (needs the validation result) ------------------
    def _field_issues(self, result: DocumentResult | None) -> list[FieldIssue]:
        if result is None:
            return []
        issues = []
        for f in result.fields:
            reasons = []
            if not f.valid:
                reasons.append(f"failed validation ({f.validation_error})")
            if f.confidence < self.threshold:
                reasons.append(f"low confidence ({f.confidence:.2f})")
            if reasons:
                issues.append(
                    FieldIssue(
                        name=f.name,
                        value=f.value,
                        confidence=f.confidence,
                        valid=f.valid,
                        reason="; ".join(reasons),
                    )
                )
        return issues

    # -- actionable recommendations ----------------------------------------
    def _recommendations(self, report: ConfidenceReport, blocks) -> list[str]:
        recs: list[str] = []
        if report.grade in ("A", "B") and not report.field_issues:
            recs.append("Recognition quality is good; no action required.")
            return recs

        # Pages that are disproportionately weak are the best rescan candidates.
        for page in report.pages:
            if page.block_count and page.weak_block_count / page.block_count >= 0.5:
                recs.append(
                    f"Page {page.page_no}: {page.weak_block_count}/{page.block_count} regions "
                    f"are low-confidence — rescan at higher DPI (≥300) or improve deskew/denoise."
                )

        if any(b.engine in ("mock", "tesseract") for b in blocks if b.confidence < self.threshold):
            recs.append(
                "Low-confidence blocks came from a single OCR engine — enable engine "
                "escalation (add a second engine) so hard regions get a stronger pass."
            )

        for fi in report.field_issues:
            if not fi.valid:
                recs.append(f"Field '{fi.name}'='{fi.value}' failed validation — flag for manual review.")
            else:
                recs.append(f"Field '{fi.name}' extracted at low confidence — verify against the source.")

        if not recs:
            recs.append(
                f"Overall confidence {report.overall_confidence:.2f} is below target; "
                "review the weak regions listed above."
            )
        return recs
