"""Agent 2 — OCR / Recognition.

Responsibilities:
  * For DIGITAL pages: trust the embedded text (confidence 1.0, zero OCR cost).
  * For SCANNED pages: run OCR per region, attaching per-token confidence.
  * Confidence-based escalation: if the primary engine returns a low-confidence
    block, retry with the next engine in the ensemble and keep whichever result
    is more confident. This is how we spend the expensive recognition path only
    on the hard regions instead of the whole document.
"""

from __future__ import annotations

from adi.agents.base import Agent
from adi.contracts import (
    DocumentLayout,
    PageKind,
    RecognizedDocument,
    TextBlock,
    Token,
)
from adi.engines import MockOCREngine
from adi.engines.base import OCREngine


class RecognitionAgent(Agent[DocumentLayout, RecognizedDocument]):
    name = "recognition"

    def __init__(self, engines: list[OCREngine] | None = None, escalate_below: float = 0.75) -> None:
        super().__init__()
        # Default to the always-available mock so the pipeline runs untouched.
        candidates = engines or [MockOCREngine()]
        self.engines = [e for e in candidates if e.available()]
        if not self.engines:
            self.engines = [MockOCREngine()]
        self.escalate_below = escalate_below

    def run(self, payload: DocumentLayout) -> RecognizedDocument:
        self.trace.clear()
        pages = {p.page_no: p for p in payload.pages}
        blocks: list[TextBlock] = []

        for region in sorted(payload.regions, key=lambda r: (r.page_no, r.order)):
            page = pages.get(region.page_no)
            if page is None:
                continue
            if page.kind == PageKind.DIGITAL:
                blocks.append(self._digital_block(page, region))
            else:
                blocks.append(self._ocr_block(page, region))

        self.log(
            f"recognized {len(blocks)} block(s) using "
            f"{', '.join(e.name for e in self.engines)}"
        )
        return RecognizedDocument(doc_id=payload.doc_id, blocks=blocks)

    # -- digital: slice embedded text for this region ----------------------
    def _digital_block(self, page, region) -> TextBlock:
        # On a digital page the region order maps 1:1 to a source line.
        lines = [ln for ln in (page.embedded_text or "").splitlines() if ln.strip()]
        text = lines[region.order] if region.order < len(lines) else ""
        tokens = [Token(text=w, confidence=1.0) for w in text.split()]
        return TextBlock(
            page_no=region.page_no,
            region_type=region.region_type,
            text=text,
            tokens=tokens,
            bbox=region.bbox,
            confidence=1.0,
            engine="embedded",
        )

    # -- scanned: OCR with confidence-based escalation ---------------------
    def _ocr_block(self, page, region) -> TextBlock:
        best: TextBlock | None = None
        for engine in self.engines:
            tokens = engine.recognize(page, region)
            block = self._assemble(tokens, region, engine.name)
            if best is None or block.confidence > best.confidence:
                best = block
            if block.confidence >= self.escalate_below:
                break  # good enough; don't pay for further engines
            self.log(
                f"page {region.page_no} region@{region.order}: {engine.name} "
                f"conf={block.confidence:.2f} below {self.escalate_below}, escalating"
            )
        return best or TextBlock(page_no=region.page_no, bbox=region.bbox, engine="none")

    @staticmethod
    def _assemble(tokens: list[Token], region, engine_name: str) -> TextBlock:
        text = " ".join(t.text for t in tokens)
        conf = sum(t.confidence for t in tokens) / len(tokens) if tokens else 0.0
        return TextBlock(
            page_no=region.page_no,
            region_type=region.region_type,
            text=text,
            tokens=tokens,
            bbox=region.bbox,
            confidence=conf,
            engine=engine_name,
        )
