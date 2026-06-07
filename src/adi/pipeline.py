"""Pipeline orchestrator — wires the three agents together.

    DocumentSource
        -> IngestionAgent   (layout + DIGITAL/SCANNED routing)
        -> RecognitionAgent (OCR with confidence escalation)
        -> ValidationAgent  (LLM correction + domain validation)
        -> DocumentResult

The orchestrator owns no business logic itself; it just connects typed contracts
and aggregates each agent's trace so a full run is auditable.
"""

from __future__ import annotations

from pathlib import Path

from adi.agents import (
    DocumentSource,
    IngestionAgent,
    RecognitionAgent,
    ReportingAgent,
    ValidationAgent,
)
from adi.contracts import ConfidenceReport, DocumentResult
from adi.engines.base import OCREngine
from adi.llm.base import LLMClient
from adi.schemas import get_schema


class DocumentIntelligencePipeline:
    def __init__(
        self,
        domain: str = "generic",
        engines: list[OCREngine] | None = None,
        corrector: LLMClient | None = None,
        escalate_below: float = 0.75,
        report_threshold: float = 0.75,
    ) -> None:
        self.schema = get_schema(domain)
        self.ingestion = IngestionAgent()
        self.recognition = RecognitionAgent(engines=engines, escalate_below=escalate_below)
        self.validation = ValidationAgent(schema=self.schema, corrector=corrector)
        self.reporting = ReportingAgent(threshold=report_threshold)

    @property
    def trace(self) -> list[str]:
        return [
            *self.ingestion.trace,
            *self.recognition.trace,
            *self.validation.trace,
            *self.reporting.trace,
        ]

    def run(self, source: DocumentSource) -> DocumentResult:
        layout = self.ingestion.run(source)
        recognized = self.recognition.run(layout)
        return self.validation.run(recognized)

    def run_with_report(
        self, source: DocumentSource
    ) -> tuple[DocumentResult, ConfidenceReport]:
        """Run the full pipeline and also produce an OCR confidence/quality report."""
        layout = self.ingestion.run(source)
        recognized = self.recognition.run(layout)
        result = self.validation.run(recognized)
        report = self.reporting.run(recognized, result=result)
        return result, report

    # Convenience entry points -------------------------------------------------
    def process_file(self, path: str, doc_id: str | None = None) -> DocumentResult:
        return self.run(DocumentSource(doc_id=doc_id or Path(path).stem, path=path))

    def process_text(self, text: str, doc_id: str = "inline") -> DocumentResult:
        return self.run(DocumentSource(doc_id=doc_id, raw_text=text))
