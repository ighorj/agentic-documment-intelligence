"""AI-native pipeline orchestrator (Claude-powered).

Mirrors `DocumentIntelligencePipeline` but routes through Claude vision +
self-correction instead of the OCR/regex agents. Same domain schemas, same
typed `DocumentResult` / `ConfidenceReport` contracts — so it's a drop-in
upgrade for callers that want maximum accuracy on hard, varied documents.
"""

from __future__ import annotations

from pathlib import Path

from adi.ai.client import ClaudeClient, build_document_content
from adi.ai.extractor import ClaudeExtractorAgent
from adi.contracts import ConfidenceReport, DocumentResult
from adi.schemas import get_schema


class ClaudeDocumentIntelligence:
    """Claude-powered document extraction with built-in validation + reporting."""

    def __init__(
        self,
        domain: str = "generic",
        client: ClaudeClient | None = None,
        model: str | None = None,
        max_corrections: int = 2,
    ) -> None:
        self.schema = get_schema(domain)
        if client is None:
            client = ClaudeClient(model=model) if model else ClaudeClient()
        self.extractor = ClaudeExtractorAgent(
            self.schema, client=client, max_corrections=max_corrections
        )

    @property
    def trace(self) -> list[str]:
        return list(self.extractor.trace)

    def process_file(
        self, path: str, doc_id: str | None = None
    ) -> tuple[DocumentResult, ConfidenceReport]:
        content = build_document_content(path=path)
        return self.extractor.run(content, doc_id or Path(path).stem)

    def process_text(
        self, text: str, doc_id: str = "inline"
    ) -> tuple[DocumentResult, ConfidenceReport]:
        content = build_document_content(raw_text=text)
        return self.extractor.run(content, doc_id)
