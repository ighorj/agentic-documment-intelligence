"""agentic-document-intelligence: a local-first, multi-agent OCR system.

Quick start
-----------
    from adi import DocumentIntelligencePipeline

    pipeline = DocumentIntelligencePipeline(domain="healthcare")
    result = pipeline.process_file("discharge.pdf")
    for field in result.fields:
        print(field.name, field.value, field.valid)
"""

from adi.agents import DocumentSource
from adi.contracts import ConfidenceReport, DocumentResult, ExtractedField
from adi.pipeline import DocumentIntelligencePipeline
from adi.schemas import available_domains

__version__ = "0.1.0"

__all__ = [
    "DocumentIntelligencePipeline",
    "DocumentSource",
    "DocumentResult",
    "ConfidenceReport",
    "ExtractedField",
    "available_domains",
]
