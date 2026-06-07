"""AI-native, Claude-powered document intelligence.

Requires the `ai` extra (`pip install -e ".[ai]"`) and an ANTHROPIC_API_KEY to
run against the real API. The agent and pipeline accept an injected client, so
they're fully unit-testable without the SDK or a key.
"""

from adi.ai.batch import BatchItem, build_batch_requests, extract_batch
from adi.ai.client import ClaudeClient, build_document_content
from adi.ai.extractor import ClaudeExtractorAgent
from adi.ai.pipeline import ClaudeDocumentIntelligence

__all__ = [
    "ClaudeClient",
    "ClaudeDocumentIntelligence",
    "ClaudeExtractorAgent",
    "build_document_content",
    "BatchItem",
    "build_batch_requests",
    "extract_batch",
]
