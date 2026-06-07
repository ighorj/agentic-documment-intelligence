"""The three connected agents that make up the pipeline."""

from adi.agents.base import Agent
from adi.agents.ingestion import DocumentSource, IngestionAgent
from adi.agents.recognition import RecognitionAgent
from adi.agents.reporting import ReportingAgent
from adi.agents.validation import ValidationAgent

__all__ = [
    "Agent",
    "DocumentSource",
    "IngestionAgent",
    "RecognitionAgent",
    "ReportingAgent",
    "ValidationAgent",
]
