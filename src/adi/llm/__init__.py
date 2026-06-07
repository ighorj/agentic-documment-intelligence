"""Local-only LLM clients for OCR post-correction."""

from adi.llm.base import LLMClient, LocalRuleCorrector
from adi.llm.ollama import OllamaCorrector

__all__ = ["LLMClient", "LocalRuleCorrector", "OllamaCorrector"]
