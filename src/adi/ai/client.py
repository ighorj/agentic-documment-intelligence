"""Claude API client wrapper for document intelligence.

Thin, dependency-isolated layer over the official Anthropic SDK. It does three
things and nothing else:

  * Build multimodal content blocks from a document (PDF / image / raw text) so
    Claude *reads the document directly* — no OCR stage, so it handles scans,
    handwriting, and arbitrary layouts.
  * Issue a single structured-extraction request: Claude vision + adaptive
    thinking + a JSON-schema-constrained response (`output_config.format`), with
    the system prompt marked for prompt caching.
  * Expose the request params builder so the same shape feeds the Batch API for
    high-volume extraction (`batch.py`).

The Anthropic SDK is imported lazily so the rest of the package (and the entire
test suite) works without it installed; only constructing a real `ClaudeClient`
requires `pip install -e ".[ai]"` and an `ANTHROPIC_API_KEY`.
"""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

# Default model: the most capable Claude model. Override per-deployment.
DEFAULT_MODEL = "claude-opus-4-8"

_IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def build_document_content(
    path: str | None = None, raw_text: str | None = None
) -> list[dict[str, Any]]:
    """Return the content blocks for one document, ready to drop into a user turn.

    PDFs become a ``document`` block, images an ``image`` block, everything else
    (and explicit ``raw_text``) a ``text`` block. Claude ingests all of these
    natively — the model is the OCR engine.
    """
    if raw_text is not None:
        return [{"type": "text", "text": raw_text}]
    if path is None:
        raise ValueError("provide either a file path or raw_text")

    suffix = Path(path).suffix.lower()
    data = Path(path).read_bytes()
    b64 = base64.standard_b64encode(data).decode("utf-8")

    if suffix == ".pdf":
        return [
            {
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": b64},
            }
        ]
    if suffix in _IMAGE_MEDIA_TYPES:
        return [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": _IMAGE_MEDIA_TYPES[suffix], "data": b64},
            }
        ]
    # Fallback: treat unknown types as UTF-8 text (csv, txt, eml, ...).
    media = mimetypes.guess_type(path)[0] or ""
    text = data.decode("utf-8", errors="replace") if media.startswith("text") or not media else None
    if text is not None:
        return [{"type": "text", "text": text}]
    raise ValueError(f"unsupported document type: {suffix or media or 'unknown'}")


class ExtractionUsage:
    """Token accounting for one or more Claude calls (cumulative)."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_input_tokens = 0
        self.cache_creation_input_tokens = 0

    def add(self, usage: Any) -> None:
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.cache_read_input_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.cache_creation_input_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
        }


class ClaudeClient:
    """Wraps the Anthropic SDK for schema-constrained extraction."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        effort: str = "high",
        max_tokens: int = 8000,
    ) -> None:
        import anthropic  # lazy: only needed for a real client

        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def build_params(
        self, system: str, messages: list[dict[str, Any]], schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Build the Messages API params. Shared by single + batch extraction so
        both paths stay byte-identical (and cache-compatible)."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "thinking": {"type": "adaptive"},
            # effort tunes thinking depth; format pins the response to our schema.
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
            "system": [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ],
            "messages": messages,
        }

    def extract(
        self, system: str, messages: list[dict[str, Any]], schema: dict[str, Any]
    ) -> tuple[dict[str, Any], Any]:
        """One structured-extraction call. Returns (parsed_json, usage)."""
        params = self.build_params(system, messages, schema)
        response = self._client.messages.create(**params)
        # output_config.format guarantees the first text block is valid JSON.
        text = next(b.text for b in response.content if b.type == "text")
        return json.loads(text), response.usage

    @property
    def raw(self) -> Any:
        """The underlying anthropic.Anthropic client (for Batch API access)."""
        return self._client
