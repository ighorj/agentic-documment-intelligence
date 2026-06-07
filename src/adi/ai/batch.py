"""High-volume extraction via the Claude Message Batches API.

For "thousands of documents per day" workloads, the Batch API processes requests
asynchronously at 50% of standard cost (up to 100k requests / batch, most finish
within the hour). This module builds one batch request per document — same schema
and cached system prompt as the single-document path — submits, polls, and maps
results back to structured `DocumentResult`s.

Note: the batch path does a single extraction pass per document (no interactive
self-correction loop — that requires multi-turn round trips). Validators still run
on the returned values, so invalid fields are flagged in the result; re-submit
just those documents through the interactive `ClaudeExtractorAgent` if needed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from adi.ai.client import ClaudeClient, build_document_content
from adi.ai.extractor import _SYSTEM_PROMPT, ClaudeExtractorAgent
from adi.contracts import DocumentResult
from adi.schemas.base import DomainSchema


@dataclass
class BatchItem:
    """One document to extract in a batch."""

    doc_id: str
    path: str | None = None
    raw_text: str | None = None


def build_batch_requests(
    client: ClaudeClient, schema: DomainSchema, items: list[BatchItem]
) -> list[dict[str, Any]]:
    """Build the per-document batch request payloads (custom_id + params)."""
    agent = ClaudeExtractorAgent(schema, client=client)
    json_schema = agent._build_schema()
    instruction = agent._instruction()
    requests = []
    for item in items:
        content = build_document_content(path=item.path, raw_text=item.raw_text)
        messages = [{"role": "user", "content": [*content, {"type": "text", "text": instruction}]}]
        requests.append(
            {
                "custom_id": item.doc_id,
                "params": client.build_params(_SYSTEM_PROMPT, messages, json_schema),
            }
        )
    return requests


def extract_batch(
    client: ClaudeClient,
    schema: DomainSchema,
    items: list[BatchItem],
    poll_interval: float = 30.0,
    timeout: float = 24 * 3600,
) -> dict[str, DocumentResult]:
    """Submit a batch, wait for completion, and return {doc_id: DocumentResult}."""
    import json

    requests = build_batch_requests(client, schema, items)
    batch = client.raw.messages.batches.create(requests=requests)

    deadline = time.monotonic() + timeout
    while True:
        batch = client.raw.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        if time.monotonic() > deadline:
            raise TimeoutError(f"batch {batch.id} did not finish within {timeout}s")
        time.sleep(poll_interval)

    agent = ClaudeExtractorAgent(schema, client=client)
    results: dict[str, DocumentResult] = {}
    for entry in client.raw.messages.batches.results(batch.id):
        if entry.result.type != "succeeded":
            continue
        msg = entry.result.message
        text = next((b.text for b in msg.content if b.type == "text"), "{}")
        data = json.loads(text)
        fields = agent._to_fields(data)
        from adi.ai.client import ExtractionUsage

        usage = ExtractionUsage()
        usage.add(msg.usage)
        results[entry.custom_id] = agent._result(entry.custom_id, fields, usage, attempts=0)
    return results
