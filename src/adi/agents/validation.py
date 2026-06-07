"""Agent 3 — Validation & Extraction (LLM-backed).

Responsibilities:
  * Locate the structured fields a domain cares about within the recognized text.
  * Repair OCR errors using a local LLM/rule corrector, focused by a per-field
    format hint (so a stray 'O' in an MRN becomes '0', but prose is left alone).
  * Validate each value against domain rules (ICD-10, NPI checksum, IBAN mod-97,
    date/amount formats) and attach provenance (bbox + confidence).
  * Emit warnings for missing required fields and failed validations rather than
    silently dropping them — auditability is mandatory for regulated data.
"""

from __future__ import annotations

import re

from adi.agents.base import Agent
from adi.contracts import DocumentResult, ExtractedField, RecognizedDocument
from adi.llm.base import LLMClient, LocalRuleCorrector
from adi.schemas.base import DomainSchema


class ValidationAgent(Agent[RecognizedDocument, DocumentResult]):
    name = "validation"

    def __init__(self, schema: DomainSchema, corrector: LLMClient | None = None) -> None:
        super().__init__()
        self.schema = schema
        self.corrector = corrector or LocalRuleCorrector()

    def run(self, payload: RecognizedDocument) -> DocumentResult:
        self.trace.clear()
        full_text = payload.full_text
        fields: list[ExtractedField] = []
        warnings: list[str] = []
        seen: set[str] = set()

        for spec in self.schema.fields:
            match = self._first_match(full_text, spec.patterns)
            if match is None:
                if spec.required:
                    warnings.append(f"required field '{spec.name}' not found")
                continue

            raw_value = match.strip()
            corrected = self.corrector.correct(raw_value, hint=spec.hint).strip()
            if corrected != raw_value:
                self.log(f"{spec.name}: corrected '{raw_value}' -> '{corrected}'")

            ok, err = spec.validator(corrected)
            block = self._source_block(payload, raw_value)
            fields.append(
                ExtractedField(
                    name=spec.name,
                    value=corrected,
                    raw_value=raw_value if corrected != raw_value else None,
                    confidence=block.confidence if block else 0.0,
                    valid=ok,
                    validation_error=err,
                    source_bbox=block.bbox if block else None,
                )
            )
            seen.add(spec.name)
            if not ok:
                warnings.append(f"field '{spec.name}' failed validation: {err}")

        result = DocumentResult(
            doc_id=payload.doc_id,
            domain=self.schema.name,
            fields=fields,
            full_text=full_text,
            mean_confidence=payload.mean_confidence,
            warnings=warnings,
            metadata={
                "corrector": self.corrector.name,
                "fields_found": str(len(seen)),
                "fields_expected": str(len(self.schema.fields)),
            },
        )
        self.log(f"extracted {len(fields)} field(s), {len(warnings)} warning(s)")
        return result

    @staticmethod
    def _first_match(text: str, patterns: list[str]) -> str | None:
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                return m.group(1)
        return None

    @staticmethod
    def _source_block(doc: RecognizedDocument, raw_value: str):
        return next((b for b in doc.blocks if raw_value and raw_value in b.text), None)
