"""The AI brain — a Claude-powered, self-correcting extraction agent.

This is the AI-native replacement for the OCR→regex pipeline. Instead of
recognizing characters and pattern-matching labels (which breaks the moment a
document uses a different layout, language, or handwriting), it hands the raw
document to Claude vision and asks for the fields *by meaning*, returning
schema-constrained JSON.

What makes it agentic rather than a single API call:

  1. **Schema-guided extraction** — the domain's fields become a JSON schema, so
     the response is guaranteed parseable and complete.
  2. **Self-correction loop** — every value is checked against real domain
     validators (IBAN mod-97, NPI Luhn, ICD-10, dates). If something fails, the
     agent shows Claude the specific error and asks it to re-examine the document
     and fix it — Claude reasons over its own output and corrects. This is the
     difference between "I piped a doc into an LLM" and a reliable system.
  3. **Confidence & evidence** — Claude returns a per-field confidence and a
     short verbatim quote it relied on, which feeds the quality report.
"""

from __future__ import annotations

from typing import Any

from adi.agents.reporting import _grade
from adi.ai.client import ClaudeClient, ExtractionUsage
from adi.contracts import (
    ConfidenceReport,
    DocumentResult,
    ExtractedField,
    FieldIssue,
)
from adi.schemas.base import DomainSchema

_SYSTEM_PROMPT = (
    "You are an expert document-intelligence extractor for regulated domains "
    "(healthcare, finance). You read documents that may be digital, scanned, "
    "photographed, or handwritten, in any layout or language.\n\n"
    "Rules:\n"
    "- Extract each requested field by MEANING, not by matching a fixed label. "
    "A name may appear without the word 'Name'; a date may be in any format.\n"
    "- Normalize values: dates as ISO-8601 (YYYY-MM-DD); keep names, codes, and "
    "identifiers exactly as written (do not reformat IBANs, MRNs, or codes).\n"
    "- If a field is genuinely absent or illegible, return null with low "
    "confidence. NEVER invent or guess a value.\n"
    "- For every field, give a confidence in [0,1] and a short verbatim quote "
    "from the document as evidence (empty string if the value is null).\n"
    "- Be precise and literal. Accuracy and calibration matter more than recall."
)

_THRESHOLD = 0.75


class ClaudeExtractorAgent:
    """Extracts a domain's fields from a document using Claude, with validation
    feedback driving an automatic correction loop."""

    name = "claude-extractor"

    def __init__(
        self,
        schema: DomainSchema,
        client: ClaudeClient | None = None,
        max_corrections: int = 2,
        threshold: float = _THRESHOLD,
    ) -> None:
        self.schema = schema
        self.client = client or ClaudeClient()
        self.max_corrections = max_corrections
        self.threshold = threshold
        self.trace: list[str] = []

    # ------------------------------------------------------------------ run
    def run(
        self, doc_content: list[dict[str, Any]], doc_id: str
    ) -> tuple[DocumentResult, ConfidenceReport]:
        self.trace.clear()
        json_schema = self._build_schema()
        usage = ExtractionUsage()

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": [*doc_content, {"type": "text", "text": self._instruction()}]}
        ]

        data, u = self.client.extract(_SYSTEM_PROMPT, messages, json_schema)
        usage.add(u)
        fields = self._to_fields(data)
        self.log(f"initial extraction: {len(fields)} field(s)")

        attempts = 0
        while self._invalid(fields) and attempts < self.max_corrections:
            attempts += 1
            bad = self._invalid(fields)
            self.log(f"correction {attempts}: re-checking {[f.name for f in bad]}")
            messages.append({"role": "assistant", "content": _dump(data)})
            messages.append({"role": "user", "content": self._correction_prompt(bad)})
            data, u = self.client.extract(_SYSTEM_PROMPT, messages, json_schema)
            usage.add(u)
            fields = self._to_fields(data)

        result = self._result(doc_id, fields, usage, attempts)
        report = self._report(doc_id, fields)
        self.log(f"done: grade {report.grade}, {len(report.field_issues)} issue(s)")
        return result, report

    def log(self, message: str) -> None:
        self.trace.append(f"[{self.name}] {message}")

    # ------------------------------------------------------- schema + prompt
    def _build_schema(self) -> dict[str, Any]:
        """Turn the domain's FieldSpecs into a strict JSON schema. Each field is
        an object with value/confidence/evidence so we get calibrated, auditable
        output rather than a bare string."""
        props: dict[str, Any] = {}
        for spec in self.schema.fields:
            props[spec.name] = {
                "type": "object",
                "description": spec.description or spec.hint or spec.name,
                "properties": {
                    "value": {
                        "type": ["string", "null"],
                        "description": f"{spec.hint or spec.name}; null if absent/illegible",
                    },
                    "confidence": {"type": "number", "description": "0.0–1.0"},
                    "evidence": {"type": "string", "description": "verbatim quote, or empty"},
                },
                "required": ["value", "confidence", "evidence"],
                "additionalProperties": False,
            }
        return {
            "type": "object",
            "properties": props,
            "required": list(props.keys()),
            "additionalProperties": False,
        }

    def _instruction(self) -> str:
        lines = [f"Extract the following fields from the attached document ({self.schema.name}):"]
        for spec in self.schema.fields:
            req = " (required)" if spec.required else ""
            lines.append(f"- {spec.name}: {spec.hint or spec.description or spec.name}{req}")
        lines.append("Return JSON matching the schema. Use null for anything not present.")
        return "\n".join(lines)

    def _correction_prompt(self, bad: list[ExtractedField]) -> str:
        lines = ["These fields failed validation. Re-examine the document and correct them:"]
        for f in bad:
            lines.append(f"- {f.name} = '{f.value}': {f.validation_error}")
        lines.append(
            "Look again carefully at the relevant part of the document. If a value is "
            "genuinely not present or unreadable, set it to null with low confidence "
            "rather than guessing. Return the full JSON again."
        )
        return "\n".join(lines)

    # ----------------------------------------------------------- conversion
    def _to_fields(self, data: dict[str, Any]) -> list[ExtractedField]:
        fields: list[ExtractedField] = []
        for spec in self.schema.fields:
            raw = data.get(spec.name) or {}
            value = raw.get("value")
            confidence = _clamp(raw.get("confidence", 0.0))
            if value is None or value == "":
                # Missing value: only an issue if the field was required.
                if spec.required:
                    fields.append(
                        ExtractedField(
                            name=spec.name, value="", confidence=confidence,
                            valid=False, validation_error="required field not found",
                        )
                    )
                continue
            ok, err = spec.validator(str(value))
            fields.append(
                ExtractedField(
                    name=spec.name,
                    value=str(value),
                    confidence=confidence,
                    valid=ok,
                    validation_error=err,
                    evidence=(raw.get("evidence") or None),
                )
            )
        return fields

    @staticmethod
    def _invalid(fields: list[ExtractedField]) -> list[ExtractedField]:
        # Only re-prompt for present-but-invalid values; a genuinely missing
        # field won't be fixed by asking again.
        return [f for f in fields if not f.valid and f.value]

    # -------------------------------------------------------------- outputs
    def _result(
        self, doc_id: str, fields: list[ExtractedField], usage: ExtractionUsage, attempts: int
    ) -> DocumentResult:
        warnings = [
            f"field '{f.name}' {f.validation_error}" for f in fields if not f.valid
        ]
        confidences = [f.confidence for f in fields if f.value]
        mean = sum(confidences) / len(confidences) if confidences else 0.0
        return DocumentResult(
            doc_id=doc_id,
            domain=self.schema.name,
            fields=fields,
            mean_confidence=round(mean, 4),
            warnings=warnings,
            metadata={
                "engine": "claude",
                "model": getattr(self.client, "model", "claude"),
                "corrections": str(attempts),
                **{k: str(v) for k, v in usage.as_dict().items()},
            },
        )

    def _report(self, doc_id: str, fields: list[ExtractedField]) -> ConfidenceReport:
        present = [f for f in fields if f.value]
        weak = [f for f in present if f.confidence < self.threshold]
        mean = sum(f.confidence for f in present) / len(present) if present else 0.0
        weak_ratio = len(weak) / len(present) if present else 0.0

        issues: list[FieldIssue] = []
        recs: list[str] = []
        for f in fields:
            reasons = []
            if not f.valid:
                reasons.append(f"failed validation ({f.validation_error})")
            if f.value and f.confidence < self.threshold:
                reasons.append(f"low confidence ({f.confidence:.2f})")
            if reasons:
                issues.append(
                    FieldIssue(name=f.name, value=f.value, confidence=f.confidence,
                               valid=f.valid, reason="; ".join(reasons))
                )
                if not f.valid:
                    recs.append(f"Field '{f.name}'='{f.value}' failed validation — manual review.")
                else:
                    recs.append(f"Field '{f.name}' extracted at low confidence — verify against source.")

        if not issues:
            recs.append("Extraction quality is high; no action required.")

        return ConfidenceReport(
            doc_id=doc_id,
            overall_confidence=round(mean, 4),
            grade=_grade(mean, weak_ratio),
            total_blocks=len(present),
            weak_block_count=len(weak),
            threshold=self.threshold,
            pages=[],
            weak_regions=[],
            field_issues=issues,
            recommendations=recs,
        )


def _clamp(x: Any) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


def _dump(data: dict[str, Any]) -> str:
    import json

    return json.dumps(data, ensure_ascii=False)
