"""Offline demo of the AI-native Claude extractor.

Runs without an API key by injecting a stub client that mimics Claude's
structured responses — including a first pass with a checksum-invalid NPI so you
can watch the self-correction loop fix it. With a real key, swap the stub for
`ClaudeClient()` and point it at a real PDF/scan.

    python examples/run_ai_demo.py
"""

from __future__ import annotations

from adi.ai.extractor import ClaudeExtractorAgent
from adi.cli import _render_human, _render_report
from adi.schemas import get_schema


class StubClaude:
    """Stand-in for ClaudeClient: returns canned structured output, then a fix."""

    model = "stub (offline demo)"

    def __init__(self, responses):
        self._responses = responses
        self.calls = 0

    def extract(self, system, messages, schema):
        data = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1

        class _U:  # minimal usage object
            input_tokens = output_tokens = cache_read_input_tokens = cache_creation_input_tokens = 0

        return data, _U()


def _resp(npi):
    # Field values mimic Claude reading a messy medical fax and normalizing
    # (synthetic sample data — not a real patient).
    return {
        "patient_name": {"value": "Alice Morgan", "confidence": 0.96, "evidence": "MORGAN, ALICE"},
        "dob": {"value": "1962-03-15", "confidence": 0.95, "evidence": "DOB: 03/15/1962"},
        "mrn": {"value": "00123456", "confidence": 0.92, "evidence": "MRN 00123456"},
        "npi": {"value": npi, "confidence": 0.9, "evidence": f"NPI {npi}"},
        "primary_diagnosis_code": {"value": "E11.9", "confidence": 0.89, "evidence": "Dx E11.9"},
    }


def main() -> None:
    # First pass returns an NPI that fails the Luhn check; the loop corrects it.
    client = StubClaude([_resp("1234567890"), _resp("1234567893")])
    agent = ClaudeExtractorAgent(get_schema("healthcare"), client=client)

    result, report = agent.run(
        [{"type": "text", "text": "(in a real run, a PDF/scan/photo goes here)"}],
        doc_id="discharge-fax",
    )

    print(_render_human(result))
    print(_render_report(report))
    print("\ntrace:")
    print("\n".join(f"  {t}" for t in agent.trace))
    print(f"\n(Claude was called {client.calls}× — initial extraction + self-correction)")


if __name__ == "__main__":
    main()
