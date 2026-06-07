"""Tests for the AI-native Claude extractor — offline, via an injected fake client.

These exercise schema building, the validation-driven self-correction loop, and
report generation without importing the Anthropic SDK or needing an API key.
"""

from adi.ai import BatchItem, ClaudeExtractorAgent, build_batch_requests, build_document_content
from adi.schemas import get_schema


class _FakeUsage:
    input_tokens = 10
    output_tokens = 5
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0


class FakeClient:
    """Returns canned structured responses in sequence; records each call."""

    model = "fake-model"

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def extract(self, system, messages, schema):
        self.calls.append({"system": system, "messages": messages, "schema": schema})
        idx = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[idx], _FakeUsage()

    def build_params(self, system, messages, schema):
        return {"system": system, "messages": messages,
                "output_config": {"format": {"type": "json_schema", "schema": schema}}}


def _hc_response(npi="1234567893"):
    return {
        "patient_name": {"value": "Alice Morgan", "confidence": 0.96, "evidence": "MORGAN, ALICE"},
        "dob": {"value": "1962-03-15", "confidence": 0.95, "evidence": "DOB: 03/15/1962"},
        "mrn": {"value": "00123456", "confidence": 0.93, "evidence": "MRN 00123456"},
        "npi": {"value": npi, "confidence": 0.9, "evidence": f"NPI {npi}"},
        "primary_diagnosis_code": {"value": "E11.9", "confidence": 0.88, "evidence": "E11.9"},
    }


def test_extracts_and_validates_clean_document():
    client = FakeClient([_hc_response()])
    agent = ClaudeExtractorAgent(get_schema("healthcare"), client=client)
    result, report = agent.run([{"type": "text", "text": "..."}], doc_id="d")

    assert result.metadata["engine"] == "claude"
    # Note: source labels were "MORGAN, ALICE" / "03/15/1962" — the model
    # normalizes semantically, which is the whole point vs. regex.
    assert result.field("patient_name").value == "Alice Morgan"
    assert result.field("dob").value == "1962-03-15"
    assert all(f.valid for f in result.fields)
    assert report.grade in ("A", "B")  # all valid, high confidence
    assert not report.field_issues
    assert len(client.calls) == 1  # no correction needed


def test_self_correction_loop_fixes_invalid_value():
    # First pass returns an NPI that fails the Luhn check; second pass is correct.
    client = FakeClient([_hc_response(npi="1234567890"), _hc_response(npi="1234567893")])
    agent = ClaudeExtractorAgent(get_schema("healthcare"), client=client)
    result, report = agent.run([{"type": "text", "text": "..."}], doc_id="d")

    assert len(client.calls) == 2  # one correction round
    npi = result.field("npi")
    assert npi.value == "1234567893" and npi.valid
    assert any("correction" in t for t in agent.trace)
    # The correction turn must reference the validation error to Claude.
    second_call_user_msg = client.calls[1]["messages"][-1]["content"]
    assert "npi" in second_call_user_msg.lower()


def test_correction_loop_is_bounded():
    # Always-invalid NPI: must stop after max_corrections, not loop forever.
    client = FakeClient([_hc_response(npi="1234567890")])
    agent = ClaudeExtractorAgent(get_schema("healthcare"), client=client, max_corrections=2)
    result, report = agent.run([{"type": "text", "text": "..."}], doc_id="d")

    assert len(client.calls) == 3  # initial + 2 corrections
    assert not result.field("npi").valid
    assert any(fi.name == "npi" and not fi.valid for fi in report.field_issues)
    assert result.metadata["corrections"] == "2"


def test_missing_required_field_is_not_retried():
    resp = _hc_response()
    resp["patient_name"] = {"value": None, "confidence": 0.0, "evidence": ""}
    client = FakeClient([resp])
    agent = ClaudeExtractorAgent(get_schema("healthcare"), client=client)
    result, report = agent.run([{"type": "text", "text": "..."}], doc_id="d")

    # A genuinely absent field shouldn't trigger the correction loop.
    assert len(client.calls) == 1
    assert any("patient_name" in w for w in result.warnings)


def test_low_confidence_surfaces_in_report():
    resp = _hc_response()
    resp["mrn"]["confidence"] = 0.4
    client = FakeClient([resp])
    agent = ClaudeExtractorAgent(get_schema("healthcare"), client=client)
    _, report = agent.run([{"type": "text", "text": "..."}], doc_id="d")

    assert any(fi.name == "mrn" and "low confidence" in fi.reason for fi in report.field_issues)
    assert report.grade != "A"


def test_build_schema_is_strict_and_complete():
    agent = ClaudeExtractorAgent(get_schema("finance"), client=FakeClient([{}]))
    schema = agent._build_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"account_iban", "total_amount", "invoice_date"}
    field = schema["properties"]["total_amount"]
    assert field["additionalProperties"] is False
    assert set(field["required"]) == {"value", "confidence", "evidence"}


def test_build_document_content_by_type(tmp_path):
    # PDF -> document block
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    assert build_document_content(path=str(pdf))[0]["type"] == "document"

    # PNG -> image block
    png = tmp_path / "x.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n fake")
    assert build_document_content(path=str(png))[0]["type"] == "image"

    # raw text -> text block
    assert build_document_content(raw_text="hello")[0] == {"type": "text", "text": "hello"}


def test_build_batch_requests_shape():
    client = FakeClient([{}])
    items = [BatchItem(doc_id="a", raw_text="foo"), BatchItem(doc_id="b", raw_text="bar")]
    reqs = build_batch_requests(client, get_schema("healthcare"), items)
    assert [r["custom_id"] for r in reqs] == ["a", "b"]
    assert reqs[0]["params"]["output_config"]["format"]["type"] == "json_schema"
