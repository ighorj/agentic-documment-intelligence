"""End-to-end pipeline tests across the three agents."""

from adi import DocumentIntelligencePipeline
from adi.agents import DocumentSource

HEALTHCARE_DOC = (
    "DISCHARGE SUMMARY\n"
    "Patient Name: Jane Doe\n"
    "DOB: 1980-04-12\n"
    "MRN: 00123456\n"
    "NPI: 1234567893\n"
    "Primary Diagnosis: E11.9\n"
)

FINANCE_DOC = (
    "INVOICE\n"
    "Invoice Date: 2026-01-15\n"
    "Account: GB82WEST12345698765432\n"
    "Total Charges: USD 1,250.00\n"
)


def test_healthcare_digital_extraction():
    pipe = DocumentIntelligencePipeline(domain="healthcare")
    result = pipe.process_text(HEALTHCARE_DOC)

    assert result.domain == "healthcare"
    assert result.field("patient_name").value == "Jane Doe"
    assert result.field("dob").value == "1980-04-12"
    assert result.field("mrn").value == "00123456"
    npi = result.field("npi")
    assert npi.value == "1234567893" and npi.valid
    dx = result.field("primary_diagnosis_code")
    assert dx.value == "E11.9" and dx.valid
    assert all(f.valid for f in result.fields)


def test_finance_extraction_and_iban_checksum():
    pipe = DocumentIntelligencePipeline(domain="finance")
    result = pipe.process_text(FINANCE_DOC)

    assert result.field("invoice_date").value == "2026-01-15"
    assert result.field("total_amount").value == "1,250.00"
    iban = result.field("account_iban")
    assert iban.value == "GB82WEST12345698765432"
    assert iban.valid, iban.validation_error


def test_invalid_npi_is_flagged():
    pipe = DocumentIntelligencePipeline(domain="healthcare")
    # 1234567890 fails the Luhn checksum.
    result = pipe.process_text("NPI: 1234567890\n")
    npi = result.field("npi")
    assert npi is not None and not npi.valid
    assert any("npi" in w for w in result.warnings)


def test_missing_required_field_warns():
    pipe = DocumentIntelligencePipeline(domain="healthcare")
    result = pipe.process_text("Primary Diagnosis: E11.9\n")
    assert any("patient_name" in w for w in result.warnings)


def test_scanned_path_uses_ocr_and_corrects_noise():
    # No raw_text + a fake path -> scanned route -> MockOCREngine injects O/l
    # noise into MRN/NPI which the corrector must repair.
    pipe = DocumentIntelligencePipeline(domain="healthcare")
    result = pipe.run(DocumentSource(doc_id="scan", path="/tmp/scan.png"))

    mrn = result.field("mrn")
    assert mrn is not None and mrn.value == "00123456"  # 'O' repaired to '0'
    assert mrn.raw_value is not None  # correction was recorded
    npi = result.field("npi")
    assert npi is not None and npi.value == "1234567893" and npi.valid
    # Scanned blocks carry sub-1.0 confidence (unlike digital).
    assert result.mean_confidence < 1.0


def test_trace_is_populated():
    pipe = DocumentIntelligencePipeline(domain="healthcare")
    pipe.process_text(HEALTHCARE_DOC)
    assert any("ingestion" in t for t in pipe.trace)
    assert any("recognition" in t for t in pipe.trace)
    assert any("validation" in t for t in pipe.trace)
