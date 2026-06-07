"""Tests for Agent 4 — the confidence/quality reporting agent."""

from adi import DocumentIntelligencePipeline
from adi.agents import DocumentSource, ReportingAgent
from adi.contracts import (
    BoundingBox,
    RecognizedDocument,
    RegionType,
    TextBlock,
    Token,
)


def _block(text, conf, page=0, engine="mock", token_confs=None):
    toks = [Token(text=w, confidence=(token_confs or {}).get(w, conf)) for w in text.split()]
    return TextBlock(
        page_no=page,
        region_type=RegionType.PARAGRAPH,
        text=text,
        tokens=toks,
        bbox=BoundingBox(page=page, x0=0, y0=0, x1=100, y1=20),
        confidence=conf,
        engine=engine,
    )


def test_high_confidence_doc_grades_well():
    doc = RecognizedDocument(doc_id="d", blocks=[_block("clean text here", 1.0)])
    report = ReportingAgent().run(doc)
    assert report.grade == "A"
    assert report.weak_block_count == 0
    assert "no action required" in " ".join(report.recommendations).lower()


def test_weak_regions_are_detected_and_sorted():
    doc = RecognizedDocument(
        doc_id="d",
        blocks=[
            _block("good line", 0.98),
            _block("bad line one", 0.40),
            _block("bad line two", 0.55),
        ],
    )
    report = ReportingAgent(threshold=0.75).run(doc)
    assert report.weak_block_count == 2
    # worst-first ordering
    confs = [r.confidence for r in report.weak_regions]
    assert confs == sorted(confs)
    assert report.weak_regions[0].confidence == 0.40


def test_weak_tokens_are_called_out():
    doc = RecognizedDocument(
        doc_id="d",
        blocks=[_block("Total 1O.OO", 0.5, token_confs={"1O.OO": 0.2, "Total": 0.9})],
    )
    report = ReportingAgent().run(doc)
    assert "1O.OO" in report.weak_regions[0].weak_tokens
    assert "Total" not in report.weak_regions[0].weak_tokens


def test_per_page_rollup():
    doc = RecognizedDocument(
        doc_id="d",
        blocks=[_block("a", 0.9, page=0), _block("b", 0.4, page=1), _block("c", 0.5, page=1)],
    )
    report = ReportingAgent().run(doc)
    p1 = next(p for p in report.pages if p.page_no == 1)
    assert p1.block_count == 2 and p1.weak_block_count == 2
    assert any("Page 1" in r for r in report.recommendations)


def test_field_issues_from_validation_result():
    # Invalid NPI should surface as a field issue + manual-review recommendation.
    pipe = DocumentIntelligencePipeline(domain="healthcare")
    result, report = pipe.run_with_report(
        DocumentSource(doc_id="d", raw_text="NPI: 1234567890\n")
    )
    assert any(fi.name == "npi" and not fi.valid for fi in report.field_issues)
    assert any("npi" in r.lower() and "manual review" in r.lower() for r in report.recommendations)


def test_pipeline_run_with_report_on_scanned_doc():
    pipe = DocumentIntelligencePipeline(domain="healthcare")
    result, report = pipe.run_with_report(DocumentSource(doc_id="scan", path="/tmp/scan.png"))
    # Scanned OCR yields sub-threshold blocks -> grade below A and recommendations present.
    assert report.overall_confidence < 1.0
    assert report.grade != "A"
    assert report.recommendations
