"""Unit tests for individual agents and their contracts."""

from adi.agents import DocumentSource, IngestionAgent, RecognitionAgent, ValidationAgent
from adi.contracts import PageKind, RegionType
from adi.engines import MockOCREngine
from adi.llm.base import LocalRuleCorrector
from adi.schemas import get_schema


def test_ingestion_marks_text_as_digital():
    layout = IngestionAgent().run(DocumentSource(doc_id="d", raw_text="Patient Name: Jane Doe"))
    assert layout.pages[0].kind == PageKind.DIGITAL
    assert layout.regions[0].region_type == RegionType.KEY_VALUE


def test_ingestion_marks_unknown_path_as_scanned():
    layout = IngestionAgent().run(DocumentSource(doc_id="d", path="/tmp/x.png"))
    assert layout.pages[0].kind == PageKind.SCANNED
    assert "deskew" in layout.pages[0].preprocessing


def test_recognition_digital_is_full_confidence():
    layout = IngestionAgent().run(DocumentSource(doc_id="d", raw_text="hello world"))
    doc = RecognitionAgent().run(layout)
    assert doc.mean_confidence == 1.0
    assert doc.blocks[0].engine == "embedded"


def test_recognition_escalates_low_confidence():
    """A primary engine returning junk should yield to a better second engine."""

    class JunkEngine:
        name = "junk"

        def available(self):
            return True

        def recognize(self, page, region):
            from adi.contracts import Token

            return [Token(text="???", confidence=0.1)]

    layout = IngestionAgent().run(DocumentSource(doc_id="d", path="/tmp/x.png"))
    agent = RecognitionAgent(engines=[JunkEngine(), MockOCREngine()])
    doc = agent.run(layout)
    # The mock's higher-confidence output should win the escalation.
    assert doc.blocks[0].engine == "mock"
    assert any("escalating" in t for t in agent.trace)


def test_local_corrector_only_touches_numeric_context():
    c = LocalRuleCorrector()
    assert c.correct("0O123456", hint="a 6-12 digit MRN") == "00123456"
    # Prose hint -> left untouched even though it contains O/l.
    assert c.correct("Hello World", hint="a person's full name") == "Hello World"


def test_validation_records_provenance_bbox():
    layout = IngestionAgent().run(DocumentSource(doc_id="d", raw_text="MRN: 00123456"))
    doc = RecognitionAgent().run(layout)
    result = ValidationAgent(schema=get_schema("healthcare")).run(doc)
    mrn = result.field("mrn")
    assert mrn.source_bbox is not None
    assert mrn.source_bbox.page == 0
