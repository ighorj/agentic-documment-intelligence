# agentic-document-intelligence

A **local-first, multi-agent OCR system** for optimized document understanding —
built for regulated domains like **healthcare** and **finance** where document
data must never leave the premises.

Small, single-purpose agents are wired together through strict typed contracts.
Each agent does one job well, is independently testable, and can be swapped
without touching the others.

```
                      DocumentSource (PDF / image / text)
                                  │
            ┌─────────────────────▼─────────────────────┐
            │  Agent 1 · Ingestion & Layout              │
            │  • DIGITAL vs SCANNED routing per page     │
            │  • deskew / denoise / DPI normalization    │
            │  • layout region + reading-order detection │
            └─────────────────────┬─────────────────────┘
                           DocumentLayout
            ┌─────────────────────▼─────────────────────┐
            │  Agent 2 · OCR / Recognition               │
            │  • digital text read losslessly (conf 1.0) │
            │  • OCR per region w/ per-token confidence   │
            │  • confidence-based engine escalation       │
            └─────────────────────┬─────────────────────┘
                        RecognizedDocument
            ┌─────────────────────▼─────────────────────┐
            │  Agent 3 · Validation & Extraction (LLM)   │
            │  • context-aware OCR error correction       │
            │  • domain validation (ICD-10/NPI/IBAN/…)    │
            │  • structured fields + provenance + warns   │
            └─────────────────────┬─────────────────────┘
                     DocumentResult │ + RecognizedDocument
            ┌─────────────────────▼─────────────────────┐
            │  Agent 4 · Confidence & Quality Reporting  │
            │  • per-page / per-region confidence rollup │
            │  • pinpoints weak regions (page+bbox+text) │
            │  • A–F grade + actionable recommendations  │
            └─────────────────────┬─────────────────────┘
                                  ▼
            DocumentResult + ConfidenceReport  (clean structured JSON)
```

## Why this design

- **Cost & accuracy lever where it matters.** Digital PDF text is extracted
  losslessly — we never pay OCR error or compute for pages that don't need it.
  Only genuinely scanned regions hit the OCR engine.
- **Spend the expensive path only on hard regions.** The recognition agent runs
  an ensemble with *confidence-based escalation*: a cheap engine first, and a
  stronger one only when a block comes back low-confidence.
- **Validation is domain-aware, not generic.** OCR output is checked against real
  rules — ICD-10 format, NPI Luhn checksum, IBAN mod-97 — so structurally
  impossible values are flagged instead of silently passed downstream.
- **Auditable.** Every value keeps provenance (bounding box + confidence), and
  each run produces a per-agent trace — essential for regulated data.
- **Knows where it's weak.** A dedicated reporting agent grades each document and
  surfaces exactly which pages/regions/fields are low-confidence, with concrete
  next steps (rescan, escalate engine, manual review) instead of a single opaque
  score.
- **Fully local.** No cloud calls, no API keys. The optional LLM correction step
  runs against a local model (Ollama) or a deterministic rule-based fallback.

## Install

```bash
# Base install — runs the full pipeline out of the box with the built-in
# MockOCREngine (zero native dependencies).
pip install -e .

# Real local OCR (Tesseract + poppler + OpenCV). Needs the `tesseract` binary.
pip install -e ".[ocr]"

# Dev tooling (pytest, ruff)
pip install -e ".[dev]"
```

## Usage

### Library

```python
from adi import DocumentIntelligencePipeline

pipeline = DocumentIntelligencePipeline(domain="healthcare")
result = pipeline.process_file("discharge.pdf")

for f in result.fields:
    print(f.name, "=", f.value, "(valid)" if f.valid else f"(BAD: {f.validation_error})")

print("warnings:", result.warnings)
```

### Confidence / quality report (Agent 4)

Get a structured report of *where OCR is lacking* alongside the extracted data:

```python
result, report = pipeline.run_with_report(  # or pass a DocumentSource
    DocumentSource(doc_id="scan", path="scan.png")
)

print(report.grade)               # 'A'..'F'
print(report.overall_confidence)  # 0.0..1.0
for region in report.weak_regions:        # worst-first
    print(region.page_no, region.bbox, region.confidence, region.weak_tokens)
for fi in report.field_issues:            # low-confidence or invalid fields
    print(fi.name, fi.reason)
for rec in report.recommendations:        # actionable next steps
    print(rec)
```

The grade blends mean confidence with how much of the document is weak, so a high
average can't hide a document where a large fraction of regions are shaky.

### CLI

```bash
ocr-extract discharge.pdf --domain healthcare
ocr-extract invoice.pdf  --domain finance --json out.json --trace
ocr-extract scan.pdf     --domain healthcare --report --report-json report.json
echo "MRN: 0O123456" | ocr-extract --text - --domain healthcare
ocr-extract --list-domains
```

The CLI exits non-zero if any extracted field fails validation, so it composes
cleanly into scripts and CI.

### Demo (no native deps required)

```bash
python examples/run_demo.py
```

This shows a digital extraction, a *scanned* document where OCR noise
(`0O123456` → `00123456`, `l234567893` → `1234567893`) is automatically
repaired, and a finance invoice with IBAN checksum validation.

## Swapping in real engines

Everything pluggable hides behind a small protocol, so going from skeleton to
production is incremental:

| Concern        | Skeleton default     | Production drop-in                                  |
| -------------- | -------------------- | -------------------------------------------------- |
| OCR engine     | `MockOCREngine`      | `TesseractEngine` (or your own `OCREngine`)        |
| OCR correction | `LocalRuleCorrector` | `OllamaCorrector` (any local `LLMClient`)          |
| Domain rules   | `healthcare/finance` | add a module under `adi/schemas/` and register it  |

```python
from adi import DocumentIntelligencePipeline
from adi.engines import TesseractEngine, MockOCREngine
from adi.llm import OllamaCorrector

pipeline = DocumentIntelligencePipeline(
    domain="finance",
    engines=[TesseractEngine(), MockOCREngine()],  # escalation fallback chain
    corrector=OllamaCorrector(model="llama3.1"),    # local model, falls back if down
)
```

## Project layout

```
src/adi/
  contracts.py        # typed models passed between agents (the backbone)
  pipeline.py         # orchestrator wiring the 3 agents together
  cli.py              # `ocr-extract` entry point
  agents/             # ingestion · recognition · validation · reporting
  engines/            # OCR backends: mock + tesseract (pluggable)
  llm/                # local correctors: rule-based + ollama (pluggable)
  schemas/            # domain field specs + validators (healthcare, finance)
tests/                # unit + end-to-end tests (33, all local)
examples/run_demo.py  # zero-dependency demo
```

## Adding a new domain

Create `src/adi/schemas/<domain>.py` with a `SCHEMA = DomainSchema(...)` of
`FieldSpec`s (locating regex + format hint + validator), register it in
`schemas/__init__._REGISTRY`, and it's immediately available via
`DocumentIntelligencePipeline(domain="<domain>")` and the CLI.

## Tests

```bash
pytest -q          # 33 tests, fully offline
ruff check src tests
```
