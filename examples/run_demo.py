"""Runnable demo — exercises the full 3-agent pipeline with zero native deps.

    python examples/run_demo.py
"""

from __future__ import annotations

from adi import DocumentIntelligencePipeline
from adi.agents import DocumentSource
from adi.cli import _render_report


def show(title: str, result, report=None) -> None:
    print(f"\n=== {title} (domain={result.domain}, conf={result.mean_confidence:.2f}) ===")
    for f in result.fields:
        flag = "✓" if f.valid else "✗"
        note = f"  [corrected from '{f.raw_value}']" if f.raw_value else ""
        print(f"  {flag} {f.name:24s} {f.value}{note}")
    for w in result.warnings:
        print(f"  ! {w}")
    if report is not None:
        print(_render_report(report))


def main() -> None:
    # 1) A digital healthcare document (text read losslessly, full confidence).
    hc = DocumentIntelligencePipeline(domain="healthcare")
    show(
        "Digital discharge summary",
        hc.process_text(
            "DISCHARGE SUMMARY\n"
            "Patient Name: Jane Doe\n"
            "DOB: 1980-04-12\n"
            "MRN: 00123456\n"
            "NPI: 1234567893\n"
            "Primary Diagnosis: E11.9\n"
        ),
    )

    # 2) A *scanned* document: the mock OCR injects O/l noise that the Validation
    #    agent's corrector repairs, plus a confidence report from Agent 4 showing
    #    where recognition was weak.
    scan_result, scan_report = hc.run_with_report(DocumentSource(doc_id="scan", path="/tmp/scan.png"))
    show("Scanned document (OCR + correction + report)", scan_result, report=scan_report)

    # 3) A finance invoice with IBAN checksum validation.
    fin = DocumentIntelligencePipeline(domain="finance")
    show(
        "Finance invoice",
        fin.process_text(
            "INVOICE\nInvoice Date: 2026-01-15\n"
            "Account: GB82WEST12345698765432\nTotal Charges: USD 1,250.00\n"
        ),
    )


if __name__ == "__main__":
    main()
