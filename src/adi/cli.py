"""Command-line interface: `ocr-extract`.

Examples
--------
    ocr-extract discharge.pdf --domain healthcare
    ocr-extract invoice.pdf --domain finance --json out.json --trace
    echo "..." | ocr-extract --text - --domain healthcare
    ocr-extract --list-domains
"""

from __future__ import annotations

import argparse
import json
import sys

from adi.pipeline import DocumentIntelligencePipeline
from adi.schemas import available_domains


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ocr-extract", description=__doc__)
    p.add_argument("input", nargs="?", help="path to a PDF/image to process")
    p.add_argument("--text", help="process raw text directly ('-' reads stdin)")
    p.add_argument("--domain", default="generic", help="domain schema to apply")
    p.add_argument("--json", dest="json_out", metavar="PATH", help="write full JSON result to PATH")
    p.add_argument("--trace", action="store_true", help="print the per-agent execution trace")
    p.add_argument("--list-domains", action="store_true", help="list available domains and exit")
    return p


def _render_human(result) -> str:
    lines = [
        f"document : {result.doc_id}",
        f"domain   : {result.domain}",
        f"confidence: {result.mean_confidence:.2f}",
        "fields:",
    ]
    if not result.fields:
        lines.append("  (none)")
    for f in result.fields:
        mark = "ok " if f.valid else "BAD"
        corr = f"  (was '{f.raw_value}')" if f.raw_value else ""
        lines.append(f"  [{mark}] {f.name}: {f.value}{corr}  conf={f.confidence:.2f}")
        if f.validation_error:
            lines.append(f"        ! {f.validation_error}")
    if result.warnings:
        lines.append("warnings:")
        lines.extend(f"  - {w}" for w in result.warnings)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.list_domains:
        print("\n".join(available_domains()))
        return 0

    if not args.input and not args.text:
        print("error: provide an input file or --text", file=sys.stderr)
        return 2

    try:
        pipeline = DocumentIntelligencePipeline(domain=args.domain)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.text is not None:
        text = sys.stdin.read() if args.text == "-" else args.text
        result = pipeline.process_text(text)
    else:
        result = pipeline.process_file(args.input)

    print(_render_human(result))

    if args.trace:
        print("\ntrace:")
        print("\n".join(f"  {line}" for line in pipeline.trace))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(result.model_dump(), fh, indent=2, default=str)
        print(f"\nwrote {args.json_out}")

    # Non-zero exit if any required field failed, so this composes in scripts/CI.
    return 0 if all(f.valid for f in result.fields) else 1


if __name__ == "__main__":
    raise SystemExit(main())
