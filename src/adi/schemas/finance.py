"""Finance document schema (invoices, statements, remittances, ...)."""

from __future__ import annotations

import re

from adi.schemas.base import DomainSchema, FieldSpec

_AMOUNT = re.compile(r"^\d{1,3}(,\d{3})*(\.\d{2})?$|^\d+(\.\d{2})?$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _valid_iban(value: str) -> tuple[bool, str | None]:
    """IBAN check via the ISO 13616 mod-97 rule."""
    iban = re.sub(r"\s", "", value).upper()
    if not re.match(r"^[A-Z]{2}\d{2}[A-Z0-9]{10,30}$", iban):
        return False, "malformed IBAN"
    rearranged = iban[4:] + iban[:4]
    digits = "".join(str(int(c, 36)) for c in rearranged)  # letters -> 10..35
    return (True, None) if int(digits) % 97 == 1 else (False, "IBAN checksum failed")


def _valid_amount(value: str) -> tuple[bool, str | None]:
    return (True, None) if _AMOUNT.match(value.strip()) else (False, "malformed amount")


def _valid_date(value: str) -> tuple[bool, str | None]:
    return (True, None) if _DATE.match(value) else (False, "expected YYYY-MM-DD")


SCHEMA = DomainSchema(
    name="finance",
    fields=[
        FieldSpec(
            name="account_iban",
            patterns=[r"(?:account|iban)\s*[:\-]?\s*([A-Z]{2}\d{2}[A-Z0-9]{10,30})"],
            validator=_valid_iban,
            hint="an IBAN bank account number",
        ),
        FieldSpec(
            name="total_amount",
            patterns=[r"(?:total\s*charges|total|amount)\s*[:\-]?\s*(?:[A-Z]{3}\s*)?([\d,]+\.\d{2})"],
            validator=_valid_amount,
            hint="a monetary amount",
            required=True,
        ),
        FieldSpec(
            name="invoice_date",
            patterns=[r"(?:invoice\s*date|date)\s*[:\-]?\s*(\d{4}-\d{2}-\d{2})"],
            validator=_valid_date,
            hint="a date in YYYY-MM-DD format",
        ),
    ],
)
