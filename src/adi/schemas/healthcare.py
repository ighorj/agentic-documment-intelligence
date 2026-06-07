"""Healthcare document schema (discharge summaries, claims, lab reports, ...).

Validators encode real domain rules so the system can flag OCR output that is
*structurally impossible* rather than silently passing garbage downstream — which
matters a great deal for clinical/billing data.
"""

from __future__ import annotations

import re

from adi.schemas.base import DomainSchema, FieldSpec

# ICD-10-CM: a letter, two alphanumerics, then optionally up to 4 more chars.
# The decimal point is optional because real systems store both "E11.9" and "E119".
_ICD10 = re.compile(r"^[A-TV-Z][0-9][0-9A-Z](\.?[0-9A-Z]{1,4})?$")
# ISO-ish date used by DOB / encounter dates.
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _valid_icd10(value: str) -> tuple[bool, str | None]:
    return (True, None) if _ICD10.match(value.upper()) else (False, "not a valid ICD-10 code")


def _valid_npi(value: str) -> tuple[bool, str | None]:
    """NPI = 10 digits validated by the Luhn algorithm over an 80840 prefix."""
    digits = re.sub(r"\D", "", value)
    if len(digits) != 10:
        return False, "NPI must be 10 digits"
    payload = "80840" + digits[:9]
    total = 0
    for i, ch in enumerate(reversed(payload)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    check = (10 - (total % 10)) % 10
    return (True, None) if check == int(digits[9]) else (False, "NPI checksum failed")


def _valid_date(value: str) -> tuple[bool, str | None]:
    return (True, None) if _DATE.match(value) else (False, "expected YYYY-MM-DD")


def _valid_mrn(value: str) -> tuple[bool, str | None]:
    digits = re.sub(r"\D", "", value)
    return (True, None) if 6 <= len(digits) <= 12 else (False, "MRN length out of range")


SCHEMA = DomainSchema(
    name="healthcare",
    fields=[
        FieldSpec(
            name="patient_name",
            # Name words must be Titlecase (upper then lower); this naturally stops
            # the match at all-caps labels such as DOB/MRN. The value is wrapped in
            # (?-i:) so it stays case-sensitive even though matching is case-
            # insensitive for the "patient name" label.
            patterns=[r"patient\s*name\s*[:\-]?\s*((?-i:[A-Z][a-z][A-Za-z'\-]*(?:[ \t]+[A-Z][a-z][A-Za-z'\-]*)+))"],
            hint="a person's full name",
            required=True,
            description="Patient full name",
        ),
        FieldSpec(
            name="dob",
            patterns=[r"dob\s*[:\-]?\s*(\d{4}-\d{2}-\d{2})"],
            validator=_valid_date,
            hint="a date in YYYY-MM-DD format",
            required=True,
        ),
        FieldSpec(
            name="mrn",
            patterns=[r"mrn\s*[:\-]?\s*([A-Za-z0-9]{6,12})"],
            validator=_valid_mrn,
            hint="a 6-12 digit medical record number",
            required=True,
        ),
        FieldSpec(
            name="npi",
            patterns=[r"npi\s*[:\-]?\s*([A-Za-z0-9]{10})"],
            validator=_valid_npi,
            hint="a 10-digit provider NPI number",
        ),
        FieldSpec(
            name="primary_diagnosis_code",
            patterns=[r"(?:primary\s*diagnosis|diagnosis)\s*[:\-]?\s*([A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?)"],
            validator=_valid_icd10,
            hint="an ICD-10 diagnosis code",
        ),
    ],
)
