"""Tests for the domain validators (the rules that catch impossible OCR output)."""

import pytest

from adi.schemas.finance import _valid_amount, _valid_iban
from adi.schemas.healthcare import _valid_icd10, _valid_npi


@pytest.mark.parametrize("code,ok", [("E11.9", True), ("I10", True), ("E119", True), ("123", False), ("", False)])
def test_icd10(code, ok):
    assert _valid_icd10(code)[0] is ok


@pytest.mark.parametrize("npi,ok", [("1234567893", True), ("1234567890", False), ("12345", False)])
def test_npi_luhn(npi, ok):
    assert _valid_npi(npi)[0] is ok


@pytest.mark.parametrize(
    "iban,ok",
    [("GB82WEST12345698765432", True), ("GB00WEST12345698765432", False), ("nonsense", False)],
)
def test_iban_mod97(iban, ok):
    assert _valid_iban(iban)[0] is ok


@pytest.mark.parametrize("amount,ok", [("1,250.00", True), ("999.99", True), ("12.3", False), ("abc", False)])
def test_amount(amount, ok):
    assert _valid_amount(amount)[0] is ok
