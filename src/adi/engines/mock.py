"""Zero-dependency OCR engine for development, demos and tests.

It produces deterministic output so the *entire* pipeline runs the moment you
install the base package, with no Tesseract/poppler/OpenCV required. It also
intentionally injects a couple of realistic OCR confusions (O<->0, l<->1) so the
Validation agent's correction logic has something to actually fix.
"""

from __future__ import annotations

from adi.contracts import BoundingBox, RawPage, Region, Token

# A small synthetic document that exercises both supported domains. Used when a
# page has no embedded text (i.e. simulating a scanned page).
_SAMPLE_SCANNED_TEXT = (
    "DISCHARGE SUMMARY\n"
    "Patient Name: Jane Doe   DOB: 1980-04-12   MRN: 00123456\n"
    "Provider NPI: 1234567893\n"
    "Primary Diagnosis: E11.9 (Type 2 diabetes mellitus)\n"
    "Secondary: I10 Essential hypertension\n"
    "Billing Account: GB82WEST12345698765432\n"
    "Total Charges: USD 1,250.00\n"
)

# Deliberate OCR-style corruptions: the validator/corrector should repair these.
_OCR_NOISE = {
    "00123456": "0O123456",  # zero -> capital O
    "1234567893": "l234567893",  # one -> lowercase L
}


class MockOCREngine:
    """Returns text either lifted from the page's embedded text or from a built-in
    synthetic sample, tokenized with plausible per-token confidences."""

    name = "mock"

    def __init__(self, sample_text: str | None = None, inject_noise: bool = True) -> None:
        self._sample = sample_text if sample_text is not None else _SAMPLE_SCANNED_TEXT
        self._inject_noise = inject_noise

    def available(self) -> bool:
        return True

    def recognize(self, page: RawPage, region: Region) -> list[Token]:
        text = page.embedded_text or self._sample
        if self._inject_noise:
            for clean, noisy in _OCR_NOISE.items():
                text = text.replace(clean, noisy)

        tokens: list[Token] = []
        # Lay tokens out left-to-right on a single synthetic baseline per region.
        x = region.bbox.x0
        y = region.bbox.y0
        for word in text.split():
            w = 8.0 * len(word)
            # Noisy/garbled-looking tokens get a lower confidence on purpose so
            # downstream low-confidence routing has signal to act on.
            conf = 0.62 if any(c in word for c in "O0l1") else 0.97
            tokens.append(
                Token(
                    text=word,
                    confidence=conf,
                    bbox=BoundingBox(page=page.page_no, x0=x, y0=y, x1=x + w, y1=y + 14),
                )
            )
            x += w + 6.0
        return tokens
