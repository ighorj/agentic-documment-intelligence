"""OCR engine abstraction.

The RecognitionAgent talks to engines only through this Protocol, so the actual
character-recognition backend (Tesseract, PaddleOCR, a mock, ...) is a pluggable
detail. Engines must report whether their native deps are `available()` so the
pipeline can degrade gracefully instead of crashing on import.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from adi.contracts import RawPage, Region, Token


@runtime_checkable
class OCREngine(Protocol):
    """Recognize characters within a single region of a page."""

    name: str

    def available(self) -> bool:
        """True if this engine's runtime dependencies are installed/usable."""
        ...

    def recognize(self, page: RawPage, region: Region) -> list[Token]:
        """Return recognized tokens (with per-token confidence) for `region`."""
        ...
