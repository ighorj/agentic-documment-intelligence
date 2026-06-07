"""Pluggable OCR engines. All run fully locally."""

from adi.engines.base import OCREngine
from adi.engines.mock import MockOCREngine
from adi.engines.tesseract import TesseractEngine

__all__ = ["OCREngine", "MockOCREngine", "TesseractEngine"]
