"""Real, fully-local OCR engine backed by Tesseract via pytesseract.

This is an *optional* engine: it only works if the `ocr` extra is installed
(`pip install -e ".[ocr]"`) and the Tesseract binary is on PATH. If those are
missing, `available()` returns False and the RecognitionAgent falls back to
another engine instead of crashing.

The image for a region is loaded from `RawPage.image_ref` (a path written by the
IngestionAgent during rasterization/preprocessing) and cropped to the region's
bounding box before recognition.
"""

from __future__ import annotations

from adi.contracts import BoundingBox, RawPage, Region, Token


class TesseractEngine:
    name = "tesseract"

    def __init__(self, lang: str = "eng", psm: int = 6) -> None:
        self.lang = lang
        self.psm = psm  # 6 = assume a uniform block of text; good default per-region.
        self._deps_ok = self._check_deps()

    @staticmethod
    def _check_deps() -> bool:
        try:
            import pytesseract  # noqa: F401
            from PIL import Image  # noqa: F401

            # Confirms the native binary is reachable, not just the bindings.
            import shutil

            return shutil.which("tesseract") is not None
        except Exception:
            return False

    def available(self) -> bool:
        return self._deps_ok

    def recognize(self, page: RawPage, region: Region) -> list[Token]:
        if not self._deps_ok:
            raise RuntimeError(
                "TesseractEngine unavailable: install the 'ocr' extra and the "
                "tesseract binary, or use MockOCREngine."
            )
        if not page.image_ref:
            return []

        import pytesseract
        from PIL import Image

        img = Image.open(page.image_ref)
        box = region.bbox
        crop = img.crop((int(box.x0), int(box.y0), int(box.x1), int(box.y1)))

        # image_to_data gives us per-word text, confidence and geometry in one pass.
        config = f"--psm {self.psm}"
        data = pytesseract.image_to_data(
            crop, lang=self.lang, config=config, output_type=pytesseract.Output.DICT
        )

        tokens: list[Token] = []
        n = len(data["text"])
        for i in range(n):
            word = (data["text"][i] or "").strip()
            if not word:
                continue
            raw_conf = float(data["conf"][i])
            conf = max(0.0, min(1.0, raw_conf / 100.0)) if raw_conf >= 0 else 0.0
            # Translate region-local geometry back into page coordinates.
            x0 = box.x0 + data["left"][i]
            y0 = box.y0 + data["top"][i]
            tokens.append(
                Token(
                    text=word,
                    confidence=conf,
                    bbox=BoundingBox(
                        page=page.page_no,
                        x0=x0,
                        y0=y0,
                        x1=x0 + data["width"][i],
                        y1=y0 + data["height"][i],
                    ),
                )
            )
        return tokens
