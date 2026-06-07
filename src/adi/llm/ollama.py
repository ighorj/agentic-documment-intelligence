"""Optional on-prem LLM corrector backed by a locally-served Ollama model.

This keeps the system fully offline (Ollama runs on localhost) while giving the
Validation agent genuine language understanding for hard OCR repairs and free-text
normalization. It is entirely optional; if Ollama isn't running, `available()` is
False and the agent falls back to `LocalRuleCorrector`.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

_PROMPT = (
    "You are an OCR post-correction engine. Fix obvious OCR recognition errors in "
    "the VALUE below. Expected format: {hint}. Return ONLY the corrected value, "
    "with no commentary.\nVALUE: {text}"
)


class OllamaCorrector:
    def __init__(self, model: str = "llama3.1", host: str = "http://localhost:11434", timeout: float = 30.0) -> None:
        self.name = f"ollama:{model}"
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def correct(self, text: str, hint: str = "") -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": _PROMPT.format(hint=hint or "free text", text=text),
                "stream": False,
                "options": {"temperature": 0.0},
            }
        ).encode()
        try:
            req = urllib.request.Request(
                f"{self.host}/api/generate", data=payload, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode())
            return (body.get("response") or text).strip()
        except (urllib.error.URLError, TimeoutError, ValueError):
            # Never let a model hiccup break extraction; fall back to the input.
            return text
