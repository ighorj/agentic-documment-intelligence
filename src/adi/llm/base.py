"""LLM client abstraction for the Validation agent.

Per the project's privacy requirement, every implementation here is local/offline.
The protocol exists so a real on-prem model (e.g. an Ollama-served Llama/Mistral)
can be dropped in for context-aware correction without changing the agent. When no
model is configured, `LocalRuleCorrector` provides a deterministic, dependency-free
fallback that handles the most common OCR confusions.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Minimal text-correction interface used by the ValidationAgent."""

    name: str

    def available(self) -> bool: ...

    def correct(self, text: str, hint: str = "") -> str:
        """Return a corrected version of `text`. `hint` describes the expected
        format (e.g. 'a 10-digit NPI number') to focus the correction."""
        ...


# Common OCR character confusions, applied only inside numeric contexts so we
# never corrupt legitimate prose. (e.g. an MRN that OCR'd 'O' instead of '0'.)
_DIGIT_CONFUSIONS = {
    "O": "0",
    "o": "0",
    "l": "1",
    "I": "1",
    "|": "1",
    "S": "5",
    "B": "8",
    "Z": "2",
    "g": "9",
}


class LocalRuleCorrector:
    """Deterministic, offline OCR corrector. No model weights, no network.

    Strategy: when the hint marks a value as numeric/code-like, map characters
    that OCR commonly confuses back to their digit forms. Otherwise leave the
    text untouched (we'd rather under-correct prose than introduce errors)."""

    name = "local-rules"

    def available(self) -> bool:
        return True

    def correct(self, text: str, hint: str = "") -> str:
        hint_l = hint.lower()
        # Only correct fields that are genuinely all-digit. We deliberately exclude
        # mixed alphanumeric codes (IBAN, ICD-10) where a letter like 'B' or 'S' is
        # meaningful and must not be rewritten to a digit.
        numeric_context = any(k in hint_l for k in ("digit", "number", "npi", "mrn", "amount"))
        if not numeric_context:
            return text

        # Transform a token only when *every* character is already a digit or a
        # known confusable, and it contains at least one digit. This repairs
        # values like '0O123456' while leaving mixed codes (IBAN: G,W,T...) and
        # prose untouched — a single foreign letter vetoes the rewrite.
        def fix_token(tok: str) -> str:
            if not any(c.isdigit() for c in tok):
                return tok
            if not all(c.isdigit() or c in _DIGIT_CONFUSIONS for c in tok):
                return tok
            return "".join(_DIGIT_CONFUSIONS.get(c, c) for c in tok)

        return re.sub(r"\S+", lambda m: fix_token(m.group(0)), text)
