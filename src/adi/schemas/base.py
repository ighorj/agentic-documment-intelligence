"""Domain schema abstraction used by the Validation agent.

A `DomainSchema` is just a named collection of `FieldSpec`s. Each FieldSpec knows
(a) how to *locate* candidate values in OCR text (regex patterns), (b) what
correction `hint` to pass the LLM, and (c) how to *validate* a value. This is the
extension point: adding support for a new vertical (insurance, legal, ...) means
adding a schema module, not editing the agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# A validator returns (is_valid, error_message_or_None).
Validator = Callable[[str], "tuple[bool, str | None]"]


def _always_valid(_: str) -> tuple[bool, str | None]:
    return True, None


@dataclass(frozen=True)
class FieldSpec:
    name: str
    # Regexes with a single capture group for the value. The label part is matched
    # case-insensitively so "MRN:", "mrn :", etc. all work.
    patterns: list[str]
    validator: Validator = _always_valid
    hint: str = ""
    required: bool = False
    description: str = ""


@dataclass(frozen=True)
class DomainSchema:
    name: str
    fields: list[FieldSpec] = field(default_factory=list)

    def required_names(self) -> list[str]:
        return [f.name for f in self.fields if f.required]
