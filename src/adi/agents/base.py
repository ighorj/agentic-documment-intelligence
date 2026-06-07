"""Base class shared by all agents.

Agents are deliberately thin and single-purpose. Each one (a) takes a typed input
contract, (b) returns a typed output contract, and (c) records human-readable
trace notes so a pipeline run is auditable end-to-end — important for regulated
healthcare/finance use where you must be able to explain every output.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


class Agent(ABC, Generic[TIn, TOut]):
    name: str = "agent"

    def __init__(self) -> None:
        self.trace: list[str] = []

    def log(self, message: str) -> None:
        self.trace.append(f"[{self.name}] {message}")

    @abstractmethod
    def run(self, payload: TIn) -> TOut:
        """Transform the input contract into the output contract."""
        raise NotImplementedError
