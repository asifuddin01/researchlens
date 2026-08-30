"""The retriever interface.

One `Protocol` rather than a base class, so an implementation need only supply
the method — there is nothing to inherit and no import cycle between a
retriever and the registry that composes it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from researchlens.types import Chunk


@runtime_checkable
class Retriever(Protocol):
    """Anything that can rank chunks against a query."""

    name: str

    def index(self, chunks: list[Chunk]) -> None:
        """Build whatever structure this retriever needs. Called once."""
        ...

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return `(chunk_id, score)` descending, at most k long."""
        ...
