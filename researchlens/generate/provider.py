"""The generation interface.

Two implementations exist because the public demo lets a visitor pick between
them, and because "swap the model, keep the retrieval" is the claim the
provider abstraction is here to make true rather than merely assert.

The comparison is worth more than a feature. Retrieval is identical on both
sides of the toggle, so the difference a visitor sees isolates generation — a
live ablation on the same evidence. The eval harness should therefore report
faithfulness and citation accuracy *per model over identical retrieved
context*, which is a table nobody can produce without this seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol, runtime_checkable

from researchlens.types import Retrieved


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    question: str
    evidence: list[Retrieved]
    #: Prior turns, already selected down. The provider receives what it should
    #: send, never the whole conversation — deciding what to carry is a
    #: retrieval concern, not a generation one.
    history: list[tuple[str, str]] | None = None
    max_tokens: int = 800
    temperature: float = 0.1


@runtime_checkable
class Provider(Protocol):
    """Anything that can turn evidence into a grounded answer."""

    name: str
    model: str

    async def generate(self, req: GenerationRequest) -> str:
        ...

    async def stream(self, req: GenerationRequest) -> AsyncIterator[str]:
        ...

    async def healthy(self) -> bool:
        """Whether this provider can serve a request right now.

        Load-bearing for the demo: the local provider is behind a container
        that scales to zero, so "not healthy yet" is a normal state that the
        page renders as a wake indicator rather than an error.
        """
        ...
