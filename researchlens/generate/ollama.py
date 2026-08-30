"""Local generation via Ollama.

The default. Everything about the repository's pitch — no API key, documents
never leave the machine — rests on this being the path that works out of the
box, so it is the one implementation that must never require configuration.

Phase III. On the public demo this provider sits behind a container that cold
-boots in roughly a minute, because a model large enough to be worth comparing
against exceeds the memory ceiling under which Fly can snapshot and resume a
machine. `healthy()` returning False is how the page knows to show a wake
state; see docs/deploy.md.
"""

from __future__ import annotations

from typing import AsyncIterator

from researchlens.generate.provider import GenerationRequest

DEFAULT_MODEL = "qwen2.5:3b-instruct"


class OllamaProvider:
    name = "local"

    def __init__(self, host: str = "http://localhost:11434", model: str = DEFAULT_MODEL) -> None:
        self.host = host
        self.model = model

    async def generate(self, req: GenerationRequest) -> str:
        raise NotImplementedError("Phase III")

    async def stream(self, req: GenerationRequest) -> AsyncIterator[str]:
        raise NotImplementedError("Phase III")

    async def healthy(self) -> bool:
        raise NotImplementedError("Phase III")
