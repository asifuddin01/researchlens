"""Hosted generation over an OpenAI-compatible endpoint.

Optional, always. The moment this becomes required, the project's central claim
stops being true, so the configuration is written so that an absent key is a
normal state rather than a startup failure.

One endpoint shape covers most hosted providers and every local server that
speaks the same protocol (vLLM, llama.cpp's server, LM Studio), which is why
this is not named after a vendor.

On the demo the key is held by the Cloudflare Worker in front, never by this
container and never by the page.
"""

from __future__ import annotations

from typing import AsyncIterator

from researchlens.generate.provider import GenerationRequest


class OpenAICompatProvider:
    name = "hosted"

    def __init__(self, base_url: str, api_key: str | None, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    async def generate(self, req: GenerationRequest) -> str:
        raise NotImplementedError("Phase III")

    async def stream(self, req: GenerationRequest) -> AsyncIterator[str]:
        raise NotImplementedError("Phase III")

    async def healthy(self) -> bool:
        raise NotImplementedError("Phase III")
