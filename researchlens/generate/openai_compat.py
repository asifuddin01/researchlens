"""Hosted generation over an OpenAI-compatible endpoint.

Optional, always. The moment this becomes required the project's central claim
stops being true, so an absent key is a supported state rather than a startup
failure — see `Settings.hosted_available`.

Named for the protocol rather than a vendor because one endpoint shape covers
most hosted providers and every local server that speaks it (vLLM, llama.cpp,
LM Studio). On the public demo the key is held by the Cloudflare Worker in
front, never by this container and never by the page.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from researchlens.generate.prompt import build_prompt
from researchlens.generate.provider import GenerationRequest


class OpenAICompatProvider:
    name = "hosted"

    def __init__(
        self, base_url: str, api_key: str | None, model: str, timeout: float = 60.0
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _body(self, req: GenerationRequest, stream: bool) -> dict:
        system, user = build_prompt(req.question, req.evidence, req.history)
        return {
            "model": self.model,
            "stream": stream,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

    async def generate(self, req: GenerationRequest) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=self._body(req, False),
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()

    async def stream(self, req: GenerationRequest) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=self._body(req, True),
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        return
                    try:
                        delta = json.loads(payload)["choices"][0].get("delta", {})
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    piece = delta.get("content")
                    if piece:
                        yield piece

    async def healthy(self) -> bool:
        # Configured is as far as this can be checked without spending a
        # request on every health poll, which on a metered endpoint is a bill
        # for saying "yes".
        return bool(self.base_url and self.api_key)
