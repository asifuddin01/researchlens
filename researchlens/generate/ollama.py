"""Local generation via Ollama.

The default, and the one implementation that must work with no configuration:
everything in this project's pitch — no API key, nothing leaving the machine —
rests on this being the path a stranger gets by default.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from researchlens.generate.prompt import build_prompt
from researchlens.generate.provider import GenerationRequest

DEFAULT_MODEL = "qwen2.5:3b-instruct"


class OllamaProvider:
    name = "local"

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = DEFAULT_MODEL,
        timeout: float = 180.0,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        # Generous, because on the public deployment this container scales to
        # zero and a cold start is a minute before a token appears. A short
        # timeout would turn a normal wake into an error.
        self.timeout = timeout

    def _body(self, req: GenerationRequest, stream: bool) -> dict:
        system, user = build_prompt(req.question, req.evidence, req.history)
        return {
            "model": self.model,
            "stream": stream,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                # Low but not zero. At 0 a 3B model repeats itself when a
                # passage is repetitive; a little sampling avoids that without
                # inviting invention.
                "temperature": req.temperature,
                "num_predict": req.max_tokens,
            },
        }

    async def generate(self, req: GenerationRequest) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{self.host}/api/chat", json=self._body(req, False))
            r.raise_for_status()
            return r.json()["message"]["content"].strip()

    async def stream(self, req: GenerationRequest) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", f"{self.host}/api/chat", json=self._body(req, True)
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    piece = chunk.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if chunk.get("done"):
                        return

    async def healthy(self) -> bool:
        """Whether this provider can serve right now.

        Load-bearing on the demo: the local model sits behind a container that
        scales to zero, so "not ready" is a normal state the page renders as a
        wake indicator rather than an error. The model must also be *present* —
        a running Ollama without the model pulled fails on the first question,
        several minutes into someone's first visit.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.host}/api/tags")
                if r.status_code != 200:
                    return False
                names = [m.get("name", "") for m in r.json().get("models", [])]
                stem = self.model.split(":")[0]
                return any(n == self.model or n.startswith(stem) for n in names)
        except Exception:
            return False
