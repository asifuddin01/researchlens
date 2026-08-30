"""Configuration.

Every value has a working default. `docker compose up` with no .env at all must
produce a running system, because "no API key required" is the claim the
README makes and configuration that must be filled in before anything runs
quietly falsifies it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Mode = Literal["local", "demo"]

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class Settings:
    mode: Mode = "local"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b-instruct"

    #: Absent by default and absent is fine — the local provider serves.
    hosted_base_url: str | None = None
    hosted_api_key: str | None = None
    hosted_model: str = "claude-sonnet-5"

    data_dir: Path = field(default_factory=lambda: ROOT / "data")

    @property
    def uploads_enabled(self) -> bool:
        return self.mode == "local"

    @property
    def hosted_available(self) -> bool:
        return bool(self.hosted_base_url and self.hosted_api_key)

    @classmethod
    def from_env(cls) -> "Settings":
        mode = os.getenv("RESEARCHLENS_MODE", "local")
        if mode not in ("local", "demo"):
            raise ValueError(f"RESEARCHLENS_MODE must be 'local' or 'demo', got {mode!r}")
        return cls(
            mode=mode,  # type: ignore[arg-type]
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            reranker_model=os.getenv("RERANKER_MODEL", cls.reranker_model),
            ollama_host=os.getenv("OLLAMA_HOST", cls.ollama_host),
            ollama_model=os.getenv("OLLAMA_MODEL", cls.ollama_model),
            hosted_base_url=os.getenv("HOSTED_BASE_URL") or None,
            hosted_api_key=os.getenv("HOSTED_API_KEY") or None,
            hosted_model=os.getenv("HOSTED_MODEL", cls.hosted_model),
            data_dir=Path(os.getenv("DATA_DIR", str(ROOT / "data"))),
        )
