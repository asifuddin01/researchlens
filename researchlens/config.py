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

# Defaults live at module level, not only as field defaults.
#
# `Settings` is a slots dataclass, and on one of those a class attribute is a
# member descriptor rather than the default value — `Settings.ollama_model`
# returns `<member 'ollama_model'>`, not the string. `from_env` used those as
# its os.getenv fallbacks, so five settings silently became descriptors
# whenever their environment variable was unset, which is the normal case.
# Nothing caught it because every retriever was constructed with its own
# module default until the engine started reading Settings.
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b-instruct"
DEFAULT_HOSTED_MODEL = "claude-sonnet-5"
# Which providers an instance offers. Both, normally. A deployment that has no
# local model — a Hugging Face Space, where there is no persistent volume to
# hold one — sets this to "hosted" so the page does not present a choice that
# can never answer.
DEFAULT_PROVIDERS = ("local", "hosted")

DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:3000",
    "http://localhost:4321",
    "http://localhost:8000",
    "https://asifuddin.com",
    "https://www.asifuddin.com",
)


@dataclass(frozen=True, slots=True)
class Settings:
    mode: Mode = "local"

    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    reranker_model: str = DEFAULT_RERANKER_MODEL

    ollama_host: str = DEFAULT_OLLAMA_HOST
    ollama_model: str = DEFAULT_OLLAMA_MODEL

    #: Absent by default and absent is fine — the local provider serves.
    hosted_base_url: str | None = None
    hosted_api_key: str | None = None
    hosted_model: str = DEFAULT_HOSTED_MODEL

    data_dir: Path = field(default_factory=lambda: ROOT / "data")

    #: Origins permitted to call the API. Listed rather than wildcarded: with
    #: "*" any third-party page could spend a metered instance's budget from a
    #: visitor's browser. Localhost is included so the local build works with
    #: no configuration at all.
    allowed_origins: tuple[str, ...] = DEFAULT_ALLOWED_ORIGINS
    providers: tuple[str, ...] = DEFAULT_PROVIDERS

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
            embedding_model=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            reranker_model=os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL),
            ollama_host=os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
            ollama_model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            hosted_base_url=os.getenv("HOSTED_BASE_URL") or None,
            hosted_api_key=os.getenv("HOSTED_API_KEY") or None,
            hosted_model=os.getenv("HOSTED_MODEL", DEFAULT_HOSTED_MODEL),
            data_dir=Path(os.getenv("DATA_DIR", str(ROOT / "data"))),
            allowed_origins=(
                tuple(o.strip() for o in origins.split(",") if o.strip())
                if (origins := os.getenv("ALLOWED_ORIGINS", ""))
                else DEFAULT_ALLOWED_ORIGINS
            ),
            providers=(
                tuple(p.strip() for p in provs.split(",") if p.strip())
                if (provs := os.getenv("PROVIDERS", ""))
                else DEFAULT_PROVIDERS
            ),
        )
