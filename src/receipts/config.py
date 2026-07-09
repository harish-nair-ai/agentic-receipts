"""Configuration management for Receipts.

All configuration is via environment variables — zero config files, zero accounts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        """Backport of StrEnum for Python 3.10."""
        pass

from pydantic import BaseModel, Field


class JudgeProvider(StrEnum):
    """Supported LLM providers for the independent judge."""

    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


# Default models per provider — cheap, fast judge models
DEFAULT_MODELS: dict[JudgeProvider, str] = {
    JudgeProvider.GEMINI: "gemini-3-flash",
    JudgeProvider.OPENAI: "gpt-4.1-mini",
    JudgeProvider.ANTHROPIC: "claude-haiku-4",
}

# API base URLs
API_URLS: dict[JudgeProvider, str] = {
    JudgeProvider.GEMINI: "https://generativelanguage.googleapis.com/v1beta",
    JudgeProvider.OPENAI: "https://api.openai.com/v1",
    JudgeProvider.ANTHROPIC: "https://api.anthropic.com/v1",
}


class Config(BaseModel):
    """Receipts configuration, resolved from environment variables.

    Environment Variables:
        RECEIPTS_PROVIDER: Judge provider ("gemini", "openai", "anthropic"). Default: auto-detect.
        RECEIPTS_MODEL: Judge model name. Default: provider-specific default.
        RECEIPTS_BLOCK: Set to "1" to block session end on unverified claims. Default: "0".
        RECEIPTS_DIR: Directory to store receipt history. Default: ~/.receipts.
        RECEIPTS_TIMEOUT: Judge API timeout in seconds. Default: 30.
        GEMINI_API_KEY: Google AI API key.
        OPENAI_API_KEY: OpenAI API key.
        ANTHROPIC_API_KEY: Anthropic API key.
    """

    provider: JudgeProvider = Field(description="LLM provider for the judge")
    model: str = Field(description="Model name for the judge")
    api_key: str = Field(description="API key for the judge provider")
    block_on_unverified: bool = Field(
        default=False, description="Block session end if unverified claims exist"
    )
    receipts_dir: Path = Field(
        default_factory=lambda: Path.home() / ".receipts",
        description="Directory to store receipt history",
    )
    timeout: int = Field(default=30, description="Judge API timeout in seconds")

    @property
    def api_url(self) -> str:
        """Base API URL for the configured judge provider."""
        return API_URLS[self.provider]

    @classmethod
    def from_env(cls) -> Config:
        """Build config from environment variables with auto-detection.

        Provider resolution order:
        1. RECEIPTS_PROVIDER env var (explicit)
        2. First available API key: GEMINI_API_KEY → OPENAI_API_KEY → ANTHROPIC_API_KEY

        Raises:
            ConfigError: If no API key is found for any supported provider.
        """
        explicit_provider = os.environ.get("RECEIPTS_PROVIDER", "").strip().lower()
        explicit_model = os.environ.get("RECEIPTS_MODEL", "").strip()
        block = os.environ.get("RECEIPTS_BLOCK", "0").strip() == "1"
        receipts_dir = os.environ.get("RECEIPTS_DIR", "").strip()
        timeout = int(os.environ.get("RECEIPTS_TIMEOUT", "30"))

        # Resolve provider and API key
        provider: JudgeProvider | None = None
        api_key: str | None = None

        if explicit_provider:
            try:
                provider = JudgeProvider(explicit_provider)
            except ValueError:
                raise ConfigError(
                    f"Unknown provider '{explicit_provider}'. "
                    f"Supported: {', '.join(p.value for p in JudgeProvider)}"
                )
            api_key = _get_api_key(provider)
            if not api_key:
                raise ConfigError(
                    f"RECEIPTS_PROVIDER is set to '{provider.value}' but no API key found. "
                    f"Set {_key_env_var(provider)} environment variable."
                )
        else:
            # Auto-detect: try each provider in order
            for p in [JudgeProvider.GEMINI, JudgeProvider.OPENAI, JudgeProvider.ANTHROPIC]:
                key = _get_api_key(p)
                if key:
                    provider = p
                    api_key = key
                    break

            if provider is None or api_key is None:
                raise ConfigError(
                    "No API key found. Receipts needs an LLM to verify agent claims.\n\n"
                    "Set one of these environment variables:\n"
                    "  export GEMINI_API_KEY=...      (recommended — free tier available)\n"
                    "  export OPENAI_API_KEY=...      \n"
                    "  export ANTHROPIC_API_KEY=...   \n\n"
                    "Get a free Gemini key at: https://aistudio.google.com/apikey"
                )

        model = explicit_model or DEFAULT_MODELS[provider]

        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            block_on_unverified=block,
            receipts_dir=Path(receipts_dir) if receipts_dir else Path.home() / ".receipts",
            timeout=timeout,
        )


class ConfigError(Exception):
    """Raised when Receipts configuration is invalid or missing."""


def _get_api_key(provider: JudgeProvider) -> str | None:
    """Get the API key for a provider from environment variables."""
    env_var = _key_env_var(provider)
    key = os.environ.get(env_var, "").strip()
    return key if key else None


def _key_env_var(provider: JudgeProvider) -> str:
    """Get the environment variable name for a provider's API key."""
    return {
        JudgeProvider.GEMINI: "GEMINI_API_KEY",
        JudgeProvider.OPENAI: "OPENAI_API_KEY",
        JudgeProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    }[provider]
