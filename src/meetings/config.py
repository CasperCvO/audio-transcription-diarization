"""Runtime configuration loaded from environment / .env.

Keys are optional at import time; each backend is expected to validate the
presence of the keys it actually needs at call time.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API keys (all optional; backends validate at use-site)
    assemblyai_api_key: str | None = Field(default=None, alias="ASSEMBLYAI_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    elevenlabs_api_key: str | None = Field(default=None, alias="ELEVENLABS_API_KEY")
    deepgram_api_key: str | None = Field(default=None, alias="DEEPGRAM_API_KEY")
    pyannoteai_api_key: str | None = Field(default=None, alias="PYANNOTEAI_API_KEY")
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")

    # Paths
    audio_dir: Path = Field(default=REPO_ROOT / "audio")
    transcription_dir: Path = Field(default=REPO_ROOT / "Transcription")

    # Timeouts (seconds)
    elevenlabs_timeout: float = Field(default=30 * 60, alias="ELEVENLABS_TIMEOUT")

    # Defaults
    default_language: str = "nl"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def require(value: str | None, key: str) -> str:
    if not value:
        raise RuntimeError(
            f"Missing required configuration: {key}. "
            f"Set it in your .env file (see .env.example)."
        )
    return value
