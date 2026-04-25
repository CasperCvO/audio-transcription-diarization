"""Summarizer Protocol shared by every backend."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ..schema import Summary, Transcript


@runtime_checkable
class Summarizer(Protocol):
    name: str

    def summarize(
        self,
        transcript: Transcript,
        *,
        log_dir: Path | None = None,
        language: str = "nl",
    ) -> Summary: ...
