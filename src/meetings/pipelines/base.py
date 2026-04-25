"""Pipeline protocol shared by every backend."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ..schema import RunResult


@runtime_checkable
class MeetingPipeline(Protocol):
    name: str

    def run(
        self,
        audio_path: Path,
        run_dir: Path,
        *,
        language: str = "nl",
    ) -> RunResult: ...
