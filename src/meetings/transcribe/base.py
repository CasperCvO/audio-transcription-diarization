"""Transcriber Protocol shared by every backend.

Every transcription backend must expose a single ``transcribe`` method
returning a ``Transcript`` with **word-level timestamps** populated in
``Transcript.segments[*].words``. Downstream diarization alignment depends
on word timestamps being present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ..schema import Transcript


@runtime_checkable
class Transcriber(Protocol):
    name: str
    model: str

    def transcribe(self, audio: Path, *, language: str = "nl") -> Transcript: ...
