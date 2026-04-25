"""Deepgram Nova-3 multilingual transcription backend.

Stub: interface matches `Transcriber`. Fill in with the Deepgram SDK call
as a follow-up (see plan B2).
"""

from __future__ import annotations

from pathlib import Path

from ..schema import Transcript


class DeepgramTranscriber:
    name: str = "deepgram"
    model: str = "nova-3"

    def transcribe(self, audio: Path, *, language: str = "nl") -> Transcript:
        raise NotImplementedError("Deepgram transcription not yet wired up. See plan B2.")
