"""ElevenLabs Scribe v1 transcription backend.

Stub: interface matches `Transcriber`. Only the default OpenAI transcriber
is fully implemented (see plan B2). Enable as a follow-up by filling in
``transcribe`` with the ElevenLabs SDK call.
"""

from __future__ import annotations

from pathlib import Path

from ..schema import Transcript


class ElevenLabsTranscriber:
    name: str = "elevenlabs"
    model: str = "scribe_v1"

    def transcribe(self, audio: Path, *, language: str = "nl") -> Transcript:
        raise NotImplementedError(
            "ElevenLabs Scribe transcription not yet wired up. See plan B2."
        )
