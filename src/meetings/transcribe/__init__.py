"""Transcription backends. See `plan/02-track-b-custom-pipeline.md` task B2."""

from .base import Transcriber
from .deepgram import DeepgramTranscriber
from .elevenlabs_scribe import ElevenLabsTranscriber
from .openai_gpt4o import OpenAITranscriber

__all__ = [
    "DeepgramTranscriber",
    "ElevenLabsTranscriber",
    "OpenAITranscriber",
    "Transcriber",
    "get_transcriber",
]


def get_transcriber(name: str) -> Transcriber:
    """Factory mapping CLI names to transcriber instances."""
    if name in {"openai_gpt4o", "openai", "gpt-4o-transcribe"}:
        return OpenAITranscriber(model="gpt-4o-transcribe")
    if name in {"whisper-1", "whisper"}:
        return OpenAITranscriber(model="whisper-1")
    if name == "elevenlabs":
        return ElevenLabsTranscriber()
    if name == "deepgram":
        return DeepgramTranscriber()
    raise ValueError(f"Unknown transcriber: {name!r}")
