"""Diarization backends. See `plan/02-track-b-custom-pipeline.md` task B3."""

from .base import Diarizer
from .builtin import BuiltinDiarizer
from .pyannote_local import PyannoteLocalDiarizer
from .pyannoteai_api import PyannoteAIDiarizer, PyannoteAIError

__all__ = [
    "BuiltinDiarizer",
    "Diarizer",
    "PyannoteAIDiarizer",
    "PyannoteAIError",
    "PyannoteLocalDiarizer",
    "get_diarizer",
]


def get_diarizer(name: str) -> Diarizer:
    if name in {"builtin", "scribe", "scribe_builtin", "elevenlabs_builtin", "none"}:
        return BuiltinDiarizer()
    if name in {"pyannoteai", "pyannote_ai", "pyannoteai_api"}:
        return PyannoteAIDiarizer()
    if name in {"pyannote_local", "local"}:
        return PyannoteLocalDiarizer()
    raise ValueError(f"Unknown diarizer: {name!r}")
