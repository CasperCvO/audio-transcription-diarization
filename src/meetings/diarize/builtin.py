"""Pass-through diarizer for ASR backends that diarize natively.

Some transcription backends (e.g. ElevenLabs Scribe v2) already emit
speaker labels alongside their word timestamps. In that case we don't need
a second diarization vendor — the transcript is already diarized.

This class exists so the pipeline can still advertise a concrete
``diarizer`` name in ``backend`` labels and metadata
(e.g. ``custom:elevenlabs+builtin+claude``). Its ``diarize`` method is a
no-op: :class:`meetings.pipelines.custom.CustomPipeline` detects the
``"builtin"`` name and skips the diarize + align stages entirely, trusting
the speakers already present on the :class:`~meetings.schema.Transcript`.
"""

from __future__ import annotations

from pathlib import Path

from ..schema import DiarizationTurn


class BuiltinDiarizer:
    """No-op diarizer: trust the transcriber's own speaker labels."""

    name: str = "builtin"

    def diarize(self, audio: Path) -> list[DiarizationTurn]:  # noqa: ARG002
        # Never called by CustomPipeline when the pipeline detects
        # ``self._diarizer.name == "builtin"``. Provided purely to satisfy
        # the ``Diarizer`` Protocol so instances can be passed through the
        # same factory / CLI wiring.
        return []


__all__ = ["BuiltinDiarizer"]
