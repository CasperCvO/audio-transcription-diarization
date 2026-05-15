"""Local pyannote.audio 3.x CPU fallback (plan B3).

This is intentionally a lazy-import stub: the heavy dependency
``pyannote.audio`` pulls in torch / torchaudio and is CPU-slow. It is only
imported when this diarizer is explicitly selected via
``--diarizer pyannote_local``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import get_settings, require
from ..schema import DiarizationTurn


class PyannoteLocalDiarizer:
    name: str = "pyannote_local"

    def __init__(
        self,
        model: str = "pyannote/speaker-diarization-3.1",
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> None:
        self.model = model
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self._pipeline: Any | None = None

    def _load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            from pyannote.audio import Pipeline  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover — optional dep
            raise RuntimeError(
                "pyannote.audio is not installed. Add it to dependencies if you "
                "want to use the local diarizer: `uv add pyannote.audio`."
            ) from exc
        hf_token = require(get_settings().hf_token, "HF_TOKEN")
        self._pipeline = Pipeline.from_pretrained(self.model, use_auth_token=hf_token)
        return self._pipeline

    def diarize(self, audio: Path) -> list[DiarizationTurn]:  # pragma: no cover — live only
        pipeline = self._load()
        kwargs: dict[str, int] = {}
        if self.num_speakers is not None:
            kwargs["num_speakers"] = self.num_speakers
        if self.min_speakers is not None:
            kwargs["min_speakers"] = self.min_speakers
        if self.max_speakers is not None:
            kwargs["max_speakers"] = self.max_speakers
        annotation = pipeline(str(audio), **kwargs)
        turns: list[DiarizationTurn] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            turns.append(
                DiarizationTurn(start=float(turn.start), end=float(turn.end), speaker=str(speaker))
            )
        return turns
