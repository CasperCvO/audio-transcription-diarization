"""Diarizer Protocol shared by every backend."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ..schema import DiarizationTurn


@runtime_checkable
class Diarizer(Protocol):
    name: str

    def diarize(self, audio: Path) -> list[DiarizationTurn]: ...
