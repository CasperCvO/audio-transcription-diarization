"""Tests for audio.py — covers what we can without invoking ffmpeg."""

from __future__ import annotations

from pathlib import Path

import pytest

from meetings.audio import AudioPrepError, prepare_audio, sha256_of


def test_sha256_of_known_bytes(tmp_path: Path) -> None:
    target = tmp_path / "x.bin"
    target.write_bytes(b"abc")
    # echo -n abc | sha256sum
    assert (
        sha256_of(target)
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_prepare_audio_missing_input_raises(tmp_path: Path) -> None:
    with pytest.raises(AudioPrepError):
        prepare_audio(tmp_path / "does-not-exist.wav", tmp_path)
