"""Regression test for backend-agnostic relabel on Track-B runs.

Confirms that AssemblyAIPipeline.relabel() works on a Track-B run directory
produced by CustomPipeline.transcribe_only(). The relabel method only reads
transcript.json + speakers.json, so it is backend-agnostic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from meetings.io import read_transcript
from meetings.pipelines.assemblyai import AssemblyAIPipeline
from meetings.pipelines.custom import CustomPipeline
from meetings.schema import DiarizationTurn, Segment, Transcript, Word


class _FakeTranscriber:
    name = "fake_asr"
    model = "fake-1"

    def transcribe(self, audio: Path, *, language: str = "nl") -> Transcript:  # noqa: ARG002
        words = [
            Word(text="Hallo", start=0.0, end=0.4, speaker=None),
            Word(text="allemaal.", start=0.4, end=1.0, speaker=None),
            Word(text="Hoi", start=1.2, end=1.5, speaker=None),
            Word(text="Casper.", start=1.5, end=1.9, speaker=None),
        ]
        seg = Segment(start=0.0, end=1.9, text="Hallo allemaal. Hoi Casper.", words=words)
        return Transcript(
            language="nl",
            duration=2.0,
            segments=[seg],
            source_audio=audio.name,
            backend="fake_asr:fake-1",
        )


class _FakeDiarizer:
    name = "fake_diar"
    model = "fake-precision"

    def diarize(self, audio: Path) -> list[DiarizationTurn]:  # noqa: ARG002
        return [
            DiarizationTurn(start=0.0, end=1.05, speaker="SPEAKER_00"),
            DiarizationTurn(start=1.05, end=2.0, speaker="SPEAKER_01"),
        ]


class _FakeSummarizer:
    name = "fake_sum"
    model = "fake-claude"


@pytest.fixture()
def fake_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide a stand-in audio file and stub audio prep + metadata."""
    audio = tmp_path / "raw" / "demo.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"RIFF0000WAVEfmt fake")

    from meetings import audio as audio_mod
    from meetings.pipelines import custom as custom_mod

    def _fake_prepare(src: Path, dst_dir: Path | None = None, **_: object) -> Path:
        return src

    def _fake_meta(path: Path) -> audio_mod.AudioMeta:
        return audio_mod.AudioMeta(path=path, sha256="0" * 64, duration=2.0, bytes_=42)

    def _fake_snippets(*args: object, **kwargs: object) -> dict[str, list[Path]]:
        # Skip actual snippet extraction - we're testing relabel, not snippets
        return {}

    monkeypatch.setattr(custom_mod, "prepare_audio", _fake_prepare)
    monkeypatch.setattr(custom_mod, "audio_meta", _fake_meta)
    monkeypatch.setattr(custom_mod, "extract_speaker_snippets", _fake_snippets)
    return audio


def test_relabel_backend_agnostic(tmp_path: Path, fake_audio: Path) -> None:
    """Confirm AssemblyAIPipeline.relabel() works on a Track-B run dir."""
    run_dir = tmp_path / "run"
    
    # Build a fake Track-B run dir using CustomPipeline.transcribe_only
    pipeline = CustomPipeline(
        transcriber=_FakeTranscriber(),
        diarizer=_FakeDiarizer(),
        summarizer=_FakeSummarizer(),  # Required by CustomPipeline even though
        # transcribe_only doesn't use it
    )
    transcript = pipeline.transcribe_only(fake_audio, run_dir, language="nl")

    # Verify transcribe_only wrote the expected files
    assert (run_dir / "transcript.json").exists()
    assert (run_dir / "speakers.json").exists()
    assert (run_dir / "meta.json").exists()

    # Verify initial speaker labels
    assert transcript.speakers == ["SPEAKER_00", "SPEAKER_01"]
    segment_speakers = {seg.speaker for seg in transcript.segments}
    assert segment_speakers == {"SPEAKER_00", "SPEAKER_01"}
    word_speakers = {w.speaker for seg in transcript.segments for w in seg.words if w.speaker}
    assert word_speakers == {"SPEAKER_00", "SPEAKER_01"}

    # Write a real-name mapping into speakers.json
    (run_dir / "speakers.json").write_text(
        '{"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}\n',
        encoding="utf-8",
    )

    # Invoke AssemblyAIPipeline().relabel(run_dir) — this should work even though
    # the run was produced by CustomPipeline, proving backend-agnosticity
    aai_pipeline = AssemblyAIPipeline()
    relabeled = aai_pipeline.relabel(run_dir)

    # Assert the rewritten transcript.json has the new speaker labels in both segments AND words
    assert relabeled.speakers == ["Alice", "Bob"]
    
    # Check segment-level labels
    segment_speakers = {seg.speaker for seg in relabeled.segments}
    assert segment_speakers == {"Alice", "Bob"}
    
    # Check word-level labels
    word_speakers = {w.speaker for seg in relabeled.segments for w in seg.words if w.speaker}
    assert word_speakers == {"Alice", "Bob"}
    
    # Verify the file was actually updated on disk
    loaded = read_transcript(run_dir)
    assert loaded.speakers == ["Alice", "Bob"]
    assert {seg.speaker for seg in loaded.segments} == {"Alice", "Bob"}
