"""Test CustomPipeline.transcribe_only with stubbed stages.

Exercises the transcribe-only entry point without hitting any external API:
a fake transcriber and diarizer feed canonical objects through the pipeline.
Verifies that the correct artefacts are written and that summary.json is NOT.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from meetings.pipelines.custom import CustomPipeline
from meetings.schema import DiarizationTurn, Segment, Transcript, Word


class _FakeTranscriber:
    name = "fake_asr"
    model = "fake-1"

    def transcribe(self, audio: Path, *, language: str = "nl") -> Transcript:  # noqa: ARG002
        words = [
            Word(text="Hallo", start=0.0, end=0.4),
            Word(text="allemaal.", start=0.4, end=1.0),
            Word(text="Hoi", start=1.2, end=1.5),
            Word(text="Casper.", start=1.5, end=1.9),
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


@pytest.fixture()
def fake_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide a stand-in audio file and stub audio prep + metadata + snippet extraction."""
    audio = tmp_path / "raw" / "demo.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"RIFF0000WAVEfmt fake")

    from meetings import audio as audio_mod
    from meetings.pipelines import custom as custom_mod

    def _fake_prepare(src: Path, dst_dir: Path | None = None, **_: object) -> Path:
        return src

    def _fake_meta(path: Path) -> audio_mod.AudioMeta:
        return audio_mod.AudioMeta(path=path, sha256="0" * 64, duration=2.0, bytes_=42)

    def _fake_snippets(
        transcript: object, audio_path: object, run_dir: Path, top_n: int = 3, **_: object
    ) -> dict[str, list[Path]]:
        # Create dummy snippet files to satisfy existence checks
        snippets_dir = run_dir / "snippets"
        snippets_dir.mkdir(parents=True, exist_ok=True)
        # Create dummy files for 2 speakers (SPEAKER_00 and SPEAKER_01)
        # with top_n files per speaker
        for speaker_idx in range(2):
            for snippet_idx in range(top_n):
                filename = f"SPEAKER_0{speaker_idx:02d}_{snippet_idx + 1:02d}.wav"
                (snippets_dir / filename).write_bytes(b"dummy")
        return {}

    monkeypatch.setattr(custom_mod, "prepare_audio", _fake_prepare)
    monkeypatch.setattr(custom_mod, "audio_meta", _fake_meta)
    monkeypatch.setattr(custom_mod, "extract_speaker_snippets", _fake_snippets)
    return audio


def test_transcribe_only_writes_correct_artefacts(
    tmp_path: Path, fake_audio: Path
) -> None:
    """transcribe_only writes transcript, speakers.json, snippets but NOT summary."""
    run_dir = tmp_path / "run"
    pipeline = CustomPipeline(
        transcriber=_FakeTranscriber(),
        diarizer=_FakeDiarizer(),
    )
    transcript = pipeline.transcribe_only(fake_audio, run_dir, language="nl")

    # transcript.json and transcript.md exist
    assert (run_dir / "transcript.json").exists()
    assert (run_dir / "transcript.md").exists()

    # speakers.json skeleton exists
    assert (run_dir / "speakers.json").exists()
    speakers_content = (run_dir / "speakers.json").read_text(encoding="utf-8")
    assert "SPEAKER_00" in speakers_content
    assert "SPEAKER_01" in speakers_content

    # snippets/ directory exists with speaker files
    snippets_dir = run_dir / "snippets"
    assert snippets_dir.exists()
    # At least one snippet file per speaker should exist
    snippet_files = list(snippets_dir.glob("*.wav"))
    assert len(snippet_files) > 0

    # meta.json exists with stage marker
    assert (run_dir / "meta.json").exists()
    meta_content = (run_dir / "meta.json").read_text(encoding="utf-8")
    assert '"stage": "transcribed"' in meta_content
    assert '"diarizer": "fake_diar"' in meta_content
    assert '"transcriber": "fake_asr"' in meta_content

    # summary.json and summary.md do NOT exist
    assert not (run_dir / "summary.json").exists()
    assert not (run_dir / "summary.md").exists()

    # Return value is the Transcript
    assert isinstance(transcript, Transcript)
    assert transcript.language == "nl"
    assert len(transcript.speakers) == 2


def test_transcribe_only_respects_snippets_per_speaker(
    tmp_path: Path, fake_audio: Path
) -> None:
    """snippets_per_speaker parameter controls how many snippets are extracted."""
    run_dir = tmp_path / "run"
    pipeline = CustomPipeline(
        transcriber=_FakeTranscriber(),
        diarizer=_FakeDiarizer(),
    )
    # Request 1 snippet per speaker instead of default 3
    pipeline.transcribe_only(fake_audio, run_dir, language="nl", snippets_per_speaker=1)

    snippets_dir = run_dir / "snippets"
    snippet_files = list(snippets_dir.glob("*.wav"))
    # With 2 speakers and 1 snippet each, we expect 2 files
    assert len(snippet_files) == 2
