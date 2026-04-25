"""Integration test for CustomPipeline with stubbed stages.

Exercises the wiring end-to-end without hitting any external API: a fake
transcriber, diarizer and summarizer feed canonical objects through the
pipeline. Verifies that all five output files are produced and that the
schema round-trips.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from meetings.io import read_run
from meetings.pipelines.custom import CustomPipeline
from meetings.schema import (
    ActionItem,
    Decision,
    DiarizationTurn,
    Segment,
    Summary,
    Topic,
    Transcript,
    Word,
)


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


class _FakeSummarizer:
    name = "fake_sum"
    model = "fake-claude"

    def summarize(
        self,
        transcript: Transcript,
        *,
        log_dir: Path | None = None,
        language: str = "nl",
    ) -> Summary:
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "fake.log").write_text("ok", encoding="utf-8")
        return Summary(
            title="Korte testvergadering",
            tldr=["Begroeting tussen twee sprekers."],
            topics=[Topic(title="Begroeting", bullets=["Hallo en hoi"])],
            decisions=[Decision(text="Geen beslissingen")],
            action_items=[ActionItem(task="Niets doen", owner="Casper")],
            open_questions=[],
            next_steps=[],
            language=language,
            summarizer_backend="fake",
            prompt_version="v0.0-test",
        )


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

    monkeypatch.setattr(custom_mod, "prepare_audio", _fake_prepare)
    monkeypatch.setattr(custom_mod, "audio_meta", _fake_meta)
    return audio


def test_custom_pipeline_end_to_end(tmp_path: Path, fake_audio: Path) -> None:
    run_dir = tmp_path / "run"
    pipeline = CustomPipeline(
        transcriber=_FakeTranscriber(),
        diarizer=_FakeDiarizer(),
        summarizer=_FakeSummarizer(),
    )
    result = pipeline.run(fake_audio, run_dir, language="nl")

    # Five canonical output files exist.
    for name in ("transcript.json", "transcript.md", "summary.json", "summary.md", "meta.json"):
        assert (run_dir / name).exists(), f"missing {name}"

    # Speakers were assigned via diarization turns.
    speaker_set = {seg.speaker for seg in result.transcript.segments}
    assert speaker_set == {"SPEAKER_00", "SPEAKER_01"}
    assert result.transcript.speakers == ["SPEAKER_00", "SPEAKER_01"]

    # Schema round-trips.
    loaded = read_run(run_dir)
    assert loaded.summary.tldr == result.summary.tldr
    assert loaded.transcript.segments == result.transcript.segments

    # Backend label combines all three stages.
    assert loaded.transcript.backend.startswith("custom:fake_asr+fake_diar+fake_sum")

    # meta.json captured timings for every stage we ran.
    assert {"prepare", "transcribe", "diarize", "align", "summarize"} <= set(
        loaded.meta.timings.keys()
    )
