"""Tests for CustomPipeline with the builtin (pass-through) diarizer.

When ``--diarizer builtin`` is selected the pipeline must trust the
transcriber's own speaker labels (e.g. ElevenLabs Scribe v2's native
diarization) and skip the external diarize + align stages entirely.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from meetings.diarize import BuiltinDiarizer, get_diarizer
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


class _DiarizedTranscriber:
    """Fake transcriber that emits a pre-diarized Transcript (like Scribe v2)."""

    name = "fake_scribe"
    model = "scribe-fake"

    def transcribe(self, audio: Path, *, language: str = "nl") -> Transcript:  # noqa: ARG002
        words = [
            Word(text="Hallo", start=0.0, end=0.4, speaker="SPEAKER_00"),
            Word(text="allemaal.", start=0.4, end=1.0, speaker="SPEAKER_00"),
            Word(text="Hoi", start=1.2, end=1.5, speaker="SPEAKER_01"),
            Word(text="Casper.", start=1.5, end=1.9, speaker="SPEAKER_01"),
        ]
        segments = [
            Segment(
                start=0.0,
                end=1.0,
                speaker="SPEAKER_00",
                text="Hallo allemaal.",
                words=words[:2],
            ),
            Segment(
                start=1.2,
                end=1.9,
                speaker="SPEAKER_01",
                text="Hoi Casper.",
                words=words[2:],
            ),
        ]
        return Transcript(
            language=language,
            duration=2.0,
            speakers=["SPEAKER_00", "SPEAKER_01"],
            segments=segments,
            source_audio=audio.name,
            backend="fake_scribe:scribe-fake",
            backend_meta={"diarization_source": "elevenlabs_builtin"},
        )


class _PlainTranscriber:
    """Fake transcriber that emits a transcript without speaker labels."""

    name = "fake_plain"
    model = "plain-1"

    def transcribe(self, audio: Path, *, language: str = "nl") -> Transcript:  # noqa: ARG002
        seg = Segment(
            start=0.0,
            end=1.0,
            text="Hallo.",
            words=[Word(text="Hallo.", start=0.0, end=1.0)],
        )
        return Transcript(
            language=language,
            duration=1.0,
            speakers=[],
            segments=[seg],
            source_audio=audio.name,
            backend="fake_plain:plain-1",
        )


class _ExplodingDiarizer:
    """Diarizer whose ``diarize()`` must never be called."""

    name = "builtin"

    def diarize(self, audio: Path) -> list[DiarizationTurn]:  # noqa: ARG002
        raise AssertionError("BuiltinDiarizer.diarize must not be called")


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
        return Summary(
            title="Test",
            tldr=["ok"],
            topics=[Topic(title="t", bullets=["b"])],
            decisions=[Decision(text="none")],
            action_items=[ActionItem(task="nothing")],
            open_questions=[],
            next_steps=[],
            language=language,
            summarizer_backend="fake",
            prompt_version="v0.0-test",
        )


@pytest.fixture()
def fake_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
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


def test_get_diarizer_aliases_return_builtin() -> None:
    for name in ("builtin", "scribe", "scribe_builtin", "elevenlabs_builtin", "none"):
        diar = get_diarizer(name)
        assert isinstance(diar, BuiltinDiarizer)
        assert diar.name == "builtin"


def test_builtin_diarizer_preserves_transcriber_labels(
    tmp_path: Path, fake_audio: Path
) -> None:
    run_dir = tmp_path / "run"
    pipeline = CustomPipeline(
        transcriber=_DiarizedTranscriber(),
        diarizer=_ExplodingDiarizer(),
        summarizer=_FakeSummarizer(),
    )
    result = pipeline.run(fake_audio, run_dir, language="nl")

    # Transcriber's speakers survived unchanged (no re-grouping from align).
    assert result.transcript.speakers == ["SPEAKER_00", "SPEAKER_01"]
    assert len(result.transcript.segments) == 2

    # Backend label reflects the builtin diarizer.
    assert result.transcript.backend == "custom:fake_scribe+builtin+fake_sum"

    # Both diarize and align stages were still timed (as no-op passes).
    loaded = read_run(run_dir)
    assert {"prepare", "transcribe", "diarize", "align", "summarize"} <= set(
        loaded.meta.timings.keys()
    )


def test_builtin_diarizer_transcribe_only(
    tmp_path: Path,
    fake_audio: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    pipeline = CustomPipeline(
        transcriber=_DiarizedTranscriber(),
        diarizer=_ExplodingDiarizer(),
        summarizer=_FakeSummarizer(),
    )

    from meetings.pipelines import custom as custom_mod

    # Stub snippets extraction to keep the test free of heavy audio deps.
    def _fake_snippets(
        transcript: object,
        audio_path: object,
        run_dir: Path,
        top_n: int = 3,
        **_: object,
    ) -> dict[str, list[Path]]:
        (run_dir / "snippets").mkdir(parents=True, exist_ok=True)
        return {}

    monkeypatch.setattr(custom_mod, "extract_speaker_snippets", _fake_snippets)
    transcript = pipeline.transcribe_only(fake_audio, run_dir, language="nl")

    # speakers preserved, no summary produced
    assert transcript.speakers == ["SPEAKER_00", "SPEAKER_01"]
    assert transcript.backend == "custom:fake_scribe+builtin+fake_sum"
    assert (run_dir / "transcript.json").exists()
    assert (run_dir / "speakers.json").exists()
    assert not (run_dir / "summary.json").exists()


def test_builtin_diarizer_rejects_transcript_without_speakers(
    tmp_path: Path, fake_audio: Path
) -> None:
    run_dir = tmp_path / "run"
    pipeline = CustomPipeline(
        transcriber=_PlainTranscriber(),
        diarizer=_ExplodingDiarizer(),
        summarizer=_FakeSummarizer(),
    )
    with pytest.raises(RuntimeError, match="builtin.*no speaker labels"):
        pipeline.run(fake_audio, run_dir, language="nl")
