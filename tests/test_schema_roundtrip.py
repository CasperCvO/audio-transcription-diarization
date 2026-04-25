"""Verify canonical schema can be written and read back unchanged."""

from __future__ import annotations

from pathlib import Path

from meetings.io import read_run, write_run
from meetings.schema import (
    ActionItem,
    Decision,
    RunMeta,
    RunResult,
    Segment,
    Summary,
    Topic,
    Transcript,
    Word,
)


def _make_result(audio: str = "test.wav") -> RunResult:
    words = [
        Word(text="Hallo", start=0.0, end=0.4, speaker="SPEAKER_00"),
        Word(text="allemaal.", start=0.4, end=1.0, speaker="SPEAKER_00"),
    ]
    seg = Segment(start=0.0, end=1.0, speaker="SPEAKER_00", text="Hallo allemaal.", words=words)
    transcript = Transcript(
        language="nl",
        duration=1.0,
        speakers=["SPEAKER_00"],
        segments=[seg],
        source_audio=audio,
        backend="test",
    )
    summary = Summary(
        title="Testvergadering",
        tldr=["Korte samenvatting."],
        topics=[Topic(title="Onderwerp", bullets=["Punt 1"])],
        decisions=[Decision(text="Beslissing 1")],
        action_items=[ActionItem(task="Doe iets", owner="Casper", due="morgen")],
        open_questions=["Wie betaalt?"],
        next_steps=["Volgende meeting plannen"],
        language="nl",
        summarizer_backend="test",
        prompt_version="v0.1.0",
    )
    meta = RunMeta(
        run_id="test__test__19700101T000000Z",
        backend="test",
        input_path=audio,
        input_sha256="0" * 64,
        duration=1.0,
    )
    return RunResult(transcript=transcript, summary=summary, meta=meta)


def test_roundtrip(tmp_path: Path) -> None:
    original = _make_result()
    write_run(tmp_path, original)
    loaded = read_run(tmp_path)
    assert loaded.transcript == original.transcript
    assert loaded.summary == original.summary
    assert loaded.meta == original.meta


def test_markdown_outputs_exist(tmp_path: Path) -> None:
    write_run(tmp_path, _make_result())
    assert (tmp_path / "transcript.md").read_text("utf-8").startswith("# Transcript")
    assert "Samenvatting" in (tmp_path / "summary.md").read_text("utf-8")
