"""Tests for notebook_helpers.py compute_gross_metrics function.

Deterministic tests against a small hand-built Transcript.
"""

from __future__ import annotations

from meetings.notebook_helpers import compute_gross_metrics
from meetings.schema import Segment, Transcript, Word


def test_compute_gross_metrics_basic() -> None:
    """Test basic metrics computation with a simple transcript."""
    words_speaker_a = [
        Word(text="Hello", start=0.0, end=0.5, speaker="SPEAKER_00"),
        Word(text="world", start=0.5, end=1.0, speaker="SPEAKER_00"),
    ]
    words_speaker_b = [
        Word(text="Goodbye", start=1.5, end=2.0, speaker="SPEAKER_01"),
    ]

    seg_a = Segment(
        start=0.0, end=1.0, speaker="SPEAKER_00", text="Hello world", words=words_speaker_a
    )
    seg_b = Segment(
        start=1.5, end=2.0, speaker="SPEAKER_01", text="Goodbye", words=words_speaker_b
    )

    transcript = Transcript(
        language="en",
        duration=2.0,
        speakers=["SPEAKER_00", "SPEAKER_01"],
        segments=[seg_a, seg_b],
        source_audio="test.wav",
        backend="test",
    )

    metrics = compute_gross_metrics(transcript)

    assert metrics["n_speakers"] == 2
    assert metrics["n_segments"] == 2
    assert metrics["n_words"] == 3
    assert metrics["duration"] == 2.0
    assert metrics["words_per_speaker"] == {"SPEAKER_00": 2, "SPEAKER_01": 1}
    assert metrics["seconds_per_speaker"] == {"SPEAKER_00": 1.0, "SPEAKER_01": 0.5}
    # SPEAKER_00: 1.0/2.0 = 50%, SPEAKER_01: 0.5/2.0 = 25%
    assert metrics["pct_per_speaker"] == {"SPEAKER_00": 50.0, "SPEAKER_01": 25.0}
    # SPEAKER_00: 2 words / 1.0 sec * 60 = 120 wpm
    # SPEAKER_01: 1 word / 0.5 sec * 60 = 120 wpm
    assert metrics["words_per_minute_per_speaker"] == {
        "SPEAKER_00": 120.0,
        "SPEAKER_01": 120.0,
    }


def test_compute_gross_metrics_empty_transcript() -> None:
    """Test metrics with an empty transcript."""
    transcript = Transcript(
        language="en",
        duration=0.0,
        speakers=[],
        segments=[],
        source_audio="test.wav",
        backend="test",
    )

    metrics = compute_gross_metrics(transcript)

    assert metrics["n_speakers"] == 0
    assert metrics["n_segments"] == 0
    assert metrics["n_words"] == 0
    assert metrics["duration"] == 0.0
    assert metrics["words_per_speaker"] == {}
    assert metrics["seconds_per_speaker"] == {}
    assert metrics["pct_per_speaker"] == {}
    assert metrics["words_per_minute_per_speaker"] == {}


def test_compute_gross_metrics_single_speaker() -> None:
    """Test metrics with a single speaker."""
    words = [
        Word(text="One", start=0.0, end=0.5, speaker="SPEAKER_00"),
        Word(text="two", start=0.5, end=1.0, speaker="SPEAKER_00"),
        Word(text="three", start=1.0, end=1.5, speaker="SPEAKER_00"),
    ]

    seg = Segment(start=0.0, end=1.5, speaker="SPEAKER_00", text="One two three", words=words)

    transcript = Transcript(
        language="en",
        duration=1.5,
        speakers=["SPEAKER_00"],
        segments=[seg],
        source_audio="test.wav",
        backend="test",
    )

    metrics = compute_gross_metrics(transcript)

    assert metrics["n_speakers"] == 1
    assert metrics["n_segments"] == 1
    assert metrics["n_words"] == 3
    assert metrics["duration"] == 1.5
    assert metrics["words_per_speaker"] == {"SPEAKER_00": 3}
    assert metrics["seconds_per_speaker"] == {"SPEAKER_00": 1.5}
    # 1.5/1.5 = 100%
    assert metrics["pct_per_speaker"] == {"SPEAKER_00": 100.0}
    # 3 words / 1.5 sec * 60 = 120 wpm
    assert metrics["words_per_minute_per_speaker"] == {"SPEAKER_00": 120.0}


def test_compute_gross_metrics_segment_without_speaker() -> None:
    """Test that segments without speaker are handled correctly (ignored)."""
    words_with_speaker = [
        Word(text="Hello", start=0.0, end=0.5, speaker="SPEAKER_00"),
    ]
    words_without_speaker = [
        Word(text="unknown", start=1.0, end=1.5, speaker=None),
    ]

    seg_with = Segment(
        start=0.0,
        end=0.5,
        speaker="SPEAKER_00",
        text="Hello",
        words=words_with_speaker,
    )
    seg_without = Segment(
        start=1.0,
        end=1.5,
        speaker=None,
        text="unknown",
        words=words_without_speaker,
    )

    transcript = Transcript(
        language="en",
        duration=1.5,
        speakers=["SPEAKER_00"],
        segments=[seg_with, seg_without],
        source_audio="test.wav",
        backend="test",
    )

    metrics = compute_gross_metrics(transcript)

    # Only SPEAKER_00 should have metrics
    assert metrics["n_speakers"] == 1
    assert metrics["words_per_speaker"] == {"SPEAKER_00": 1}
    assert metrics["seconds_per_speaker"] == {"SPEAKER_00": 0.5}
    assert metrics["pct_per_speaker"] == {"SPEAKER_00": 0.5 / 1.5 * 100.0}
