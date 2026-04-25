"""Unit tests for word-level speaker assignment and segment grouping."""

from __future__ import annotations

from meetings.align import assign_speakers, group_into_segments
from meetings.schema import DiarizationTurn, Word


def _w(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end)


def test_assign_speakers_pure_overlap() -> None:
    words = [_w("Hi", 0.0, 0.5), _w("there", 0.6, 1.0), _w("Bob", 1.5, 1.9)]
    turns = [
        DiarizationTurn(start=0.0, end=1.2, speaker="A"),
        DiarizationTurn(start=1.3, end=2.0, speaker="B"),
    ]
    out = assign_speakers(words, turns)
    assert [w.speaker for w in out] == ["A", "A", "B"]


def test_assign_speakers_no_overlap_uses_nearest_center() -> None:
    words = [_w("orphan", 5.0, 5.2)]
    turns = [
        DiarizationTurn(start=0.0, end=1.0, speaker="A"),
        DiarizationTurn(start=4.0, end=4.5, speaker="B"),
    ]
    assert assign_speakers(words, turns)[0].speaker == "B"


def test_group_segments_speaker_change_splits() -> None:
    words = [
        Word(text="Hallo", start=0.0, end=0.3, speaker="A"),
        Word(text="daar", start=0.3, end=0.6, speaker="A"),
        Word(text="Hoi", start=0.7, end=1.0, speaker="B"),
    ]
    segs = group_into_segments(words)
    assert len(segs) == 2
    assert segs[0].speaker == "A"
    assert segs[1].speaker == "B"


def test_group_segments_silence_gap_splits() -> None:
    words = [
        Word(text="een", start=0.0, end=0.3, speaker="A"),
        Word(text="twee", start=2.0, end=2.3, speaker="A"),  # gap > 0.7s
    ]
    segs = group_into_segments(words)
    assert len(segs) == 2
