"""Additional alignment edge cases (plan B4 unit tests)."""

from __future__ import annotations

from meetings.align import assign_speakers, group_into_segments
from meetings.schema import DiarizationTurn, Word


def _w(text: str, start: float, end: float, speaker: str | None = None) -> Word:
    return Word(text=text, start=start, end=end, speaker=speaker)


def test_tiny_word_inside_long_turn() -> None:
    turns = [DiarizationTurn(start=0.0, end=10.0, speaker="A")]
    words = [_w("ja", 4.5, 4.6)]
    out = assign_speakers(words, turns)
    assert out[0].speaker == "A"


def test_speaker_change_mid_sentence() -> None:
    turns = [
        DiarizationTurn(start=0.0, end=1.0, speaker="A"),
        DiarizationTurn(start=1.0, end=2.0, speaker="B"),
    ]
    words = [
        _w("ik", 0.0, 0.3),
        _w("denk", 0.3, 0.6),
        _w("nee", 1.05, 1.3),
    ]
    out = assign_speakers(words, turns)
    assert [w.speaker for w in out] == ["A", "A", "B"]


def test_group_segments_sentence_punctuation_splits() -> None:
    words = [
        Word(text="Hallo.", start=0.0, end=0.3, speaker="A"),
        Word(text="Hoe", start=0.4, end=0.6, speaker="A"),
        Word(text="gaat", start=0.6, end=0.8, speaker="A"),
        Word(text="het", start=0.8, end=1.0, speaker="A"),
    ]
    segs = group_into_segments(words)
    assert len(segs) == 2
    assert segs[0].text.startswith("Hallo")


def test_group_segments_empty_input() -> None:
    assert group_into_segments([]) == []
