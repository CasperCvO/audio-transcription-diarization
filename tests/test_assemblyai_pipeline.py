"""Tests for the AssemblyAI Track A pipeline.

Live tests are skipped automatically when ``ASSEMBLYAI_API_KEY`` is not set
or when no Dutch sample audio file is present, so this test file is safe to
run on a clean checkout.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from meetings.config import get_settings
from meetings.io import new_run_id
from meetings.pipelines.assemblyai import _to_canonical_transcript
from meetings.schema import Segment, Transcript, Word
from meetings.speakers import (
    SpeakerMappingError,
    apply_mapping,
    validate_mapping,
)
from meetings.summarize._utils import (
    parse_action_items,
    parse_decisions,
    parse_json,
    parse_summary_payload,
    render_transcript_for_llm,
)

# --------------------------------------------------------------------------- #
# Pure-function unit tests (no network, always run)
# --------------------------------------------------------------------------- #


def test_parse_json_from_fenced_block() -> None:
    text = """Hier is het resultaat:
```json
{"title": "Test", "tldr": ["a", "b"]}
```
einde."""
    data = parse_json(text)
    assert data["title"] == "Test"
    assert data["tldr"] == ["a", "b"]


def test_parse_json_from_raw_object() -> None:
    text = 'noise before {"title": "X", "next_steps": []} noise after'
    data = parse_json(text)
    assert data["title"] == "X"
    assert data["next_steps"] == []


def test_parse_json_raises_on_garbage() -> None:
    with pytest.raises(ValueError):
        parse_json("absolutely no json here")


def test_parse_action_items_filters_empty_and_normalizes() -> None:
    raw = [
        {"task": "Stuur agenda", "owner": "Casper", "due": "vrijdag"},
        {"task": "", "owner": "X"},
        {"task": "Plan vervolg", "owner": "  ", "due": None},
        "not a dict",
    ]
    items = parse_action_items(raw)
    assert [a.task for a in items] == ["Stuur agenda", "Plan vervolg"]
    assert items[0].owner == "Casper"
    assert items[1].owner is None
    assert items[1].due is None


def test_parse_decisions_accepts_dict_or_string() -> None:
    raw = [{"text": "Begroting goedgekeurd"}, "Volgende meeting verzet", {}]
    decs = parse_decisions(raw)
    assert [d.text for d in decs] == ["Begroting goedgekeurd", "Volgende meeting verzet"]


def test_parse_summary_payload_fills_defaults() -> None:
    summary = parse_summary_payload(
        {},
        language="nl",
        summarizer_backend="anthropic-batch:claude-test",
        prompt_version="single-call-v1",
    )
    assert summary.title == "Vergaderingsamenvatting"
    assert summary.tldr == []
    assert summary.action_items == []
    assert summary.summarizer_backend == "anthropic-batch:claude-test"


# --------------------------------------------------------------------------- #
# Canonicalization — duration units (regression for /1000 bug)
# --------------------------------------------------------------------------- #


def _fake_aai_transcript(
    *,
    audio_duration: int | None,
    utterances: list | None = None,
) -> SimpleNamespace:
    """Minimal stand-in for `aai.Transcript` accepted by `_to_canonical_transcript`.

    Only the attributes actually read by the canonicaliser are provided.
    """
    return SimpleNamespace(
        id="fake-id",
        audio_url="https://example/fake.wav",
        audio_duration=audio_duration,
        text=None,
        utterances=utterances or [],
        speech_model_used="universal",
    )


def test_canonical_transcript_keeps_audio_duration_in_seconds() -> None:
    """`Transcript.audio_duration` is in seconds in the AssemblyAI SDK.

    Regression for a bug where it was divided by 1000 (because word/utterance
    timestamps *are* in ms), turning a 2 h meeting's `transcript.duration`
    into ~7.2 s instead of 7200 s.
    """
    fake = _fake_aai_transcript(audio_duration=7200, utterances=[])
    canonical = _to_canonical_transcript(
        fake,  # type: ignore[arg-type]
        source_audio="meeting.wav",
        language="nl",
        speech_models=["universal"],
    )
    assert canonical.duration == 7200.0


def test_canonical_transcript_falls_back_to_last_segment_when_duration_missing() -> None:
    """If `audio_duration` is None, fall back to the last segment's end time."""
    word = SimpleNamespace(
        text="hi", start=0, end=500, speaker="A", confidence=0.9
    )
    utterance = SimpleNamespace(
        speaker="A",
        start=0,
        end=2000,  # ms -> 2.0 s
        text="hi",
        words=[word],
    )
    fake = _fake_aai_transcript(audio_duration=None, utterances=[utterance])
    canonical = _to_canonical_transcript(
        fake,  # type: ignore[arg-type]
        source_audio="meeting.wav",
        language="nl",
        speech_models=["universal"],
    )
    assert canonical.duration == 2.0
    assert canonical.segments[0].end == 2.0
    assert canonical.segments[0].words[0].end == 0.5


# --------------------------------------------------------------------------- #
# Speaker relabel + transcript rendering
# --------------------------------------------------------------------------- #


def _toy_transcript() -> Transcript:
    seg_a = Segment(
        start=0.0,
        end=5.0,
        speaker="SPEAKER_A",
        text="Hallo allemaal.",
        words=[
            Word(text="Hallo", start=0.0, end=0.5, speaker="SPEAKER_A"),
            Word(text="allemaal.", start=0.6, end=1.0, speaker="SPEAKER_A"),
        ],
    )
    seg_b = Segment(
        start=5.5,
        end=10.0,
        speaker="SPEAKER_B",
        text="Goedemorgen.",
        words=[
            Word(text="Goedemorgen.", start=5.5, end=6.0, speaker="SPEAKER_B"),
        ],
    )
    return Transcript(
        language="nl",
        duration=10.0,
        speakers=["SPEAKER_A", "SPEAKER_B"],
        segments=[seg_a, seg_b],
        source_audio="dummy.wav",
        backend="assemblyai",
    )


def test_apply_mapping_rewrites_segments_and_words() -> None:
    transcript = _toy_transcript()
    renamed = apply_mapping(
        transcript, {"SPEAKER_A": "Casper", "SPEAKER_B": "Anna"}
    )
    assert renamed.speakers == ["Casper", "Anna"]
    assert renamed.segments[0].speaker == "Casper"
    assert renamed.segments[0].words[0].speaker == "Casper"
    assert renamed.segments[1].speaker == "Anna"
    assert renamed.segments[1].words[0].speaker == "Anna"
    assert renamed.backend_meta["speaker_mapping_applied"] is True


def test_apply_mapping_keeps_label_when_value_is_null() -> None:
    transcript = _toy_transcript()
    renamed = apply_mapping(
        transcript, {"SPEAKER_A": "Casper", "SPEAKER_B": None}
    )
    assert renamed.segments[1].speaker == "SPEAKER_B"


def test_validate_mapping_rejects_missing_labels() -> None:
    transcript = _toy_transcript()
    with pytest.raises(SpeakerMappingError):
        validate_mapping({"SPEAKER_A": "Casper"}, transcript)


def test_validate_mapping_rejects_unset_when_strict() -> None:
    transcript = _toy_transcript()
    with pytest.raises(SpeakerMappingError):
        validate_mapping(
            {"SPEAKER_A": "Casper", "SPEAKER_B": None},
            transcript,
            require_all_named=True,
        )


def test_validate_mapping_allows_unset_when_lax() -> None:
    transcript = _toy_transcript()
    validate_mapping(
        {"SPEAKER_A": "Casper", "SPEAKER_B": None},
        transcript,
        require_all_named=False,
    )


def test_render_transcript_for_llm_uses_current_speaker_labels() -> None:
    transcript = apply_mapping(
        _toy_transcript(),
        {"SPEAKER_A": "Casper", "SPEAKER_B": "Anna"},
    )
    rendered = render_transcript_for_llm(transcript)
    assert "Casper: Hallo allemaal." in rendered
    assert "Anna: Goedemorgen." in rendered
    assert "SPEAKER_A" not in rendered


# --------------------------------------------------------------------------- #
# Live smoke test (skipped without key + sample)
# --------------------------------------------------------------------------- #


def _find_sample_audio() -> Path | None:
    settings = get_settings()
    candidates = [
        settings.audio_dir / "sample_nl_short.wav",
        Path("audio/test_sample/sample_nl_short.wav"),
        Path("audio/raw/sample_nl_short.wav"),
        Path("audio/processed/sample_nl_short.16k.mono.wav"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


@pytest.mark.skipif(
    not os.environ.get("ASSEMBLYAI_API_KEY") and not get_settings().assemblyai_api_key,
    reason="ASSEMBLYAI_API_KEY is not set",
)
def test_assemblyai_transcribe_smoke(tmp_path: Path) -> None:
    """Stage 1 only: live AssemblyAI transcribe + diarize + snippet export.

    Requires a Dutch sample audio file. Does NOT call any LLM, so it is safe
    to run without ANTHROPIC_API_KEY / GOOGLE_API_KEY.
    """
    sample = _find_sample_audio()
    if sample is None:
        pytest.skip(
            "No Dutch sample audio found. Drop a 30–60s wav at "
            "audio/test_sample/sample_nl_short.wav to enable this test."
        )

    from meetings.pipelines.assemblyai import AssemblyAIPipeline

    pipeline = AssemblyAIPipeline()
    run_dir = tmp_path / new_run_id(sample, pipeline.name)
    run_dir.mkdir(parents=True, exist_ok=True)

    transcript = pipeline.transcribe(sample, run_dir, language="nl")

    assert len(transcript.segments) > 0
    assert len(transcript.speakers) >= 1
    assert (run_dir / "transcript.json").exists()
    assert (run_dir / "transcript.md").exists()
    assert (run_dir / "speakers.json").exists()
    assert (run_dir / "meta.json").exists()
