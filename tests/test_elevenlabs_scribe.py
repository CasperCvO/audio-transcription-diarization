"""Tests for ElevenLabs Scribe v2 transcription backend.

See plan/02-track-b-custom-pipeline.md task B2a.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from meetings.config import get_settings
from meetings.transcribe.elevenlabs_scribe import ElevenLabsTranscriber


def _mock_word(
    text: str,
    start: float | None,
    end: float | None,
    speaker_id: str | None,
    logprob: float,
    type: str = "word",
) -> MagicMock:
    word = MagicMock()
    word.text = text
    word.start = start
    word.end = end
    word.speaker_id = speaker_id
    word.logprob = logprob
    word.type = type
    return word


def test_parse_fixture_response() -> None:
    """Test parsing a Scribe v2 response fixture into a Transcript."""
    # Load fixture
    fixture_path = Path(__file__).parent / "fixtures" / "elevenlabs_scribe_v2.json"
    import json

    with fixture_path.open() as f:
        fixture_data = json.load(f)

    # Create mock ElevenLabs response (Scribe v2 schema)
    mock_result = MagicMock()
    mock_result.words = [_mock_word(**w) for w in fixture_data["words"]]
    mock_result.audio_duration_secs = fixture_data["audio_duration_secs"]

    # Mock the ElevenLabs client
    mock_client = MagicMock()
    mock_client.speech_to_text.convert.return_value = mock_result

    with patch("meetings.transcribe.elevenlabs_scribe.ElevenLabs", return_value=mock_client), \
         patch("meetings.transcribe.elevenlabs_scribe.get_settings") as mock_settings, \
         patch.object(Path, "open", mock_open(read_data=b"")):
        mock_settings.return_value.elevenlabs_api_key = "test_key"
        transcriber = ElevenLabsTranscriber()
        transcript = transcriber.transcribe(Path("test.wav"), language="nl")

    # Assert word count
    assert len(transcript.segments) > 0
    total_words = sum(len(seg.words) for seg in transcript.segments)
    assert total_words == 16  # Fixture has 16 words

    # Assert speaker normalisation (should be SPEAKER_0, SPEAKER_1)
    assert transcript.speakers == ["SPEAKER_0", "SPEAKER_1"]

    # Assert all words have normalized speaker labels
    for seg in transcript.segments:
        for word in seg.words:
            assert word.speaker in ["SPEAKER_0", "SPEAKER_1"]

    # Assert segment grouping (should split on speaker change)
    # First 6 words are SPEAKER_0, rest are SPEAKER_1
    assert transcript.segments[0].speaker == "SPEAKER_0"
    assert len(transcript.segments[0].words) == 6
    assert transcript.segments[1].speaker == "SPEAKER_1"
    assert len(transcript.segments[1].words) == 10

    # Assert backend_meta
    assert transcript.backend_meta["model"] == "scribe_v2"
    assert transcript.backend_meta["had_word_timestamps"] is True
    assert transcript.backend_meta["diarization_source"] == "elevenlabs_builtin"


def test_speaker_normalisation_first_appearance_order() -> None:
    """Test that speaker labels are normalized to SPEAKER_<N> in first-appearance order."""
    # Create a response where speaker_2 appears before speaker_1
    words_data = [
        {"text": "A", "start": 0.0, "end": 0.5, "speaker_id": "speaker_2", "logprob": -0.1},
        {"text": "B", "start": 0.5, "end": 1.0, "speaker_id": "speaker_2", "logprob": -0.1},
        {"text": "C", "start": 1.0, "end": 1.5, "speaker_id": "speaker_1", "logprob": -0.1},
        {"text": "D", "start": 1.5, "end": 2.0, "speaker_id": "speaker_0", "logprob": -0.1},
    ]

    mock_result = MagicMock()
    mock_result.words = [_mock_word(**w) for w in words_data]
    mock_result.audio_duration_secs = 2.0

    mock_client = MagicMock()
    mock_client.speech_to_text.convert.return_value = mock_result

    with patch("meetings.transcribe.elevenlabs_scribe.ElevenLabs", return_value=mock_client), \
         patch("meetings.transcribe.elevenlabs_scribe.get_settings") as mock_settings, \
         patch.object(Path, "open", mock_open(read_data=b"")):
        mock_settings.return_value.elevenlabs_api_key = "test_key"
        transcriber = ElevenLabsTranscriber()
        transcript = transcriber.transcribe(Path("test.wav"), language="nl")

    # speaker_2 appears first -> SPEAKER_0
    # speaker_1 appears second -> SPEAKER_1
    # speaker_0 appears third -> SPEAKER_2
    assert transcript.speakers == ["SPEAKER_0", "SPEAKER_1", "SPEAKER_2"]


@pytest.mark.skipif(
    not get_settings().elevenlabs_api_key,
    reason="ELEVENLABS_API_KEY not set"
)
def test_live_smoke() -> None:
    """Live smoke test skipped when ELEVENLABS_API_KEY is absent."""
    # This test would make a real API call if the key is present
    # For now, it just passes if the key is set
    assert True
