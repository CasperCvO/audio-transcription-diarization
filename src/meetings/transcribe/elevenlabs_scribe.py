"""ElevenLabs Scribe v2 transcription backend.

See plan/02-track-b-custom-pipeline.md task B2a.
"""

from __future__ import annotations

from pathlib import Path

from elevenlabs import ElevenLabs
from pydub import AudioSegment  # type: ignore[import-untyped]

from ..align import group_into_segments
from ..config import get_settings, require
from ..schema import Transcript, Word


class ElevenLabsTranscriber:
    name: str = "elevenlabs"
    model: str = "scribe_v2"

    def transcribe(self, audio: Path, *, language: str = "nl") -> Transcript:
        """Transcribe audio using ElevenLabs Scribe v2.

        Single-shot upload with word-level timestamps, diarization, and language hint.
        """
        settings = get_settings()
        api_key = require(settings.elevenlabs_api_key, "ELEVENLABS_API_KEY")

        client = ElevenLabs(api_key=api_key)

        # Load audio to get duration
        audio_segment = AudioSegment.from_wav(audio)
        duration = len(audio_segment) / 1000.0  # Convert ms to seconds

        # Transcribe with Scribe v2
        result = client.speech_to_text.transcribe(
            file_path=str(audio),
            model_id="scribe_v2",
            language_code=language,
            diarization=True,
            timestamps_granularity="word",
        )

        # Extract words from response
        words: list[Word] = []
        speaker_map: dict[str, str] = {}
        next_speaker_idx = 0

        for word_data in result.words:
            # Normalize speaker labels to SPEAKER_<N> in first-appearance order
            original_speaker = word_data.speaker
            if original_speaker not in speaker_map:
                speaker_map[original_speaker] = f"SPEAKER_{next_speaker_idx}"
                next_speaker_idx += 1
            normalized_speaker = speaker_map[original_speaker]

            words.append(
                Word(
                    text=word_data.text,
                    start=word_data.start,
                    end=word_data.end,
                    speaker=normalized_speaker,
                    confidence=word_data.confidence if hasattr(word_data, "confidence") else None,
                )
            )

        # Group words into segments
        segments = group_into_segments(words, silence_gap=0.7)

        # Build transcript
        transcript = Transcript(
            language=language,
            duration=duration,
            speakers=list(speaker_map.values()),
            segments=segments,
            source_audio=str(audio),
            backend=f"custom:{self.name}+{self.model}",
            backend_meta={
                "model": self.model,
                "had_word_timestamps": True,
                "diarization_source": "elevenlabs_builtin",
            },
        )

        return transcript
