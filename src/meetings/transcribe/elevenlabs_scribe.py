"""ElevenLabs Scribe v2 transcription backend.

See plan/02-track-b-custom-pipeline.md task B2a.

Response schema reference (Scribe v2, single-channel):
    https://elevenlabs.io/docs/api-reference/speech-to-text/convert

Each word object has fields:
    text, start, end, type ("word" | "spacing" | "audio_event"),
    speaker_id, logprob (in [-inf, 0]), characters.
"""

from __future__ import annotations

import math
from pathlib import Path

from elevenlabs import ElevenLabs
from pydub import AudioSegment  # type: ignore[import-untyped]

from ..align import group_into_segments
from ..config import get_settings, require
from ..schema import Transcript, Word

# Scribe v2 is fully synchronous: the API returns only after the whole file
# has been transcribed. For a 2 h Dutch meeting that easily exceeds the SDK's
# default 240 s HTTP read timeout (we've observed 8–15 min server-side
# compute). Default to 30 min and let callers / env override it for longer
# audio without code changes.
DEFAULT_TIMEOUT_SECONDS: float = 30 * 60


class ElevenLabsTranscriber:
    name: str = "elevenlabs"
    model: str = "scribe_v2"

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        num_speakers: int | None = None,
    ) -> None:
        self.timeout = timeout
        self.num_speakers = num_speakers

    def transcribe(self, audio: Path, *, language: str = "nl") -> Transcript:
        """Transcribe audio using ElevenLabs Scribe v2.

        Single-shot upload with word-level timestamps, diarization, and language hint.
        """
        settings = get_settings()
        api_key = require(settings.elevenlabs_api_key, "ELEVENLABS_API_KEY")

        client = ElevenLabs(api_key=api_key, timeout=self.timeout)

        # Transcribe with Scribe v2
        with audio.open("rb") as audio_file:
            result = client.speech_to_text.convert(
                file=audio_file,
                model_id="scribe_v2",
                language_code=language,
                diarize=True,
                num_speakers=self.num_speakers,
                timestamps_granularity="word",
            )

        # Prefer the API-reported duration; fall back to pydub if absent.
        duration = getattr(result, "audio_duration_secs", None)
        if duration is None:
            audio_segment = AudioSegment.from_wav(audio)
            duration = len(audio_segment) / 1000.0  # Convert ms to seconds

        # Extract words from response
        words: list[Word] = []
        speaker_map: dict[str, str] = {}
        next_speaker_idx = 0

        for word_data in result.words:
            # Skip non-word entries (spacing, audio_event) — only true words carry
            # transcript content with reliable timestamps.
            if getattr(word_data, "type", "word") != "word":
                continue

            # Skip entries without timestamps (shouldn't happen for type=word, but be safe).
            if word_data.start is None or word_data.end is None:
                continue

            # Normalize speaker labels to SPEAKER_<N> in first-appearance order.
            # speaker_id is the official Scribe v2 field name; may be None when
            # diarization is disabled or the model can't attribute the word.
            original_speaker = word_data.speaker_id
            if original_speaker is None:
                normalized_speaker = None
            else:
                if original_speaker not in speaker_map:
                    speaker_map[original_speaker] = f"SPEAKER_{next_speaker_idx}"
                    next_speaker_idx += 1
                normalized_speaker = speaker_map[original_speaker]

            # Convert logprob (in [-inf, 0]) to a 0..1 confidence via exp().
            logprob = getattr(word_data, "logprob", None)
            confidence = math.exp(logprob) if logprob is not None else None

            words.append(
                Word(
                    text=word_data.text,
                    start=word_data.start,
                    end=word_data.end,
                    speaker=normalized_speaker,
                    confidence=confidence,
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
