"""Pure helper functions for notebook metrics computation.

These functions are used by the comparison notebooks to compute gross
metrics from a Transcript without performing any I/O.
"""

from __future__ import annotations

from .schema import Transcript


def compute_gross_metrics(transcript: Transcript) -> dict[str, object]:
    """Compute gross metrics from a Transcript.

    Returns a dict with:
    - n_speakers: number of unique speakers
    - n_segments: total number of segments
    - n_words: total number of words
    - duration: total duration in seconds
    - words_per_speaker: dict mapping speaker label to word count
    - seconds_per_speaker: dict mapping speaker label to total speaking time
    - pct_per_speaker: dict mapping speaker label to percentage of total
      duration
    - words_per_minute_per_speaker: dict mapping speaker label to words per
      minute

    Pure function, no I/O.
    """
    # Count words per speaker
    words_per_speaker: dict[str, int] = {}
    # Count speaking time per speaker (sum of segment durations)
    seconds_per_speaker: dict[str, float] = {}

    for seg in transcript.segments:
        if seg.speaker:
            # Count words in this segment
            words_per_speaker[seg.speaker] = words_per_speaker.get(seg.speaker, 0) + len(seg.words)
            # Add segment duration
            seconds_per_speaker[seg.speaker] = (
                seconds_per_speaker.get(seg.speaker, 0.0) + (seg.end - seg.start)
            )

    n_speakers = len(transcript.speakers)
    n_segments = len(transcript.segments)
    n_words = sum(len(seg.words) for seg in transcript.segments)
    duration = transcript.duration

    # Convert to regular dicts
    words_per_speaker_dict = dict(words_per_speaker)
    seconds_per_speaker_dict = dict(seconds_per_speaker)

    # Compute percentage per speaker
    pct_per_speaker: dict[str, float] = {}
    for speaker in transcript.speakers:
        if duration > 0:
            pct_per_speaker[speaker] = (
                seconds_per_speaker_dict.get(speaker, 0.0) / duration
            ) * 100.0
        else:
            pct_per_speaker[speaker] = 0.0

    # Compute words per minute per speaker
    words_per_minute_per_speaker: dict[str, float] = {}
    for speaker in transcript.speakers:
        seconds = seconds_per_speaker_dict.get(speaker, 0.0)
        if seconds > 0:
            words_per_minute_per_speaker[speaker] = (
                words_per_speaker_dict.get(speaker, 0) / seconds
            ) * 60.0
        else:
            words_per_minute_per_speaker[speaker] = 0.0

    return {
        "n_speakers": n_speakers,
        "n_segments": n_segments,
        "n_words": n_words,
        "duration": duration,
        "words_per_speaker": words_per_speaker_dict,
        "seconds_per_speaker": seconds_per_speaker_dict,
        "pct_per_speaker": pct_per_speaker,
        "words_per_minute_per_speaker": words_per_minute_per_speaker,
    }


__all__ = ["compute_gross_metrics"]
