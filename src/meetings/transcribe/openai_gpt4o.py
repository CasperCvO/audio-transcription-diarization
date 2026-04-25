"""OpenAI transcription backend (plan/02-track-b-custom-pipeline.md B2).

Defaults to the ``whisper-1`` model because — as of the OpenAI Python SDK
2.x — only ``whisper-1`` returns *word-level* timestamps via
``timestamp_granularities=["word"]``. ``gpt-4o-transcribe`` and
``gpt-4o-mini-transcribe`` give stronger Dutch text quality but currently
only return ``text`` / segment-level timestamps, so using them requires
synthesizing word timestamps via proportional alignment. That fallback is
implemented here and activates automatically when the selected model does
not return ``words``.

The module name retains ``openai_gpt4o`` per the plan for consistency with
the directory layout; the ``model`` attribute carries the actual model id.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ..audio import probe_duration
from ..config import get_settings, require
from ..schema import Segment, Transcript, Word


class OpenAITranscriber:
    """Transcribe Dutch meetings via the OpenAI Audio API."""

    name: str = "openai"

    def __init__(self, model: str = "whisper-1", client: OpenAI | None = None) -> None:
        self.model = model
        self._client = client

    def _get_client(self) -> OpenAI:
        if self._client is not None:
            return self._client
        api_key = require(get_settings().openai_api_key, "OPENAI_API_KEY")
        self._client = OpenAI(api_key=api_key)
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        reraise=True,
    )
    def _call_api(self, audio: Path, language: str) -> dict[str, Any]:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "language": language,
            "response_format": "verbose_json",
        }
        # Only whisper-1 currently supports word-level timestamp granularities;
        # gpt-4o-transcribe ignores the parameter and returns text only.
        if self.model == "whisper-1":
            kwargs["timestamp_granularities"] = ["word", "segment"]

        with audio.open("rb") as f:
            response = client.audio.transcriptions.create(file=f, **kwargs)
        # openai SDK returns a pydantic-like model; normalize to dict.
        if hasattr(response, "model_dump"):
            return response.model_dump()  # type: ignore[no-any-return]
        return dict(response)  # pragma: no cover — defensive fallback

    def transcribe(self, audio: Path, *, language: str = "nl") -> Transcript:
        payload = self._call_api(audio, language)

        raw_words = payload.get("words") or []
        raw_segments = payload.get("segments") or []
        duration = float(payload.get("duration") or probe_duration(audio))
        detected_language = str(payload.get("language") or language)

        if raw_words:
            words = [
                Word(text=str(w["word"]), start=float(w["start"]), end=float(w["end"]))
                for w in raw_words
            ]
            segments = _segments_from_words_and_api(words, raw_segments)
        else:
            # Fallback: synthesize word timestamps via proportional alignment
            # over each segment's text. Documented in plan B2.
            segments = _segments_with_synthetic_words(raw_segments, payload.get("text", ""))

        return Transcript(
            language=detected_language,
            duration=duration,
            speakers=[],
            segments=segments,
            source_audio=audio.name,
            backend=f"openai:{self.model}",
            backend_meta={
                "model": self.model,
                "had_word_timestamps": bool(raw_words),
                "raw_segment_count": len(raw_segments),
            },
        )


def _segments_from_words_and_api(
    words: list[Word], raw_segments: list[dict[str, Any]]
) -> list[Segment]:
    """Group words into segments using API-provided segment boundaries.

    Falls back to a single segment if the API returned no segments.
    """
    if not raw_segments:
        if not words:
            return []
        return [
            Segment(
                start=words[0].start,
                end=words[-1].end,
                text=" ".join(w.text for w in words).strip(),
                words=words,
            )
        ]

    segments: list[Segment] = []
    i = 0
    for seg in raw_segments:
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        bucket: list[Word] = []
        while i < len(words) and words[i].end <= seg_end + 1e-3:
            if words[i].start >= seg_start - 1e-3:
                bucket.append(words[i])
            i += 1
        text = str(seg.get("text") or " ".join(w.text for w in bucket)).strip()
        segments.append(
            Segment(
                start=seg_start,
                end=seg_end,
                text=text,
                words=bucket,
            )
        )
    # Any trailing words (shouldn't happen) get appended to the last segment.
    if i < len(words) and segments:
        tail = words[i:]
        last = segments[-1]
        merged = Segment(
            start=last.start,
            end=max(last.end, tail[-1].end),
            text=(last.text + " " + " ".join(w.text for w in tail)).strip(),
            words=[*last.words, *tail],
            speaker=last.speaker,
        )
        segments[-1] = merged
    return segments


def _segments_with_synthetic_words(
    raw_segments: list[dict[str, Any]], full_text: str
) -> list[Segment]:
    """Produce Segments with proportionally-timed synthetic words.

    Used when the backend does not emit word timestamps (e.g.
    ``gpt-4o-transcribe``). Words inside a segment are distributed evenly
    across the segment's time span, weighted by character length. This is
    imperfect but sufficient for diarization alignment to choose the correct
    speaker for each word when turns are longer than a few seconds.
    """
    segments: list[Segment] = []
    if not raw_segments and full_text:
        # Single implicit segment spanning [0, nan]. Without duration we cannot
        # reasonably synthesize — emit a single zero-duration segment.
        return [Segment(start=0.0, end=0.0, text=full_text.strip(), words=[])]

    for seg in raw_segments:
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        text = str(seg.get("text", "")).strip()
        tokens = text.split()
        span = max(seg_end - seg_start, 1e-3)
        char_total = sum(len(t) for t in tokens) or max(len(tokens), 1)
        cursor = seg_start
        words: list[Word] = []
        for tok in tokens:
            weight = len(tok) if char_total else 1
            dur = span * (weight / char_total) if char_total else span / max(len(tokens), 1)
            w_start = cursor
            w_end = min(seg_end, cursor + dur)
            if math.isclose(w_start, w_end):
                w_end = w_start + 1e-3
            words.append(Word(text=tok, start=w_start, end=w_end))
            cursor = w_end
        segments.append(Segment(start=seg_start, end=seg_end, text=text, words=words))
    return segments
