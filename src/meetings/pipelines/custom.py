"""Track B: composable custom pipeline (plan B7).

Composes audio prep → transcription → diarization → word-speaker
alignment → (optional) speaker name resolution → summarization → output
writing. Each stage is timed and the timings are recorded in
``meta.json``.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..align import assign_speakers, group_into_segments
from ..audio import audio_meta, prepare_audio
from ..diarize import Diarizer, get_diarizer
from ..io import new_run_id, write_meta, write_run, write_transcript
from ..schema import RunMeta, RunResult, Segment, Transcript
from ..snippets import extract_speaker_snippets
from ..speakers import write_skeleton
from ..summarize import Summarizer, get_summarizer
from ..summarize.names import apply_name_mapping, resolve_speaker_names
from ..transcribe import Transcriber, get_transcriber


class CustomPipeline:
    name: str = "custom"

    def __init__(
        self,
        transcriber: str | Transcriber = "whisper-1",
        diarizer: str | Diarizer = "pyannoteai",
        summarizer: str | Summarizer = "claude",
        cleanup: bool = False,
        name_resolution: bool = False,
    ) -> None:
        self._transcriber = (
            transcriber if not isinstance(transcriber, str) else get_transcriber(transcriber)
        )
        self._diarizer = diarizer if not isinstance(diarizer, str) else get_diarizer(diarizer)
        self._summarizer = (
            summarizer if not isinstance(summarizer, str) else get_summarizer(summarizer)
        )
        self.cleanup = cleanup
        self.name_resolution = name_resolution
        # Compose a precise backend label for the canonical Transcript.backend.
        self.name = (
            f"custom:{self._transcriber.name}+{self._diarizer.name}+{self._summarizer.name}"
        )

    # ------------------------------------------------------------------ run

    def run(
        self,
        audio_path: Path,
        run_dir: Path,
        *,
        language: str = "nl",
    ) -> RunResult:
        run_dir.mkdir(parents=True, exist_ok=True)
        llm_log_dir = run_dir / "llm"
        timings: dict[str, float] = {}

        with _stage("prepare", timings):
            processed = prepare_audio(audio_path)
            meta = audio_meta(processed)

        with _stage("transcribe", timings):
            transcript = self._transcriber.transcribe(processed, language=language)

        with _stage("diarize", timings):
            turns = self._diarizer.diarize(processed)

        with _stage("align", timings):
            words = [w for seg in transcript.segments for w in seg.words]
            words_with_speakers = assign_speakers(words, turns)
            segments = group_into_segments(words_with_speakers)
            transcript = _replace_segments(transcript, segments, backend=self.name)

        if self.name_resolution:
            with _stage("name_resolution", timings):
                mapping = resolve_speaker_names(transcript, log_dir=llm_log_dir)
                transcript = apply_name_mapping(transcript, mapping)

        with _stage("summarize", timings):
            summary = self._summarizer.summarize(
                transcript, log_dir=llm_log_dir, language=language
            )

        run_meta = RunMeta(
            run_id=new_run_id(audio_path, self.name),
            backend=self.name,
            input_path=str(audio_path),
            input_sha256=meta.sha256,
            duration=meta.duration,
            timings=timings,
            model_versions={
                "transcriber": getattr(self._transcriber, "model", self._transcriber.name),
                "diarizer": getattr(self._diarizer, "model", self._diarizer.name),
                "summarizer": getattr(self._summarizer, "model", self._summarizer.name),
            },
            prompt_version=summary.prompt_version,
            extra={
                "audio_processed": str(processed),
                "audio_bytes": meta.bytes_,
                "name_resolution": self.name_resolution,
            },
        )

        result = RunResult(transcript=transcript, summary=summary, meta=run_meta)
        write_run(run_dir, result)
        return result

    # ---------------------------------------------------------- transcribe_only

    def transcribe_only(
        self,
        audio: Path,
        run_dir: Path,
        *,
        language: str = "nl",
        snippets_per_speaker: int = 3,
    ) -> Transcript:
        """Run prepare → transcribe → diarize → align, write artefacts, return Transcript.

        Writes:
        - ``transcript.json`` + ``transcript.md`` via :func:`io.write_transcript`
        - ``speakers.json`` skeleton via :func:`speakers.write_skeleton`
        - ``snippets/SPEAKER_*.wav`` via :func:`snippets.extract_speaker_snippets`
        - ``meta.json`` via :func:`io.write_meta`` with ``extra={"stage": "transcribed",
          "diarizer": <name>, "transcriber": <name>}``

        Does NOT summarize. Returns the Transcript for human-in-the-loop relabelling.
        """
        run_dir.mkdir(parents=True, exist_ok=True)
        timings: dict[str, float] = {}

        with _stage("prepare", timings):
            processed = prepare_audio(audio)
            meta = audio_meta(processed)

        with _stage("transcribe", timings):
            transcript = self._transcriber.transcribe(processed, language=language)

        with _stage("diarize", timings):
            turns = self._diarizer.diarize(processed)

        with _stage("align", timings):
            words = [w for seg in transcript.segments for w in seg.words]
            words_with_speakers = assign_speakers(words, turns)
            segments = group_into_segments(words_with_speakers)
            transcript = _replace_segments(transcript, segments, backend=self.name)

        # Write transcript outputs
        write_transcript(run_dir, transcript)

        # Write speakers.json skeleton
        write_skeleton(run_dir, transcript)

        # Extract speaker snippets
        extract_speaker_snippets(
            transcript, processed, run_dir, top_n=snippets_per_speaker
        )

        # Write meta.json with stage marker
        run_meta = RunMeta(
            run_id=new_run_id(audio, self.name),
            backend=self.name,
            input_path=str(audio),
            input_sha256=meta.sha256,
            duration=meta.duration,
            timings=timings,
            model_versions={
                "transcriber": getattr(self._transcriber, "model", self._transcriber.name),
                "diarizer": getattr(self._diarizer, "model", self._diarizer.name),
            },
            extra={
                "stage": "transcribed",
                "diarizer": self._diarizer.name,
                "transcriber": self._transcriber.name,
                "audio_processed": str(processed),
                "audio_bytes": meta.bytes_,
            },
        )
        write_meta(run_dir, run_meta)

        return transcript


# --------------------------------------------------------------- helpers ---


class _stage:
    """Context manager that records elapsed wall time into ``timings``."""

    def __init__(self, name: str, timings: dict[str, float]) -> None:
        self._name = name
        self._timings = timings
        self._start = 0.0

    def __enter__(self) -> _stage:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self._timings[self._name] = round(time.perf_counter() - self._start, 3)


def _replace_segments(
    transcript: Transcript, segments: list[Segment], *, backend: str
) -> Transcript:
    """Return a new Transcript with re-grouped segments and updated metadata."""
    speakers: list[str] = []
    seen: set[str] = set()
    for seg in segments:
        for w in seg.words:
            if w.speaker and w.speaker not in seen:
                seen.add(w.speaker)
                speakers.append(w.speaker)
    return transcript.model_copy(
        update={
            "segments": segments,
            "speakers": speakers,
            "backend": backend,
        }
    )


__all__ = ["CustomPipeline"]

