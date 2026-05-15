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
from ..config import get_settings
from ..diarize import Diarizer, get_diarizer
from ..diarize.pyannoteai_api import PyannoteAIDiarizer
from ..diarize.pyannote_local import PyannoteLocalDiarizer
from ..io import new_run_id, write_meta, write_run, write_transcript
from ..schema import RunMeta, RunResult, Segment, Transcript
from ..snippets import extract_speaker_snippets
from ..speakers import write_skeleton
from ..summarize import Summarizer, get_summarizer
from ..summarize.names import apply_name_mapping, resolve_speaker_names
from ..transcribe import Transcriber, get_transcriber
from ..transcribe.elevenlabs_scribe import ElevenLabsTranscriber


class CustomPipeline:
    name: str = "custom"

    def __init__(
        self,
        transcriber: str | Transcriber = "elevenlabs",
        diarizer: str | Diarizer = "builtin",
        summarizer: str | Summarizer = "claude",
        cleanup: bool = False,
        name_resolution: bool = False,
        *,
        num_speakers: int | None = None,
    ) -> None:
        # Resolve transcriber; if builtin diarizer is in use, the transcriber
        # owns diarization, so that's where the hint must go.
        if isinstance(transcriber, str):
            if (
                transcriber == "elevenlabs"
                and num_speakers is not None
                and isinstance(diarizer, str)
                and diarizer
                in {
                    "builtin",
                    "scribe",
                    "scribe_builtin",
                    "elevenlabs_builtin",
                    "none",
                }
            ):
                settings = get_settings()
                self._transcriber: Transcriber = ElevenLabsTranscriber(
                    num_speakers=num_speakers,
                    timeout=settings.elevenlabs_timeout,
                )
            else:
                self._transcriber = get_transcriber(transcriber)
        else:
            self._transcriber = transcriber

        # Resolve diarizer with num_speakers hint where applicable.
        if isinstance(diarizer, str):
            if (
                diarizer in {"pyannoteai", "pyannote_ai", "pyannoteai_api"}
                and num_speakers is not None
            ):
                self._diarizer: Diarizer = PyannoteAIDiarizer(num_speakers=num_speakers)
            elif (
                diarizer in {"pyannote_local", "local"}
                and num_speakers is not None
            ):
                self._diarizer = PyannoteLocalDiarizer(num_speakers=num_speakers)
            else:
                self._diarizer = get_diarizer(diarizer)
        else:
            self._diarizer = diarizer

        self._summarizer = (
            summarizer if not isinstance(summarizer, str) else get_summarizer(summarizer)
        )
        self.cleanup = cleanup
        self.name_resolution = name_resolution
        self.num_speakers = num_speakers
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

        transcript = self._diarize_and_align(transcript, processed, timings)

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
                "num_speakers": self.num_speakers,
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

        transcript = self._diarize_and_align(transcript, processed, timings)

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
                "num_speakers": self.num_speakers,
            },
        )
        write_meta(run_dir, run_meta)

        return transcript

    # ------------------------------------------------------------ internals

    def _diarize_and_align(
        self,
        transcript: Transcript,
        processed: Path,
        timings: dict[str, float],
    ) -> Transcript:
        """Run the diarize + align stages (or short-circuit for builtin).

        When the selected diarizer is :class:`BuiltinDiarizer` the transcriber
        has already produced speaker labels (e.g. ElevenLabs Scribe v2's
        native diarization). In that case we skip the external diarization
        call and trust the existing labels, only updating ``transcript.backend``
        so the composed backend label is still recorded.
        """
        if self._diarizer.name == "builtin":
            with _stage("diarize", timings):
                pass  # no-op; transcriber already diarized
            with _stage("align", timings):
                if not transcript.speakers:
                    raise RuntimeError(
                        "Diarizer 'builtin' was selected but the transcript "
                        f"from {self._transcriber.name!r} has no speaker labels. "
                        "Use a transcriber that diarizes natively (e.g. "
                        "'elevenlabs') or select a different --diarizer."
                    )
                transcript = transcript.model_copy(update={"backend": self.name})
            return transcript

        with _stage("diarize", timings):
            turns = self._diarizer.diarize(processed)

        with _stage("align", timings):
            words = [w for seg in transcript.segments for w in seg.words]
            words_with_speakers = assign_speakers(words, turns)
            segments = group_into_segments(words_with_speakers)
            transcript = _replace_segments(transcript, segments, backend=self.name)
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

