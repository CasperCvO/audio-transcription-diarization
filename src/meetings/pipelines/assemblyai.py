"""Track A: AssemblyAI transcription + diarization, summarization via direct
provider APIs (Anthropic Claude or Google Gemini), batch by default.

This pipeline is split into three stages so the user can review and
relabel speakers before paying for a summary:

1. :meth:`AssemblyAIPipeline.transcribe` — uploads audio to AssemblyAI,
   gets back a diarized transcript using the Universal-2 speech model.
   Writes ``transcript.json``/``transcript.md`` plus a ``speakers.json``
   skeleton and per-speaker audio snippets under ``snippets/``.

2. :meth:`AssemblyAIPipeline.relabel` — reads the user-edited
   ``speakers.json`` and rewrites the transcript with real names.

3. :meth:`AssemblyAIPipeline.summarize` — feeds the (renamed) transcript
   to Claude or Gemini through their batch APIs (50% cost) and writes
   ``summary.json``/``summary.md``. AssemblyAI's LeMUR is **no longer
   used**; the summarization is now an Anthropic / Google API call so it
   can run on tiers without LeMUR / LLM Gateway access.

The convenience :meth:`AssemblyAIPipeline.run` chains all three stages
without any human-in-the-loop pause, for backward compatibility with the
``meetings run`` CLI command.

Notes on AssemblyAI feature usage:
- Speech model is fixed to **Universal-2** (``aai.SpeechModel.universal``)
  because the Dutch primary language is not supported by Universal-3 Pro.
- Native ``auto_chapters`` and ``summarization`` are English-only and
  therefore disabled — we no longer rely on them.
"""

from __future__ import annotations

import time
from pathlib import Path

import assemblyai as aai  # type: ignore[import-untyped]

from ..config import get_settings, require
from ..io import (
    read_meta,
    read_transcript,
    sha256_of,
    write_meta,
    write_summary,
    write_transcript,
)
from ..schema import RunMeta, RunResult, Segment, Summary, Transcript, Word
from ..snippets import extract_speaker_snippets
from ..speakers import (
    SpeakerMappingError,
    apply_mapping,
    read_mapping,
    validate_mapping,
    write_skeleton,
)
from ..summarize.anthropic_batch import AnthropicSummarizer
from ..summarize.base import Summarizer
from ..summarize.gemini_batch import GeminiSummarizer

DEFAULT_SPEECH_MODEL: aai.SpeechModel = aai.SpeechModel.universal


def _track_a_summarizer(name: str, *, batch: bool = True) -> Summarizer:
    """Track-A-specific summarizer resolution.

    ``claude``/``anthropic`` -> single-call Claude (Message Batches API).
    ``gemini``/``google`` -> single-call Gemini (Batch API).
    """
    key = name.lower()
    if key in {"claude", "anthropic", "claude-sonnet"}:
        return AnthropicSummarizer(batch=batch)
    if key in {"gemini", "google"}:
        return GeminiSummarizer(batch=batch)
    raise ValueError(
        f"Unknown Track A summarizer: {name!r}. Choose 'claude' or 'gemini'."
    )


class AssemblyAIPipeline:
    """Pipeline that runs Track A as three explicit stages."""

    name = "assemblyai"

    def __init__(
        self,
        speech_model: aai.SpeechModel = DEFAULT_SPEECH_MODEL,
    ) -> None:
        self.speech_model = speech_model

    # ------------------------------------------------------------------ #
    # Stage 1: transcribe
    # ------------------------------------------------------------------ #

    def transcribe(
        self,
        audio_path: Path,
        run_dir: Path,
        *,
        language: str = "nl",
        snippets_per_speaker: int = 3,
    ) -> Transcript:
        """Run AssemblyAI transcription + diarization and write artefacts.

        Outputs in ``run_dir``:
        - ``transcript.json``, ``transcript.md`` — the raw diarized transcript.
        - ``speakers.json`` — a ``{label: null}`` skeleton for the user to fill in.
        - ``snippets/SPEAKER_*.wav`` — short audio clips per speaker.
        - ``meta.json`` — run metadata, no summary yet.
        """
        settings = get_settings()
        api_key = require(settings.assemblyai_api_key, "ASSEMBLYAI_API_KEY")
        aai.settings.api_key = api_key

        run_dir.mkdir(parents=True, exist_ok=True)

        config = aai.TranscriptionConfig(
            speech_model=self.speech_model,
            language_code=language,
            speaker_labels=True,
            punctuate=True,
            format_text=True,
        )

        t0 = time.perf_counter()
        transcript = aai.Transcriber(config=config).transcribe(str(audio_path))
        transcribe_seconds = time.perf_counter() - t0

        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(
                f"AssemblyAI transcription failed: {transcript.error}"
            )

        canonical = _to_canonical_transcript(
            transcript,
            source_audio=str(audio_path),
            language=language,
            speech_model=str(self.speech_model),
        )

        write_transcript(run_dir, canonical)
        write_skeleton(run_dir, canonical)

        # Best-effort snippet extraction; not fatal if ffmpeg is missing.
        try:
            extract_speaker_snippets(
                canonical,
                audio_path,
                run_dir,
                top_n=snippets_per_speaker,
            )
        except Exception as exc:  # noqa: BLE001 — surfaced to the user, not fatal.
            (run_dir / "snippets" / "EXTRACTION_FAILED.txt").parent.mkdir(
                parents=True, exist_ok=True
            )
            (run_dir / "snippets" / "EXTRACTION_FAILED.txt").write_text(
                f"Snippet extraction failed: {exc}\n", encoding="utf-8"
            )

        meta = RunMeta(
            run_id=run_dir.name,
            backend=self.name,
            input_path=str(audio_path),
            input_sha256=sha256_of(audio_path),
            duration=canonical.duration,
            timings={"transcribe_s": transcribe_seconds},
            model_versions={"assemblyai_speech": str(self.speech_model)},
            extra={
                "transcript_id": transcript.id,
                "language_code": language,
                "stage": "transcribed",
            },
        )
        write_meta(run_dir, meta)
        return canonical

    # ------------------------------------------------------------------ #
    # Stage 2: relabel
    # ------------------------------------------------------------------ #

    def relabel(self, run_dir: Path, *, require_all_named: bool = True) -> Transcript:
        """Apply the user-edited ``speakers.json`` to the saved transcript.

        Overwrites ``transcript.json``/``transcript.md`` with the renamed
        version. Updates ``meta.extra`` with the mapping for auditing.
        """
        transcript = read_transcript(run_dir)
        mapping = read_mapping(run_dir)
        validate_mapping(
            mapping, transcript, require_all_named=require_all_named
        )
        renamed = apply_mapping(transcript, mapping)
        write_transcript(run_dir, renamed)

        meta = read_meta(run_dir)
        meta_extra = dict(meta.extra)
        meta_extra["speaker_mapping"] = {k: v for k, v in mapping.items()}
        meta_extra["stage"] = "relabelled"
        meta = meta.model_copy(update={"extra": meta_extra})
        write_meta(run_dir, meta)
        return renamed

    # ------------------------------------------------------------------ #
    # Stage 3: summarize
    # ------------------------------------------------------------------ #

    def summarize(
        self,
        run_dir: Path,
        *,
        summarizer: str | Summarizer = "claude",
        batch: bool = True,
        language: str | None = None,
    ) -> Summary:
        """Summarize the (relabelled) transcript via Claude or Gemini.

        Writes ``summary.json``/``summary.md`` and updates ``meta.json``
        with summarizer info, prompt version, and timings.
        """
        transcript = read_transcript(run_dir)
        meta = read_meta(run_dir)
        lang = language or str(meta.extra.get("language_code") or transcript.language)

        impl = (
            _track_a_summarizer(summarizer, batch=batch)
            if isinstance(summarizer, str)
            else summarizer
        )

        log_dir = run_dir / "llm"
        t0 = time.perf_counter()
        summary = impl.summarize(transcript, log_dir=log_dir, language=lang)
        summarize_seconds = time.perf_counter() - t0

        write_summary(run_dir, summary)

        timings = dict(meta.timings)
        timings["summarize_s"] = summarize_seconds
        model_versions = dict(meta.model_versions)
        model_versions["summarizer"] = summary.summarizer_backend
        meta_extra = dict(meta.extra)
        meta_extra["stage"] = "summarized"
        meta = meta.model_copy(
            update={
                "timings": timings,
                "model_versions": model_versions,
                "prompt_version": summary.prompt_version,
                "extra": meta_extra,
            }
        )
        write_meta(run_dir, meta)
        return summary

    # ------------------------------------------------------------------ #
    # Convenience: chain all three stages (no human-in-the-loop pause)
    # ------------------------------------------------------------------ #

    def run(
        self,
        audio_path: Path,
        run_dir: Path,
        *,
        language: str = "nl",
        summarizer: str | Summarizer = "claude",
        batch: bool = True,
    ) -> RunResult:
        """Run all three stages back-to-back.

        Skips relabelling because no ``speakers.json`` has been edited
        yet — generic ``SPEAKER_A/B/C`` labels are passed through to the
        summarizer. For human-in-the-loop usage prefer the staged CLI
        commands ``meetings transcribe`` / ``relabel`` / ``summarize``.
        """
        transcript = self.transcribe(audio_path, run_dir, language=language)
        summary = self.summarize(
            run_dir, summarizer=summarizer, batch=batch, language=language
        )
        meta = read_meta(run_dir)
        return RunResult(transcript=transcript, summary=summary, meta=meta)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _ms_to_s(ms: int | float | None) -> float:
    return float(ms or 0) / 1000.0


def _speaker_label(raw: str | None) -> str | None:
    if not raw:
        return None
    raw = str(raw).strip()
    if raw.upper().startswith("SPEAKER"):
        return raw.upper()
    return f"SPEAKER_{raw}"


def _to_canonical_transcript(
    t: aai.Transcript,
    *,
    source_audio: str,
    language: str,
    speech_model: str,
) -> Transcript:
    segments: list[Segment] = []
    speaker_order: list[str] = []
    seen: set[str] = set()
    duration = _ms_to_s(getattr(t, "audio_duration", 0))

    utterances = getattr(t, "utterances", None) or []
    if utterances:
        for u in utterances:
            sp = _speaker_label(getattr(u, "speaker", None))
            if sp and sp not in seen:
                seen.add(sp)
                speaker_order.append(sp)
            words: list[Word] = []
            for w in getattr(u, "words", None) or []:
                w_sp = _speaker_label(getattr(w, "speaker", None)) or sp
                words.append(
                    Word(
                        text=w.text,
                        start=_ms_to_s(w.start),
                        end=_ms_to_s(w.end),
                        speaker=w_sp,
                        confidence=getattr(w, "confidence", None),
                    )
                )
            segments.append(
                Segment(
                    start=_ms_to_s(u.start),
                    end=_ms_to_s(u.end),
                    speaker=sp,
                    text=(u.text or "").strip(),
                    words=words,
                )
            )
    elif getattr(t, "text", None):
        segments.append(
            Segment(start=0.0, end=duration, speaker=None, text=t.text or "", words=[])
        )

    if not duration and segments:
        duration = max(s.end for s in segments)

    return Transcript(
        language=language,
        duration=duration,
        speakers=speaker_order,
        segments=segments,
        source_audio=source_audio,
        backend="assemblyai",
        backend_meta={
            "transcript_id": t.id,
            "audio_url": getattr(t, "audio_url", None),
            "speech_model": speech_model,
        },
    )


# Re-export for callers that previously relied on these from this module.
__all__ = [
    "AssemblyAIPipeline",
    "DEFAULT_SPEECH_MODEL",
    "SpeakerMappingError",
]
