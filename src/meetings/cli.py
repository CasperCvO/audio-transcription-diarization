"""Typer CLI entry point. Installed as the `meetings` console script.

Track A's pipeline is split into three stages:
- ``meetings transcribe`` — AssemblyAI transcription + diarization.
- ``meetings relabel`` — apply user-edited ``speakers.json`` to the transcript.
- ``meetings summarize`` — call Claude or Gemini (batch by default) to produce
  the structured summary.

The legacy ``meetings run`` chains all three back-to-back without the manual
relabel step (kept for parity with Track B).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from .audio import audio_meta, prepare_audio
from .config import get_settings
from .io import new_run_id, read_meta, read_run, read_transcript
from .pipelines.assemblyai import AssemblyAIPipeline
from .pipelines.base import MeetingPipeline
from .pipelines.custom import CustomPipeline
from .snippets import SNIPPETS_DIRNAME
from .speakers import SPEAKERS_FILENAME, SpeakerMappingError

app = typer.Typer(
    add_completion=False,
    help="Transcribe, diarize and summarize meeting recordings.",
    no_args_is_help=True,
)
console = Console()


def _select_pipeline(
    backend: str,
    transcriber: str,
    diarizer: str,
    summarizer: str,
    cleanup: bool,
    name_resolution: bool,
) -> MeetingPipeline:
    if backend == "assemblyai":
        return AssemblyAIPipeline()
    if backend == "custom":
        return CustomPipeline(
            transcriber=transcriber,
            diarizer=diarizer,
            summarizer=summarizer,
            cleanup=cleanup,
            name_resolution=name_resolution,
        )
    raise typer.BadParameter(f"Unknown backend: {backend!r}")


@app.command()
def run(
    audio: Annotated[Path, typer.Option("--audio", exists=True, dir_okay=False, readable=True)],
    backend: Annotated[str, typer.Option("--backend", help="assemblyai | custom")] = "assemblyai",
    language: Annotated[str, typer.Option("--language")] = "nl",
    summarizer: Annotated[
        str,
        typer.Option(
            "--summarizer",
            help=(
                "Track A: claude | gemini (single-call, batch by default). "
                "Track B: claude (map-reduce)."
            ),
        ),
    ] = "claude",
    transcriber: Annotated[
        str,
        typer.Option(
            "--transcriber",
            help="openai_gpt4o | whisper-1 | elevenlabs | deepgram (custom backend only)",
        ),
    ] = "elevenlabs",
    diarizer: Annotated[
        str,
        typer.Option(
            "--diarizer", help="pyannoteai | pyannote_local (custom backend only)"
        ),
    ] = "pyannoteai",
    cleanup: Annotated[bool, typer.Option("--cleanup/--no-cleanup")] = False,
    name_resolution: Annotated[
        bool,
        typer.Option(
            "--name-resolution/--no-name-resolution",
            help="Try to map SPEAKER_XX labels to real names via Claude (custom backend).",
        ),
    ] = False,
    sync: Annotated[
        bool,
        typer.Option(
            "--sync/--batch",
            help=(
                "Track A only: --sync forces a synchronous Claude/Gemini call instead "
                "of the (cheaper, default) batch API."
            ),
        ),
    ] = False,
) -> None:
    """Run a pipeline end-to-end on a single audio file.

    For Track A this skips the manual relabel step. Use the staged commands
    (`transcribe`, `relabel`, `summarize`) for the human-in-the-loop flow.
    """
    settings = get_settings()
    pipeline = _select_pipeline(
        backend, transcriber, diarizer, summarizer, cleanup, name_resolution
    )
    run_id = new_run_id(audio, pipeline.name)
    run_dir = settings.transcription_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]Backend:[/] {pipeline.name}")
    console.print(f"[bold]Run dir:[/] {run_dir}")

    if isinstance(pipeline, AssemblyAIPipeline):
        pipeline.run(
            audio,
            run_dir,
            language=language,
            summarizer=summarizer,
            batch=not sync,
        )
    else:
        pipeline.run(audio, run_dir, language=language)
    console.print(f"[green]Done.[/] Outputs in {run_dir}")


# --------------------------------------------------------------------------- #
# Track A staged commands
# --------------------------------------------------------------------------- #


@app.command()
def transcribe(
    audio: Annotated[
        Path, typer.Option("--audio", exists=True, dir_okay=False, readable=True)
    ],
    backend: Annotated[
        str,
        typer.Option(
            "--backend",
            help="assemblyai (Track A) or custom (Track B with separate transcriber/diarizer)",
        ),
    ] = "assemblyai",
    language: Annotated[str, typer.Option("--language")] = "nl",
    snippets_per_speaker: Annotated[
        int,
        typer.Option(
            "--snippets",
            help="How many audio snippets to extract per speaker for manual labelling.",
        ),
    ] = 3,
    transcriber: Annotated[
        str,
        typer.Option(
            "--transcriber",
            help="openai_gpt4o | whisper-1 | elevenlabs | deepgram (custom backend only)",
        ),
    ] = "elevenlabs",
    diarizer: Annotated[
        str,
        typer.Option(
            "--diarizer",
            help="pyannoteai | pyannote_local (custom backend only)",
        ),
    ] = "pyannoteai",
) -> None:
    """Stage 1: transcribe + diarize; emit speakers.json + snippets.

    Track A (default): uses AssemblyAI's unified transcription+diarization.
    Track B (--backend custom): composes separate transcriber and diarizer models
    for best-of-breed quality (e.g. ElevenLabs Scribe v2 + pyannoteAI).

    After this completes, listen to the per-speaker clips in
    ``<run_dir>/snippets/`` and edit ``<run_dir>/speakers.json`` to assign
    real names. Then run ``meetings relabel`` and ``meetings summarize``.
    """
    # Validate custom-only flags
    if backend == "assemblyai":
        if transcriber != "elevenlabs":
            raise typer.BadParameter(
                "--transcriber is only available with --backend custom"
            )
        if diarizer != "pyannoteai":
            raise typer.BadParameter(
                "--diarizer is only available with --backend custom"
            )

    settings = get_settings()
    pipeline: AssemblyAIPipeline | CustomPipeline
    run_dir: Path

    if backend == "assemblyai":
        pipeline = AssemblyAIPipeline()
        run_id = new_run_id(audio, pipeline.name)
        run_dir = settings.transcription_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[bold]Run dir:[/] {run_dir}")

        transcript = pipeline.transcribe(
            audio, run_dir, language=language, snippets_per_speaker=snippets_per_speaker
        )

        console.print(
            f"[green]Transcribed[/] {len(transcript.segments)} segments, "
            f"{len(transcript.speakers)} speakers."
        )
        console.print(f"  Edit [cyan]{run_dir / SPEAKERS_FILENAME}[/]")
        console.print(f"  Listen to [cyan]{run_dir / SNIPPETS_DIRNAME}/*.wav[/]")
        console.print(f"  Then: [cyan]meetings relabel {run_dir}[/]")
    elif backend == "custom":
        pipeline = CustomPipeline(
            transcriber=transcriber,
            diarizer=diarizer,
            summarizer="claude",  # Required by constructor but unused in transcribe_only
        )
        run_id = new_run_id(audio, pipeline.name)
        run_dir = settings.transcription_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[bold]Run dir:[/] {run_dir}")
        console.print(f"[bold]Backend:[/] {pipeline.name}")

        transcript = pipeline.transcribe_only(
            audio,
            run_dir,
            language=language,
            snippets_per_speaker=snippets_per_speaker,
        )

        console.print(
            f"[green]Transcribed[/] {len(transcript.segments)} segments, "
            f"{len(transcript.speakers)} speakers."
        )
        console.print(f"  Edit [cyan]{run_dir / SPEAKERS_FILENAME}[/]")
        console.print(f"  Listen to [cyan]{run_dir / SNIPPETS_DIRNAME}/*.wav[/]")
        console.print(f"  Then: [cyan]meetings relabel {run_dir}[/]")
    else:
        raise typer.BadParameter(f"Unknown backend: {backend!r}")


@app.command()
def relabel(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    allow_unset: Annotated[
        bool,
        typer.Option(
            "--allow-unset/--require-all-named",
            help="Allow null entries in speakers.json (keeps the original SPEAKER_X label).",
        ),
    ] = False,
) -> None:
    """Stage 2: apply the user-edited speakers.json to the transcript."""
    pipeline = AssemblyAIPipeline()
    try:
        renamed = pipeline.relabel(run_dir, require_all_named=not allow_unset)
    except SpeakerMappingError as exc:
        raise typer.BadParameter(str(exc)) from exc

    console.print(
        f"[green]Relabelled[/] — speakers now: {', '.join(renamed.speakers) or '(none)'}"
    )
    console.print(f"  Next: [cyan]meetings summarize {run_dir}[/]")


@app.command()
def summarize(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    summarizer: Annotated[
        str,
        typer.Option(
            "--summarizer",
            help="claude (Anthropic) or gemini (Google). Both single-call.",
        ),
    ] = "claude",
    sync: Annotated[
        bool,
        typer.Option(
            "--sync/--batch",
            help=(
                "Use the synchronous API (immediate response, full price) instead "
                "of the batch API (50%% cheaper, up to 24h SLO; usually <1h)."
            ),
        ),
    ] = False,
    language: Annotated[
        str | None,
        typer.Option(
            "--language",
            help="Override the transcript language for the summary prompt.",
        ),
    ] = None,
) -> None:
    """Stage 3: summarize the (relabelled) transcript via Claude or Gemini."""
    pipeline = AssemblyAIPipeline()
    summary = pipeline.summarize(
        run_dir,
        summarizer=summarizer,
        batch=not sync,
        language=language,
    )
    console.print(
        f"[green]Summarized[/] via [cyan]{summary.summarizer_backend}[/] — "
        f"{len(summary.tldr)} tldr bullets, {len(summary.action_items)} actions."
    )
    console.print(f"  See [cyan]{run_dir / 'summary.md'}[/]")


# --------------------------------------------------------------------------- #
# Misc commands
# --------------------------------------------------------------------------- #


@app.command()
def validate(run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)]) -> None:
    """Re-validate an existing run directory against the current schema."""
    if (run_dir / "summary.json").exists():
        result = read_run(run_dir)
        console.print(
            f"[green]OK[/] — {len(result.transcript.segments)} segments, "
            f"{len(result.summary.action_items)} action items."
        )
        return
    transcript = read_transcript(run_dir)
    meta = read_meta(run_dir)
    console.print(
        f"[yellow]Partial run[/] (stage={meta.extra.get('stage', '?')}) — "
        f"{len(transcript.segments)} segments, no summary yet."
    )


@app.command()
def preprocess(
    src: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    dst_dir: Annotated[
        Path | None,
        typer.Option("--dst-dir", help="Output directory (default: audio/processed/)."),
    ] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = False,
) -> None:
    """Convert SRC to canonical 16 kHz mono PCM WAV per plan/05-audio-preprocessing.md."""
    out_dir = dst_dir or (Path.cwd() / "audio" / "processed")
    out = prepare_audio(src, out_dir, overwrite=overwrite)
    meta = audio_meta(out)
    console.print(
        f"[green]OK[/] {out} — {meta.duration:.1f}s, "
        f"{meta.bytes_/1024:.0f} KiB, sha256={meta.sha256[:12]}…"
    )


@app.command()
def compare(
    run_a: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    run_b: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Compare two runs (stub — see plan/04-evaluation-and-comparison.md)."""
    a = read_run(run_a)
    b = read_run(run_b)
    console.print(f"A: {a.transcript.backend} — {len(a.transcript.segments)} segments")
    console.print(f"B: {b.transcript.backend} — {len(b.transcript.segments)} segments")
    console.print("[yellow]Full comparison not yet implemented.[/]")


if __name__ == "__main__":
    app()
