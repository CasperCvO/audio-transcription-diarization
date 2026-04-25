"""Typer CLI entry point. Installed as the `meetings` console script."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from .audio import audio_meta, prepare_audio
from .config import get_settings
from .io import new_run_id, read_run
from .pipelines.assemblyai import AssemblyAIPipeline
from .pipelines.base import MeetingPipeline
from .pipelines.custom import CustomPipeline

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
    transcriber: Annotated[
        str,
        typer.Option(
            "--transcriber",
            help="openai_gpt4o | whisper-1 | elevenlabs | deepgram (custom backend only)",
        ),
    ] = "whisper-1",
    diarizer: Annotated[
        str,
        typer.Option(
            "--diarizer", help="pyannoteai | pyannote_local (custom backend only)"
        ),
    ] = "pyannoteai",
    summarizer: Annotated[
        str, typer.Option("--summarizer", help="claude (custom backend only)")
    ] = "claude",
    cleanup: Annotated[bool, typer.Option("--cleanup/--no-cleanup")] = False,
    name_resolution: Annotated[
        bool,
        typer.Option(
            "--name-resolution/--no-name-resolution",
            help="Try to map SPEAKER_XX labels to real names via Claude (custom backend).",
        ),
    ] = False,
) -> None:
    """Run a pipeline end-to-end on a single audio file."""
    settings = get_settings()
    pipeline = _select_pipeline(
        backend, transcriber, diarizer, summarizer, cleanup, name_resolution
    )
    run_id = new_run_id(audio, pipeline.name)
    run_dir = settings.transcription_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]Backend:[/] {pipeline.name}")
    console.print(f"[bold]Run dir:[/] {run_dir}")
    pipeline.run(audio, run_dir, language=language)
    console.print(f"[green]Done.[/] Outputs in {run_dir}")


@app.command()
def validate(run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)]) -> None:
    """Re-validate an existing run directory against the current schema."""
    result = read_run(run_dir)
    console.print(
        f"[green]OK[/] — {len(result.transcript.segments)} segments, "
        f"{len(result.summary.action_items)} action items."
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
