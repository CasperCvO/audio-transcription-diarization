"""Per-speaker audio snippet extraction for the manual relabelling step.

Extracts short audio clips from the source recording — one set per
speaker — so the user can listen and assign real names in
``speakers.json``. Uses ffmpeg without re-encoding (``-c copy``) when
possible, falling back to PCM re-encoding so output is always a valid
playable file regardless of the input container.

Public API
----------
- ``extract_speaker_snippets(transcript, audio_path, run_dir, top_n=3)``
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .schema import Segment, Transcript

SNIPPETS_DIRNAME = "snippets"


class SnippetExtractionError(RuntimeError):
    """Raised when ffmpeg fails or no segments are available."""


def _require_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if not found:
        raise SnippetExtractionError(
            "ffmpeg not found on PATH. Install it (e.g. `winget install ffmpeg`) "
            "to enable speaker snippet extraction."
        )
    return found


def _pick_segments_per_speaker(
    transcript: Transcript,
    *,
    top_n: int,
    min_seconds: float,
    max_seconds: float,
) -> dict[str, list[Segment]]:
    """For every speaker, pick the ``top_n`` longest segments within bounds.

    Falls back to the longest segments below ``min_seconds`` if none meet the
    minimum length, so we always emit something the user can listen to.
    """
    by_speaker: dict[str, list[Segment]] = {}
    for seg in transcript.segments:
        if not seg.speaker:
            continue
        by_speaker.setdefault(seg.speaker, []).append(seg)

    picks: dict[str, list[Segment]] = {}
    for speaker, segs in by_speaker.items():
        in_range = [s for s in segs if min_seconds <= (s.end - s.start) <= max_seconds]
        chosen = in_range if in_range else segs
        chosen = sorted(chosen, key=lambda s: s.end - s.start, reverse=True)[:top_n]
        chosen.sort(key=lambda s: s.start)  # natural play order
        picks[speaker] = chosen
    return picks


def _extract_clip(
    ffmpeg: str,
    src: Path,
    dst: Path,
    *,
    start: float,
    duration: float,
) -> None:
    """Extract ``[start, start+duration]`` from ``src`` into ``dst`` as WAV."""
    # Re-encode to mono 16 kHz PCM. We don't use ``-c copy`` because lossy
    # containers (m4a/mp3) can't be cut with frame-accurate accuracy without
    # re-encoding, and the clips are short — the cost is negligible.
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{max(start, 0.0):.3f}",
        "-t",
        f"{max(duration, 0.1):.3f}",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SnippetExtractionError(
            f"ffmpeg failed extracting {dst.name}: {result.stderr.strip()[-500:]}"
        )


def extract_speaker_snippets(
    transcript: Transcript,
    audio_path: Path,
    run_dir: Path,
    *,
    top_n: int = 3,
    min_seconds: float = 4.0,
    max_seconds: float = 20.0,
) -> dict[str, list[Path]]:
    """Write up to ``top_n`` snippets per speaker into ``run_dir/snippets/``.

    Returns a mapping ``{speaker_label: [snippet_path, ...]}``.

    Idempotent: existing snippet files are overwritten so re-running gives
    fresh clips after a transcript update.
    """
    if not audio_path.exists():
        raise SnippetExtractionError(f"Source audio not found: {audio_path}")
    ffmpeg = _require_ffmpeg()

    out_dir = run_dir / SNIPPETS_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)

    picks = _pick_segments_per_speaker(
        transcript,
        top_n=top_n,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
    )

    written: dict[str, list[Path]] = {}
    for speaker, segs in picks.items():
        paths: list[Path] = []
        # Safe filename: speakers come from `_speaker_label` already (e.g.
        # ``SPEAKER_A``) but we still sanitize defensively.
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in speaker)
        for i, seg in enumerate(segs, start=1):
            dst = out_dir / f"{safe}_{i:02d}.wav"
            _extract_clip(
                ffmpeg,
                audio_path,
                dst,
                start=seg.start,
                duration=max(seg.end - seg.start, 0.5),
            )
            paths.append(dst)
        written[speaker] = paths
    return written
