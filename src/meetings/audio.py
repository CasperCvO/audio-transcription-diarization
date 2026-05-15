"""Audio preparation: ffmpeg conversion to canonical 16 kHz mono PCM WAV.

Implements plan/02-track-b-custom-pipeline.md task B1, aligned with the
stricter policy in plan/05-audio-preprocessing.md: convert only (sample rate
+ channels + codec). Never apply loudnorm / denoise / EQ / AGC.

Public API
----------
- `prepare_audio(src, dst_dir)` -> Path to 16 kHz mono WAV. Idempotent.
- `probe_duration(path)` -> float seconds (via ffprobe).
- `sha256_of(path)` -> hex digest.
- `audio_meta(path)` -> dict with `sha256`, `duration`, `bytes`.

All functions raise `AudioPrepError` on failure with a clear message.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

MAX_DURATION_SECONDS: float = 4 * 60 * 60  # reject files > 4 hours (plan B1).


class AudioPrepError(RuntimeError):
    """Raised when audio preparation / probing fails."""


@dataclass(frozen=True)
class AudioMeta:
    path: Path
    sha256: str
    duration: float  # seconds
    bytes_: int


def _require_binary(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise AudioPrepError(
            f"Required binary '{name}' not found on PATH. Install ffmpeg "
            f"(e.g. `scoop install ffmpeg` or `winget install ffmpeg`)."
        )
    return found


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """Compute sha256 of a file in streaming fashion."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def probe_duration(path: Path) -> float:
    """Return audio duration in seconds using ffprobe."""
    ffprobe = _require_binary("ffprobe")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AudioPrepError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
        return float(payload["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise AudioPrepError(f"Could not parse ffprobe output for {path}: {exc}") from exc


def audio_meta(path: Path) -> AudioMeta:
    """Return canonical metadata for a file: sha256 + duration + size."""
    if not path.exists():
        raise AudioPrepError(f"Audio file not found: {path}")
    duration = probe_duration(path)
    return AudioMeta(
        path=path,
        sha256=sha256_of(path),
        duration=duration,
        bytes_=path.stat().st_size,
    )


def _is_canonical(src: Path) -> bool:
    """Return True if ``src`` already conforms to 16 kHz mono PCM WAV.

    A fast check via ffprobe lets us short-circuit `prepare_audio` when the
    user passes an already-processed file (e.g. ``audio/processed/x.16k.mono.wav``),
    avoiding wasted ffmpeg work and the ``x.16k.mono.16k.mono.wav`` double-suffix
    that ``dst = dst_dir / f"{src.stem}.16k.mono.wav"`` would otherwise produce.
    """
    if src.suffix.lower() != ".wav":
        return False
    ffprobe = _require_binary("ffprobe")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,sample_rate,channels",
            "-of",
            "json",
            str(src),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    try:
        streams = json.loads(result.stdout).get("streams") or []
    except json.JSONDecodeError:
        return False
    if not streams:
        return False
    s = streams[0]
    return (
        s.get("codec_name") == "pcm_s16le"
        and int(s.get("sample_rate", 0)) == 16000
        and int(s.get("channels", 0)) == 1
    )


def prepare_audio(
    src: Path,
    dst_dir: Path | None = None,
    *,
    max_duration: float = MAX_DURATION_SECONDS,
    overwrite: bool = False,
) -> Path:
    """Convert ``src`` to 16 kHz mono ``pcm_s16le`` WAV in ``dst_dir``.

    Idempotent: if the output already exists and ``overwrite`` is False, the
    existing path is returned. If the input is **already** 16 kHz mono PCM
    WAV, the source path is returned unchanged (no ffmpeg pass, no
    ``x.16k.mono.16k.mono.wav`` duplication). No loudnorm / denoise / EQ /
    AGC is applied — see plan/05-audio-preprocessing.md for the rationale.

    Raises ``AudioPrepError`` if the input is missing, longer than
    ``max_duration`` seconds, or ffmpeg fails.
    """
    if not src.exists():
        raise AudioPrepError(f"Input audio not found: {src}")

    duration = probe_duration(src)
    if duration > max_duration:
        raise AudioPrepError(
            f"Input {src.name} is {duration/3600:.2f} h long, exceeds the "
            f"{max_duration/3600:.1f} h limit."
        )

    # Fast path: input already conforms; nothing to do.
    if not overwrite and _is_canonical(src):
        return src

    dst_dir = dst_dir or (src.parent.parent / "processed")
    dst_dir.mkdir(parents=True, exist_ok=True)
    # Avoid `<stem>.16k.mono.16k.mono.wav` if a caller forces overwrite on
    # an already-canonical name.
    base_stem = src.stem
    if base_stem.endswith(".16k.mono"):
        base_stem = base_stem[: -len(".16k.mono")]
    dst = dst_dir / f"{base_stem}.16k.mono.wav"
    if dst.exists() and not overwrite:
        return dst

    ffmpeg = _require_binary("ffmpeg")
    cmd = [
        ffmpeg,
        "-y",
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
        raise AudioPrepError(
            f"ffmpeg failed converting {src.name}:\n{result.stderr.strip()[-2000:]}"
        )
    return dst
