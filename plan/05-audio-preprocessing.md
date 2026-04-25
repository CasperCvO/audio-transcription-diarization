# 05 — Audio Preprocessing (input normalization)

**Run this before any track.** All pipelines consume audio from
`audio/processed/`. The original recording stays untouched in `audio/raw/`.

## Objective

Convert any incoming meeting recording into a single canonical format that
ASR and diarization models expect, with **no quality-altering DSP**. The goal
is *normalization*, not *enhancement*. Speech models (Whisper, pyannote,
AssemblyAI Universal-2) are trained on real-world noisy audio and degrade
when fed denoised / EQ'd / compressed material.

## Canonical output format

| Property        | Value              |
|-----------------|--------------------|
| Container       | `wav`              |
| Codec           | `pcm_s16le`        |
| Sample rate     | `16000 Hz`         |
| Channels        | `1` (mono)         |
| Bit depth       | `16-bit`           |
| Loudness        | unchanged (no AGC, no normalization) |
| Naming          | `<basename>.16k.mono.wav` |

Rationale:
- 16 kHz mono 16-bit PCM is the input expected by Whisper, pyannote 3.x, and
  is what AssemblyAI resamples to internally. Doing it once upfront keeps
  every backend deterministic and reduces file size ~6× vs 48 kHz stereo.
- We **do not** apply normalization, denoise, EQ, gating or compression.
  Phone recorders (e.g. Pixel Recorder) already apply their own NS/AGC;
  stacking more processing reliably hurts WER and diarization.

## Directory layout

```
audio/
├── raw/         # Original recordings (m4a, mp3, wav, …) — read-only
└── processed/   # Canonical 16 kHz mono PCM WAVs produced by this step
```

`audio/raw/` is the source of truth. `audio/processed/` is reproducible from
it and may be deleted at any time.

## Steps

### A1. Inspect the source file

Before converting, confirm what you have:

```powershell
ffprobe -v error -show_entries stream=codec_name,sample_rate,channels,bits_per_sample -show_entries format=duration,size,bit_rate -of default "audio/raw/<file>"
```

Record: codec, sample rate, channels, duration. If the file is already
`pcm_s16le` / 16 kHz / mono, you can symlink/copy it into `audio/processed/`
and skip A2.

### A2. Convert to canonical format

One ffmpeg invocation, no filters beyond channel/sample-rate conversion:

```powershell
ffmpeg -y -i "audio/raw/<name>.<ext>" `
  -ac 1 -ar 16000 -c:a pcm_s16le `
  "audio/processed/<name>.16k.mono.wav"
```

Flags:
- `-ac 1` — downmix to mono. For phone recordings the L/R channels are just
  the device's mic array, not per-speaker tracks; downmixing is lossless
  for ASR purposes.
- `-ar 16000` — resample to 16 kHz.
- `-c:a pcm_s16le` — uncompressed 16-bit PCM.
- No `-af` filter chain. No loudnorm, no highpass, no afftdn.

### A3. Verify the output

```powershell
ffprobe -v error -show_entries stream=codec_name,sample_rate,channels,bits_per_sample -show_entries format=duration -of default "audio/processed/<name>.16k.mono.wav"
```

Expect: `pcm_s16le`, `16000`, `1`, `16`, duration within ±0.1 s of source.

## When (and only when) to deviate

| Situation                                  | Action |
|--------------------------------------------|--------|
| Per-speaker channels (e.g. Zoom dual-track)| **Do not downmix.** Process each channel separately and merge speaker labels afterwards. |
| Source is already mono 16 kHz PCM          | Copy as-is. |
| Catastrophic noise (HVAC roar, constant hum) | Try a *single* light pass with a dedicated speech enhancer (e.g. `resemble-enhance`, Demucs `htdemucs` vocals). Never use Ableton/iZotope-style multi-effect chains. Keep the un-enhanced version too and A/B both. |
| Very long file (>2 h) and Track B local    | Optional: split into ≤30-min chunks at silence boundaries with `ffmpeg silenceremove` / `silencedetect`. Track A handles long files in one upload. |
| Clipped audio (peaks at 0 dBFS continuously) | Nothing fixes clipping after the fact. Note it in `meta.json` and accept the WER hit. |

## Things explicitly *not* done here

- ❌ Loudness normalization (`loudnorm`, `dynaudnorm`).
- ❌ Noise reduction (`afftdn`, RNNoise, Krisp, iZotope RX).
- ❌ EQ / high-pass beyond what the recorder already applies.
- ❌ Compression / limiting / AGC.
- ❌ Reverb removal.
- ❌ Re-encoding through a lossy codec.

If a future experiment shows one of these helps on *your* data, document it
as a separate variant under `audio/processed/<name>.<variant>.wav` and
compare via `04-evaluation-and-comparison.md`. Never overwrite the canonical
output.

## Future automation (tracked in `03-repo-modernization.md`)

`src/meetings/audio.py` will expose:

```python
def preprocess(src: Path, dst_dir: Path = Path("audio/processed")) -> Path:
    """Convert src to 16 kHz mono pcm_s16le WAV in dst_dir. Idempotent."""
```

The CLI command `meetings preprocess audio/raw/<file>` will wrap it. Until
that lands, run the ffmpeg command above manually.

## Acceptance criteria

- Every file in `audio/raw/` has a matching `<name>.16k.mono.wav` in
  `audio/processed/`.
- `ffprobe` on each processed file reports `pcm_s16le`, `16000 Hz`, `1` ch,
  `16-bit`.
- Duration of processed file matches source within ±0.1 s.
- No filter chain other than `-ac 1 -ar 16000` was applied.
