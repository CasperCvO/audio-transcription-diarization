# meetings

Transcribe, diarize and summarize meeting recordings — Dutch-first.

This repo started as a notebook-based Whisper + pyannote experiment and is
being modernized into a small Python package with two pluggable backends:

- **Track A** — AssemblyAI Universal-2 (single API: transcribe + diarize + summarize).
- **Track B** — Custom composition of best-in-class models per stage,
  optimized for **summary quality** on Dutch meetings.

Implementation plan and design notes live in [`plan/`](./plan/README.md).

## Quick start

Requirements: Python 3.12, [UV](https://github.com/astral-sh/uv), `ffmpeg` on `PATH`.

```bash
uv sync
cp .env.example .env   # fill in only the keys you need (see below)
uv run meetings --help
```

Drop a recording into `audio/raw/`, normalize it to canonical 16 kHz mono WAV,
then run a backend against the processed file:

```bash
uv run meetings preprocess audio/raw/your_meeting.m4a
uv run meetings run --backend assemblyai --audio audio/processed/your_meeting.16k.mono.wav
uv run meetings run --backend custom     --audio audio/processed/your_meeting.16k.mono.wav
```

Each run writes its outputs under `Transcription/<run_id>/`:

| File | Contents |
|------|----------|
| `transcript.json` | Canonical structured transcript (speakers, segments, words). |
| `transcript.md`   | Human-readable transcript with speakers and timestamps. |
| `summary.json`    | Structured summary (TL;DR, topics, decisions, actions, …). |
| `summary.md`      | Human-readable Dutch summary. |
| `meta.json`       | Run metadata (backend, models, prompt version, timings). |

## CLI

| Command | Purpose |
|---------|---------|
| `meetings preprocess SRC [--dst-dir DIR] [--overwrite]` | Convert any input to 16 kHz mono PCM WAV (`audio/processed/` by default). |
| `meetings run --audio FILE [--backend ...] [options]`   | Run a full pipeline (transcribe + diarize + summarize). |
| `meetings validate RUN_DIR`                              | Re-validate an existing run directory against the current schema. |
| `meetings compare RUN_A RUN_B`                           | Compare two runs (stub — see `plan/04-evaluation-and-comparison.md`). |

### `run` options

- `--backend` — `assemblyai` (default) or `custom`.
- `--language` — BCP-47 language code, default `nl`.
- Custom-backend-only:
  - `--transcriber` — `elevenlabs` (default, Scribe v2), `openai_gpt4o`, `whisper-1`, `deepgram`.
  - `--diarizer` — `builtin` (default; trust the transcriber's own speaker labels, e.g. Scribe v2), `pyannoteai`, `pyannote_local`.
  - `--summarizer` — `claude` (default).
  - `--name-resolution/--no-name-resolution` — try to map `SPEAKER_XX` to real names via Claude.
  - `--cleanup/--no-cleanup` — remove intermediate artifacts after a successful run.

## Configuration

Copy `.env.example` to `.env` and fill in only what you plan to use:

| Variable | Used by |
|----------|---------|
| `ASSEMBLYAI_API_KEY` | Track A (AssemblyAI Universal-2). |
| `OPENAI_API_KEY`     | Track B transcription (`whisper-1`, `openai_gpt4o`). |
| `ELEVENLABS_API_KEY` | Track B transcription **and** default diarization (Scribe v2, `--diarizer builtin`). |
| `DEEPGRAM_API_KEY`   | Track B transcription (`deepgram`). |
| `PYANNOTEAI_API_KEY` | Optional Track B diarization (`--diarizer pyannoteai`) for overlap-heavy audio. |
| `ANTHROPIC_API_KEY`  | Track B summarization (`claude`) and optional name resolution. |
| `GOOGLE_API_KEY`     | Reserved for alternative summarizers. |
| `HF_TOKEN`           | Optional local pyannote fallback (`pyannote_local`). |

## Project layout

```
src/meetings/        # the package (cli, pipelines, transcribe, diarize, summarize, ...)
plan/                # implementation plan and design notes
notebooks/legacy/    # original Whisper + pyannote notebooks (reference only)
audio/raw/           # input recordings — gitignored
audio/processed/     # canonical 16 kHz mono WAVs from `meetings preprocess` — gitignored
Transcription/       # run outputs — gitignored
tests/
```

## Status

- [x] Repo modernization scaffold (`plan/03-repo-modernization.md`)
- [x] Track A — AssemblyAI implementation (`plan/01-track-a-assemblyai.md`)
- [x] Track B — custom pipeline (`plan/02-track-b-custom-pipeline.md`)
- [ ] Evaluation & comparison (`plan/04-evaluation-and-comparison.md`)
- [ ] Live 2 h Dutch acceptance run (Scribe v2 + `builtin` + Claude/Gemini)

## Development

```bash
uv run ruff check .
uv run mypy src
uv run pytest
```

## License

MIT — see `LICENSE`.
