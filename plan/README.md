# Meeting Transcription, Diarization & Summarization — Plan

This directory contains the implementation plan for modernizing the existing
Whisper/pyannote codebase into a meeting-focused tool with **two parallel tracks**:

- **Track A** — Test AssemblyAI Universal-2 (all-in-one API).
- **Track B** — Build a custom pipeline composed of best-in-class models for
  transcription, diarization, and summarization. Quality of the meeting
  **summary** is the top priority. Cost/speed do not matter (1–2 runs total).
  Learning project. No local GPU available → prefer APIs.

Both tracks share a common project skeleton (see `03-repo-modernization.md`)
and a shared evaluation method (see `04-evaluation-and-comparison.md`).

## File index

| File | Purpose |
|------|---------|
| `00-architecture.md`            | Big-picture decisions, data contracts, directory layout. |
| `01-track-a-assemblyai.md`      | Step-by-step plan for AssemblyAI Universal-2 PoC. |
| `02-track-b-custom-pipeline.md` | Step-by-step plan for the multi-model custom pipeline (2026 refresh: Scribe v2 default). |
| `03-repo-modernization.md`      | Shared modernization: deps, lint, tests, CLI, IO. |
| `04-evaluation-and-comparison.md` | How to compare Track A vs Track B output quality. |
| `05-audio-preprocessing.md`     | Canonical input format and ffmpeg conversion steps. |
| `06-testing-and-comparison-notebooks.md` | Notebook workflow: smoke-test on the 3 short samples + metrics/relabel/review of the 2 h CLI run. |

## How to use this with sub-agents

Each numbered file is self-contained and lists concrete tasks with acceptance
criteria. Hand a single file to an agent and it should be able to execute the
work end-to-end. Tasks are ordered; dependencies between files are stated
explicitly at the top of each file.

## High-level execution order

1. `03-repo-modernization.md` — set up the new repo skeleton (do this first).
2. `05-audio-preprocessing.md` — normalize each new recording into `audio/processed/`.
3. `01-track-a-assemblyai.md` — fastest path to a working baseline.
4. `02-track-b-custom-pipeline.md` — build the custom pipeline (2026 refresh: Scribe v2 + pyannoteAI + Claude/Gemini).
5. `06-testing-and-comparison-notebooks.md` — smoke-test both pipelines on the 3 short samples; run the 2 h meeting from the CLI; review + relabel + summarize via the notebooks.
6. `04-evaluation-and-comparison.md` — final apples-to-apples comparison of summary quality.

## Inputs / outputs (shared)

- **Input**: a meeting audio file in `audio/raw/` (wav, mp3, m4a, mp4).
  Preprocessed into `audio/processed/<name>.16k.mono.wav` per
  `05-audio-preprocessing.md` before any pipeline runs.
- **Outputs** (per run, written under `Transcription/<run_id>/`):
  - `transcript.json` — canonical structured transcript (see `00-architecture.md`).
  - `transcript.md` — human-readable transcript with speakers and timestamps.
  - `summary.md` — structured meeting summary.
  - `summary.json` — machine-readable summary.
  - `meta.json` — run metadata (track, models, prompts, timings, cost).
