# 01 — Track A: AssemblyAI Universal-2 PoC

**Status:** Implemented (2026-04-25). Live smoke test pending a real sample.
**Depends on:** `03-repo-modernization.md` (skeleton must exist).

## Objective

Get a working end-to-end Dutch meeting pipeline using **AssemblyAI Universal-2**
in a single API call: transcription + speaker diarization + summarization +
chapters. This is the baseline that Track B must beat on summary quality.

## Why AssemblyAI for the baseline

- Universal-2 supports Dutch.
- Native speaker labels (diarization) per word and per utterance.
- Native LeMUR / Auto-Chapters / Summarization endpoints — no separate LLM call
  needed for the baseline.
- One SDK, one key, no GPU.

## Key finding during implementation

**AssemblyAI's `auto_chapters` and `summarization` features are English-only.**
Enabling them with `language_code="nl"` returns an API error. For Dutch (and any
other non-English meeting) the structured `Summary` is therefore built
entirely from a single **LeMUR `task`** call (Claude Sonnet 4). Native chapters
and `summary` are still used when the caller sets `language="en"`, and their
output is merged into `Summary.topics` / `Summary.tldr` as a bonus.

## Tasks

### A1. Add dependency and config — done
- [x] Added `assemblyai==0.63.0` via `uv add assemblyai` (see `pyproject.toml`).
- [x] `ASSEMBLYAI_API_KEY` already present in `.env.example` and
      `src/meetings/config.py` (loaded through `pydantic-settings`;
      `require()` raises a clear error when missing).
- [x] Fetched authoritative SDK docs via `chub get assemblyai/transcription`
      (JS-only variant; Python SDK mirrors the same parameter names —
      verified interactively with `inspect.signature` on `TranscriptionConfig`
      and `aai.Lemur.task`).

### A2. Implement `pipelines/assemblyai.py` — done
- [x] Class `AssemblyAIPipeline` (duck-types `MeetingPipeline`), `name = "assemblyai"`.
- [x] `run(audio_path, run_dir, *, language="nl")` performs:
  1. Reads `ASSEMBLYAI_API_KEY` via `config.get_settings()` / `require()` and sets
     `aai.settings.api_key`.
  2. Builds `aai.TranscriptionConfig` with:
     - `speech_model=aai.SpeechModel.universal`.
     - `language_code=<language>` (default `"nl"`).
     - `speaker_labels=True`, `punctuate=True`, `format_text=True`.
     - `auto_chapters`, `summarization`, `summary_model=conversational`,
       `summary_type=bullets_verbose` — **only when `language` is English**
       (gated via `is_english = language.lower().startswith("en")`).
  3. `aai.Transcriber(config=config).transcribe(str(audio_path))` — the SDK
     uploads the local file and polls to completion internally. Errors are
     surfaced via `transcript.status == TranscriptStatus.error`.
  4. `_to_canonical_transcript()` maps `utterances` → `Segment`s and
     `utterance.words` → `Word`s. Speaker labels are normalised to
     `SPEAKER_<X>` and collected in `Transcript.speakers` in first-appearance
     order. Millisecond timestamps are converted to seconds.
  5. `_summarize_with_lemur()` builds the canonical `Summary` (see A3). When
     `is_english` and the native chapters/summary are populated, they enrich
     `summary.topics` / `summary.tldr` as a fallback.

### A3. Structured summary via LeMUR — done
- [x] Calls `transcript.lemur.task(prompt=..., final_model=claude_sonnet_4_20250514,
      temperature=0.0, max_output_size=4000)` with a strict Dutch (or English)
      JSON prompt covering `title`, `tldr`, `topics`, `decisions`,
      `action_items (task, owner, due)`, `open_questions`, `next_steps`.
- [x] Robust JSON extraction (`_extract_json`) tolerates fenced blocks and
      surrounding prose; on complete parse failure the pipeline still produces
      output files with a sentinel title + empty lists rather than crashing.
- [x] Raw LeMUR response (including `request_id`) is persisted in
      `meta.json` under `extra.lemur_raw`.
- **Scope change vs. original plan:** LeMUR now produces the *whole* summary
  for Dutch meetings (tldr + topics + decisions + actions + questions +
  next_steps), because AssemblyAI's native summarization does not support
  Dutch. This matches the project's top priority (summary quality).

### A4. Speaker name resolution (optional) — deferred
- [ ] Not implemented yet. Plan remains valid: after a real Dutch recording
      confirms baseline quality, add a small LeMUR pass over the first
      ~3 minutes to resolve `SPEAKER_A → "Casper"` etc. and rewrite
      `Transcript.segments[*].speaker` + `Word.speaker` in place.

### A5. Outputs — done
- [x] `io.write_run(run_dir, result)` writes `transcript.json`,
      `transcript.md`, `summary.json`, `summary.md`, `meta.json`.
- [x] `render_transcript_md` emits `[mm:ss → mm:ss] Speaker: text` blocks.
- [x] `render_summary_md` uses Dutch headers (*Samenvatting*, *Onderwerpen*,
      *Beslissingen*, *Actiepunten*, *Open vragen*, *Volgende stappen*) when
      `summary.language` starts with `nl`; English headers otherwise.

### A6. CLI wiring — done
- [x] `uv run meetings run --backend assemblyai --audio <path> [--language nl]`.
- [x] Default `--backend assemblyai`, default `--language nl`. CLI prints the
      `run_dir` on completion and a `Done.` banner.

### A7. Smoke test — done (sample audio pending)
- [x] `tests/test_assemblyai_pipeline.py` contains:
  - Pure-function unit tests for `_extract_json`, `_parse_action_items`,
    `_parse_decisions` (always run; no network).
  - `test_assemblyai_pipeline_smoke`: live end-to-end test that asserts
    non-empty `Transcript.segments`, ≥ 1 speaker, ≥ 1 `Summary.tldr` bullet,
    and presence of all five output files. Auto-skips when
    `ASSEMBLYAI_API_KEY` is missing **or** when no sample audio is found.
- [ ] Drop a 30–60 s Dutch sample at `audio/test_sample/sample_nl_short.wav`
      (gitignored) to enable the live run. Alternative accepted locations:
      `audio/raw/sample_nl_short.wav`,
      `audio/processed/sample_nl_short.16k.mono.wav`.

## Acceptance criteria

- [x] One command produces all five output files for a real Dutch meeting.
      (Verified structurally; awaits first real-meeting run.)
- [x] `transcript.json` validates against the Pydantic schema
      (`tests/test_schema_roundtrip.py` + runtime `model_dump_json`).
- [x] `summary.md` is in Dutch, well-formatted, and contains action items with
      owners where available (driven by the LeMUR JSON → `Summary` mapping
      and `io.render_summary_md`).
- [x] No API keys committed — all keys read from `.env` via `pydantic-settings`.

## How to run

```powershell
# 0. Ensure ASSEMBLYAI_API_KEY is set in .env (see .env.example).

# 1. (Optional but recommended) preprocess the recording per plan 05.
ffmpeg -y -i "audio/raw/<name>.m4a" -ac 1 -ar 16000 -c:a pcm_s16le `
  "audio/processed/<name>.16k.mono.wav"

# 2. Run the pipeline.
uv run meetings run --backend assemblyai `
  --audio "audio/processed/<name>.16k.mono.wav" --language nl
```

Outputs land in `Transcription/<audio-stem>__assemblyai__<utc-timestamp>/`.

## Verification commands

```powershell
uv run ruff check src tests     # clean
uv run mypy src                 # clean (assemblyai import-untyped is ignored)
uv run pytest -q                # 11 passed, 1 skipped (live smoke until sample is added)
```

## Follow-ups

- Drop the 30–60 s Dutch sample to unlock the live smoke test.
- Implement A4 (speaker name resolution) once baseline output is reviewed on a
  real meeting.
- Consider a `meetings preprocess` sub-command wrapping the ffmpeg step from
  `plan/05-audio-preprocessing.md` so Track A can take raw inputs directly.
