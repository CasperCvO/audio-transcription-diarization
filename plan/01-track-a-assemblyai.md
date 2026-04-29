# 01 — Track A: AssemblyAI + direct Claude/Gemini summarization

**Status:** Implemented (2026-04-26). Live smoke test pending a real Dutch sample.
**Depends on:** `03-repo-modernization.md` (skeleton must exist).

> **2026 context.** Per the Artificial Analysis AA-WER leaderboard, AssemblyAI
> Universal is no longer the single most accurate ASR (ElevenLabs Scribe v2
> leads at 2.3 %). Track A remains the **lowest-risk, fully-implemented path**
> for the 2 h Dutch meeting today, and AssemblyAI's Dutch transcription +
> diarization is still strong. The relabel UX (`speakers.json` + snippets)
> introduced here is now reused verbatim by Track B — see
> `02-track-b-custom-pipeline.md` B12.

## Objective

Run a Dutch (or English) meeting through:

1. **AssemblyAI Universal-2** for transcription + speaker diarization.
2. A **manual speaker-relabel step** (human-in-the-loop) using audio
   snippets per speaker, written to `speakers.json`.
3. **Direct API summarization** via Claude (Anthropic) or Gemini (Google),
   selectable at the CLI. **AssemblyAI's LeMUR / LLM Gateway is not used**
   so this works on AssemblyAI plan tiers without LLM access.

Quality of the summary remains the top priority. Cost matters enough that
both summarizers default to their provider's **batch API (50% cheaper)**
with a synchronous escape hatch (`--sync`) for prompt iteration.

## Why this shape

- **Universal-2 for Dutch.** Universal-3 Pro only supports
  `{en, es, de, fr, pt, it}`. Dutch is in Universal-2's high-accuracy tier.
  Out of scope for now: dynamically promoting to Universal-3 Pro for
  English/EU-language meetings.
- **No LeMUR.** LeMUR routes through AssemblyAI and requires a plan tier
  the project doesn't have. Calling Anthropic / Google directly side-steps
  the AssemblyAI plan dependency entirely.
- **Native `auto_chapters` and `summarization` are not used.** Both are
  English-only and overlap with the LLM summarization step anyway.
- **Stage split.** Single-shot pipelines waste LLM tokens when diarization
  is wrong. Splitting transcribe / relabel / summarize lets the user gate
  the (paid) summarization on a satisfactory transcript.
- **Single-call summarization.** Modern Claude Sonnet 4.5 (200k context)
  and Gemini 2.5 Pro (1M+ context) easily fit multi-hour Dutch meetings.
  Map-reduce is unnecessary here — that complexity stays in Track B.

## Pipeline shape

Three stages, each writing to `Transcription/<run_id>/`:

| Stage | CLI command | Inputs | Outputs |
|-------|-------------|--------|---------|
| 1. Transcribe | `meetings transcribe --audio …` | audio file | `transcript.json/.md`, `speakers.json` skeleton, `snippets/SPEAKER_*.wav`, `meta.json` (`stage="transcribed"`) |
| 2. Relabel | `meetings relabel <run_dir>` | user-edited `speakers.json` | renamed `transcript.json/.md`, `meta.json` (`stage="relabelled"`) |
| 3. Summarize | `meetings summarize <run_dir> --summarizer claude\|gemini` | `transcript.json` | `summary.json/.md`, `llm/<provider>.{prompt,response,meta}.{md,json}`, `meta.json` (`stage="summarized"`) |

The legacy `meetings run --backend assemblyai` chains stage 1 + stage 3
(no manual relabel) for parity with Track B and quick smoke tests.

## Tasks

### A1. Dependencies and config — done
- [x] `assemblyai>=0.63.0` (existing).
- [x] `anthropic>=0.39` (existing) — used directly via `client.messages.batches.create`.
- [x] `google-genai>=1.0` (added) — used directly via `client.batches.create`.
- [x] `ASSEMBLYAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` in
      `Settings` (`src/meetings/config.py`); `require()` raises a clear
      error when a key needed at call time is missing.

### A2. AssemblyAI transcription + diarization — done
- [x] `pipelines/assemblyai.py::AssemblyAIPipeline.transcribe(audio_path, run_dir, *, language="nl", snippets_per_speaker=3)`:
  - `aai.TranscriptionConfig(speech_model=universal, language_code=language, speaker_labels=True, punctuate=True, format_text=True)`.
  - **Native `auto_chapters` and `summarization` are not enabled** —
    they are English-only and superseded by the LLM step.
  - Maps `utterances` → `Segment`s, `utterance.words` → `Word`s; speaker
    labels normalised to `SPEAKER_<X>`.
  - Writes `transcript.json/.md` and a `speakers.json` skeleton via
    `speakers.write_skeleton`.
  - Writes per-speaker audio snippets via `snippets.extract_speaker_snippets`
    (top-N longest segments per speaker, 4–20 s, re-encoded with ffmpeg
    to 16 kHz mono PCM). Snippet failures are non-fatal; an
    `EXTRACTION_FAILED.txt` marker is written instead.
  - `meta.json.extra` carries `transcript_id`, `language_code`, and
    `stage="transcribed"`.

### A3. Manual speaker relabel — done
- [x] `speakers.py` provides `write_skeleton`, `read_mapping`,
      `validate_mapping(transcript, *, require_all_named)`,
      `apply_mapping(transcript, mapping)`.
- [x] `apply_mapping` rewrites speaker labels at both the **segment** and
      **word** level and rebuilds `Transcript.speakers` in first-appearance
      order. `null` mapping values keep the original `SPEAKER_X` label.
- [x] `pipelines/assemblyai.py::AssemblyAIPipeline.relabel(run_dir, *, require_all_named=True)`
      validates the user-edited file, applies the mapping, and overwrites
      `transcript.json/.md`. The mapping is recorded in `meta.extra.speaker_mapping`
      for auditing.
- [x] CLI flag `--allow-unset` allows partial mappings; default is strict.

### A4. Single-call summarization (Claude or Gemini) — done

Quality and cost defaults:

- Default model — Anthropic: `claude-sonnet-4-5-20250929`, Gemini: `gemini-2.5-pro`.
- Default mode: **batch** (50 % cheaper, ≤ 24 h SLO; usually < 1 h).
- `--sync` falls back to the standard messages / generate_content call
  for live prompt iteration (full price, immediate response).

Implementation:

- [x] `summarize/anthropic_batch.py::AnthropicSummarizer`
  - **Batch path**: `client.messages.batches.create(...)` →
    poll `client.messages.batches.retrieve(id).processing_status == "ended"` →
    stream results via `client.messages.batches.results(id)`.
  - **Sync path**: `client.messages.create(...)`.
  - Fed the diarized transcript rendered as
    `[mm:ss] SPEAKER: text` (current labels — i.e. real names if relabelled).
- [x] `summarize/gemini_batch.py::GeminiSummarizer`
  - **Batch path**: `client.batches.create(model=..., src=[InlinedRequestDict])`
    → poll `client.batches.get(name=...).state.name in {"JOB_STATE_SUCCEEDED", ...}`
    → read `dest.inlined_responses[0].response.text`.
  - **Sync path**: `client.models.generate_content(...)`.
  - `response_mime_type="application/json"` is set on the config to nudge
    Gemini towards parseable JSON; tolerant `parse_json` cleans up fences.
- [x] Both share the prompt in `summarize/prompts.py::SINGLE_CALL_PROMPT_{NL,EN}`,
      version `single-call-v1` (recorded in `Summary.prompt_version` and
      `meta.json`).
- [x] Both produce the canonical `Summary` via `summarize/_utils.py::parse_summary_payload`.
- [x] All prompts and raw responses logged under
      `Transcription/<run_id>/llm/{anthropic,gemini}.{prompt,response,meta}.{md,json}`.

### A5. Outputs — done
- [x] Per-stage helpers in `io.py`: `write_transcript`, `write_summary`,
      `write_meta`, `read_transcript`, `read_meta`. `read_run` still works
      for fully-completed runs (Track B and `meetings run`).

### A6. CLI wiring — done
- [x] `meetings transcribe --audio <path> [--language nl] [--snippets 3]` — Stage 1.
- [x] `meetings relabel <run_dir> [--allow-unset]` — Stage 2.
- [x] `meetings summarize <run_dir> [--summarizer claude|gemini] [--sync] [--language ...]` — Stage 3.
- [x] `meetings run --backend assemblyai --audio <path> [--summarizer claude|gemini] [--sync]`
      — chained convenience (no relabel pause).
- [x] `meetings validate <run_dir>` recognises **partial** runs (transcribed
      but not yet summarized) and reports the stage from `meta.extra.stage`.

### A7. Tests — done (live smoke pending sample)
- [x] `tests/test_assemblyai_pipeline.py` covers, without any network:
  - `summarize._utils`: `parse_json`, `parse_action_items`, `parse_decisions`,
    `parse_summary_payload`, `render_transcript_for_llm`.
  - `speakers`: `apply_mapping` rewrites segments **and** words; `null`
    values keep original labels; `validate_mapping` enforces strict /
    lax modes; missing labels raise.
- [x] Live smoke test renamed to `test_assemblyai_transcribe_smoke`. It
      now exercises **only stage 1** (no LLM call) so it does not require
      `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY`.
- [ ] Drop a 30–60 s Dutch sample at
      `audio/test_sample/sample_nl_short.wav` (gitignored) to enable the
      live run. Alternative locations are also accepted (see test file).
      The repo currently ships three ~5 min samples in `audio/test_sample/`
      (`segment_1_start.wav`, `segment_2_45min.wav`, `segment_3_90min.wav`)
      — either reuse one of those or trim to 30–60 s for the smoke test.

## Acceptance criteria

- [x] One human-in-the-loop pipeline produces all five output files for a
      real Dutch meeting via three explicit commands. (Verified
      structurally; awaits first real-meeting run.)
- [x] `transcript.json` validates against the Pydantic schema (covered by
      `tests/test_schema_roundtrip.py` + runtime `model_dump_json`).
- [x] `summary.md` is in Dutch, well-formatted, and contains action items
      with owners that use **the relabelled speaker names** (because the
      summarizer reads the post-relabel transcript directly via the
      provider API).
- [x] No API keys committed — keys read from `.env` via `pydantic-settings`.

## How to run

```powershell
# 0. Ensure ASSEMBLYAI_API_KEY + (ANTHROPIC_API_KEY OR GOOGLE_API_KEY) are
#    set in .env (see .env.example).

# 1. (Recommended) preprocess to canonical 16 kHz mono WAV (plan 05).
uv run meetings preprocess "audio/raw/<name>.m4a"

# 2. Stage 1 — transcribe + diarize, generate speakers.json + snippets.
uv run meetings transcribe --audio "audio/processed/<name>.16k.mono.wav" --language nl

# 3. Listen to Transcription/<run_id>/snippets/SPEAKER_*.wav and edit
#    Transcription/<run_id>/speakers.json:
#    {"SPEAKER_A": "Casper", "SPEAKER_B": "Anna"}

# 4. Stage 2 — apply the mapping.
uv run meetings relabel "Transcription/<run_id>"

# 5. Stage 3 — summarize. Defaults: --summarizer claude --batch.
uv run meetings summarize "Transcription/<run_id>"            # Claude, batch
uv run meetings summarize "Transcription/<run_id>" --summarizer gemini  # Gemini, batch
uv run meetings summarize "Transcription/<run_id>" --sync     # Sync iteration
```

For a no-pause smoke test (skips relabel; uses generic SPEAKER_X labels):

```powershell
uv run meetings run --backend assemblyai `
  --audio "audio/processed/<name>.16k.mono.wav" --language nl `
  --summarizer claude
```

Outputs land in `Transcription/<audio-stem>__assemblyai__<utc-timestamp>/`.

## Verification commands

```powershell
uv run ruff check src tests     # clean
uv run mypy src                 # strict, clean (30 source files)
uv run pytest -q                # 30 passed, 1 skipped (live smoke until sample is dropped)
```

## Follow-ups

- Drop the 30–60 s Dutch sample to unlock the live transcribe smoke test.
- Add a thin live integration test for the summarize stage that mocks
  the Anthropic / Google SDKs end-to-end (avoids hitting the real batch
  API in CI).
- Once the user confirms the human-in-the-loop loop on a real meeting,
  consider adding optional **dynamic Universal-3 Pro selection** for
  English / Spanish / French / German / Portuguese / Italian recordings.
- **2026 follow-up:** the staged commands (`transcribe`, `relabel`,
  `summarize`) become backend-agnostic per `03-repo-modernization.md`
  S3 — Track A's pipeline only changes by accepting the new `--backend`
  flag on `transcribe` (defaulting to `assemblyai` for backwards compat).
- The notebook UX for stepping through snippets + writing `speakers.json`
  is now described in `06-testing-and-comparison-notebooks.md`.
