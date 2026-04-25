# 02 — Track B: Custom Best-Quality Pipeline

**Depends on:** `03-repo-modernization.md`. Independent of Track A but reuses
the canonical schema from `00-architecture.md`.

**Status:** *implemented* (April 2026). The pipeline runs end-to-end via
`meetings run --backend custom`. See [Implementation status](#implementation-status)
at the bottom for the per-task checklist, file map, and known deviations.

## Objective

Compose **separate best-in-class models** for each stage to maximize
**summary quality** on Dutch meetings. APIs preferred (no local GPU). This is
a learning project — the agent should make the seams obvious so individual
stages can be swapped without touching the rest.

## Stage choices (defaults)

The defaults below are starting points. Each module must be swappable.

| Stage | Default | Reason | Alt to try |
|-------|---------|--------|------------|
| Audio prep | `ffmpeg` to 16 kHz mono `pcm_s16le` WAV (no loudnorm — see plan 05) | Predictable input, no DSP that hurts WER | — |
| Transcription | **OpenAI `whisper-1`** (`verbose_json` + `timestamp_granularities=["word","segment"]`) | Only OpenAI model that emits **word-level** timestamps required for diarization alignment | `gpt-4o-transcribe` (better Dutch text, no word timestamps — falls back to proportional synthesis); ElevenLabs Scribe v1; Deepgram Nova-3 |
| Diarization | **pyannoteAI Premium API** (`precision-2` model) | SOTA diarization, no GPU needed | Local `pyannote/speaker-diarization-3.1` (CPU-slow) |
| Word-speaker alignment | Custom `align.py` | Combine word timestamps + diarization turns | — |
| Transcript cleanup (optional) | Claude Sonnet 4.5 lightweight pass | Fix Dutch punctuation, remove fillers, keep timestamps | Skip for purity |
| Summarization | **Claude Sonnet 4.5** (`claude-sonnet-4-5-20250929`) with map-reduce + critique | Strongest long-context structured Dutch output | GPT-5 / Gemini 2.5 Pro (factory hooks ready, not yet implemented) |

> Before coding any stage, fetch live API docs:
> `chub search "<provider>"` then `chub get <id> --lang py`.

## Tasks

### B1. Audio preparation (`audio.py`)
- [x] `prepare_audio(src: Path, dst_dir: Path | None = None) -> Path` → 16 kHz
      mono `pcm_s16le` WAV via ffmpeg. **Idempotent.**
- [x] Reject files > 4 h with a clear `AudioPrepError`.
- [x] Compute and store sha256 + duration (via ffprobe) in `meta.json`
      (`audio_meta()` returns the canonical tuple).
- ⚠️ **Deviation from original plan:** loudnorm is *not* applied. Plan
      `05-audio-preprocessing.md` (written later as the canonical preprocessing
      contract) explicitly forbids loudness normalization, denoise, EQ, gating,
      and AGC because real-world ASR models degrade on pre-processed audio.
      The implementation follows plan 05.

### B2. Transcription module (`transcribe/openai_gpt4o.py`)
- [x] Class `OpenAITranscriber` exposing
      `transcribe(audio, *, language="nl") -> Transcript` with word-level
      timestamps populated.
- [x] Native word timestamps when `model="whisper-1"` (via
      `response_format="verbose_json"`,
      `timestamp_granularities=["word", "segment"]`).
- [x] Fallback: when the selected model omits `words` (e.g.
      `gpt-4o-transcribe`), per-segment text is split into tokens and word
      timestamps are synthesized by proportional alignment weighted by token
      length. The fallback path is documented inline and exercised by the
      `had_word_timestamps` flag in `Transcript.backend_meta`.
- [x] `Transcriber` Protocol in `transcribe/base.py`; factory
      `get_transcriber(name)` accepts `openai_gpt4o`/`whisper-1`/`elevenlabs`/`deepgram`.
- [x] Stubs `ElevenLabsTranscriber` and `DeepgramTranscriber` raise
      `NotImplementedError` until wired.
- [ ] **Not yet implemented:** silence-based chunking for long files
      (>~25 MB OpenAI upload limit). Tracked as a follow-up; the OpenAI SDK
      currently accepts files up to 25 MB which covers most ≤30 min meetings
      at 16 kHz mono.

### B3. Diarization module (`diarize/pyannoteai_api.py`)
- [x] `PyannoteAIDiarizer` implements the full pyannoteAI v1 flow:
      1. `POST /v1/media/input` with `{"url": "media://<key>"}` → pre-signed PUT URL.
      2. `PUT` raw bytes to that URL.
      3. `POST /v1/diarize` with `{model: "precision-2", exclusive: true, ...}` → `jobId`.
      4. Poll `GET /v1/jobs/{jobId}` every 8 s until `status` is terminal.
      5. Map `output.exclusiveDiarization` (preferred) or `output.diarization`
         → `list[DiarizationTurn]`.
- [x] All HTTP calls retried with `tenacity` (exp backoff, 3 attempts).
      Hard timeout: 1 h per job.
- [x] `PyannoteLocalDiarizer` provided as a lazy-import fallback that loads
      `pyannote.audio` only when explicitly selected via `--diarizer pyannote_local`.
- [x] Selection is via the `--diarizer` CLI flag and the `get_diarizer(name)`
      factory (not an env var).

### B4. Word-level speaker alignment (`align.py`)
- [x] `assign_speakers(words, turns)` picks the turn with max temporal
      overlap; tie-breaks by nearest turn center.
- [x] `group_into_segments(words, silence_gap=0.7)` splits on speaker change,
      gap > 0.7 s, or sentence-final punctuation.
- [x] Unit tests in `tests/test_align.py` and `tests/test_align_extra.py`
      cover pure overlap, no-overlap nearest-center fallback, speaker change
      mid-sentence, tiny words inside long turns, sentence-punctuation splits,
      and empty-input handling.

### B5. Speaker naming (optional)
- [x] `summarize/names.resolve_speaker_names(transcript, window_seconds=300)`
      sends the first ~5 min to Claude and parses a JSON mapping.
- [x] `apply_name_mapping(transcript, mapping)` immutably rewrites speaker
      labels on every `Word` and `Segment` and updates `Transcript.speakers`.
- [x] Toggled by the CLI flag `--name-resolution`.
      The mapping is recorded indirectly via the rewritten transcript; the raw
      prompt + response are saved to `Transcription/<run_id>/llm/names.*.md`.
- [ ] **Follow-up:** also persist the explicit mapping under `meta.extra` for
      easy auditing without re-parsing the transcript.

### B6. Summarization module (`summarize/claude.py`)

Quality is the priority. Implement a **map-reduce + critique** strategy:

1. **Map** — split transcript into ≤ 8k-token windows on speaker boundaries
   (never mid-sentence). For each window, ask Claude (Dutch system prompt) to
   produce a JSON with: `local_topics`, `local_decisions`, `local_actions`,
   `local_questions`, `quotes` (with `segment_idx`).
2. **Reduce** — feed all local JSONs back to Claude with a consolidation
   prompt to produce the final `Summary` JSON. Deduplicate decisions and
   merge action items by `(owner, task)` similarity.
3. **Critique pass** — second Claude call: "You are a meticulous reviewer.
   Given the transcript and this summary, list missing decisions, missing
   action items, or hallucinated content. Return JSON." Apply patches.
4. **Render** — convert final `Summary` to `summary.md` (Dutch) and `summary.json`.

Implementation status:
- [x] Map → Reduce → Critique pipeline in `summarize/claude.py::ClaudeSummarizer`.
- [x] Windowing on segment boundaries (default ~24k chars), never mid-sentence.
- [x] All prompts live in `summarize/prompts.py`; `PROMPT_VERSION` (`v0.1.0`)
      baked into `Summary.prompt_version` and `meta.json`.
- [x] Temperatures: `0.2` map+reduce, `0.0` critique.
- [x] Diarized transcript (`[idx] HH:MM:SS SPEAKER: text`) is what gets sent —
      not raw text — so action-item attribution can resolve owners.
- [x] All prompts and raw responses logged under
      `Transcription/<run_id>/llm/{map_NN,reduce,critique}.{prompt,response}.md`.
- [x] Critique pass applies **only additive patches** (missing decisions /
      missing actions). Field-level corrections from the critique are saved
      to disk for human review but **not auto-applied**, to avoid silent
      regressions on otherwise-good drafts. This was a deliberate choice; if
      you want stricter behaviour, extend `_apply_critique` in `claude.py`.
- [ ] Anthropic structured-output / tool-use to force valid JSON: not yet
      adopted. Current implementation uses a tolerant `parse_json` that strips
      code fences and surrounding prose. Tests in `tests/test_summarize_helpers.py`
      cover the parser. Migrating to tool-use is a low-risk follow-up.

### B7. Pipeline composition (`pipelines/custom.py`)
- [x] `CustomPipeline(transcriber, diarizer, summarizer, cleanup=False,
      name_resolution=False)` accepts either factory names (strings) **or**
      pre-built instances (handy for tests).
- [x] `run()` orchestrates: prepare → transcribe → diarize → align →
      (optional name resolution) → summarize → `write_run`.
- [x] Each stage timed via a `_stage` context manager; timings stored in
      `RunMeta.timings` keyed by stage name.
- [x] `Transcript.backend` is set to `custom:<asr>+<diar>+<sum>` so a run is
      self-describing.
- [ ] **Not yet wired:** the optional transcript-cleanup stage from the
      stage-choices table. Constructor arg `cleanup` is accepted but unused —
      add a `Cleaner` Protocol and stage when desired.

### B8. CLI wiring
- [x] `meetings run --backend custom --audio audio/processed/foo.wav` uses defaults.
- [x] Flags:
      `--transcriber {openai_gpt4o,whisper-1,elevenlabs,deepgram}` (default `whisper-1`),
      `--diarizer {pyannoteai,pyannote_local}` (default `pyannoteai`),
      `--summarizer {claude}` (default `claude`),
      `--cleanup/--no-cleanup`,
      `--name-resolution/--no-name-resolution`.
- [x] Bonus: `meetings preprocess SRC` wraps `prepare_audio` per plan 05.
- ⚠️ `--summarizer openai` and `--summarizer gemini` are not implemented yet;
      `get_summarizer` raises `ValueError` for unknown names.

### B9. Tests
- [x] Unit tests for `align.py` in `tests/test_align.py` and
      `tests/test_align_extra.py` (fast, deterministic).
- [x] Schema round-trip test in `tests/test_schema_roundtrip.py`.
- [x] `tests/test_audio.py` — sha256 known-vector + missing-input error path.
- [x] `tests/test_summarize_helpers.py` — JSON extractor + timestamp formatter.
- [x] `tests/test_custom_pipeline.py` — full pipeline run with fake stages
      (no API keys required). Verifies all 5 output files, schema round-trip,
      speaker assignment via diarization, backend label, and that timings are
      recorded for every stage.
- [ ] **Live integration test** on the same short Dutch sample as Track A:
      not yet added. Reuse Track A's smoke-test pattern: skip when
      `OPENAI_API_KEY` / `PYANNOTEAI_API_KEY` / `ANTHROPIC_API_KEY` are
      absent.

## Acceptance criteria

- [x] `meetings run --backend custom --audio …` produces the same five output
      files as Track A (`transcript.json`, `transcript.md`, `summary.json`,
      `summary.md`, `meta.json`) plus an `llm/` subdirectory with prompts and
      responses. Verified by `tests/test_custom_pipeline.py`.
- [x] Each stage is independently swappable via a CLI flag and a factory in
      the corresponding sub-package.
- [ ] `summary.md` in Dutch with action-item ownership: ready end-to-end but
      not yet validated on a real Dutch meeting (pending live run with API
      keys; see B9 follow-up).
- [x] All LLM prompts and responses are logged under
      `Transcription/<run_id>/llm/` for inspection.

## Implementation status

### File map

| Stage | Module | Class / Entry point |
|-------|--------|---------------------|
| Audio prep | `src/meetings/audio.py` | `prepare_audio`, `audio_meta` |
| Transcription Protocol | `src/meetings/transcribe/base.py` | `Transcriber` |
| Transcription factory | `src/meetings/transcribe/__init__.py` | `get_transcriber` |
| OpenAI ASR | `src/meetings/transcribe/openai_gpt4o.py` | `OpenAITranscriber` |
| ElevenLabs ASR (stub) | `src/meetings/transcribe/elevenlabs_scribe.py` | `ElevenLabsTranscriber` |
| Deepgram ASR (stub) | `src/meetings/transcribe/deepgram.py` | `DeepgramTranscriber` |
| Diarization Protocol | `src/meetings/diarize/base.py` | `Diarizer` |
| Diarization factory | `src/meetings/diarize/__init__.py` | `get_diarizer` |
| pyannoteAI API | `src/meetings/diarize/pyannoteai_api.py` | `PyannoteAIDiarizer` |
| Local pyannote | `src/meetings/diarize/pyannote_local.py` | `PyannoteLocalDiarizer` |
| Word-speaker alignment | `src/meetings/align.py` | `assign_speakers`, `group_into_segments` |
| Summarizer Protocol | `src/meetings/summarize/base.py` | `Summarizer` |
| Summarizer factory | `src/meetings/summarize/__init__.py` | `get_summarizer` |
| Claude summarizer | `src/meetings/summarize/claude.py` | `ClaudeSummarizer` |
| Prompts | `src/meetings/summarize/prompts.py` | `PROMPT_VERSION = "v0.1.0"` |
| Speaker naming | `src/meetings/summarize/names.py` | `resolve_speaker_names`, `apply_name_mapping` |
| Pipeline | `src/meetings/pipelines/custom.py` | `CustomPipeline` |
| CLI | `src/meetings/cli.py` | `meetings run`, `meetings preprocess` |

### Required API keys (in `.env`)

- `OPENAI_API_KEY` — transcription.
- `PYANNOTEAI_API_KEY` — diarization (Premium API).
- `ANTHROPIC_API_KEY` — summarization and optional speaker-name resolution.
- `HF_TOKEN` — only when `--diarizer pyannote_local` is selected.

### Known deviations from this plan

1. **No loudnorm in audio prep.** Plan 05 supersedes the original B1 wording.
2. **Default ASR is `whisper-1`, not `gpt-4o-transcribe`.** Word timestamps
   are required by alignment; only `whisper-1` returns them natively in the
   OpenAI Audio API today. `gpt-4o-transcribe` is still available via
   `--transcriber openai_gpt4o` and falls back to proportional word synthesis.
3. **Critique applies only additive patches.** Field corrections are logged
   for review but not auto-applied. Tighten this in `_apply_critique` if you
   want stricter behaviour.
4. **No structured-output / tool-use** on Anthropic calls yet — a tolerant
   JSON parser (`summarize/_utils.parse_json`) handles fenced / prose-wrapped
   responses. Migrating is a low-risk follow-up.
5. **`--summarizer openai` and `--summarizer gemini` are placeholders** in
   the plan; only `claude` is implemented. The factory raises a clear error
   for unknown names.
6. **No long-file silence-chunking** in the OpenAI transcriber yet; meetings
   above ~25 MB at 16 kHz mono need to be sliced manually until this lands.
7. **`--cleanup` flag is accepted but unused.** Add a `Cleaner` Protocol and
   wire a stage when there's evidence cleanup actually improves Dutch summary
   quality on this dataset.

### Verification snapshot

At the time of this implementation:

- `uv run pytest -q` → **24 passed, 1 skipped** (the skipped one is Track A's
  live AssemblyAI smoke test, awaiting `audio/test_sample/sample_nl_short.wav`).
- `uv run ruff check .` → clean.
- `uv run mypy src` → strict, clean across 26 source files.
- `uv run meetings --help` lists `run`, `validate`, `preprocess`, `compare`.
