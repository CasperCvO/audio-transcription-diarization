# 02 — Track B: Custom Best-Quality Pipeline

**Depends on:** `03-repo-modernization.md`. Independent of Track A but reuses
the canonical schema from `00-architecture.md`.

**Status:** *partially implemented* (April 2026). The pipeline runs end-to-end
via `meetings run --backend custom` with OpenAI ASR + pyannoteAI + Claude.
The **2026 refresh** — Scribe v2 default, transcribe-only CLI, and shared
relabel UX — is tracked below as tasks B10–B13. See
[Implementation status](#implementation-status) for the per-task checklist,
file map, and known deviations.

## Objective

Compose **separate best-in-class models** for each stage to maximize
**summary quality** on Dutch meetings. APIs preferred (no local GPU). This is
a learning project — the agent should make the seams obvious so individual
stages can be swapped without touching the rest.

### 2026 reality check (why the defaults are changing)

The original Track B thesis (compose Whisper + pyannote + Claude) was written
before the 2025/2026 ASR step-change. Per the **Artificial Analysis AA-WER
leaderboard** (late 2026, https://artificialanalysis.ai/speech-to-text):

1. **ElevenLabs Scribe v2 — 2.3 %** (best). 90+ languages incl. Dutch, native
   diarization (up to 32 speakers), word-level timestamps, single-upload
   long-file handling, key-term prompting.
2. Gemini 3 Pro (High) — 2.9 %. Up to ~8 h audio per request via the Gemini API.
3. Voxtral Small (Mistral) — 2.9 %. Open weights.
4. Gemini 3.1 Pro Preview — 2.9 %.
5. MAI-Transcribe-1 (Azure) — 3.0 %.

OpenAI Whisper / `gpt-4o-transcribe` no longer rank in the top tier **and**
still cap uploads at 25 MB, which makes any 2 h meeting a chunking exercise.
The new Track B default ASR is therefore **ElevenLabs Scribe v2**; OpenAI is
demoted to short-files only (≤ ~30 min at 16 kHz mono).

The "best-of-breed" thesis narrows but does not disappear: **pyannoteAI
Precision-2** still beats Scribe v2's built-in diarization on real-world
meeting audio with overlap (vendor-reported ~28 % DER improvement vs
open-source SOTA). So Track B's strongest 2026 composition is
**Scribe v2 (ASR) + pyannoteAI Precision-2 (diarization) + Claude/Gemini
(summary)**. A simpler "Scribe-only" composition (Scribe v2 emits both ASR
and diarization) is the documented alternative — A/B them on the test
samples before committing to the 2 h run.

## Stage choices (defaults — 2026 refresh)

The defaults below are starting points. Each module must be swappable.

| Stage | Default | Reason | Alt to try |
|-------|---------|--------|------------|
| Audio prep | `ffmpeg` to 16 kHz mono `pcm_s16le` WAV (no loudnorm — see plan 05) | Predictable input, no DSP that hurts WER | — |
| Transcription | **ElevenLabs Scribe v2** (`scribe_v2`) | Top of the AA-WER leaderboard (2.3 %); native word-level timestamps + diarization; **single upload, no chunking** for the 2 h meeting | Gemini 3 Pro audio (reuses `GOOGLE_API_KEY`, ~8 h per request); OpenAI `whisper-1` / `gpt-4o-transcribe` (≤ 25 MB only — short files); Deepgram Nova-3 |
| Diarization | **pyannoteAI Premium API** (`precision-2` model) | SOTA diarization, no GPU needed, ~28 % DER edge over open-source SOTA on overlap-heavy audio | Scribe v2 built-in diarization (one less vendor; competitive on clean 2-speaker audio); local `pyannote/speaker-diarization-3.1` (CPU-slow) |
| Word-speaker alignment | Custom `align.py` | Combine word timestamps + diarization turns | — |
| Transcript cleanup (optional) | Claude Sonnet 4.5 lightweight pass | Fix Dutch punctuation, remove fillers, keep timestamps | Skip for purity |
| Summarization | **Claude Sonnet 4.5** (`claude-sonnet-4-5-20250929`) with map-reduce + critique | Strongest long-context structured Dutch output | Gemini 2.5 Pro / Gemini 3 Pro (single-call, factory hook to add) |

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

### B2. Transcription module — **default changed to Scribe v2 (2026)**

#### B2a. `transcribe/elevenlabs_scribe.py` — new default *(completed)*
- [x] Replace the current stub with a working `ElevenLabsTranscriber`
      (`name="elevenlabs"`, `model="scribe_v2"`).
- [x] Use the official `elevenlabs` Python SDK; auth via `ELEVENLABS_API_KEY`
      from `Settings`. Add `elevenlabs>=1.0` to `pyproject.toml`.
- [x] Single-shot upload of the canonical 16 kHz mono WAV. Request
      `diarize=True`, `timestamps_granularity="word"`, `language_code=language`.
- [x] Map Scribe's response → canonical `Transcript`:
      - `words[]` → `Word(text, start, end, speaker, confidence)`.
      - Speaker labels normalised to `SPEAKER_<N>` (in first-appearance order).
      - Group words into `Segment`s on speaker change + sentence-final
        punctuation + `silence_gap > 0.7 s` (reuse `align.group_into_segments`).
      - `Transcript.backend_meta` records `model="scribe_v2"`,
        `had_word_timestamps=True`, `diarization_source="elevenlabs_builtin"`.
- [x] When the user selects `--diarizer pyannoteai` (default), Scribe's
      built-in speaker labels are **discarded** and re-assigned via
      `align.assign_speakers` against the pyannoteAI turns. This keeps the
      "best-of-breed" composition. Record `diarization_source="pyannoteai"`
      in `backend_meta` in that case.
- [x] Tests in `tests/test_elevenlabs_scribe.py`: response-shape parsing
      (using a fixture JSON, no network), speaker label normalisation, and
      segment grouping correctness. Live smoke skipped when
      `ELEVENLABS_API_KEY` is absent.

#### B2b. `transcribe/openai_gpt4o.py` — demoted to short-files-only
- [x] Class `OpenAITranscriber` exposing
      `transcribe(audio, *, language="nl") -> Transcript` with word-level
      timestamps populated.
- [x] Native word timestamps when `model="whisper-1"` (via
      `response_format="verbose_json"`,
      `timestamp_granularities=["word", "segment"]`).
- [x] Fallback: when the selected model omits `words` (e.g.
      `gpt-4o-transcribe`), per-segment text is split into tokens and word
      timestamps are synthesized by proportional alignment weighted by token
      length.
- [ ] Add an explicit pre-flight size check: if the input WAV is > 24 MB,
      raise a clear `TranscribeError` pointing at `--transcriber elevenlabs`
      / `--transcriber gemini_audio` instead of silently failing on the
      OpenAI 25 MB upload limit.
- [ ] **Silence-based chunking for long files is dropped from the roadmap.**
      In 2026 the long-file path goes through Scribe v2 / Gemini 3 Pro /
      AssemblyAI — all single-upload — so chunking + boundary-word
      reconciliation is no longer worth implementing here.

#### B2c. `transcribe/gemini_audio.py` — new alternative *(task, optional)*
- [ ] New module `GeminiAudioTranscriber` (`name="gemini_audio"`,
      `model="gemini-3-pro"`). Reuses the existing `google-genai` SDK and
      `GOOGLE_API_KEY` already in `Settings`.
- [ ] Upload via `client.files.upload(...)` then
      `client.models.generate_content(...)` with a structured-output prompt
      that returns JSON with words + word timestamps + speaker labels.
- [ ] Verify Dutch quality + timestamp granularity on a 5-min sample
      *before* trusting it on the 2 h run. Treat as alternative, not default.

#### B2d. `transcribe/__init__.py` — factory + protocol
- [x] `Transcriber` Protocol in `transcribe/base.py`.
- [x] Update `get_transcriber(name)` so the **default name is `elevenlabs`
      / `scribe_v2`** and OpenAI options are still selectable.
      Accepts: `elevenlabs` (default) | `scribe_v2` | `scribe` | `gemini_audio` | `whisper-1` |
      `openai_gpt4o` | `deepgram` (stub).
- [x] Stubs `DeepgramTranscriber` raise `NotImplementedError` until wired.

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

### B5. Speaker naming

**2026 update:** the manual snippet + `speakers.json` UX from Track A is now
the canonical relabel flow for Track B as well (see B12). The auto-resolve
path below is kept as a convenience for unattended runs.

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
- [x] **2026:** add a `transcribe_only(audio, run_dir, *, language)` entry
      point that runs prepare → transcribe → diarize → align and writes
      `transcript.json/.md`, `speakers.json` skeleton, and per-speaker audio
      snippets, then **stops** (no summarize). Mirrors
      `AssemblyAIPipeline.transcribe`. Unlocks the human-in-the-loop relabel
      flow for Track B (B12) and the long-file CLI run from B11.
- [ ] **Not yet wired:** the optional transcript-cleanup stage from the
      stage-choices table. Constructor arg `cleanup` is accepted but unused —
      add a `Cleaner` Protocol and stage when desired.

### B8. CLI wiring
- [x] `meetings run --backend custom --audio audio/processed/foo.wav` uses defaults.
- [x] Flags:
      `--transcriber {openai_gpt4o,whisper-1,elevenlabs,deepgram}`,
      `--diarizer {pyannoteai,pyannote_local}` (default `pyannoteai`),
      `--summarizer {claude}` (default `claude`),
      `--cleanup/--no-cleanup`,
      `--name-resolution/--no-name-resolution`.
- [x] **2026 update:** flip the `--transcriber` default to `elevenlabs`
      (Scribe v2) and add `gemini_audio` as a documented choice.
- [x] Bonus: `meetings preprocess SRC` wraps `prepare_audio` per plan 05.
- [ ] `--summarizer gemini` for Track B: add a thin wrapper around
      `summarize/gemini_batch.py` (single-call, already used by Track A) so
      the custom pipeline can also be summarized with Gemini for A/B testing
      against Claude's map-reduce. Updates `get_summarizer` to dispatch
      `claude` → map-reduce, `gemini` → single-call batch.
- ⚠️ `--summarizer openai` is still not implemented; `get_summarizer` raises
      `ValueError` for unknown names.

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
- [x] `tests/test_elevenlabs_scribe.py` — fixture-based parsing of a Scribe v2
      response into `Transcript`, plus speaker normalisation and segment
      grouping. Live smoke skipped without `ELEVENLABS_API_KEY`.
- [ ] `tests/test_custom_transcribe_only.py` — exercise the new
      `transcribe_only` entry point with fake stages; assert that
      `transcript.json`, `speakers.json`, and `snippets/SPEAKER_*.wav` exist
      and that **no** `summary.json` is written.
- [ ] **Live integration test** on the same short Dutch sample as Track A:
      not yet added. Reuse Track A's smoke-test pattern: skip when
      `ELEVENLABS_API_KEY` / `PYANNOTEAI_API_KEY` / `ANTHROPIC_API_KEY` are
      absent.

### B10. Long-file readiness — the 2 h Dutch meeting *(task)*

Goal: run the full 2 h recording through Track B from the CLI without
babysitting a notebook kernel and without manual chunking.

- [ ] Confirm the canonical input (`audio/processed/<name>.16k.mono.wav`,
      ~230 MB at 2 h) goes through Scribe v2 in a **single upload**. Document
      the upper bound observed in `backend_meta`.
- [ ] If pyannoteAI is selected as the diarizer, confirm the pre-signed PUT
      flow handles the same file size (it should — pyannoteAI v1 is built on
      object-storage uploads).
- [ ] Add a wall-clock budget assertion in the CLI (`--max-stage-seconds`)
      so a runaway transcribe / diarize job aborts cleanly instead of
      hanging the terminal.
- [ ] Acceptance: `meetings transcribe --backend custom --audio
      audio/processed/<2h-meeting>.16k.mono.wav` produces
      `transcript.json/.md`, `speakers.json` skeleton, and
      `snippets/SPEAKER_*.wav` in one shot.

### B11. CLI symmetry: `meetings transcribe --backend custom` *(completed)*

Today `meetings transcribe` is hard-wired to AssemblyAI (Track A stage 1).
Make it backend-aware so Track B has the same human-in-the-loop entry point.

- [x] Add `--backend {assemblyai|custom}` to `meetings transcribe`
      (default `assemblyai` for backwards compat).
- [x] When `--backend custom`, dispatch to
      `CustomPipeline.transcribe_only(...)` from B7.
- [x] Forward the relevant flags: `--transcriber`, `--diarizer`, `--language`,
      `--snippets`. Reject Track-A-only flags with a clear error.
- [x] Update `cli.py` help text to document the new shape.

### B12. Shared relabel UX *(task)*

Reuse Track A's snippets + `speakers.json` workflow for Track B verbatim.

- [x] Generalise `snippets.extract_speaker_snippets(transcript, audio,
      run_dir, ...)` to accept any `Transcript` (it already does — confirm).
- [x] Have `CustomPipeline.transcribe_only` write the `speakers.json`
      skeleton via `speakers.write_skeleton` and the per-speaker WAVs via
      `snippets.extract_speaker_snippets`, identical to Track A.
- [x] `meetings relabel <run_dir>` is **already** backend-agnostic (it only
      reads `transcript.json` + `speakers.json`). Verify with a Track-B run
      and add a regression test (`tests/test_relabel_custom.py`).
- [ ] Keep the existing `--name-resolution` auto-resolve flag as an
      unattended alternative; document in the README that the manual flow is
      now the recommended path for both tracks.

### B13. Track-B summarize via Gemini *(task)*

Make `--summarizer gemini` work for Track B too, so Claude vs Gemini can be
A/B-tested inside the custom pipeline (not only across tracks).

- [x] Extend `summarize/__init__.py::get_summarizer` to return
      `GeminiSummarizer` (single-call batch, reuses `gemini_batch.py`) when
      `name == "gemini"`.
- [x] Document: Track B with Claude uses **map-reduce + critique**; Track B
      with Gemini uses **single-call** (Gemini 2.5 Pro / 3 Pro have enough
      context for a 2 h transcript in one pass — same rationale as Track A).
- [x] Test: `tests/test_custom_summarizer_gemini.py` — full run with a fake
      `GeminiSummarizer`, asserting `Summary.summarizer_backend.startswith("gemini")`.

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
- [ ] **2026 acceptance:** `meetings transcribe --backend custom --audio
      <2h>.wav` followed by manual `speakers.json` editing, then
      `meetings relabel <run_dir>` and `meetings summarize <run_dir>
      --summarizer {claude|gemini}` produces the same five output files for
      a real Dutch 2 h meeting using **Scribe v2 + pyannoteAI + Claude/Gemini**.

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

- **`ELEVENLABS_API_KEY`** — transcription (default, Scribe v2).
- `OPENAI_API_KEY` — transcription (only when `--transcriber whisper-1`/`openai_gpt4o`, ≤ 25 MB).
- `GOOGLE_API_KEY` — transcription (`--transcriber gemini_audio`) **and** summarization (`--summarizer gemini`).
- `PYANNOTEAI_API_KEY` — diarization (Premium API, default).
- `ANTHROPIC_API_KEY` — summarization (default Claude) and optional speaker-name auto-resolution.
- `HF_TOKEN` — only when `--diarizer pyannote_local` is selected.

### Known deviations from this plan

1. **No loudnorm in audio prep.** Plan 05 supersedes the original B1 wording.
2. **Implemented default ASR is currently `whisper-1`.** The 2026 refresh
   (B2a) flips the default to `elevenlabs` / Scribe v2; until that ships,
   anything > ~25 MB needs `--transcriber elevenlabs` once it's wired.
3. **Critique applies only additive patches.** Field corrections are logged
   for review but not auto-applied. Tighten this in `_apply_critique` if you
   want stricter behaviour.
4. **No structured-output / tool-use** on Anthropic calls yet — a tolerant
   JSON parser (`summarize/_utils.parse_json`) handles fenced / prose-wrapped
   responses. Migrating is a low-risk follow-up.
5. **`--summarizer openai` is a placeholder.** `--summarizer gemini` is wired
   for Track A today and added for Track B in B13. The factory raises a
   clear error for other unknown names.
6. **OpenAI silence-chunking dropped from the roadmap.** With Scribe v2 /
   Gemini 3 Pro / AssemblyAI all handling the full 2 h file in a single
   upload, implementing chunking is no longer worth the boundary-error cost.
   OpenAI ASR stays available for ≤ ~30 min recordings only.
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
