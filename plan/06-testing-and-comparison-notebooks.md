# 06 — Testing & Comparison Notebooks

**Depends on:** `01-track-a-assemblyai.md`, `02-track-b-custom-pipeline.md`,
`05-audio-preprocessing.md`. Runs **after** the staged CLI commands from
`03-repo-modernization.md` S3 are backend-agnostic (Track A done, Track B
tasks B11–B13 pending).

## Objective

Provide two notebooks that wrap the *interactive* parts of the workflow
without driving long-running jobs from a Jupyter kernel:

1. `notebooks/01_test_samples.ipynb` — smoke-test both pipelines on the
   three ~5 min Dutch samples in `audio/test_sample/`. Decide which
   transcriber + diarizer combination to use for the 2 h run.
2. `notebooks/02_metrics_and_review.ipynb` — load the run directories
   produced by the CLI for the full 2 h meeting, compute gross metrics
   (speakers, segments, words, durations), and write the manual
   `speakers.json` mapping while listening to per-speaker snippets.

The actual long-running work (full-meeting transcription, diarization,
summarization) **always runs from the CLI**, never from a notebook
kernel. The notebooks only orchestrate inspection, A/B review, and the
human-in-the-loop relabel step.

## End-to-end workflow (canonical)

The order assumes Scribe v2 + pyannoteAI for Track B (post-2026 refresh)
and AssemblyAI for Track A. Replace the `--backend` flag on
`meetings transcribe` to switch tracks.

```powershell
# 0. Preprocess raw recording → canonical 16 kHz mono PCM (plan 05).
uv run meetings preprocess "audio/raw/<meeting>.m4a"

# 1. Sanity-check both pipelines on the three 5-min samples.
#    See notebooks/01_test_samples.ipynb. It calls `meetings transcribe`
#    on each sample with --backend assemblyai and --backend custom and
#    renders the resulting transcripts side-by-side.

# 2. Once a track is chosen, run stage 1 on the full 2 h meeting from
#    the terminal. This is transcription + diarization only — no summary.
uv run meetings transcribe `
  --backend custom `
  --transcriber elevenlabs `
  --diarizer pyannoteai `
  --audio "audio/processed/<meeting>.16k.mono.wav" `
  --language nl

# 3. Inspect the run dir and write speakers.json.
#    Open notebooks/02_metrics_and_review.ipynb, point it at
#    Transcription/<run_id>/, listen to snippets/SPEAKER_*.wav, and
#    fill speakers.json with real names.

# 4. Apply the mapping.
uv run meetings relabel "Transcription/<run_id>"

# 5. Summarize — also from the terminal (batch mode, kernel-independent).
uv run meetings summarize "Transcription/<run_id>" --summarizer claude   # or gemini

# 6. Review summary.md / summary.json in
#    notebooks/02_metrics_and_review.ipynb (it auto-detects when
#    summary.json appears in the run dir and renders the comparison).
```

## Notebook 01 — `notebooks/01_test_samples.ipynb`

**Goal:** smoke-test both pipelines on the three samples in
`audio/test_sample/` (`segment_1_start.wav`, `segment_2_45min.wav`,
`segment_3_90min.wav`, ~9.6 MB / ~5 min each).

### Cells

1. **Setup** — load `dotenv`, instantiate `Settings`, list the three
   sample paths, sanity-check that they are 16 kHz mono PCM (call
   `audio.audio_meta` per file). If not, suggest running
   `meetings preprocess` first.
2. **Run AssemblyAI on all three samples** — invoke
   `AssemblyAIPipeline().transcribe(...)` directly (in-process, the
   samples are short enough that kernel time is fine). Write to a
   per-sample run dir under `Transcription/_test/aai__<sample>__<ts>/`.
3. **Run Track B on all three samples** — invoke
   `CustomPipeline(transcriber="elevenlabs", diarizer="pyannoteai")
   .transcribe_only(...)` once B11/B12 land. Write to
   `Transcription/_test/custom__<sample>__<ts>/`.
4. **Side-by-side render** — for each sample, print:
   - Number of speakers, segments, total words, duration.
   - Speaker-overlap % (sum of overlapping segment durations / total).
   - First 30 seconds of each transcript with speaker labels — verify
     diarization placement makes sense.
5. **Decision cell** — markdown cell where you note which combination
   you'll use for the 2 h run, e.g.
   `Track B: elevenlabs (scribe_v2) + pyannoteai`.

### Acceptance

- Notebook runs end-to-end on all three samples without errors when
  every required `*_API_KEY` is set in `.env`.
- Cells gracefully `print` and skip when a key is missing instead of
  crashing the kernel.
- Output cells contain enough detail for the user to make the Track A
  vs Track B decision before committing to the 2 h run.

## Notebook 02 — `notebooks/02_metrics_and_review.ipynb`

**Goal:** post-processing UI for the long-running CLI run. Two phases:
metrics + manual relabel (before summarization), then summary review
(after summarization).

### Inputs

- `RUN_DIR = Path("Transcription/<run_id>")` — set in the first cell.

### Cells

1. **Setup** — `RUN_DIR`, `read_transcript(RUN_DIR)`, `read_meta(RUN_DIR)`.
   Show `meta.extra.stage` so you know where in the workflow you are
   (`transcribed` / `relabelled` / `summarized`).
2. **Gross metrics** — print and chart:
   - Number of speakers (`len(transcript.speakers)`).
   - Number of segments and words (`len(transcript.segments)`,
     `sum(len(s.words) for s in transcript.segments)`).
   - Duration (`transcript.duration`).
   - Words per minute per speaker.
   - Speaker turn distribution (segments per speaker, total seconds per
     speaker, % of meeting per speaker).
   - Optional: histogram of segment lengths.
3. **Snippet listener (relabel UI)** — for each `SPEAKER_*` in
   `transcript.speakers`:
   - Find `RUN_DIR / "snippets" / f"{speaker}.wav"` (top-N already
     extracted by stage 1 per Track A's logic / Track B's B12 work).
   - Render an HTML5 `<audio>` widget so the user can play it inline.
   - Show the longest two utterances of that speaker as text.
   - Provide a small form / dict cell where the user types real names.
4. **Write `speakers.json`** — serialize the dict from step 3 to
   `RUN_DIR / "speakers.json"`. Then print the exact CLI command to run
   next: `uv run meetings relabel "<run_dir>"`.
5. **Post-summary review** *(only runs once `summary.json` exists)* —
   load `read_run(RUN_DIR)`, render `summary.md` inline, list action
   items as a dataframe, and show the LLM prompt/response under
   `RUN_DIR / "llm/"` for inspection.
6. **Optional A/B cell** — if two run dirs are provided (Claude vs
   Gemini, or Track A vs Track B), render both summaries side-by-side
   for human rubric scoring per `04-evaluation-and-comparison.md`.

### Acceptance

- Notebook works on a partial run (post-`transcribe`, pre-`relabel`)
  and on a complete run (post-`summarize`) — cells gate on file
  existence, not raise.
- The relabel cell writes a valid `speakers.json` that
  `meetings relabel` accepts without `--allow-unset`.
- No long-running API calls are made from the notebook (all calls are
  to local files in `Transcription/<run_id>/`).

## Why CLI for the 2 h run, not notebook

- Stage 1 (transcribe + diarize) on a 2 h Dutch meeting is **single
  API call per provider, but each provider call is minutes-to-hours**
  (AssemblyAI: ~5–15 min; pyannoteAI: similar with polling). A kernel
  restart loses that progress and there's no resume; the CLI writes
  artifacts incrementally per stage and is safe to interrupt.
- Stage 3 (summarize) defaults to **batch mode** — Claude / Gemini
  batch APIs return within an hour but have a 24 h SLO. Polling that
  from a kernel is wasteful; the CLI poll loop is short, idempotent,
  and writes the result to disk when it lands.
- Notebook iteration on prompt design uses `meetings summarize
  --sync` (immediate, full price) — you only do this on a few
  hundred-token slice, not the full transcript.

## Implementation tasks

- [x] Scaffold `notebooks/01_test_samples.ipynb` per the cell list above.
      Both Track A (AssemblyAI) and Track B (Scribe v2 + pyannoteAI) are
      implemented with graceful degradation for missing API keys.
- [x] Scaffold `notebooks/02_metrics_and_review.ipynb` per the cell list
      above. The metrics + relabel halves can be written today against
      Track A; the summary-review half works for both tracks already.
- [x] Add a tiny helper module `src/meetings/notebook_helpers.py` (or
      keep it inline) with the gross-metrics computations so the
      notebook stays small and the same code is unit-testable.
- [x] Tests: `tests/test_notebook_helpers.py` — fast, deterministic
      tests for `compute_gross_metrics(transcript)` and the segment/word
      aggregators. No notebook execution in CI.

## Acceptance criteria for plan 06

- The CLI command sequence above runs end-to-end on a real 2 h Dutch
  meeting using either Track A or Track B.
- Both notebooks open, execute, and produce the documented outputs on
  the run dir from that meeting.
- No long-running API call lives in a notebook cell — the notebooks
  only read run-dir artifacts and play snippets.
