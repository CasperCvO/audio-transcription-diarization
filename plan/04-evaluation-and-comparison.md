# 04 — Evaluation & Comparison: Track A vs Track B

**Depends on:** Track A and Track B both produce runs under `Transcription/<run_id>/`.

## Objective

Decide whether the custom pipeline (Track B) actually beats the AssemblyAI
baseline (Track A) on **summary quality** for the user's Dutch meetings.
Since this is a learning, low-volume project, we use lightweight evaluation:
human review + LLM-as-judge, no large benchmark.

## What to evaluate

### Transcript quality (sanity check, not the priority)
- Word Error Rate vs a hand-corrected reference for a 2–3 minute slice of
  one meeting. Compute with `jiwer`.
- Diarization Error Rate (DER) on the same slice using `pyannote.metrics` if
  a reference RTTM is available; otherwise visual inspection of speaker
  changes against ground truth.

### Summary quality (the priority)

Define a rubric (0–5 per criterion):

1. **Coverage** — are all major decisions and action items present?
2. **Accuracy** — no hallucinated facts, owners, or due dates.
3. **Attribution** — actions correctly assigned to the speaker who took them.
4. **Structure** — sections present, action items have owner + task + due
   when the meeting stated them.
5. **Language quality** — natural Dutch, professional register.
6. **Brevity vs completeness** — TL;DR is tight; details live in topics.

## Tasks

### E1. Build a reference set
- [ ] Pick **one** real meeting (15–60 min Dutch).
- [ ] Manually correct a 2–3 minute slice for transcript reference.
- [ ] Manually list the meeting's true decisions and action items (gold
      summary). Save under `Transcription/_gold/<meeting>/`.

### E2. Run both pipelines on the same audio
- [ ] `meetings run --backend assemblyai --audio Audio/<meeting>.wav`
- [ ] `meetings run --backend custom --audio Audio/<meeting>.wav`

### E3. `meetings compare` command (`cli.py`)
- [ ] Loads two `run_dir`s.
- [ ] Computes WER + DER on the reference slice when available.
- [ ] Prints side-by-side rubric scores (filled by the user) into a single
      Markdown report at `Transcription/_compare/<timestamp>.md`.

### E4. LLM-as-judge (assist, not replace, human review)
- [ ] Add `meetings/eval/judge.py`: feed the gold summary + both candidate
      summaries to Claude Opus (or strongest available model) with a strict
      Dutch rubric prompt. Returns per-criterion scores + justifications as JSON.
- [ ] Append the judge output to the comparison report.
- [ ] **Always** keep the human rubric column — LLM judges are biased toward
      verbose answers; the user is the final arbiter.

### E5. Decision
- [ ] Document the verdict in `Transcription/_compare/<timestamp>.md`:
      keep AssemblyAI, keep custom, or use custom only when X.
- [ ] If custom wins, note which **stage** drove the win (transcription vs
      summarization) so future swaps are informed.

## Acceptance criteria

- A single Markdown report contains: WER, DER (if computed), rubric scores
  from both human and LLM judge, and a one-paragraph verdict.
- The verdict references concrete examples (quoted decisions/actions) from
  the gold summary to justify the score.
