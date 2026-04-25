# Cost estimation — Custom pipeline (Track B)

This document estimates the per-run API cost of the **custom pipeline**
(`src/meetings/pipelines/custom.py`) for a representative **2-hour audio file**.
The pipeline composes four paid stages:

```
prepare → transcribe → diarize → (optional name-resolution) → summarize
```

Local audio prep (`prepare_audio`) and `align.assign_speakers` are pure CPU work
and have no API cost. All cost is driven by the three external APIs selected
through the `--transcriber`, `--diarizer`, `--summarizer` CLI flags.

> **Disclaimer.** Vendor prices change. Treat all numbers as **±20 %** and
> re-check the relevant pricing pages before making billing decisions.
> Prices below reflect public list prices as of late 2025 / early 2026.

## Assumptions

- 2 h audio = 120 min = 7 200 s.
- Speech rate ≈ 150 words/min ⇒ ~18 000 words ≈ 110 k chars of spoken text.
- With timestamp + `[idx] HH:MM:SS SPEAKER_XX:` formatting overhead from
  `_format_segment` (`src/meetings/summarize/claude.py`) → ~135 k chars of
  diarized transcript ≈ **~34 k input tokens** for the full transcript.
- Default summarizer config: `window_chars=24_000` ⇒ **~6 map windows**.
- 1 token ≈ 4 chars (English/Dutch mix). Used only as a rough conversion.

---

## 1. Transcription

Default backend: `OpenAITranscriber("whisper-1")`.

| Backend (CLI `--transcriber`) | List price | 2 h cost |
|---|---|---|
| `whisper-1` *(default)* | $0.006 / min | **≈ $0.72** |
| `gpt-4o-transcribe` | ~$0.006 / min equiv. | ≈ $0.72 |
| `gpt-4o-mini-transcribe` | ~$0.003 / min | ≈ $0.36 |
| `elevenlabs` (Scribe) | ~$0.40 / hour | ≈ $0.80 |
| `deepgram` (Nova-3) | ~$0.0043 / min | ≈ $0.52 |

## 2. Diarization

Default backend: `PyannoteAIDiarizer("precision-2")`.

| Backend (CLI `--diarizer`) | List price | 2 h cost |
|---|---|---|
| `pyannoteai` (`precision-2`) *(default)* | ~$0.40 / hour | **≈ $0.80** |
| `pyannote_local` | $0 (self-hosted) | $0 — needs GPU + HF token |

## 3. Summarization — Claude (Anthropic)

`ClaudeSummarizer` runs a **map → reduce → critique** loop
(`src/meetings/summarize/claude.py`). The critique step re-sends the entire
diarized transcript, which dominates the input-token bill on long meetings.

### Per-stage token estimate (2 h transcript, 6 map windows)

| Stage | Calls | Input tok (each) | Output tok (each) | Subtotal in | Subtotal out |
|---|---:|---:|---:|---:|---:|
| Map | 6 | ~7 000 | ~1 500 | 42 000 | 9 000 |
| Reduce | 1 | ~10 000 | ~2 500 | 10 000 | 2 500 |
| Critique | 1 | ~37 000 | ~1 500 | 37 000 | 1 500 |
| **Total** | | | | **~89 000** | **~13 000** |

> `max_tokens` caps in code (4 000 / 6 000 / 4 000) are *upper bounds*. Real
> outputs for a 2 h meeting are typically far below them. If the model fills
> the caps, output cost can roughly double for the summarize stage.

### Cost per Claude model

Pricing in $ per **1 M tokens**:

| Model | Input | Output |
|---|---:|---:|
| Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`) — *current default* | $3.00 | $15.00 |
| Claude Sonnet 4.6 (assumed same Sonnet tier pricing) | $3.00 | $15.00 |

> **Note on Sonnet 4.6.** At time of writing, Anthropic has historically kept
> the Sonnet tier at $3 / $15 per 1 M tokens across `3.5`, `3.7`, `4`, `4.5`.
> The numbers below assume Sonnet 4.6 will follow the same tier pricing.
> **Verify on https://www.anthropic.com/pricing before relying on this.**
> If a future Sonnet 4.6 ships at a different tier, scale linearly.

#### Sonnet 4.5 — 2 h meeting

```
input : 89 000 tok × $3 / 1 M  =  $0.267
output: 13 000 tok × $15 / 1 M =  $0.195
total                          ≈  $0.46
```

#### Sonnet 4.6 — 2 h meeting (assumed same tier pricing)

```
input : 89 000 tok × $3 / 1 M  =  $0.267
output: 13 000 tok × $15 / 1 M =  $0.195
total                          ≈  $0.46
```

Identical to Sonnet 4.5 under the stated assumption. If Sonnet 4.6 ends up
priced higher (e.g. $4 / $20 per 1 M), recompute as:

```
total ≈ 0.089 × <input $/M> + 0.013 × <output $/M>
```

| Hypothetical Sonnet 4.6 pricing | 2 h summarize cost |
|---|---:|
| $3 / $15 (same as 4.5)  | ~$0.46 |
| $4 / $20                | ~$0.62 |
| $5 / $25                | ~$0.77 |

## 4. Optional — speaker name resolution

`resolve_speaker_names` (`src/meetings/summarize/names.py`) sends only the first
5 min of segments and caps output at 600 tokens. On Sonnet 4.5/4.6:

- ~1 500 input + ~400 output tokens
- **≈ $0.01** — negligible.

---

## Totals — common configurations (2 h audio)

| Config | Transcribe | Diarize | Summarize | **Total** |
|---|---:|---:|---:|---:|
| **Default** (`whisper-1` + `pyannoteai` + Sonnet 4.5) | $0.72 | $0.80 | $0.46 | **≈ $1.98** |
| Default but with **Sonnet 4.6** (same-tier pricing) | $0.72 | $0.80 | $0.46 | **≈ $1.98** |
| Cheap cloud (`gpt-4o-mini-transcribe` + `pyannoteai` + Sonnet 4.5) | $0.36 | $0.80 | $0.46 | ≈ $1.62 |
| GPU-local diarization (`whisper-1` + `pyannote_local` + Sonnet 4.5) | $0.72 | $0.00 | $0.46 | ≈ $1.18 |
| Cheapest CLI combo (`gpt-4o-mini-transcribe` + `pyannote_local` + Sonnet 4.5) | $0.36 | $0.00 | $0.46 | ≈ $0.82 |

Add **+$0.01** to any row when running with `--name-resolution`.

## Caveats

- The critique stage re-sends the full transcript. For very long meetings
  (>3–4 h) it dominates cost; reducing `window_chars` *increases* total
  cost (more map calls + same critique).
- Audio is uploaded raw to pyannoteAI; egress bandwidth is on you.
- `prepare_audio` may transcode/downsample, but billable duration is the
  processed file's duration ≈ original duration.
- Token counts are estimated via `chars / 4`. Real Anthropic / OpenAI
  tokenizers will differ by ±10 %.

## Quick recompute formula

For a meeting of duration `D` hours, default backends, Sonnet-tier summarizer
at `(p_in, p_out)` $/M tokens:

```
transcribe ≈ 0.36 · D                 # whisper-1
diarize    ≈ 0.40 · D                 # pyannoteai precision-2
summarize  ≈ (45·D)·p_in/1000 + (6·D)·p_out/1000   # rough scaling
```

For `D=2`, `p_in=3`, `p_out=15` this reproduces ≈ $1.98.
