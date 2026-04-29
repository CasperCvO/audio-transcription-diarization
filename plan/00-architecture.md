# 00 — Architecture & Shared Contracts

## Goals

- Modernize the existing repo from notebook-only Whisper+pyannote into a small,
  testable Python package with a CLI and two pluggable backends (Track A and B).
- Keep Dutch meetings as the primary use case.
- Prioritize **summary quality**. Speed and cost are non-goals.

## Non-goals

- Real-time / streaming transcription.
- Multi-tenant deployment.
- Web UI (optional, only if trivially added via Streamlit later).

## Directory layout (target)

```
.
├── Audio/                       # Input meeting recordings
├── Transcription/<run_id>/      # All outputs for one run
├── plan/                        # This directory
├── src/meetings/
│   ├── __init__.py
│   ├── audio.py                 # ffmpeg conversion to 16 kHz mono WAV (no DSP)
│   ├── snippets.py              # Per-speaker audio snippet extraction (Track A relabel UX)
│   ├── speakers.py              # speakers.json read/write/validate/apply (Track A)
│   ├── schema.py                # Pydantic models for Transcript, Segment, Word, Summary
│   ├── io.py                    # Per-stage read/write helpers (transcript / summary / meta)
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── base.py              # MeetingPipeline protocol (monolithic run())
│   │   ├── assemblyai.py        # Track A — also exposes transcribe/relabel/summarize stages
│   │   └── custom.py            # Track B (composes transcribe + diarize + summarize)
│   ├── transcribe/
│   │   ├── __init__.py
│   │   ├── openai_gpt4o.py
│   │   ├── elevenlabs_scribe.py
│   │   └── deepgram.py
│   ├── diarize/
│   │   ├── __init__.py
│   │   ├── pyannote_local.py
│   │   └── pyannoteai_api.py    # Premium API by the pyannote team
│   ├── summarize/
│   │   ├── __init__.py          # get_summarizer() factory
│   │   ├── base.py              # Summarizer Protocol
│   │   ├── prompts.py           # Versioned prompt templates (Dutch + English)
│   │   ├── _utils.py            # JSON parsing + transcript rendering shared helpers
│   │   ├── claude.py            # Track B map-reduce ClaudeSummarizer
│   │   ├── anthropic_batch.py   # Track A single-call AnthropicSummarizer (Message Batches API)
│   │   ├── gemini_batch.py      # Track A single-call GeminiSummarizer (Batch API)
│   │   └── names.py             # Track B optional auto speaker-name resolution
│   ├── align.py                 # Word-level speaker assignment from word + diarization
│   ├── cli.py                   # Typer CLI entry point
│   └── config.py                # Settings via pydantic-settings + .env
├── tests/
├── pyproject.toml
├── README.md
└── .env.example
```

## Canonical data contracts (Pydantic)

Defined once in `src/meetings/schema.py`. All backends MUST produce/consume
these. This is the seam that lets Track A and Track B be compared apples-to-apples.

```python
class Word(BaseModel):
    text: str
    start: float           # seconds
    end: float
    speaker: str | None    # e.g. "SPEAKER_00" or resolved name
    confidence: float | None = None

class Segment(BaseModel):
    start: float
    end: float
    speaker: str | None
    text: str
    words: list[Word] = []

class Transcript(BaseModel):
    language: str          # ISO code, e.g. "nl"
    duration: float
    speakers: list[str]    # unique labels in order of first appearance
    segments: list[Segment]
    source_audio: str
    backend: str           # "assemblyai" | "custom:<transcribe>+<diarize>"
    backend_meta: dict = {}

class ActionItem(BaseModel):
    task: str
    owner: str | None
    due: str | None        # ISO date or natural language
    source_segment_idx: int | None

class Decision(BaseModel):
    text: str
    source_segment_idx: int | None

class Topic(BaseModel):
    title: str
    bullets: list[str] = []
    segment_range: tuple[float, float] | None = None

class Summary(BaseModel):
    title: str
    tldr: list[str]                # 3–5 bullets
    topics: list[Topic]
    decisions: list[Decision]
    action_items: list[ActionItem]
    open_questions: list[str]
    next_steps: list[str]
    language: str
    summarizer_backend: str        # e.g. "anthropic-batch:claude-sonnet-4-5-..."
    prompt_version: str
```

## Run identifier and run-dir contract

- `run_id = f"{audio_basename}__{backend_short}__{utc_timestamp}"`
- Every run writes its output under `Transcription/<run_id>/`. Files
  produced depend on whether the pipeline is monolithic (Track B,
  `meetings run`) or staged (Track A's `transcribe` / `relabel` /
  `summarize`):

| File / dir | Stage written | Required by | Notes |
|------------|---------------|-------------|-------|
| `transcript.json` | transcribe | summarize, validate | Canonical `Transcript`. |
| `transcript.md` | transcribe | humans | `[mm:ss → mm:ss] Speaker: text`. |
| `speakers.json` | transcribe (Track A) | relabel | `{label: name | null}`. User-edited. |
| `snippets/SPEAKER_*.wav` | transcribe (Track A) | humans | Top-N longest utterances per speaker, ffmpeg-extracted. |
| `summary.json` | summarize | validate, compare | Canonical `Summary`. |
| `summary.md` | summarize | humans | NL/EN headers per `Summary.language`. |
| `meta.json` | every stage | validate | `extra.stage` ∈ {`transcribed`, `relabelled`, `summarized`}. |
| `llm/*.{prompt,response,meta}.{md,json}` | summarize | debugging | All LLM I/O for inspection. |

- A `meta.json` records: input path + sha256, backend, model versions,
  prompt version, wall time per stage, and any provider-specific
  identifiers (e.g. `transcript_id`, `speaker_mapping`, batch job IDs).

## Configuration

- `.env` for API keys: `ASSEMBLYAI_API_KEY`, `OPENAI_API_KEY`,
  `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `ELEVENLABS_API_KEY`,
  `DEEPGRAM_API_KEY`, `PYANNOTEAI_API_KEY`, `HF_TOKEN`.
- `pydantic-settings` in `config.py` to load and validate.
- No keys in code, no keys in notebooks.

## Pipeline protocols

The base monolithic protocol — implemented by both tracks:

```python
class MeetingPipeline(Protocol):
    name: str
    def run(self, audio_path: Path, run_dir: Path, *, language: str = "nl") -> RunResult: ...
```

Track A additionally exposes a **staged** interface to support the
human-in-the-loop relabel workflow:

```python
class AssemblyAIPipeline:
    name = "assemblyai"

    def transcribe(self, audio_path: Path, run_dir: Path, *,
                   language: str = "nl",
                   snippets_per_speaker: int = 3) -> Transcript: ...

    def relabel(self, run_dir: Path, *,
                require_all_named: bool = True) -> Transcript: ...

    def summarize(self, run_dir: Path, *,
                  summarizer: str | Summarizer = "claude",
                  batch: bool = True,
                  language: str | None = None) -> Summary: ...

    def run(self, audio_path: Path, run_dir: Path, *,
            language: str = "nl",
            summarizer: str | Summarizer = "claude",
            batch: bool = True) -> RunResult:
        """transcribe() + summarize() back-to-back; no relabel pause."""
```

`RunResult` bundles the `Transcript`, `Summary`, and `meta`. Both tracks
implement `MeetingPipeline.run`; the CLI selects which one. Track A's
staged methods are accessed via the `meetings transcribe` / `relabel` /
`summarize` subcommands.
