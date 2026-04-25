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
│   ├── audio.py                 # ffmpeg conversion, loudness normalization, chunking
│   ├── schema.py                # Pydantic models for Transcript, Segment, Word, Summary
│   ├── io.py                    # Read/write transcript.json/md, summary.json/md
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── base.py              # Pipeline protocol
│   │   ├── assemblyai.py        # Track A
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
│   │   ├── __init__.py
│   │   ├── prompts.py           # Versioned prompt templates (Dutch + English)
│   │   ├── claude.py
│   │   ├── openai.py
│   │   └── gemini.py
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

class Summary(BaseModel):
    title: str
    tldr: list[str]                # 3–5 bullets
    topics: list[dict]             # [{title, bullets:[...], segment_range:[start,end]}]
    decisions: list[Decision]
    action_items: list[ActionItem]
    open_questions: list[str]
    next_steps: list[str]
    language: str
    summarizer_backend: str        # e.g. "claude-sonnet-4.5"
    prompt_version: str
```

## Run identifier

- `run_id = f"{audio_basename}__{backend_short}__{utc_timestamp}"`
- Every run writes its complete output under `Transcription/<run_id>/`.
- A `meta.json` records: input path + sha256, backend, model versions, prompt
  versions, wall time per stage, token counts, API cost estimate.

## Configuration

- `.env` for API keys: `ASSEMBLYAI_API_KEY`, `OPENAI_API_KEY`,
  `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `ELEVENLABS_API_KEY`,
  `DEEPGRAM_API_KEY`, `PYANNOTEAI_API_KEY`, `HF_TOKEN`.
- `pydantic-settings` in `config.py` to load and validate.
- No keys in code, no keys in notebooks.

## Pipeline protocol

```python
class MeetingPipeline(Protocol):
    name: str
    def run(self, audio_path: Path, run_dir: Path, *, language: str = "nl") -> RunResult: ...
```

`RunResult` bundles the `Transcript`, `Summary`, and `meta` dict. Both Track A
and Track B implement this. The CLI just selects which one.
