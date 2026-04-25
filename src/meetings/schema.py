"""Canonical Pydantic data contracts shared by every pipeline backend.

All transcription/diarization/summarization backends must produce or consume
these types so that Track A (AssemblyAI) and Track B (custom) can be compared
apples-to-apples. See `plan/00-architecture.md`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Word(BaseModel):
    text: str
    start: float  # seconds
    end: float
    speaker: str | None = None
    confidence: float | None = None


class Segment(BaseModel):
    start: float
    end: float
    speaker: str | None = None
    text: str
    words: list[Word] = Field(default_factory=list)


class Transcript(BaseModel):
    language: str  # ISO code, e.g. "nl"
    duration: float
    speakers: list[str] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)
    source_audio: str
    backend: str
    backend_meta: dict[str, object] = Field(default_factory=dict)


class DiarizationTurn(BaseModel):
    start: float
    end: float
    speaker: str


class ActionItem(BaseModel):
    task: str
    owner: str | None = None
    due: str | None = None  # ISO date or natural language
    source_segment_idx: int | None = None


class Decision(BaseModel):
    text: str
    source_segment_idx: int | None = None


class Topic(BaseModel):
    title: str
    bullets: list[str] = Field(default_factory=list)
    segment_range: tuple[float, float] | None = None


class Summary(BaseModel):
    title: str
    tldr: list[str] = Field(default_factory=list)
    topics: list[Topic] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    language: str
    summarizer_backend: str
    prompt_version: str


class RunMeta(BaseModel):
    run_id: str
    backend: str
    input_path: str
    input_sha256: str
    duration: float
    timings: dict[str, float] = Field(default_factory=dict)
    model_versions: dict[str, str] = Field(default_factory=dict)
    prompt_version: str | None = None
    extra: dict[str, object] = Field(default_factory=dict)


class RunResult(BaseModel):
    transcript: Transcript
    summary: Summary
    meta: RunMeta
