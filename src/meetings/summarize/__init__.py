"""Summarization backends. See `plan/02-track-b-custom-pipeline.md` task B6."""

from .base import Summarizer
from .claude import ClaudeSummarizer

__all__ = ["ClaudeSummarizer", "Summarizer", "get_summarizer"]


def get_summarizer(name: str) -> Summarizer:
    if name in {"claude", "anthropic", "claude-sonnet"}:
        return ClaudeSummarizer()
    raise ValueError(f"Unknown summarizer: {name!r}")
