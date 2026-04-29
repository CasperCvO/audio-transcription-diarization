"""Summarization backends.

See `plan/02-track-b-custom-pipeline.md` (task B6) for Track B's
map-reduce :class:`ClaudeSummarizer`, and `plan/01-track-a-assemblyai.md`
for the single-call batch summarizers used by Track A.
"""

from .anthropic_batch import AnthropicSummarizer
from .base import Summarizer
from .claude import ClaudeSummarizer
from .gemini_batch import GeminiSummarizer

__all__ = [
    "AnthropicSummarizer",
    "ClaudeSummarizer",
    "GeminiSummarizer",
    "Summarizer",
    "get_summarizer",
]


def get_summarizer(name: str) -> Summarizer:
    """Resolve a summarizer name to an instance (used by Track B).

    Track B keeps the existing ``"claude"`` alias pointing at the
    map-reduce :class:`ClaudeSummarizer` for backwards compatibility.
    Track A's CLI maps ``claude``/``gemini`` to the single-call batch
    summarizers directly (see ``pipelines/assemblyai.py``).
    """
    key = name.lower()
    if key in {"claude", "claude-mapreduce", "claude-mr"}:
        return ClaudeSummarizer()
    if key in {"anthropic", "anthropic-batch", "claude-batch"}:
        return AnthropicSummarizer(batch=True)
    if key == "anthropic-sync":
        return AnthropicSummarizer(batch=False)
    if key in {"gemini", "gemini-batch", "google"}:
        return GeminiSummarizer(batch=True)
    if key == "gemini-sync":
        return GeminiSummarizer(batch=False)
    raise ValueError(f"Unknown summarizer: {name!r}")
