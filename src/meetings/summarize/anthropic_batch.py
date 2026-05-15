"""Single-call meeting summarizer backed by Anthropic Claude.

Used by Track A (AssemblyAI) as a direct replacement for the LeMUR ``task``
call. The whole transcript is fed to Claude in one prompt; the response is
parsed into a canonical :class:`~meetings.schema.Summary`.

Two execution modes:

- **batch** (default): submits the request to the
  `Message Batches API <https://platform.claude.com/docs/en/build-with-claude/batch-processing>`_
  for **50% lower cost**. Most batches finish well under an hour, but the
  SLO is 24 h. The summarizer polls until the batch ends and returns the
  result inline.
- **sync**: falls back to ``client.messages.create`` for immediate response,
  useful while iterating on prompts.

Configuration: ``ANTHROPIC_API_KEY`` must be set in ``.env``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from ..config import get_settings, require
from ..schema import Summary, Transcript
from ._utils import (
    parse_json,
    parse_summary_payload,
    render_transcript_for_llm,
)
from .base import Summarizer
from .prompts import SINGLE_CALL_PROMPT_VERSION, single_call_prompt

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4000
POLL_INTERVAL_SECONDS = 30
MAX_POLL_SECONDS = 24 * 60 * 60  # Anthropic's hard 24h batch SLA.

CUSTOM_ID = "meeting-summary"


class AnthropicSummarizer:
    """Single-call summarizer using Claude (batch by default)."""

    name = "anthropic"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        batch: bool = True,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.batch = batch
        self.max_tokens = max_tokens
        self.temperature = temperature

    # ------------------------------------------------------------------ #
    # Summarizer Protocol
    # ------------------------------------------------------------------ #

    def summarize(
        self,
        transcript: Transcript,
        *,
        log_dir: Path | None = None,
        language: str = "nl",
    ) -> Summary:
        api_key = require(get_settings().anthropic_api_key, "ANTHROPIC_API_KEY")
        client = Anthropic(api_key=api_key)

        prompt = single_call_prompt(language)
        rendered = render_transcript_for_llm(transcript)
        user_content = (
            f"{prompt}\n\n----- TRANSCRIPT START -----\n{rendered}\n"
            "----- TRANSCRIPT END -----"
        )

        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "anthropic.prompt.md").write_text(user_content, encoding="utf-8")

        if self.batch:
            raw_text, raw_meta = self._run_batch(client, user_content)
        else:
            raw_text, raw_meta = self._run_sync(client, user_content)

        if log_dir is not None:
            (log_dir / "anthropic.response.md").write_text(raw_text, encoding="utf-8")
            (log_dir / "anthropic.meta.json").write_text(
                json.dumps(raw_meta, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        try:
            data = parse_json(raw_text)
        except ValueError:
            data = {}

        return parse_summary_payload(
            data,
            language=language,
            summarizer_backend=f"anthropic-{'batch' if self.batch else 'sync'}:{self.model}",
            prompt_version=SINGLE_CALL_PROMPT_VERSION,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _run_sync(
        self, client: Anthropic, user_content: str
    ) -> tuple[str, dict[str, Any]]:
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": user_content}],
        )
        text = _join_text_blocks(message.content)
        meta: dict[str, Any] = {
            "mode": "sync",
            "model": self.model,
            "id": message.id,
            "usage": message.usage.model_dump() if message.usage else None,
        }
        return text, meta

    def _run_batch(
        self, client: Anthropic, user_content: str
    ) -> tuple[str, dict[str, Any]]:
        batch = client.messages.batches.create(
            requests=[
                Request(
                    custom_id=CUSTOM_ID,
                    params=MessageCreateParamsNonStreaming(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        messages=[{"role": "user", "content": user_content}],
                    ),
                )
            ]
        )

        deadline = time.monotonic() + MAX_POLL_SECONDS
        while True:
            batch = client.messages.batches.retrieve(batch.id)
            if batch.processing_status == "ended":
                break
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"Anthropic batch {batch.id} did not complete within "
                    f"{MAX_POLL_SECONDS // 3600} h. Final status: "
                    f"{batch.processing_status}."
                )
            time.sleep(POLL_INTERVAL_SECONDS)

        text: str | None = None
        for result in client.messages.batches.results(batch.id):
            if result.custom_id != CUSTOM_ID:
                continue
            if result.result.type == "succeeded":
                text = _join_text_blocks(result.result.message.content)
                break
            raise RuntimeError(
                f"Anthropic batch {batch.id} request {result.custom_id} "
                f"failed with type={result.result.type}."
            )

        if text is None:
            raise RuntimeError(
                f"No result returned for custom_id {CUSTOM_ID!r} in batch {batch.id}."
            )

        meta: dict[str, Any] = {
            "mode": "batch",
            "model": self.model,
            "batch_id": batch.id,
            "request_counts": (
                batch.request_counts.model_dump() if batch.request_counts else None
            ),
            "created_at": str(batch.created_at) if batch.created_at else None,
            "ended_at": str(batch.ended_at) if batch.ended_at else None,
        }
        return text, meta


def _join_text_blocks(blocks: Any) -> str:
    """Concatenate the ``text`` from Claude content blocks."""
    parts: list[str] = []
    for block in blocks or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


# Static type-check the Protocol conformance.
_check: Summarizer = AnthropicSummarizer()
