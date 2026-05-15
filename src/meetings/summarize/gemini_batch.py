"""Single-call meeting summarizer backed by Google Gemini.

Mirror of :mod:`meetings.summarize.anthropic_batch` but using the Gemini
``google-genai`` SDK. Submits the whole transcript to Gemini in a single
prompt and parses the JSON response into a canonical
:class:`~meetings.schema.Summary`.

Two execution modes:

- **batch** (default): submits via Gemini's
  `Batch API <https://ai.google.dev/gemini-api/docs/batch-api>`_ for **50%
  lower cost** with a 24 h SLO (typically completes much faster).
- **sync**: ``client.models.generate_content`` for immediate response,
  useful while iterating on prompts.

Configuration: ``GOOGLE_API_KEY`` must be set in ``.env``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, cast

from google import genai
from google.genai import types as genai_types

from ..config import get_settings, require
from ..schema import Summary, Transcript
from ._utils import (
    parse_json,
    parse_summary_payload,
    render_transcript_for_llm,
)
from .base import Summarizer
from .prompts import SINGLE_CALL_PROMPT_VERSION, single_call_prompt

DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_MAX_TOKENS = 4000
POLL_INTERVAL_SECONDS = 30
MAX_POLL_SECONDS = 24 * 60 * 60  # Gemini batch SLO.

_COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


class GeminiSummarizer:
    """Single-call summarizer using Google Gemini (batch by default)."""

    name = "gemini"

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
        api_key = require(get_settings().google_api_key, "GOOGLE_API_KEY")
        client = genai.Client(api_key=api_key)

        prompt = single_call_prompt(language)
        rendered = render_transcript_for_llm(transcript)
        user_content = (
            f"{prompt}\n\n----- TRANSCRIPT START -----\n{rendered}\n"
            "----- TRANSCRIPT END -----"
        )

        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "gemini.prompt.md").write_text(user_content, encoding="utf-8")

        if self.batch:
            raw_text, raw_meta = self._run_batch(client, user_content)
        else:
            raw_text, raw_meta = self._run_sync(client, user_content)

        if log_dir is not None:
            (log_dir / "gemini.response.md").write_text(raw_text, encoding="utf-8")
            (log_dir / "gemini.meta.json").write_text(
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
            summarizer_backend=f"gemini-{'batch' if self.batch else 'sync'}:{self.model}",
            prompt_version=SINGLE_CALL_PROMPT_VERSION,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _generate_config(self) -> genai_types.GenerateContentConfig:
        return genai_types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            response_mime_type="application/json",
        )

    def _run_sync(
        self, client: genai.Client, user_content: str
    ) -> tuple[str, dict[str, Any]]:
        response = client.models.generate_content(
            model=self.model,
            contents=user_content,
            config=self._generate_config(),
        )
        text = response.text or ""
        meta: dict[str, Any] = {
            "mode": "sync",
            "model": self.model,
            "usage_metadata": (
                response.usage_metadata.model_dump()
                if response.usage_metadata
                else None
            ),
        }
        return text, meta

    def _run_batch(
        self, client: genai.Client, user_content: str
    ) -> tuple[str, dict[str, Any]]:
        # Gemini's inline-batch input is a list of InlinedRequestDicts. We
        # submit a batch of size 1; the cost discount still applies and the
        # API is exactly the same as for larger batches.
        cfg_dump = self._generate_config().model_dump(
            exclude_none=True, by_alias=True
        )
        cfg_dict = cast(genai_types.GenerateContentConfigDict, cfg_dump)
        inline_requests: list[genai_types.InlinedRequestDict] = [
            {
                "contents": [
                    {"parts": [{"text": user_content}], "role": "user"},
                ],
                "config": cfg_dict,
            }
        ]

        job = client.batches.create(
            model=self.model,
            src=inline_requests,
            config={"display_name": "meeting-summary"},
        )

        deadline = time.monotonic() + MAX_POLL_SECONDS
        while True:
            state_name = _state_name(job)
            if state_name in _COMPLETED_STATES:
                break
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"Gemini batch {job.name!r} did not complete within "
                    f"{MAX_POLL_SECONDS // 3600} h. Final state: {state_name}."
                )
            time.sleep(POLL_INTERVAL_SECONDS)
            if not job.name:
                raise RuntimeError("Gemini batch returned without a job name.")
            job = client.batches.get(name=job.name)

        final_state = _state_name(job)
        if final_state != "JOB_STATE_SUCCEEDED":
            error = getattr(job, "error", None)
            raise RuntimeError(
                f"Gemini batch {job.name!r} ended in state {final_state}. "
                f"Error: {error}"
            )

        text = _extract_inline_text(job)
        meta: dict[str, Any] = {
            "mode": "batch",
            "model": self.model,
            "job_name": job.name,
            "state": final_state,
        }
        return text, meta


def _state_name(job: Any) -> str:
    state = getattr(job, "state", None)
    return getattr(state, "name", "") or ""


def _extract_inline_text(job: Any) -> str:
    """Pull the text out of a successful inline-batch job."""
    dest = getattr(job, "dest", None)
    inlined = getattr(dest, "inlined_responses", None) if dest else None
    if not inlined:
        raise RuntimeError(
            f"Gemini batch {getattr(job, 'name', '?')} returned no inline responses."
        )
    first = inlined[0]
    response = getattr(first, "response", None)
    if response is None:
        err = getattr(first, "error", None)
        raise RuntimeError(
            f"Gemini batch returned an error for the first request: {err}"
        )
    text = getattr(response, "text", None)
    if isinstance(text, str) and text:
        return text
    # Fallback: drill into candidates / parts manually.
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", None) or []:
            t = getattr(part, "text", None)
            if isinstance(t, str) and t:
                return t
    raise RuntimeError("Gemini batch response contained no text part.")


# Static type-check the Protocol conformance.
_check: Summarizer = GeminiSummarizer()
