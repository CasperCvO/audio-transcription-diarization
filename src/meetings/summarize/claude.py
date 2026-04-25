"""Claude (Anthropic) summarizer with map-reduce + critique (plan B6).

Strategy
--------
1. **Map** — split transcript into windows of ~``window_chars`` characters
   on speaker boundaries. For each window, ask Claude for a JSON with local
   topics/decisions/actions/questions/quotes.
2. **Reduce** — consolidate local JSONs into one `Summary` JSON.
3. **Critique** — second pass that flags missing decisions, missing action
   items, and hallucinations, returning patch suggestions.
4. Apply safe patches (adding missing decisions / actions).

Every prompt and raw response is written under ``log_dir`` when provided,
because this is a learning project and transparency beats brevity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import get_settings, require
from ..schema import ActionItem, Decision, Segment, Summary, Topic, Transcript
from ._utils import fmt_ts, parse_json
from .prompts import (
    CRITIQUE_PROMPT_NL,
    MAP_PROMPT_NL,
    PROMPT_VERSION,
    REDUCE_PROMPT_NL,
    SYSTEM_PROMPT_NL,
)


class ClaudeSummarizer:
    """Summarize a diarized transcript with map-reduce + critique on Claude."""

    name: str = "claude"

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        *,
        window_chars: int = 24_000,
        map_max_tokens: int = 4_000,
        reduce_max_tokens: int = 6_000,
        critique_max_tokens: int = 4_000,
        map_temperature: float = 0.2,
        reduce_temperature: float = 0.2,
        critique_temperature: float = 0.0,
        client: Anthropic | None = None,
    ) -> None:
        self.model = model
        self.window_chars = window_chars
        self.map_max_tokens = map_max_tokens
        self.reduce_max_tokens = reduce_max_tokens
        self.critique_max_tokens = critique_max_tokens
        self.map_temperature = map_temperature
        self.reduce_temperature = reduce_temperature
        self.critique_temperature = critique_temperature
        self._client = client

    def _get_client(self) -> Anthropic:
        if self._client is not None:
            return self._client
        api_key = require(get_settings().anthropic_api_key, "ANTHROPIC_API_KEY")
        self._client = Anthropic(api_key=api_key)
        return self._client

    # ------------------------------------------------------------------ API

    def summarize(
        self,
        transcript: Transcript,
        *,
        log_dir: Path | None = None,
        language: str = "nl",
    ) -> Summary:
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)

        windows = _chunk_transcript(transcript, max_chars=self.window_chars)
        local_summaries: list[dict[str, Any]] = []
        for idx, window_text in enumerate(windows):
            result = self._map_window(window_text, idx, log_dir)
            local_summaries.append(result)

        draft = self._reduce(local_summaries, log_dir)

        full_transcript_text = _render_diarized(transcript)
        critique = self._critique(full_transcript_text, draft, log_dir)
        patched = _apply_critique(draft, critique)

        return _to_summary(
            patched,
            language=language,
            summarizer_backend=f"anthropic:{self.model}",
            prompt_version=PROMPT_VERSION,
        )

    # ----------------------------------------------------------------- Map

    def _map_window(
        self, window_text: str, idx: int, log_dir: Path | None
    ) -> dict[str, Any]:
        user_content = f"{MAP_PROMPT_NL}\n\n---\n\n{window_text}"
        response_text = self._call(
            user_content,
            max_tokens=self.map_max_tokens,
            temperature=self.map_temperature,
            log_dir=log_dir,
            log_stem=f"map_{idx:02d}",
        )
        return parse_json(response_text)

    # -------------------------------------------------------------- Reduce

    def _reduce(
        self, local_summaries: list[dict[str, Any]], log_dir: Path | None
    ) -> dict[str, Any]:
        bundled = json.dumps(local_summaries, ensure_ascii=False, indent=2)
        user_content = (
            f"{REDUCE_PROMPT_NL}\n\n"
            "Lokale samenvattingen (in chronologische volgorde):\n"
            f"```json\n{bundled}\n```"
        )
        response_text = self._call(
            user_content,
            max_tokens=self.reduce_max_tokens,
            temperature=self.reduce_temperature,
            log_dir=log_dir,
            log_stem="reduce",
        )
        return parse_json(response_text)

    # ------------------------------------------------------------ Critique

    def _critique(
        self,
        transcript_text: str,
        draft: dict[str, Any],
        log_dir: Path | None,
    ) -> dict[str, Any]:
        draft_json = json.dumps(draft, ensure_ascii=False, indent=2)
        user_content = (
            f"{CRITIQUE_PROMPT_NL}\n\n"
            "Transcript:\n"
            f"```\n{transcript_text}\n```\n\n"
            "Conceptsamenvatting:\n"
            f"```json\n{draft_json}\n```"
        )
        response_text = self._call(
            user_content,
            max_tokens=self.critique_max_tokens,
            temperature=self.critique_temperature,
            log_dir=log_dir,
            log_stem="critique",
        )
        try:
            return parse_json(response_text)
        except ValueError:
            return {}

    # -------------------------------------------------------------- Driver

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        reraise=True,
    )
    def _call(
        self,
        user_content: str,
        *,
        max_tokens: int,
        temperature: float,
        log_dir: Path | None,
        log_stem: str,
    ) -> str:
        client = self._get_client()
        message = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=SYSTEM_PROMPT_NL,
            messages=[{"role": "user", "content": user_content}],
        )
        text = "".join(
            block.text  # type: ignore[union-attr]
            for block in message.content
            if getattr(block, "type", None) == "text"
        )
        if log_dir:
            (log_dir / f"{log_stem}.prompt.md").write_text(user_content, encoding="utf-8")
            (log_dir / f"{log_stem}.response.md").write_text(text, encoding="utf-8")
        return text


# --------------------------------------------------------------- helpers ---


def _chunk_transcript(t: Transcript, *, max_chars: int) -> list[str]:
    """Group segments into contiguous chunks up to ``max_chars``."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for idx, seg in enumerate(t.segments):
        line = _format_segment(idx, seg)
        if current and current_len + len(line) > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    if not chunks:
        chunks = [""]
    return chunks


def _format_segment(idx: int, seg: Segment) -> str:
    ts = fmt_ts(seg.start)
    speaker = seg.speaker or "?"
    return f"[{idx}] {ts} {speaker}: {seg.text.strip()}"


def _render_diarized(t: Transcript) -> str:
    return "\n".join(_format_segment(i, s) for i, s in enumerate(t.segments))


def _apply_critique(draft: dict[str, Any], critique: dict[str, Any]) -> dict[str, Any]:
    """Merge safe additions from the critique back into the draft.

    Only additive patches are applied here (missing decisions / actions).
    Field-level corrections are left for human review; they are logged but
    not auto-applied to avoid silent quality regressions.
    """
    patched = dict(draft)
    decisions = list(patched.get("decisions") or [])
    for miss in critique.get("missing_decisions") or []:
        text = str(miss.get("text", "")).strip()
        if text and not any(d.get("text") == text for d in decisions):
            decisions.append({"text": text})
    patched["decisions"] = decisions

    actions = list(patched.get("action_items") or [])
    existing = {(a.get("task"), a.get("owner")) for a in actions}
    for miss in critique.get("missing_actions") or []:
        key = (miss.get("task"), miss.get("owner"))
        if miss.get("task") and key not in existing:
            actions.append(
                {
                    "task": miss.get("task"),
                    "owner": miss.get("owner"),
                    "due": miss.get("due"),
                }
            )
    patched["action_items"] = actions
    return patched


def _to_summary(
    data: dict[str, Any],
    *,
    language: str,
    summarizer_backend: str,
    prompt_version: str,
) -> Summary:
    topics = [
        Topic(title=str(t.get("title", "")), bullets=[str(b) for b in t.get("bullets") or []])
        for t in data.get("topics") or []
    ]
    decisions = [Decision(text=str(d.get("text", ""))) for d in data.get("decisions") or []]
    actions = [
        ActionItem(
            task=str(a.get("task", "")),
            owner=_opt_str(a.get("owner")),
            due=_opt_str(a.get("due")),
        )
        for a in data.get("action_items") or []
    ]
    return Summary(
        title=str(data.get("title") or "Vergadering"),
        tldr=[str(b) for b in data.get("tldr") or []],
        topics=topics,
        decisions=decisions,
        action_items=actions,
        open_questions=[str(q) for q in data.get("open_questions") or []],
        next_steps=[str(n) for n in data.get("next_steps") or []],
        language=language,
        summarizer_backend=summarizer_backend,
        prompt_version=prompt_version,
    )


def _opt_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return str(value)
