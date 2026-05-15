"""Internal helpers shared across summarize.claude and summarize.names."""

from __future__ import annotations

import json
import re
from typing import Any, cast

from ..schema import ActionItem, Decision, Summary, Topic, Transcript

_JSON_FENCE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def parse_json(text: str) -> dict[str, Any]:
    """Parse a JSON object out of a model response, tolerating code fences
    and trailing junk (e.g. Gemini occasionally emits an extra ``}`` after a
    well-formed object)."""
    candidates: list[str] = []
    m = _JSON_FENCE.search(text)
    if m:
        candidates.append(m.group(1))
    candidates.append(text)
    decoder = json.JSONDecoder()
    for body in candidates:
        body = body.strip()
        start = body.find("{")
        if start == -1:
            continue
        # raw_decode parses the first JSON value and ignores any trailing
        # characters, which makes us robust to stray braces / extra prose.
        try:
            obj, _ = decoder.raw_decode(body[start:])
        except json.JSONDecodeError:
            # Fallback: try the legacy start..rfind('}') slice in case the
            # response had leading prose followed by valid JSON we can't
            # otherwise locate.
            end = body.rfind("}")
            if end > start:
                try:
                    return cast(dict[str, Any], json.loads(body[start : end + 1]))
                except json.JSONDecodeError:
                    pass
            continue
        if isinstance(obj, dict):
            return cast(dict[str, Any], obj)
    raise ValueError(f"Could not parse JSON from model response: {text[:300]!r}")


# --------------------------------------------------------------------------- #
# Helpers reused by the single-call (Track A) summarizers
# --------------------------------------------------------------------------- #


def render_transcript_for_llm(transcript: Transcript) -> str:
    """Render a diarized transcript as plain text suitable for an LLM prompt.

    Format: ``[mm:ss] SPEAKER: text`` per segment, one segment per line. Uses
    whatever speaker labels are currently on the transcript — if the user has
    relabelled SPEAKER_A → "Casper", that's what the model sees.
    """
    lines: list[str] = []
    for seg in transcript.segments:
        speaker = seg.speaker or "?"
        text = (seg.text or "").strip()
        if not text:
            continue
        lines.append(f"[{fmt_ts(seg.start)}] {speaker}: {text}")
    return "\n".join(lines)


def parse_decisions(items: Any) -> list[Decision]:
    out: list[Decision] = []
    if not isinstance(items, list):
        return out
    for d in items:
        text = (
            str(d.get("text") or "").strip() if isinstance(d, dict) else str(d).strip()
        )
        if text:
            out.append(Decision(text=text))
    return out


def parse_action_items(items: Any) -> list[ActionItem]:
    out: list[ActionItem] = []
    if not isinstance(items, list):
        return out
    for a in items:
        if not isinstance(a, dict):
            continue
        task = str(a.get("task") or "").strip()
        if not task:
            continue
        owner_raw = a.get("owner")
        due_raw = a.get("due")
        owner = (
            str(owner_raw).strip()
            if isinstance(owner_raw, str) and owner_raw.strip()
            else None
        )
        due = (
            str(due_raw).strip()
            if isinstance(due_raw, str) and due_raw.strip()
            else None
        )
        out.append(ActionItem(task=task, owner=owner, due=due))
    return out


def parse_topics(items: Any) -> list[Topic]:
    out: list[Topic] = []
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        bullets_raw = item.get("bullets") or []
        bullets = [
            str(b).strip()
            for b in bullets_raw
            if isinstance(bullets_raw, list) and str(b).strip()
        ]
        if title or bullets:
            out.append(Topic(title=title, bullets=bullets))
    return out


def parse_summary_payload(
    data: dict[str, Any],
    *,
    language: str,
    summarizer_backend: str,
    prompt_version: str,
) -> Summary:
    """Convert a JSON payload (already parsed) into a canonical ``Summary``."""
    fallback_title = (
        "Vergaderingsamenvatting"
        if language.lower().startswith("nl")
        else "Meeting summary"
    )
    return Summary(
        title=str(data.get("title") or fallback_title),
        tldr=[str(b).strip() for b in (data.get("tldr") or []) if str(b).strip()],
        topics=parse_topics(data.get("topics")),
        decisions=parse_decisions(data.get("decisions")),
        action_items=parse_action_items(data.get("action_items")),
        open_questions=[
            str(q).strip() for q in (data.get("open_questions") or []) if str(q).strip()
        ],
        next_steps=[
            str(s).strip() for s in (data.get("next_steps") or []) if str(s).strip()
        ],
        language=language,
        summarizer_backend=summarizer_backend,
        prompt_version=prompt_version,
    )
