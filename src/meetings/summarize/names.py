"""Optional speaker name resolution (plan B5).

Feed the first ``window_seconds`` of the diarized transcript to Claude and
ask it to map ``SPEAKER_XX`` labels onto real names it can extract from
self-introductions. Returns a mapping that callers can apply to a
``Transcript`` via :func:`apply_name_mapping`.
"""

from __future__ import annotations

from pathlib import Path

from anthropic import Anthropic

from ..config import get_settings, require
from ..schema import Segment, Transcript, Word
from ._utils import fmt_ts, parse_json
from .prompts import NAME_RESOLUTION_PROMPT_NL, SYSTEM_PROMPT_NL


def resolve_speaker_names(
    transcript: Transcript,
    *,
    window_seconds: float = 300.0,
    model: str = "claude-sonnet-4-5-20250929",
    log_dir: Path | None = None,
    client: Anthropic | None = None,
) -> dict[str, str]:
    """Return a mapping ``{SPEAKER_00: "Casper", ...}``, possibly partial."""
    snippet_lines: list[str] = []
    for idx, seg in enumerate(transcript.segments):
        if seg.start > window_seconds:
            break
        speaker = seg.speaker or "?"
        snippet_lines.append(f"[{idx}] {fmt_ts(seg.start)} {speaker}: {seg.text.strip()}")
    if not snippet_lines:
        return {}

    user_content = f"{NAME_RESOLUTION_PROMPT_NL}\n\n" + "\n".join(snippet_lines)
    if client is None:
        api_key = require(get_settings().anthropic_api_key, "ANTHROPIC_API_KEY")
        client = Anthropic(api_key=api_key)

    message = client.messages.create(
        model=model,
        max_tokens=600,
        temperature=0.0,
        system=SYSTEM_PROMPT_NL,
        messages=[{"role": "user", "content": user_content}],
    )
    text = "".join(
        block.text  # type: ignore[union-attr]
        for block in message.content
        if getattr(block, "type", None) == "text"
    )

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "names.prompt.md").write_text(user_content, encoding="utf-8")
        (log_dir / "names.response.md").write_text(text, encoding="utf-8")

    try:
        raw = parse_json(text)
    except ValueError:
        return {}

    mapping: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(value, str) and value.strip():
            mapping[str(key)] = value.strip()
    return mapping


def apply_name_mapping(transcript: Transcript, mapping: dict[str, str]) -> Transcript:
    """Return a new ``Transcript`` with speaker labels replaced by real names."""
    if not mapping:
        return transcript

    def remap(label: str | None) -> str | None:
        if label is None:
            return None
        return mapping.get(label, label)

    new_segments: list[Segment] = []
    for seg in transcript.segments:
        new_words = [
            Word(
                text=w.text,
                start=w.start,
                end=w.end,
                speaker=remap(w.speaker),
                confidence=w.confidence,
            )
            for w in seg.words
        ]
        new_segments.append(
            Segment(
                start=seg.start,
                end=seg.end,
                speaker=remap(seg.speaker),
                text=seg.text,
                words=new_words,
            )
        )
    speakers = list(dict.fromkeys(remap(s) or s for s in transcript.speakers))
    return transcript.model_copy(update={"segments": new_segments, "speakers": speakers})

