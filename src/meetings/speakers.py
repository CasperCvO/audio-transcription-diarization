"""Manual speaker-name labelling utilities.

After Track A's `transcribe` stage produces a diarized `Transcript` with
generic labels (``SPEAKER_A``, ``SPEAKER_B``, ...), the user listens to a
handful of audio snippets and writes a `speakers.json` mapping their real
names. This module reads / writes / validates that file and applies the
mapping to a `Transcript` (both at the segment and word level).

The mapping is **append-only** with respect to existing labels: every
label in the transcript must appear in `speakers.json`, but values may be
``null`` to indicate "I do not know who this is" (the original label is
kept in that case).
"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import Segment, Transcript, Word

SPEAKERS_FILENAME = "speakers.json"


class SpeakerMappingError(ValueError):
    """Raised when speakers.json is malformed, incomplete, or inconsistent."""


def write_skeleton(run_dir: Path, transcript: Transcript) -> Path:
    """Write a `speakers.json` skeleton with every label set to ``null``.

    If the file already exists it is left untouched so the user does not lose
    edits when re-running `transcribe`.
    """
    path = run_dir / SPEAKERS_FILENAME
    if path.exists():
        return path
    skeleton = {label: None for label in transcript.speakers}
    path.write_text(
        json.dumps(skeleton, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def read_mapping(run_dir: Path) -> dict[str, str | None]:
    """Read the raw speakers.json mapping from a run directory."""
    path = run_dir / SPEAKERS_FILENAME
    if not path.exists():
        raise SpeakerMappingError(
            f"{SPEAKERS_FILENAME} not found in {run_dir}. "
            f"Run `meetings transcribe` first."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SpeakerMappingError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SpeakerMappingError(
            f"{path} must contain a JSON object {{label: name}}."
        )
    out: dict[str, str | None] = {}
    for label, value in data.items():
        if value is None:
            out[str(label)] = None
        elif isinstance(value, str):
            stripped = value.strip()
            out[str(label)] = stripped or None
        else:
            raise SpeakerMappingError(
                f"Value for {label!r} in {path} must be a string or null; "
                f"got {type(value).__name__}."
            )
    return out


def validate_mapping(
    mapping: dict[str, str | None],
    transcript: Transcript,
    *,
    require_all_named: bool = True,
) -> None:
    """Check the mapping covers every speaker label in ``transcript``.

    With ``require_all_named=True`` (the default for the relabel CLI),
    every value must be a non-empty string. With ``False`` (the default for
    the summarize CLI) ``null`` values are allowed and the original label
    is kept on those segments.
    """
    transcript_labels = set(transcript.speakers)
    mapping_labels = set(mapping)

    missing = transcript_labels - mapping_labels
    if missing:
        raise SpeakerMappingError(
            f"speakers.json is missing entries for: {sorted(missing)}. "
            f"Re-run `meetings transcribe` or add them by hand."
        )

    extra = mapping_labels - transcript_labels
    if extra:
        # Not fatal but worth flagging — keep going so the user can iterate.
        # (We surface this through a print in the CLI rather than raising.)
        pass

    if require_all_named:
        unset = sorted(label for label, value in mapping.items() if not value)
        if unset:
            raise SpeakerMappingError(
                f"speakers.json still has unset entries: {unset}. "
                f"Edit the file to assign a name to every speaker."
            )

    # Duplicate names are allowed (e.g. two diarized labels turn out to be
    # the same person), but warn if every speaker collapses to a single name —
    # almost always a mistake.
    named_values = [v for v in mapping.values() if v]
    if named_values and len(set(named_values)) == 1 and len(named_values) > 1:
        # Soft warning; surfaced via the CLI, not raised.
        pass


def apply_mapping(
    transcript: Transcript, mapping: dict[str, str | None]
) -> Transcript:
    """Return a new ``Transcript`` with speaker labels rewritten.

    Word-level speaker labels are rewritten alongside segment-level labels.
    Labels mapped to ``None`` are kept as-is. The ``speakers`` list is
    rebuilt in first-appearance order using the mapped names.
    """
    def _resolve(label: str | None) -> str | None:
        if label is None:
            return None
        new = mapping.get(label)
        return new if new else label

    new_segments: list[Segment] = []
    seen: list[str] = []
    for seg in transcript.segments:
        new_speaker = _resolve(seg.speaker)
        if new_speaker and new_speaker not in seen:
            seen.append(new_speaker)
        new_words = [
            Word(
                text=w.text,
                start=w.start,
                end=w.end,
                speaker=_resolve(w.speaker),
                confidence=w.confidence,
            )
            for w in seg.words
        ]
        new_segments.append(
            Segment(
                start=seg.start,
                end=seg.end,
                speaker=new_speaker,
                text=seg.text,
                words=new_words,
            )
        )

    return Transcript(
        language=transcript.language,
        duration=transcript.duration,
        speakers=seen,
        segments=new_segments,
        source_audio=transcript.source_audio,
        backend=transcript.backend,
        backend_meta={**transcript.backend_meta, "speaker_mapping_applied": True},
    )
