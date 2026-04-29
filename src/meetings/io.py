"""Read/write canonical run outputs under `Transcription/<run_id>/`."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from .schema import RunMeta, RunResult, Summary, Transcript


def new_run_id(audio_path: Path, backend: str) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    short_backend = backend.replace(":", "-").replace("/", "-")
    return f"{audio_path.stem}__{short_backend}__{ts}"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_run(run_dir: Path, result: RunResult) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_transcript(run_dir, result.transcript)
    write_summary(run_dir, result.summary)
    write_meta(run_dir, result.meta)


def write_transcript(run_dir: Path, transcript: Transcript) -> None:
    """Write `transcript.json` + `transcript.md` for an in-progress run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "transcript.json").write_text(
        transcript.model_dump_json(indent=2), encoding="utf-8"
    )
    (run_dir / "transcript.md").write_text(
        render_transcript_md(transcript), encoding="utf-8"
    )


def write_summary(run_dir: Path, summary: Summary) -> None:
    """Write `summary.json` + `summary.md` for a completed run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        summary.model_dump_json(indent=2), encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(
        render_summary_md(summary), encoding="utf-8"
    )


def write_meta(run_dir: Path, meta: RunMeta) -> None:
    """Write `meta.json` for an in-progress or completed run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(
        meta.model_dump_json(indent=2), encoding="utf-8"
    )


def read_transcript(run_dir: Path) -> Transcript:
    return Transcript.model_validate_json(
        (run_dir / "transcript.json").read_text("utf-8")
    )


def read_meta(run_dir: Path) -> RunMeta:
    return RunMeta.model_validate_json((run_dir / "meta.json").read_text("utf-8"))


def read_run(run_dir: Path) -> RunResult:
    """Read a fully completed run.

    Raises ``FileNotFoundError`` if ``summary.json`` is missing — use
    :func:`read_transcript` / :func:`read_meta` for partial runs that
    have transcribed but not yet summarized.
    """
    transcript = read_transcript(run_dir)
    summary = Summary.model_validate_json((run_dir / "summary.json").read_text("utf-8"))
    meta = read_meta(run_dir)
    return RunResult(transcript=transcript, summary=summary, meta=meta)


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def render_transcript_md(t: Transcript) -> str:
    lines = [f"# Transcript ({t.language}) — {t.source_audio}", ""]
    lines.append(f"_Backend: `{t.backend}` — duration: {_fmt_ts(t.duration)}_")
    lines.append("")
    for seg in t.segments:
        speaker = seg.speaker or "?"
        ts = f"[{_fmt_ts(seg.start)} \u2192 {_fmt_ts(seg.end)}]"
        lines.append(f"**{ts} {speaker}:** {seg.text.strip()}")
        lines.append("")
    return "\n".join(lines)


def render_summary_md(s: Summary) -> str:
    nl = s.language.lower().startswith("nl")
    h = {
        "tldr": "Samenvatting" if nl else "TL;DR",
        "topics": "Onderwerpen" if nl else "Topics",
        "decisions": "Beslissingen" if nl else "Decisions",
        "actions": "Actiepunten" if nl else "Action items",
        "questions": "Open vragen" if nl else "Open questions",
        "next": "Volgende stappen" if nl else "Next steps",
    }
    out: list[str] = [f"# {s.title}", ""]
    out.append(f"_Summarizer: `{s.summarizer_backend}` (prompt `{s.prompt_version}`)_\n")

    out.append(f"## {h['tldr']}")
    for b in s.tldr:
        out.append(f"- {b}")
    out.append("")

    if s.topics:
        out.append(f"## {h['topics']}")
        for topic in s.topics:
            out.append(f"### {topic.title}")
            for b in topic.bullets:
                out.append(f"- {b}")
            out.append("")

    if s.decisions:
        out.append(f"## {h['decisions']}")
        for d in s.decisions:
            out.append(f"- {d.text}")
        out.append("")

    if s.action_items:
        out.append(f"## {h['actions']}")
        for a in s.action_items:
            bits = [a.task]
            if a.owner:
                bits.append(f"— **{a.owner}**")
            if a.due:
                bits.append(f"_(due: {a.due})_")
            out.append(f"- {' '.join(bits)}")
        out.append("")

    if s.open_questions:
        out.append(f"## {h['questions']}")
        for q in s.open_questions:
            out.append(f"- {q}")
        out.append("")

    if s.next_steps:
        out.append(f"## {h['next']}")
        for n in s.next_steps:
            out.append(f"- {n}")
        out.append("")

    return "\n".join(out)
