"""Track A: AssemblyAI Universal-2 pipeline.

End-to-end Dutch (or English) meeting pipeline using a single AssemblyAI
transcription call for transcription + speaker diarization, plus a LeMUR
``task`` call for the structured summary (decisions, action items, etc.).

Notes on AssemblyAI feature compatibility:
- Native ``auto_chapters`` and ``summarization`` are English-only on
  AssemblyAI's API. They are enabled automatically when ``language == "en"``
  and skipped otherwise; LeMUR (Claude) handles the structured Summary in
  every supported language including Dutch.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import assemblyai as aai  # type: ignore[import-untyped]

from ..config import get_settings, require
from ..io import sha256_of, write_run
from ..schema import (
    ActionItem,
    Decision,
    RunMeta,
    RunResult,
    Segment,
    Summary,
    Topic,
    Transcript,
    Word,
)

PROMPT_VERSION = "assemblyai-lemur-v1"

DEFAULT_LEMUR_MODEL: str = aai.LemurModel.claude_sonnet_4_20250514
DEFAULT_SPEECH_MODEL: aai.SpeechModel = aai.SpeechModel.universal


_LEMUR_PROMPT_NL = """Je bent een notulist die een Nederlandse vergadering samenvat.

Geef een gestructureerde samenvatting in het Nederlands. Antwoord met
UITSLUITEND geldige JSON, zonder uitleg en zonder markdown-fences. Het
JSON-object volgt exact dit schema:

{
  "title": "korte titel van de vergadering",
  "tldr": ["3 tot 5 bullets met de kern van de vergadering"],
  "topics": [
    {"title": "onderwerp", "bullets": ["belangrijkste punten over dit onderwerp"]}
  ],
  "decisions": [
    {"text": "wat is besloten"}
  ],
  "action_items": [
    {"task": "concrete actie", "owner": "naam of null", "due": "datum/termijn of null"}
  ],
  "open_questions": ["nog openstaande vragen"],
  "next_steps": ["geplande vervolgstappen"]
}

Vereisten:
- tldr: 3 tot 5 korte bullets.
- Verzin niets dat niet uit de transcriptie blijkt; laat een lijst leeg als
  de inhoud ontbreekt.
- Houd Nederlandse termen en eigennamen exact aan.
"""

_LEMUR_PROMPT_EN = """You are taking minutes for a meeting.

Reply with ONLY valid JSON, no prose, no markdown fences. JSON schema:

{
  "title": "short meeting title",
  "tldr": ["3 to 5 bullets capturing the essence"],
  "topics": [{"title": "topic", "bullets": ["key points"]}],
  "decisions": [{"text": "what was decided"}],
  "action_items": [{"task": "...", "owner": "name or null", "due": "date/term or null"}],
  "open_questions": ["..."],
  "next_steps": ["..."]
}

Do not invent content not present in the transcript. Leave lists empty when
appropriate.
"""


class AssemblyAIPipeline:
    """Pipeline that runs Track A: a single AssemblyAI call + LeMUR summary."""

    name = "assemblyai"

    def __init__(
        self,
        speech_model: aai.SpeechModel = DEFAULT_SPEECH_MODEL,
        lemur_model: str = DEFAULT_LEMUR_MODEL,
    ) -> None:
        self.speech_model = speech_model
        self.lemur_model = lemur_model

    def run(
        self,
        audio_path: Path,
        run_dir: Path,
        *,
        language: str = "nl",
    ) -> RunResult:
        settings = get_settings()
        api_key = require(settings.assemblyai_api_key, "ASSEMBLYAI_API_KEY")
        aai.settings.api_key = api_key

        timings: dict[str, float] = {}
        is_english = language.lower().startswith("en")

        config = aai.TranscriptionConfig(
            speech_model=self.speech_model,
            language_code=language,
            speaker_labels=True,
            punctuate=True,
            format_text=True,
            auto_chapters=is_english,
            summarization=is_english,
            summary_model=(
                aai.types.SummarizationModel.conversational if is_english else None
            ),
            summary_type=(
                aai.types.SummarizationType.bullets_verbose if is_english else None
            ),
        )

        t0 = time.perf_counter()
        transcript = aai.Transcriber(config=config).transcribe(str(audio_path))
        timings["transcribe_s"] = time.perf_counter() - t0

        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(
                f"AssemblyAI transcription failed: {transcript.error}"
            )

        canonical_transcript = _to_canonical_transcript(
            transcript, source_audio=str(audio_path), language=language
        )

        t1 = time.perf_counter()
        lemur_raw, summary = _summarize_with_lemur(
            transcript, language=language, model=self.lemur_model
        )
        timings["lemur_s"] = time.perf_counter() - t1

        # Enrich tldr/topics from native AssemblyAI summary/chapters when
        # available (English only).
        if is_english and getattr(transcript, "summary", None) and not summary.tldr:
            summary.tldr = [
                line.lstrip("-•* ").strip()
                for line in str(transcript.summary).splitlines()
                if line.strip()
            ][:5]

        if is_english and getattr(transcript, "chapters", None) and not summary.topics:
            summary.topics = [
                Topic(
                    title=(c.headline or c.gist or "Chapter"),
                    bullets=[
                        b.strip()
                        for b in (c.summary or "").split(". ")
                        if b.strip()
                    ],
                    segment_range=(_ms_to_s(c.start), _ms_to_s(c.end)),
                )
                for c in transcript.chapters
            ]

        meta = RunMeta(
            run_id=run_dir.name,
            backend=self.name,
            input_path=str(audio_path),
            input_sha256=sha256_of(audio_path),
            duration=canonical_transcript.duration,
            timings=timings,
            model_versions={
                "assemblyai_speech": str(self.speech_model),
                "lemur": str(self.lemur_model),
            },
            prompt_version=PROMPT_VERSION,
            extra={
                "transcript_id": transcript.id,
                "language_code": language,
                "lemur_raw": lemur_raw,
            },
        )

        result = RunResult(
            transcript=canonical_transcript, summary=summary, meta=meta
        )
        write_run(run_dir, result)
        return result


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _ms_to_s(ms: int | float | None) -> float:
    return float(ms or 0) / 1000.0


def _speaker_label(raw: str | None) -> str | None:
    if not raw:
        return None
    raw = str(raw).strip()
    if raw.upper().startswith("SPEAKER"):
        return raw.upper()
    return f"SPEAKER_{raw}"


def _to_canonical_transcript(
    t: aai.Transcript, *, source_audio: str, language: str
) -> Transcript:
    segments: list[Segment] = []
    speaker_order: list[str] = []
    seen: set[str] = set()
    duration = _ms_to_s(getattr(t, "audio_duration", 0))

    utterances = getattr(t, "utterances", None) or []
    if utterances:
        for u in utterances:
            sp = _speaker_label(getattr(u, "speaker", None))
            if sp and sp not in seen:
                seen.add(sp)
                speaker_order.append(sp)
            words: list[Word] = []
            for w in getattr(u, "words", None) or []:
                w_sp = _speaker_label(getattr(w, "speaker", None)) or sp
                words.append(
                    Word(
                        text=w.text,
                        start=_ms_to_s(w.start),
                        end=_ms_to_s(w.end),
                        speaker=w_sp,
                        confidence=getattr(w, "confidence", None),
                    )
                )
            segments.append(
                Segment(
                    start=_ms_to_s(u.start),
                    end=_ms_to_s(u.end),
                    speaker=sp,
                    text=(u.text or "").strip(),
                    words=words,
                )
            )
    elif getattr(t, "text", None):
        segments.append(
            Segment(start=0.0, end=duration, speaker=None, text=t.text or "", words=[])
        )

    if not duration and segments:
        duration = max(s.end for s in segments)

    return Transcript(
        language=language,
        duration=duration,
        speakers=speaker_order,
        segments=segments,
        source_audio=source_audio,
        backend="assemblyai",
        backend_meta={
            "transcript_id": t.id,
            "audio_url": getattr(t, "audio_url", None),
            "speech_model": str(DEFAULT_SPEECH_MODEL),
        },
    )


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction from a possibly noisy LLM response."""
    candidates: list[str] = []
    for m in _JSON_FENCE_RE.finditer(text):
        candidates.append(m.group(1))
    candidates.append(text)
    obj_match = _JSON_OBJECT_RE.search(text)
    if obj_match:
        candidates.append(obj_match.group(0))

    last_err: Exception | None = None
    for c in candidates:
        try:
            data = json.loads(c)
        except json.JSONDecodeError as e:
            last_err = e
            continue
        if isinstance(data, dict):
            return data
    raise ValueError(f"LeMUR did not return valid JSON. Last error: {last_err}")


def _summarize_with_lemur(
    transcript: aai.Transcript, *, language: str, model: str
) -> tuple[dict[str, Any], Summary]:
    prompt = _LEMUR_PROMPT_NL if language.lower().startswith("nl") else _LEMUR_PROMPT_EN
    response = transcript.lemur.task(
        prompt=prompt,
        final_model=model,
        max_output_size=4000,
        temperature=0.0,
    )
    raw_text = getattr(response, "response", None) or str(response)
    raw: dict[str, Any] = {
        "request_id": getattr(response, "request_id", None),
        "response": raw_text,
    }

    try:
        data = _extract_json(raw_text)
    except ValueError:
        data = {}

    fallback_title = (
        "Vergaderingsamenvatting"
        if language.lower().startswith("nl")
        else "Meeting summary"
    )

    summary = Summary(
        title=str(data.get("title") or fallback_title),
        tldr=[str(b) for b in (data.get("tldr") or []) if str(b).strip()],
        topics=[
            Topic(
                title=str(item.get("title") or ""),
                bullets=[str(b) for b in (item.get("bullets") or []) if str(b).strip()],
            )
            for item in (data.get("topics") or [])
            if isinstance(item, dict) and (item.get("title") or item.get("bullets"))
        ],
        decisions=_parse_decisions(data.get("decisions") or []),
        action_items=_parse_action_items(data.get("action_items") or []),
        open_questions=[str(q) for q in (data.get("open_questions") or []) if str(q).strip()],
        next_steps=[str(s) for s in (data.get("next_steps") or []) if str(s).strip()],
        language=language,
        summarizer_backend=f"assemblyai-lemur:{model}",
        prompt_version=PROMPT_VERSION,
    )
    return raw, summary


def _parse_decisions(items: list[Any]) -> list[Decision]:
    out: list[Decision] = []
    for d in items:
        text = (
            str(d.get("text") or "").strip()
            if isinstance(d, dict)
            else str(d).strip()
        )
        if text:
            out.append(Decision(text=text))
    return out


def _parse_action_items(items: list[Any]) -> list[ActionItem]:
    out: list[ActionItem] = []
    for a in items:
        if not isinstance(a, dict):
            continue
        task = str(a.get("task") or "").strip()
        if not task:
            continue
        owner = a.get("owner")
        due = a.get("due")
        out.append(
            ActionItem(
                task=task,
                owner=str(owner).strip() if isinstance(owner, str) and owner.strip() else None,
                due=str(due).strip() if isinstance(due, str) and due.strip() else None,
            )
        )
    return out
