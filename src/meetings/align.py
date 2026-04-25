"""Word-level speaker assignment from word timestamps + diarization turns.

See `plan/02-track-b-custom-pipeline.md` task B4.
"""

from __future__ import annotations

from .schema import DiarizationTurn, Segment, Word


def assign_speakers(words: list[Word], turns: list[DiarizationTurn]) -> list[Word]:
    """Assign each word the diarization turn with maximum temporal overlap.

    Tie-break by the turn whose center is closest to the word center.
    """
    out: list[Word] = []
    for w in words:
        best_speaker: str | None = None
        best_overlap = 0.0
        best_center_dist = float("inf")
        w_center = (w.start + w.end) / 2
        for t in turns:
            overlap = max(0.0, min(w.end, t.end) - max(w.start, t.start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = t.speaker
                best_center_dist = abs(((t.start + t.end) / 2) - w_center)
            elif overlap == best_overlap and overlap == 0.0:
                # No overlap anywhere yet: track nearest by center distance.
                d = abs(((t.start + t.end) / 2) - w_center)
                if d < best_center_dist:
                    best_center_dist = d
                    best_speaker = t.speaker
        out.append(w.model_copy(update={"speaker": best_speaker}))
    return out


def group_into_segments(
    words: list[Word],
    *,
    silence_gap: float = 0.7,
) -> list[Segment]:
    """Group consecutive same-speaker words into segments.

    Splits when:
    - speaker changes
    - inter-word gap exceeds `silence_gap`
    - previous word ends with sentence-final punctuation
    """
    if not words:
        return []
    segments: list[Segment] = []
    cur: list[Word] = [words[0]]
    for prev, w in zip(words, words[1:], strict=False):
        gap = w.start - prev.end
        speaker_change = w.speaker != prev.speaker
        sentence_end = prev.text.rstrip().endswith((".", "?", "!"))
        if speaker_change or gap > silence_gap or sentence_end:
            segments.append(_segment_from_words(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        segments.append(_segment_from_words(cur))
    return segments


def _segment_from_words(ws: list[Word]) -> Segment:
    return Segment(
        start=ws[0].start,
        end=ws[-1].end,
        speaker=ws[0].speaker,
        text=" ".join(w.text for w in ws).strip(),
        words=ws,
    )
