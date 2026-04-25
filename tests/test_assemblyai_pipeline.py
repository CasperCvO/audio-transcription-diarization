"""Tests for the AssemblyAI Track A pipeline.

Live tests are skipped automatically when ``ASSEMBLYAI_API_KEY`` is not set
or when no Dutch sample audio file is present, so this test file is safe to
run on a clean checkout.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from meetings.config import get_settings
from meetings.io import new_run_id
from meetings.pipelines.assemblyai import (
    _extract_json,
    _parse_action_items,
    _parse_decisions,
    _summarize_with_lemur,  # noqa: F401  (imported to ensure module loads)
)

# --------------------------------------------------------------------------- #
# Pure-function unit tests (no network, always run)
# --------------------------------------------------------------------------- #


def test_extract_json_from_fenced_block() -> None:
    text = """Hier is het resultaat:
```json
{"title": "Test", "tldr": ["a", "b"]}
```
einde."""
    data = _extract_json(text)
    assert data["title"] == "Test"
    assert data["tldr"] == ["a", "b"]


def test_extract_json_from_raw_object() -> None:
    text = 'noise before {"title": "X", "next_steps": []} noise after'
    data = _extract_json(text)
    assert data["title"] == "X"
    assert data["next_steps"] == []


def test_extract_json_raises_on_garbage() -> None:
    with pytest.raises(ValueError):
        _extract_json("absolutely no json here")


def test_parse_action_items_filters_empty_and_normalizes() -> None:
    raw = [
        {"task": "Stuur agenda", "owner": "Casper", "due": "vrijdag"},
        {"task": "", "owner": "X"},  # filtered out
        {"task": "Plan vervolg", "owner": "  ", "due": None},
        "not a dict",  # filtered out
    ]
    items = _parse_action_items(raw)
    assert [a.task for a in items] == ["Stuur agenda", "Plan vervolg"]
    assert items[0].owner == "Casper"
    assert items[1].owner is None
    assert items[1].due is None


def test_parse_decisions_accepts_dict_or_string() -> None:
    raw = [{"text": "Begroting goedgekeurd"}, "Volgende meeting verzet", {}]
    decs = _parse_decisions(raw)
    assert [d.text for d in decs] == ["Begroting goedgekeurd", "Volgende meeting verzet"]


# --------------------------------------------------------------------------- #
# Live smoke test (skipped without key + sample)
# --------------------------------------------------------------------------- #


def _find_sample_audio() -> Path | None:
    """Look for a short Dutch sample audio file in expected locations."""
    settings = get_settings()
    candidates = [
        settings.audio_dir / "sample_nl_short.wav",
        Path("audio/test_sample/sample_nl_short.wav"),
        Path("audio/raw/sample_nl_short.wav"),
        Path("audio/processed/sample_nl_short.16k.mono.wav"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


@pytest.mark.skipif(
    not os.environ.get("ASSEMBLYAI_API_KEY") and not get_settings().assemblyai_api_key,
    reason="ASSEMBLYAI_API_KEY is not set",
)
def test_assemblyai_pipeline_smoke(tmp_path: Path) -> None:
    sample = _find_sample_audio()
    if sample is None:
        pytest.skip(
            "No Dutch sample audio found. Drop a 30–60s wav at "
            "audio/test_sample/sample_nl_short.wav to enable this test."
        )

    from meetings.pipelines.assemblyai import AssemblyAIPipeline

    pipeline = AssemblyAIPipeline()
    run_dir = tmp_path / new_run_id(sample, pipeline.name)
    run_dir.mkdir(parents=True, exist_ok=True)

    result = pipeline.run(sample, run_dir, language="nl")

    # Acceptance criteria from plan A7
    assert len(result.transcript.segments) > 0
    assert len(result.transcript.speakers) >= 1
    assert len(result.summary.tldr) >= 1

    for fname in ("transcript.json", "transcript.md", "summary.json", "summary.md", "meta.json"):
        assert (run_dir / fname).exists(), f"missing output: {fname}"
