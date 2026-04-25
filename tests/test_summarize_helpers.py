"""Tests for the JSON parser used to extract Claude responses."""

from __future__ import annotations

import pytest

from meetings.summarize._utils import fmt_ts, parse_json


def test_parse_json_plain() -> None:
    assert parse_json('{"a": 1}') == {"a": 1}


def test_parse_json_inside_code_fence() -> None:
    text = 'Hier is je samenvatting:\n```json\n{"x": [1, 2]}\n```\nKlaar.'
    assert parse_json(text) == {"x": [1, 2]}


def test_parse_json_strips_prose() -> None:
    text = 'Even nadenken... {"ok": true} (klaar)'
    assert parse_json(text) == {"ok": True}


def test_parse_json_invalid_raises() -> None:
    with pytest.raises(ValueError):
        parse_json("totaal geen json hier")


def test_fmt_ts_minutes_seconds() -> None:
    assert fmt_ts(0) == "00:00"
    assert fmt_ts(65) == "01:05"
    assert fmt_ts(3661) == "01:01:01"
