"""Internal helpers shared across summarize.claude and summarize.names."""

from __future__ import annotations

import json
import re
from typing import Any, cast

_JSON_FENCE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def parse_json(text: str) -> dict[str, Any]:
    """Parse a JSON object out of a model response, tolerating code fences."""
    candidates: list[str] = []
    m = _JSON_FENCE.search(text)
    if m:
        candidates.append(m.group(1))
    candidates.append(text)
    for body in candidates:
        body = body.strip()
        start = body.find("{")
        end = body.rfind("}")
        if start != -1 and end != -1 and end > start:
            body = body[start : end + 1]
        try:
            return cast(dict[str, Any], json.loads(body))
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Could not parse JSON from model response: {text[:300]!r}")
