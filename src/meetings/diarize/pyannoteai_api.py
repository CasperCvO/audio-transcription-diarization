"""pyannoteAI Premium API diarization backend (plan B3).

Flow:
1. Request a pre-signed PUT URL for an opaque object key via
   ``POST /v1/media/input`` with ``{"url": "media://<key>"}``.
2. Upload the local audio file to that URL (``PUT`` with raw bytes).
3. Submit a diarization job via ``POST /v1/diarize`` with
   ``{"url": "media://<key>", "model": "precision-2", ...}``.
4. Poll ``GET /v1/jobs/{jobId}`` until ``status`` is terminal
   (``succeeded`` / ``failed`` / ``canceled``).
5. Map ``output.diarization`` → ``list[DiarizationTurn]``.

All keys and endpoints verified against docs.pyannote.ai (April 2026).
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import get_settings, require
from ..schema import DiarizationTurn

API_BASE = "https://api.pyannote.ai/v1"
DEFAULT_MODEL = "precision-2"
POLL_INTERVAL = 8.0  # seconds
POLL_TIMEOUT = 60 * 60  # 1 hour hard cap per job


class PyannoteAIError(RuntimeError):
    """Raised when the pyannoteAI API returns an error or times out."""


class PyannoteAIDiarizer:
    """Diarize audio through the pyannoteAI Premium REST API."""

    name: str = "pyannoteai"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        exclusive: bool = True,
        confidence: bool = False,
        timeout: float = POLL_TIMEOUT,
        poll_interval: float = POLL_INTERVAL,
    ) -> None:
        self.model = model
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.exclusive = exclusive
        self.confidence = confidence
        self.timeout = timeout
        self.poll_interval = poll_interval

    def _headers(self) -> dict[str, str]:
        key = require(get_settings().pyannoteai_api_key, "PYANNOTEAI_API_KEY")
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        reraise=True,
    )
    def _create_presigned(self, client: httpx.Client, object_key: str) -> str:
        resp = client.post(
            f"{API_BASE}/media/input",
            json={"url": f"media://{object_key}"},
            headers=self._headers(),
        )
        if resp.status_code >= 400:
            raise PyannoteAIError(f"media/input failed ({resp.status_code}): {resp.text}")
        data = resp.json()
        url = data.get("url")
        if not url:
            raise PyannoteAIError(f"media/input returned no url: {data}")
        return str(url)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        reraise=True,
    )
    def _upload(self, client: httpx.Client, put_url: str, audio: Path) -> None:
        with audio.open("rb") as f:
            resp = client.put(
                put_url,
                content=f.read(),
                headers={"Content-Type": "application/octet-stream"},
            )
        if resp.status_code >= 400:
            raise PyannoteAIError(f"PUT upload failed ({resp.status_code}): {resp.text}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        reraise=True,
    )
    def _submit_job(self, client: httpx.Client, media_url: str) -> str:
        body: dict[str, Any] = {"url": media_url, "model": self.model}
        if self.num_speakers is not None:
            body["numSpeakers"] = self.num_speakers
        if self.min_speakers is not None:
            body["minSpeakers"] = self.min_speakers
        if self.max_speakers is not None:
            body["maxSpeakers"] = self.max_speakers
        if self.exclusive:
            body["exclusive"] = True
        if self.confidence:
            body["confidence"] = True

        resp = client.post(f"{API_BASE}/diarize", json=body, headers=self._headers())
        if resp.status_code >= 400:
            raise PyannoteAIError(f"diarize submit failed ({resp.status_code}): {resp.text}")
        data = resp.json()
        job_id = data.get("jobId")
        if not job_id:
            raise PyannoteAIError(f"diarize submit returned no jobId: {data}")
        return str(job_id)

    def _poll(self, client: httpx.Client, job_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout
        while True:
            resp = client.get(
                f"{API_BASE}/jobs/{job_id}",
                headers={"Authorization": self._headers()["Authorization"]},
            )
            if resp.status_code >= 400:
                raise PyannoteAIError(f"job poll failed ({resp.status_code}): {resp.text}")
            data: dict[str, Any] = resp.json()
            status = data.get("status")
            if status == "succeeded":
                return data
            if status in {"failed", "canceled"}:
                raise PyannoteAIError(f"diarization job {status}: {data}")
            if time.monotonic() > deadline:
                raise PyannoteAIError(f"diarization job {job_id} exceeded {self.timeout}s timeout")
            time.sleep(self.poll_interval)

    def diarize(self, audio: Path) -> list[DiarizationTurn]:
        if not audio.exists():
            raise FileNotFoundError(audio)

        object_key = f"meetings/{uuid.uuid4().hex}/{audio.name}"
        media_url = f"media://{object_key}"

        with httpx.Client(timeout=httpx.Timeout(60.0, connect=30.0)) as client:
            put_url = self._create_presigned(client, object_key)
            self._upload(client, put_url, audio)
            job_id = self._submit_job(client, media_url)
            result = self._poll(client, job_id)

        output = result.get("output") or {}
        # Prefer exclusiveDiarization when we requested it (no overlapping speech).
        raw_turns = output.get("exclusiveDiarization") or output.get("diarization") or []
        turns: list[DiarizationTurn] = []
        for t in raw_turns:
            turns.append(
                DiarizationTurn(
                    start=float(t["start"]),
                    end=float(t["end"]),
                    speaker=str(t["speaker"]),
                )
            )
        return turns
