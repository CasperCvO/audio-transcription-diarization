"""Test CLI transcribe command with custom backend.

Uses typer.testing.CliRunner with monkeypatched CustomPipeline.transcribe_only
to verify that the CLI dispatches with the correct kwargs and that custom-only
flags are rejected when using the assemblyai backend.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from meetings.cli import app

runner = CliRunner()


@pytest.fixture()
def fake_audio(tmp_path: Path) -> Path:
    """Provide a tiny WAV fixture for CLI testing."""
    audio = tmp_path / "demo.wav"
    audio.parent.mkdir(parents=True, exist_ok=True)
    # Write a minimal WAV header (44 bytes) to satisfy existence checks
    audio.write_bytes(
        b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
        b"\x01\x00\x01\x00\x44\xAC\x00\x00\x88\x58\x01\x00"
        b"\x02\x00\x10\x00data\x00\x00\x00\x00"
    )
    return audio


@pytest.fixture()
def mock_transcribe_only(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Monkeypatch CustomPipeline.transcribe_only to track calls."""
    from meetings.pipelines import custom

    mock = MagicMock()
    # Make the mock return a dummy Transcript
    from meetings.schema import Segment, Transcript, Word

    dummy_transcript = Transcript(
        language="nl",
        duration=1.0,
        segments=[
            Segment(
                start=0.0,
                end=1.0,
                speaker="SPEAKER_00",
                text="Test.",
                words=[Word(text="Test.", start=0.0, end=1.0, speaker="SPEAKER_00")],
            )
        ],
        source_audio="demo.wav",
        backend="custom:test",
    )
    mock.return_value = dummy_transcript
    monkeypatch.setattr(custom.CustomPipeline, "transcribe_only", mock)
    return mock


def test_transcribe_custom_backend_dispatches_correctly(
    tmp_path: Path,
    fake_audio: Path,
    mock_transcribe_only: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI with --backend custom dispatches to CustomPipeline.transcribe_only with right kwargs."""
    # Also need to stub prepare_audio and audio_meta to avoid actual file processing
    from meetings import audio as audio_mod
    from meetings.config import get_settings
    from meetings.pipelines import custom as custom_mod

    def _fake_prepare(src: Path, dst_dir: Path | None = None, **_: object) -> Path:
        return src

    def _fake_meta(path: Path) -> audio_mod.AudioMeta:
        return audio_mod.AudioMeta(path=path, sha256="0" * 64, duration=1.0, bytes_=42)

    monkeypatch.setattr(custom_mod, "prepare_audio", _fake_prepare)
    monkeypatch.setattr(custom_mod, "audio_meta", _fake_meta)

    # Stub extract_speaker_snippets to avoid actual audio processing
    def _fake_snippets(
        transcript: object,
        audio_path: object,
        run_dir: Path,
        top_n: int = 3,
        **_: object,
    ) -> dict[str, list[Path]]:
        snippets_dir = run_dir / "snippets"
        snippets_dir.mkdir(parents=True, exist_ok=True)
        return {}

    monkeypatch.setattr(custom_mod, "extract_speaker_snippets", _fake_snippets)

    # Mock settings to use tmp_path as transcription directory
    def _fake_settings():
        settings = get_settings()
        return settings.model_copy(update={"transcription_dir": tmp_path})

    monkeypatch.setattr("meetings.cli.get_settings", _fake_settings)

    # Run the CLI command
    result = runner.invoke(
        app,
        [
            "transcribe",
            "--audio",
            str(fake_audio),
            "--backend",
            "custom",
            "--transcriber",
            "whisper-1",
            "--diarizer",
            "pyannoteai",
            "--language",
            "de",
            "--snippets",
            "5",
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert mock_transcribe_only.called

    # Verify the call kwargs
    call_args = mock_transcribe_only.call_args
    assert call_args[0][0] == fake_audio  # audio argument
    assert call_args[0][1].is_relative_to(tmp_path)  # run_dir is under tmp_path

    # Verify keyword arguments
    kwargs = call_args[1]
    assert kwargs["language"] == "de"
    assert kwargs["snippets_per_speaker"] == 5


def test_transcribe_custom_backend_with_defaults(
    tmp_path: Path,
    fake_audio: Path,
    mock_transcribe_only: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI with --backend custom uses default transcriber/diarizer when not specified."""
    from meetings import audio as audio_mod
    from meetings.config import get_settings
    from meetings.pipelines import custom as custom_mod

    def _fake_prepare(
        src: Path, dst_dir: Path | None = None, **_: object
    ) -> Path:
        return src

    def _fake_meta(path: Path) -> audio_mod.AudioMeta:
        return audio_mod.AudioMeta(
            path=path, sha256="0" * 64, duration=1.0, bytes_=42
        )

    def _fake_snippets(
        transcript: object,
        audio_path: object,
        run_dir: Path,
        top_n: int = 3,
        **_: object,
    ) -> dict[str, list[Path]]:
        snippets_dir = run_dir / "snippets"
        snippets_dir.mkdir(parents=True, exist_ok=True)
        return {}

    monkeypatch.setattr(custom_mod, "prepare_audio", _fake_prepare)
    monkeypatch.setattr(custom_mod, "audio_meta", _fake_meta)
    monkeypatch.setattr(custom_mod, "extract_speaker_snippets", _fake_snippets)

    # Mock settings to use tmp_path as transcription directory
    def _fake_settings():
        settings = get_settings()
        return settings.model_copy(update={"transcription_dir": tmp_path})

    monkeypatch.setattr("meetings.cli.get_settings", _fake_settings)

    result = runner.invoke(
        app,
        [
            "transcribe",
            "--audio",
            str(fake_audio),
            "--backend",
            "custom",
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert mock_transcribe_only.called


def test_transcribe_assemblyai_backend_with_custom_flags_errors(
    tmp_path: Path, fake_audio: Path
) -> None:
    """Passing --transcriber with --backend assemblyai raises BadParameter."""
    result = runner.invoke(
        app,
        [
            "transcribe",
            "--audio",
            str(fake_audio),
            "--backend",
            "assemblyai",
            "--transcriber",
            "whisper-1",
        ],
    )

    assert result.exit_code != 0
    assert "--transcriber is only available with --backend custom" in result.stdout


def test_transcribe_assemblyai_backend_with_diarizer_flag_errors(
    tmp_path: Path, fake_audio: Path
) -> None:
    """Passing --diarizer with --backend assemblyai raises BadParameter."""
    result = runner.invoke(
        app,
        [
            "transcribe",
            "--audio",
            str(fake_audio),
            "--backend",
            "assemblyai",
            "--diarizer",
            "pyannote_local",
        ],
    )

    assert result.exit_code != 0
    assert "--diarizer is only available with --backend custom" in result.stdout


def test_transcribe_assemblyai_backend_default_works(
    tmp_path: Path, fake_audio: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AssemblyAI backend works without custom flags (backward compatibility)."""
    from meetings.config import get_settings
    from meetings.pipelines import assemblyai

    # Mock AssemblyAIPipeline.transcribe to avoid actual API call
    mock_transcribe = MagicMock()
    from meetings.schema import Segment, Transcript, Word

    dummy_transcript = Transcript(
        language="nl",
        duration=1.0,
        segments=[
            Segment(
                start=0.0,
                end=1.0,
                speaker="A",
                text="Test.",
                words=[Word(text="Test.", start=0.0, end=1.0, speaker="A")],
            )
        ],
        source_audio="demo.wav",
        backend="assemblyai",
    )
    mock_transcribe.return_value = dummy_transcript
    monkeypatch.setattr(assemblyai.AssemblyAIPipeline, "transcribe", mock_transcribe)

    # Also stub extract_speaker_snippets
    from meetings.pipelines import assemblyai as assemblyai_mod

    def _fake_snippets(
        transcript: object,
        audio_path: object,
        run_dir: Path,
        top_n: int = 3,
        **_: object,
    ) -> dict[str, list[Path]]:
        snippets_dir = run_dir / "snippets"
        snippets_dir.mkdir(parents=True, exist_ok=True)
        return {}

    monkeypatch.setattr(assemblyai_mod, "extract_speaker_snippets", _fake_snippets)

    # Mock settings to use tmp_path as transcription directory
    def _fake_settings():
        settings = get_settings()
        return settings.model_copy(update={"transcription_dir": tmp_path})

    monkeypatch.setattr("meetings.cli.get_settings", _fake_settings)

    result = runner.invoke(
        app,
        [
            "transcribe",
            "--audio",
            str(fake_audio),
            # No --backend specified, should default to assemblyai
        ],
    )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert mock_transcribe.called
