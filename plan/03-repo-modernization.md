# 03 — Repo Modernization (shared foundation)

**Run this first.** Both tracks depend on the skeleton, schema, and CLI built here.

## Objective

Convert the current notebook-only repo into a clean, testable Python package
with `src/` layout, Pydantic schema, Typer CLI, and UV-managed deps. Keep the
existing notebooks for reference but stop relying on them as the entry point.

## Tasks

### S1. UV environment refresh
- [ ] Bump `pyproject.toml`:
  - Keep `requires-python = ">=3.12,<3.13"`.
  - Replace dependency list with the new core deps (see below).
  - Add `[project.scripts] meetings = "meetings.cli:app"`.
- [ ] Move to `src/` layout: configure `[tool.hatch.build.targets.wheel]
      packages = ["src/meetings"]` (or use `uv init --package`).
- [ ] `uv sync` clean.

**Core deps**
```
pydantic
pydantic-settings
typer[all]
rich
python-dotenv
httpx
tenacity            # retries for API calls
pydub               # audio chunking helper
```

**Stage-specific deps (added by the relevant track)**
```
assemblyai          # Track A
openai              # transcription + summarization options
anthropic           # summarization
google-genai        # alt summarization
elevenlabs          # alt transcription
deepgram-sdk        # alt transcription
pyannote.audio      # Track B fallback diarizer
```

**Dev deps**
```
ruff
mypy
pytest
pytest-asyncio
```

### S2. Source layout
- [ ] Create directories per `00-architecture.md`.
- [ ] Implement `schema.py` with the Pydantic models exactly as specified
      in `00-architecture.md` — both tracks must import from here.
- [ ] Implement `config.py` (pydantic-settings) reading `.env`.
- [ ] Implement `io.py` with `write_run(run_dir, transcript, summary, meta)`
      and `read_run(run_dir)`.

### S3. CLI
- [ ] `meetings/cli.py` with Typer:
  - `run --backend {assemblyai,custom} --audio PATH [--language nl] [stage flags]`
  - `validate PATH` — re-validate an existing `Transcription/<run_id>/` folder.
  - `compare RUN_A RUN_B` — see `04-evaluation-and-comparison.md`.
- [ ] Pretty progress with `rich`.

### S4. Lint, types, tests
- [ ] `ruff` config: line length 100, `select = ["E","F","I","UP","B","SIM"]`.
- [ ] `mypy --strict` passes for `src/meetings`.
- [ ] `pytest` discovers `tests/`. CI not required for this learning project.

### S5. Documentation
- [ ] Rewrite top-level `README.md` to describe the new tool and link to
      `plan/` for design notes.
- [ ] Add `.env.example` listing every supported key with empty values and
      a one-line comment per key.

### S6. Legacy handling
- [ ] Move existing notebooks into `notebooks/legacy/` and add a one-line
      header explaining they are kept for reference only.
- [ ] Keep `Audio/` and `Transcription/` at repo root and gitignored.

## Acceptance criteria

- `uv sync` succeeds on a fresh checkout.
- `uv run meetings --help` prints the CLI.
- `uv run pytest` is green (with live tests skipped when keys absent).
- `uv run ruff check .` and `uv run mypy src` are clean.
