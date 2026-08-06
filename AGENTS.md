# Project guidance

## Purpose

This repository contains a Python command-line tool that detects lightning
events in videos, ranks frames containing visible lightning channels, and
exports reviewable stills and machine-readable results.

## Architecture

- Production code lives in `src/lightning_extractor/`.
- The CLI entry point is `lightning_extractor.cli:main` and is exposed as
  `lightning` through `pyproject.toml`.
- Default analysis settings live in `config/default.toml`.
- Tests live in `tests/` and use the standard-library `unittest` runner.
- Root-level Python scripts are the original research prototype. Preserve them
  as reference material unless a task explicitly asks to migrate or remove
  them.
- `analysis/`, `output/`, and `previews/` contain legacy generated artifacts and
  are not production source code.

## Inputs and outputs

- Never commit raw videos, generated runs, caches, or exported media.
- Original videos are read in place and must never be modified.
- `data/inbox/` is an optional local inbox and is ignored by Git except for its
  `.gitkeep` file.
- New analyses write isolated runs below `runs/<video>-<source-id>/`.
- Keep regenerable caches, machine-readable results, review material, and final
  exports conceptually separate.
- JSON is the canonical structured output; CSV is a convenience export.
- Use ffprobe-derived metadata instead of hard-coding frame rate, duration,
  resolution, codecs, or audio availability.

## Development commands

Preferred setup and checks:

```bash
uv sync --extra dev
uv run python -m unittest discover -s tests -v
uv run ruff check .
```

Useful CLI smoke test against a short known section of the local reference
video, when that video is available:

```bash
uv run lightning inspect GX010422.mp4
uv run lightning analyze GX010422.mp4 --start 260 --end 264 --top 5
```

The known event around `262.05s` should be detected; the clear channel frame
around `262.24s` is a useful ranking regression reference.

## Implementation principles

- Keep Python as the orchestration and analysis language unless profiling shows
  a native-language rewrite is justified.
- Prefer FFmpeg/ffprobe for media inspection and rendering, OpenCV/NumPy for
  image analysis, and the Python standard library for lightweight plumbing.
- Process long videos sequentially where practical. Avoid repeated random seeks
  in HEVC footage and avoid storing huge text-based per-frame metrics.
- Express time windows in seconds and derive frame counts from the probed FPS.
- Keep thresholds configurable because cameras, exposure, noise, frame rate,
  and weather conditions vary.
- Preserve or improve detection behavior with focused tests and short reference
  clips before optimizing algorithms.
- Handle videos without audio, empty detections, interrupted runs, and invalid
  media gracefully.

## Git and commits

- Use Conventional Commits for every commit.
- Split commits by coherent theme whenever meaningful. Do not combine unrelated
  scaffolding, features, tests, documentation, or fixes into one catch-all
  commit.
- Typical commit types include `feat:`, `fix:`, `refactor:`, `test:`, `docs:`,
  `chore:`, and `perf:`.
- Use concise imperative subjects, for example:
  - `chore: scaffold python project`
  - `feat: add configurable lightning analysis pipeline`
  - `test: cover channel geometry scoring`
  - `docs: document command line usage`
- Stage explicit paths when the worktree contains multiple themes.
- Run the relevant tests before committing and report any check that could not
  be run.
- Never commit large media or generated artifacts even if they are locally
  present.

