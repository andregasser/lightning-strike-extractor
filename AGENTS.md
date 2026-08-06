# Project guidance

## Purpose

This repository contains a Python command-line tool that detects lightning
events in videos, ranks frames containing visible lightning channels, and
exports reviewable stills and machine-readable results.

## Architecture

- Production code lives in `src/lightning_extractor/`.
- Batch discovery, manifests, scheduling, and summaries live in
  `src/lightning_extractor/batch.py` and `discovery.py`.
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
- New analyses write isolated video runs below
  `runs/videos/<video>-<source-id>-<analysis-id>/` and batch state below
  `runs/batches/<batch-id>/`. The analysis identity includes the source,
  selected time range, full configuration, and tool version.
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

## Quality and documentation

- The project must be well supported by automated tests. Add or update focused
  tests whenever behavior changes, a bug is fixed, or a new edge case is
  discovered.
- Prefer deterministic unit tests for scoring and configuration logic, plus
  short integration fixtures for end-to-end video behavior. Do not rely on the
  large local reference video as the only validation path.
- Treat detection-ranking regressions as product regressions. Preserve known
  reference behavior unless an intentional algorithm change is documented and
  validated.
- Review `README.md` for accuracy regularly, especially after changes to the
  CLI, installation steps, dependencies, configuration keys, output layout,
  supported media, project status, or roadmap.
- Do not leave planned or partially implemented behavior presented as finished
  functionality in the README. Examples and commands must be runnable against
  the current codebase.

## Git and commits

- Use Conventional Commits for every commit.
- Follow the canonical structure
  `<type>[optional scope][!]: <description>`, followed by a blank line, the
  required body, and optional footers after another blank line.
- Every commit must include both a concise Conventional Commit subject and a
  meaningful body. The body must explain what changed, why it changed, relevant
  user or developer impact, and how the change was validated. Prefer explaining
  intent and tradeoffs over restating the diff. Do not create subject-only
  commits, even though the upstream specification permits an optional body.
- Split commits by coherent theme whenever meaningful. Do not combine unrelated
  scaffolding, features, tests, documentation, or fixes into one catch-all
  commit.
- Use `feat` for user-visible features and `fix` for defects. Other accepted
  types include `build`, `chore`, `ci`, `docs`, `perf`, `refactor`, `revert`,
  `style`, and `test`. Use a short noun scope such as `detection`, `cli`, or
  `export` when it adds useful context; do not invent a scope merely to fill the
  field.
- Use concise imperative subjects without a trailing period, for example:
  - `chore: scaffold python project`
  - `feat(detection): rank saturated event frames`
  - `test(export): cover channel geometry scoring`
  - `docs: document command line usage`
- Mark breaking changes with `!` in the subject and a
  `BREAKING CHANGE: <description>` footer. Use Git-trailer-style footers for
  issue references or other metadata, for example `Refs: #42` or `Closes: #42`.
- A complete commit message should resemble:

  ```text
  feat(detection): rank saturated event frames

  Measure original-frame luminance through a channel template taken from
  adjacent confirmed frames. This keeps a saturated but visible return stroke
  eligible without allowing distant exposure flashes to inherit its geometry.

  Validate with the positive reference clips, synthetic motion negatives, the
  full unit suite, and Ruff.
  ```
- Stage explicit paths when the worktree contains multiple themes.
- Run the relevant tests before committing and report any check that could not
  be run.
- Documentation-only changes may skip runtime tests when they cannot affect
  behavior, but must still pass formatting and link/path sanity checks.
- Never commit large media or generated artifacts even if they are locally
  present.
