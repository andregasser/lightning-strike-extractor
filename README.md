# Lightning Strike Extractor

Command-line tool for finding lightning flashes and ranking frames with visible,
thin lightning channels. It uses FFmpeg/ffprobe for media inspection and OpenCV
for frame analysis.

## Requirements

- Python 3.11 or newer
- FFmpeg including `ffprobe`

## Installation

Using [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
uv run lightning inspect /path/to/video.mp4
```

Or with a regular virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
lightning inspect /path/to/video.mp4
```

## Analyze a video

```bash
lightning analyze /path/to/video.mp4 --config config/default.toml
```

For a quick test or a selected portion of a long recording:

```bash
lightning analyze /path/to/video.mp4 --start 250 --end 270 --top 10
```

Every source video gets an isolated run directory:

```text
runs/<video>-<source-id>/
├── run.json
├── source.json
├── config.json
├── results/
│   ├── events.csv
│   ├── events.json
│   ├── candidates.csv
│   ├── candidates.json
│   └── summary.json
└── exports/
    ├── contact-sheet.jpg
    └── stills/
```

Original videos are read in place and are never modified or copied. `data/inbox/`
is an optional ignored convenience folder. Large inputs, generated runs and the
legacy prototype outputs are excluded from Git.

## Detection pipeline

1. Read the video sequentially at a reduced analysis resolution.
2. Compare luminance and frame differences with a rolling baseline.
3. Collapse nearby flash frames into events.
4. Inspect the frames around every event for newly appearing line structures.
5. Rank candidates and export the best frames from the original video.

The default values live in `config/default.toml`. Thresholds are deliberately
configurable because camera noise, exposure, frame rate and weather differ.

## Development

```bash
uv run python -m unittest discover -s tests
uv run ruff check .
```

The root-level Python scripts and existing `analysis/` and `output/` directories
are the original prototype and remain available as reference material. New work
belongs in `src/lightning_extractor/`.
