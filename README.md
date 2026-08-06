<p align="center">
  <img src="docs/assets/lightning-extractor-hero.png" alt="Lightning Strike Extractor detecting a lightning channel in video footage" width="100%">
</p>

<h1 align="center">⚡ Lightning Strike Extractor</h1>

<p align="center">
  <strong>Turn hours of storm footage into ranked lightning events, sharp stills, and structured data.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/"><img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white"></a>
  <a href="https://opencv.org/"><img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-powered-5C3EE8?logo=opencv&logoColor=white"></a>
  <a href="https://ffmpeg.org/"><img alt="FFmpeg" src="https://img.shields.io/badge/FFmpeg-required-007808?logo=ffmpeg&logoColor=white"></a>
  <img alt="Status: experimental" src="https://img.shields.io/badge/status-experimental-f59e0b">
</p>

---

Lightning Strike Extractor is a Python command-line tool for scanning video,
detecting sudden lightning flashes, and ranking the frames most likely to show
a thin, visible lightning channel. It reads the original video in place and
exports full-resolution stills alongside JSON and CSV results.

Built for demanding footage such as **4K, 100 fps HEVC recordings**—without
uploading the video or modifying the source file.

> [!NOTE]
> The project is currently experimental. Detection thresholds work well on the
> reference nighttime footage but will need tuning for other cameras, exposure
> settings, weather conditions, and shooting environments.

## What it does

```text
video.mp4
   │
   ├── inspect media metadata with ffprobe
   ├── detect abrupt luminance and frame changes
   ├── group nearby detections into lightning events
   ├── score short-lived line geometry around every event
   └── export ranked full-resolution frames and structured results
```

- Detects flash candidates against a rolling luminance baseline
- Separates broad cloud illumination from thin channel-like structures
- Adapts time windows to the video's probed frame rate
- Accepts arbitrary local video paths
- Supports partial time ranges for fast experiments
- Produces reproducible, isolated run directories
- Keeps thresholds in a human-readable TOML configuration

## Quick start

### Requirements

- Python 3.11 or newer
- [FFmpeg](https://ffmpeg.org/) with `ffprobe` available on `PATH`
- [uv](https://docs.astral.sh/uv/) is recommended

### Install

```bash
git clone https://github.com/andregasser/lightning-strike-extractor.git
cd lightning-strike-extractor
uv sync --extra dev
```

### Inspect a video

```bash
uv run lightning inspect /path/to/storm.mp4
```

Example metadata:

```json
{
  "duration_seconds": 3072.0,
  "video": {
    "codec_name": "hevc",
    "width": 3840,
    "height": 2160,
    "fps": 100.0
  },
  "has_audio": true
}
```

### Analyze it

```bash
uv run lightning analyze /path/to/storm.mp4 \
  --config config/default.toml
```

The same command scales to any number of files and directories:

```bash
uv run lightning analyze \
  /Volumes/Storms/camera-a.mp4 \
  /Volumes/Storms/camera-b.mov \
  /Volumes/Storms/archive \
  --recursive
```

Inspect discovery without decoding video:

```bash
uv run lightning analyze /Volumes/Storms \
  --recursive \
  --include "*.mp4" \
  --exclude "*preview*" \
  --dry-run
```

Use a short range while tuning thresholds:

```bash
uv run lightning analyze /path/to/storm.mp4 \
  --start 250 \
  --end 270 \
  --top 10
```

Long runs write regular checkpoints. If a run is interrupted, continue the
exact same source, range, and configuration with `--resume`:

```bash
uv run lightning analyze /path/to/storm.mp4 \
  --config config/default.toml \
  --resume
```

Progress includes processed frames or events, throughput, elapsed time, and an
estimated time remaining. Re-running an existing analysis without `--resume`
is rejected to prevent accidental overwrites.

Videos run sequentially by default, which is safest for large HEVC sources on
one disk. Controlled concurrency is available when the hardware can support it:

```bash
uv run lightning analyze /Volumes/Storms \
  --recursive \
  --jobs 2 \
  --resume
```

One invalid video is recorded as failed while the remaining batch continues.
Use `--fail-fast` when the batch should stop scheduling new videos after the
first failure. Progress can be emitted as `auto`, `interactive`, `plain`,
`json`, or `quiet`.

## Output

Every source and analysis setup gets a stable run directory based on the source,
time range, configuration, and tool version:

```text
runs/
├── batches/
│   └── batch-248f7728284d/
│       ├── batch.json       # batch lifecycle and scheduler settings
│       ├── inputs.json      # resolved, reproducible input set
│       ├── summary.json
│       └── summary.csv
└── videos/
    └── storm-a84f29c1-2bb55de739/
        ├── run.json         # status, phase, identity, timestamps, errors
        ├── source.json      # probed codec, dimensions, FPS, duration
        ├── config.json      # exact settings used for this run
        ├── cache/
        │   ├── flash-scan/
        │   └── channel-ranking.json
        ├── results/
        │   ├── events.json
        │   ├── events.csv
        │   ├── candidates.json
        │   ├── candidates.csv
        │   └── summary.json
        └── exports/
            ├── contact-sheet.jpg
            └── stills/
```

Original media is never copied or changed. Raw videos, run outputs, caches, and
legacy generated artifacts are excluded from Git.

Inspect accumulated state at any time:

```bash
uv run lightning runs list
uv run lightning runs list --status failed
uv run lightning runs show runs/batches/batch-248f7728284d
```

## Batch manifests

For large or repeatable collections, use a TOML manifest:

```bash
uv run lightning analyze --manifest storm-campaign.toml
```

```toml
[batch]
output = "runs"
jobs = 2
config = "config/default.toml"
resume = true

[[video]]
path = "/Volumes/Storms/camera-a.mp4"
start = 120.0
end = 1800.0

[[input]]
path = "/Volumes/Storms/archive"
recursive = true
include = ["*.mp4", "*.mov"]
exclude = ["*preview*", "*proxy*"]
```

Paths in a manifest are resolved relative to the manifest file. Each `video` or
`input` entry can override `config`, `start`, and `end`. See
[`examples/batch.example.toml`](examples/batch.example.toml) for a complete
template.

## Exit codes

| Code | Meaning |
| ---: | --- |
| `0` | All videos completed or were skipped |
| `1` | At least one video failed |
| `2` | Invalid arguments, configuration, or media |
| `3` | No supported videos were discovered |
| `4` | ffprobe is missing |
| `5` | Insufficient disk space |
| `130` | Interrupted by the user |

## How detection works

### 1. Flash detection

Frames are decoded sequentially at a reduced analysis resolution. Average
luminance, bright-pixel response, and frame-to-frame difference are compared
with a rolling baseline. Adaptive percentiles select exceptional changes, and
nearby hits collapse into a single event.

### 2. Channel ranking

The analyzer revisits a small window around each event. Local background
subtraction and temporal differencing isolate newly appearing bright ridges.
Hough line segments reward long, thin, multi-segment structures while broad
illumination is penalized.

### 3. Full-resolution export

Only the highest-ranked timestamps are read from the original video for final
JPEG export. This preserves source resolution without making the entire
analysis operate on 4K frames.

## Configuration

Defaults live in [`config/default.toml`](config/default.toml):

```toml
[analysis]
width = 960
baseline_seconds = 0.30
event_gap_seconds = 0.75
rise_percentile = 0.995
diff_percentile = 0.995
keep_frames_per_event = 3

[export]
top = 50
jpeg_quality = 96
```

Copy the file to create camera- or scenario-specific profiles. Available
settings cover event windows, adaptive cutoffs, geometry thresholds, event
limits, and export quality.

## Supported media

Media support follows the locally installed FFmpeg build. Common containers
include MP4, MOV, MKV, M4V, and AVI; common codecs include H.264, HEVC, ProRes,
AV1, and VP9. The analyzer probes the actual streams instead of trusting the
file extension.

## Development

```bash
uv sync --extra dev
uv run python -m unittest discover -s tests -v
uv run ruff check .
```

The root-level Python scripts are the original research prototype. Production
code lives in [`src/lightning_extractor`](src/lightning_extractor), and project
conventions are documented in [`AGENTS.md`](AGENTS.md).

## Roadmap

The public, continuously maintained roadmap lives in [`TODO.md`](TODO.md).
Current priorities are full-length validation on real footage, an interactive
review workflow, and exports for accepted stills and clips.

## Contributing

Issues and focused pull requests are welcome. Please keep changes thematic, add
tests for detection behavior, and use [Conventional
Commits](https://www.conventionalcommits.org/).

If you test the extractor with a new camera or storm scene, sharing the relevant
metadata and configuration is especially valuable—even when the result is a
false positive.
