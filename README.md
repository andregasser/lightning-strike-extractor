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

The repository also contains an optional, fixed-configuration object-detector
runtime and offline tools for exporting candidate frames, generating COCO box
proposals, and handing annotation work to Label Studio. The currently pinned detector
checkpoint is a bootstrap model for dataset creation, not yet the final
lightning-trained DINO model.

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
├── labels/
│   └── review.json       # atomic manual-review decisions
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
        ├── review/
        │   └── previews/   # generated five-frame manual-review strips
        └── exports/
            ├── contact-sheet.jpg
            ├── stills/
            └── events/
                └── evt_000001_000075.630s/
                    ├── frame_-02.jpg
                    ├── frame_-01.jpg
                    ├── frame_peak.jpg
                    ├── frame_+01.jpg
                    ├── frame_+02.jpg
                    └── slow-motion.mp4
```

Original media is never copied or changed. Raw videos, run outputs, caches, and
legacy generated artifacts are excluded from Git.

Inspect accumulated state at any time:

```bash
uv run lightning runs list
uv run lightning runs list --status failed
uv run lightning runs show runs/batches/batch-248f7728284d
```

## Manual verification

Label the currently selected channel candidates after an analysis:

```bash
uv run lightning review runs
```

Audit recall by reviewing the strongest proposed frame from every raw flash
event, including events rejected by the export filter:

```bash
uv run lightning review runs --scope all-events
```

For each event, the command creates a five-frame strip from two frames before
the selected peak through two frames after it, opens the strip in the system
image viewer, and prompts for a label:

```text
Lightning channel? [y]es/[n]o/[u]ncertain/[q]uit:
```

Each answer is written atomically to `runs/labels/review.json`. Re-running the
command skips completed items whose reviewed frame is unchanged and continues
with the first pending event. If another scope proposes a different winner for
an event, that event is automatically presented again. Use
`--include-reviewed` to revisit existing labels, `--labels /private/path.json`
to choose another label file, or `--no-open` when preview paths should only be
printed. Review data and previews live below ignored run directories and are
never committed automatically.

## Fixed object-detector runtime

Install the optional detector dependencies without changing the lightweight core installation:

```bash
uv sync --extra detector --extra dev
```

Run the detector on an exported frame:

```bash
uv run lightning detector detect frame.jpg \
  --output detections.json \
  --preview detections.jpg
```

The runtime model, revision, class and validated confidence threshold are fixed by
`src/lightning_extractor/model_manifest.json`. They are included in every JSON result. The product
CLI intentionally provides no model, prompt, device, training or threshold controls, so the same
release produces reproducible results.

The current `0.1.0` manifest is explicitly marked `bootstrap`: it pins the Apache-2.0-licensed
`IDEA-Research/grounding-dino-tiny` checkpoint for integration work, but it is not the final
lightning-trained model. A production manifest will replace it only after the dedicated detector
has passed the reference-video evaluation. The first detector invocation downloads the pinned
checkpoint from Hugging Face; subsequent runs use the local model cache.

The bootstrap detector is suitable for annotation proposals only. Its boxes are not ground truth,
and its detections must not be treated as validated product results.

## Building the training dataset

For the current implementation status and the remaining work, see
[`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md). Label Studio is the active
annotation workflow; CVAT helpers are retained only for legacy compatibility.

Fine-tuning data uses standard COCO detection JSON. Each visible channel is one
`lightning_channel` bounding box in `[x, y, width, height]` format. Images with no annotations are
intentional negative examples.

Annotation, validation and training helpers live below `tools/model_development/`; they are not
part of the installed `lightning` CLI or runtime contract.

Run the complete atomic preparation workflow with the fixed proposal model:

```bash
uv run python -m tools.model_development.prepare_training_dataset runs \
  --output dataset
```

This produces `manifest.json`, exported images, unverified COCO proposals in
`annotations/proposals.json`, a validated `preparation.json` summary, and the ready-to-upload
`cvat/import.zip`. The output directory is published only after every step succeeds.

In CVAT, create a project or task by importing `cvat/import.zip` as `COCO 1.0`. Review every
`lightning_channel` box, delete false proposals, add missing channels, and leave true negative
images without a box. Export the corrected task again as `COCO 1.0` with images, then validate the
exported archive before training:

```bash
unzip corrected-coco.zip -d corrected-coco
uv run python -m tools.model_development.validate_coco \
  corrected-coco/annotations/instances_default.json \
  --images corrected-coco/images/default
```

CVAT may choose a subset name other than `default`; in that case use the corresponding
`instances_<subset>.json` and `images/<subset>` paths. Never train directly from the unverified
proposal file.

After that validation, import the manually corrected archive and create deterministic,
source-grouped splits:

```bash
uv run python -m tools.model_development.import_cvat_dataset \
  corrected-coco.zip \
  --output verified-dataset
```

The importer requires exactly one COCO instances subset and the `lightning_channel` category. It
recovers each source ID from the CVAT filename, rejects missing or ambiguous source identity, marks
the corrected annotations as verified, and atomically writes:

```text
verified-dataset/
├── manifest.json
├── annotations/
│   ├── instances_train.json
│   ├── instances_validation.json
│   └── instances_test.json
└── images/
    ├── train/
    ├── validation/
    └── test/
```

Default ratios are 70% training, 20% validation, and 10% test by image count. Whole source videos
are assigned to one split, so the exact ratios are approximate—especially with only a few source
videos. Override them with `--train-ratio`, `--validation-ratio`, and `--test-ratio`; the three
values must sum to `1.0`. Only run this command after completing the manual CVAT review because all
imported annotations are treated as verified.

Export event peaks and nearby context frames from completed analysis runs:

```bash
uv run python -m tools.model_development.export_training_frames runs \
  --output dataset
```

The exporter processes each source sequentially, keeps at most 100 event winners per video by
default, includes two adjacent frames on either side, deduplicates repeated absolute frames across
analysis runs, and records source IDs, candidate metrics, review labels, roles, and SHA-256 image
checksums in `dataset/manifest.json`. The generated `dataset/` directory is ignored by Git.

Create an initial COCO file of unverified box proposals from a recursively scanned image root:

```bash
uv run python -m tools.model_development.preannotate_coco \
  dataset/images --output dataset/annotations/proposals.json
```

Put each source video below its own first-level directory, for example
`dataset/images/storm-a/frame-001.jpg`. The proposal file records that directory as `source_id` and
marks every generated annotation as `verified: false`; proposals must be manually corrected before
they become training labels.

Package an already prepared dataset separately when needed:

```bash
uv run python -m tools.model_development.package_cvat_dataset dataset \
  --output dataset/cvat/import.zip
```

Label Studio is supported as an alternative review interface. Export its JSON prediction tasks,
label configuration, and locally served image directory with:

```bash
uv run python -m tools.model_development.export_label_studio dataset \
  --output label-studio-dataset
```

Start a local image server from another terminal:

```bash
uv run python -m tools.model_development.serve_label_studio \
  label-studio-dataset/serve
```

Create a Label Studio project using `label-studio-dataset/project/label-config.xml`, then import
`label-studio-dataset/import/tasks.json`. The `import/` directory deliberately contains no images;
do not upload `serve/` through the Label Studio import dialog. The default task URLs point to
`http://localhost:8001/images`; use `--image-base-url` when Label Studio must reach the images at a
different HTTP or HTTPS address. Keep the image server running while labeling. Imported boxes are
model predictions, not verified annotations, and must all be reviewed.

Export completed work from Label Studio as full `JSON`—not `JSON_MIN`, COCO, or CSV—and convert the
human annotations to verified, source-grouped COCO splits:

```bash
uv run python -m tools.model_development.import_label_studio_dataset \
  label-studio-export.json \
  --images dataset/images \
  --output verified-dataset
```

The importer requires exactly one completed, non-cancelled annotation per task. An empty human
annotation is preserved as a verified negative image; model predictions stored alongside it are
never promoted to ground truth. Rectangle percentages are converted back to pixel coordinates and
validated against the original image dimensions before the atomic split output is published.

Keep frames from the same source video in one split. Randomly distributing adjacent frames across
training, validation, and test sets would produce misleadingly strong evaluation results. The
source ID is preserved in the CVAT and Label Studio handoffs for this purpose.

Training is not implemented yet. The current development pipeline ends with a manually corrected,
validated, and source-grouped COCO dataset.

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

Only candidates above the configured geometry threshold are exported, with one
representative frame per event by default. `top` is an upper limit rather than
a target that is filled with weak candidates. All candidates remain available
in JSON and CSV. Exported JPEGs preserve source resolution without making the
entire analysis operate on 4K frames.

## Configuration

Defaults live in [`config/default.toml`](config/default.toml):

```toml
[analysis]
width = 960
baseline_seconds = 0.30
event_gap_seconds = 0.75
event_window_before = 0.40
event_window_after = 0.40
rise_percentile = 0.995
diff_percentile = 0.995
minimum_rise = 1.0
minimum_difference = 0.8
minimum_high_rise = 3.0
max_events = 0
keep_frames_per_event = 0
checkpoint_seconds = 30.0

[channel]
analysis_width = 1920
stabilization_enabled = true
stabilization_width = 640
stabilization_max_features = 1200
stabilization_min_matches = 24
stabilization_min_inlier_ratio = 0.45
stabilization_ransac_threshold = 2.5
stabilization_orb_max_residual = 3.0
stabilization_ecc_enabled = true
stabilization_min_ecc_correlation = 0.90
stabilization_max_translation_fraction = 0.08
stabilization_max_rotation_degrees = 5.0
stabilization_max_scale_change = 0.05
stabilization_mask_aligned_edges = true
stabilization_edge_mask_dilation = 5
multiframe_enabled = true
multiframe_width = 640
multiframe_window_seconds = 0.06
multiframe_dilation_pixels = 3
multiframe_bonus_weight = 0.25
multiframe_template_min_support = 0.5
multiframe_peak_radius_frames = 2
ridge_threshold = 10
bright_area_threshold = 20
minimum_line_length = 12
maximum_line_gap = 8

[export]
top = 50
minimum_geometry_score = 50.0
minimum_channel_length = 100.0
minimum_line_segments = 3
minimum_channel_strength = 15.0
maximum_channel_thickness = 3.5
minimum_strong_geometry_score = 500.0
minimum_long_channel_length = 300.0
maximum_clean_channel_bright_area = 5000.0
minimum_clean_line_segments = 5
minimum_peak_geometry_score = 500.0
minimum_peak_channel_length = 200.0
one_frame_per_event = true
minimum_winner_geometry_ratio = 0.0
jpeg_quality = 96
contact_sheet_columns = 5
contact_sheet_context_frames = 2
contact_sheet_context_stride = 1
contact_sheet_include_overlay = true
event_frames_enabled = true
slow_motion_enabled = true
slow_motion_before_seconds = 0.25
slow_motion_after_seconds = 0.25
slow_motion_factor = 4.0
slow_motion_output_fps = 25
slow_motion_crf = 18
```

`keep_frames_per_event = 0` enables exhaustive selection: every decoded frame
from the dynamic interval around an event is measured and retained in the
candidate data. The exporter first rejects geometrically implausible frames,
then chooses the remaining frame with the greatest combined channel length,
quadratically weighted original-frame luminance, connected branching, and
clarity. Temporal response validates that a channel is new, but no longer acts
as a proxy for its visible brightness. Thickness receives only a linear clarity
penalty so that natural bloom around a powerful channel is not punished twice.
A channel must meet minimum length, line-segment, strength, and thickness
requirements. It must additionally be geometrically strong, genuinely long, or
cleanly isolated from a diffuse frame-wide brightening. Multi-frame overlap can
no longer replace missing channel geometry. A bright or saturated peak remains
eligible only when its exact adjacent template frame independently passes the
same strict channel test. This preserves powerful sky-illuminating return
strokes without promoting ordinary cloud illumination. The selected peak must
also retain visible evidence of its own: a 200-pixel channel, a geometry score
of 500, or a clean locally isolated channel. This prevents a valid neighbouring
template from promoting a peak that contains only diffuse cloud light.
A positive value restores an
optional early per-event limit for faster exploratory runs.

Before comparing an event frame with its pre-event background, the channel
ranker aligns the background with an affine camera-motion estimate. ORB feature
matches and RANSAC make the estimate robust against a newly appearing lightning
channel. When feature matches are ambiguous or leave excessive residual error,
ECC intensity alignment provides a second estimate. Translation, rotation,
scale, match quality, and correlation are accepted only within configured safety
limits; an implausible transform causes a safe fallback to the unaligned
comparison. Static-edge masking removes sub-pixel interpolation remnants after
successful alignment. Stabilization is analysis-only: exported JPEGs always
contain the original, unwarped video frame.

Multi-frame support compares each detected channel mask with the masks from the
immediately surrounding frames. Spatially recurring channel geometry receives
a bounded event-ranking bonus, helping distinguish a real developing discharge
from a one-frame artifact. Isolated channels are not penalized, so an extremely
short lightning strike remains eligible. Within each event, single-frame
quality remains the primary winner criterion; multi-frame support only resolves
a tie. Compact masks limit memory use, and the selected export is the strongest
original single frame rather than a synthetic frame stack.

For saturated peak frames where local differencing can temporarily lose the
channel itself, a strongly supported neighbouring channel mask acts as an event
template. The tool measures every original frame through that shared geometry,
allowing the truly brightest instant to win even when sky illumination reduces
local contrast. Template transfer is limited to two direct neighbouring frames
so a nearby broad exposure flash cannot inherit unrelated channel geometry. The
exported image remains the untouched original frame.

The contact sheet uses one row per selected event by default. Two original
frames before and after the winner surround a yellow-marked `PEAK` frame. The
peak label includes event ID, quality, geometry, channel length and strength,
branch count, and multi-frame support. An adjacent debug view marks the exact
channel mask used by the ranker in magenta with a yellow outline. The context
count, frame stride, and overlay are configurable, making temporal evolution
and scoring errors easy to verify without exporting review images as final
stills.

Each selected event also receives a high-resolution export directory. By
default it contains the same five original frames shown around the contact-sheet
peak, plus a two-second H.264 slow-motion clip made from 0.25 seconds before and
after the peak at four-times slowdown. Context count and stride control both the
contact sheet and high-resolution frame sequence. Slow-motion window, factor,
output frame rate, and quality are independently configurable. Audio is omitted
from these short review clips.

Copy the file to create camera- or scenario-specific profiles. Available
settings cover event windows, adaptive cutoffs, geometry thresholds, event
limits, and export quality.

## Supported media

Input support has two separate layers:

1. Discovery accepts an explicit set of filename extensions.
2. The contained video stream must be readable by both the local `ffprobe`
   installation and OpenCV's video backend.

### Explicitly discovered containers

| Container | Extensions | Status |
| --- | --- | --- |
| MPEG-4 / ISO Base Media | `.mp4`, `.m4v` | Explicitly supported |
| QuickTime | `.mov` | Explicitly supported |
| Matroska | `.mkv` | Explicitly supported |
| WebM | `.webm` | Explicitly supported |
| AVI | `.avi` | Explicitly supported |

Extension matching is case-insensitive. Files with other extensions are not
selected during file or directory discovery, even when the installed FFmpeg
build could technically decode them.

### Codecs

The project does not maintain a codec whitelist. Codec availability depends on
the OpenCV wheel/platform and the locally installed media backend. The current
reference footage verifies **HEVC/H.265 in MP4 at 4K and 100 fps** on macOS, and
the automated fixtures verify **Motion JPEG in AVI**.

These common combinations are expected to work when the local OpenCV backend
supports them, but are not all covered by project fixtures yet:

- H.264/AVC in MP4, MOV, or MKV
- HEVC/H.265 in MP4, MOV, or MKV
- VP8 or VP9 in WebM or MKV
- AV1 in MP4, WebM, or MKV
- ProRes in MOV
- Motion JPEG in AVI or MOV

`ffprobe` successfully reading a file is necessary but not sufficient: OpenCV
performs the actual frame decoding. Run both a metadata check and a short
analysis before committing to a long batch:

```bash
uv run lightning inspect /path/to/video.mp4
uv run lightning analyze /path/to/video.mp4 --start 0 --end 30
```

### Common formats not explicitly supported

The following common containers are currently excluded by discovery and must
not be presented as supported:

| Format | Typical extensions | Current behavior |
| --- | --- | --- |
| MPEG transport stream / AVCHD | `.ts`, `.mts`, `.m2ts` | Not discovered |
| MPEG program stream / DVD video | `.mpg`, `.mpeg`, `.vob` | Not discovered |
| Windows Media / ASF | `.wmv`, `.asf` | Not discovered |
| Flash Video | `.flv`, `.f4v` | Not discovered |
| 3GPP | `.3gp`, `.3g2` | Not discovered |
| Ogg video | `.ogv` | Not discovered |
| Material Exchange Format | `.mxf` | Not discovered |
| Animated images | `.gif`, animated WebP | Not supported as video input |
| Image sequences | numbered JPEG, PNG, TIFF, EXR | Not supported |
| Camera RAW video | `.braw`, `.r3d`, and similar | Not supported |
| Network streams | RTSP, HLS URLs, HTTP URLs | Not supported; inputs must be local files |

Renaming one of these files to a supported extension is not a reliable
workaround. Support requires adding the extension to discovery, verifying
ffprobe and OpenCV decoding, and adding a regression fixture.

### Additional limitations

- A readable video stream and a positive reported frame rate are required.
- Audio is optional and ignored during analysis.
- Constant-frame-rate video is the tested path. Variable-frame-rate input uses
  the reported average frame rate and is not yet fully validated.
- HDR, 10/12-bit, alpha-channel, and unusual pixel formats may be converted by
  the local decoder; detection quality for them is not currently guaranteed.
- Encrypted or DRM-protected media is not supported.
- Corrupt or truncated input is marked as failed instead of silently producing
  a complete run.

## Development

```bash
uv sync --extra dev
uv run python -m unittest discover -s tests -v
uv run ruff check .
```

Production code lives in [`src/lightning_extractor`](src/lightning_extractor),
model-development utilities in [`tools/model_development`](tools/model_development),
and the original research scripts in [`legacy/prototypes`](legacy/prototypes).
Project conventions are documented in [`AGENTS.md`](AGENTS.md).

Generated data stays out of the source tree: current analyses use ignored
`runs/`, training exports use ignored `dataset/`, and historical local results
are consolidated below ignored `artifacts/archive/`. The archive is local and
reversible; it is not part of the repository or production source code.

## Roadmap

The public, continuously maintained roadmap lives in [`TODO.md`](TODO.md).
Current priorities are full-length validation on real footage, an interactive
annotation pass in CVAT, source-grouped dataset splits, DINO training and evaluation, and replacing
the bootstrap manifest with a validated production checkpoint.

## Contributing

Issues and focused pull requests are welcome. Please keep changes thematic, add
tests for detection behavior, and use [Conventional
Commits](https://www.conventionalcommits.org/).

If you test the extractor with a new camera or storm scene, sharing the relevant
metadata and configuration is especially valuable—even when the result is a
false positive.
