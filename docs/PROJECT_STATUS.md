# Project status

Last updated: 2026-08-08

This file is the handoff note for continuing work in another chat. It records
what is actually in the repository; generated runs, datasets, and raw videos
are local and are not part of Git.

## Current state

- The production detector can inspect and analyze local videos and writes
  isolated, reproducible results below `runs/`.
- The pinned Grounding-DINO checkpoint is a bootstrap model for annotation
  proposals. It is not a lightning-trained production detector.
- The dataset preparation path extracts event frames and creates unverified
  COCO box proposals.
- Label Studio is the supported review interface. The exporter creates tasks,
  a label configuration, and a local image-serving directory. The server
  defaults to `http://localhost:8001/images`.
- The Label Studio importer accepts a full JSON export, preserves empty
  annotations as verified negatives, validates rectangles, and creates
  source-grouped COCO train/validation/test splits.
- CVAT is legacy compatibility code only. New workflow and documentation
  should use Label Studio; do not add new CVAT features.
- Model training and the immutable dataset-release builder are not implemented
  yet. The current pipeline ends at `verified-dataset/`.

## End-to-end workflow

```text
raw video
  -> lightning analyze --output runs
  -> prepare_training_dataset runs --output dataset
  -> export_label_studio dataset --output label-studio-dataset
  -> review every task in Label Studio
  -> export full Label Studio JSON
  -> import_label_studio_dataset ... --output verified-dataset
  -> (future) register campaign and build immutable dataset release
  -> (future) train and evaluate the lightning detector
```
`runs/` is analysis evidence, not a training dataset. `dataset/` contains
copied frames and model guesses and is not trustworthy until reviewed.
`verified-dataset/` contains human-corrected annotations and deterministic
source-grouped splits. A future release builder should copy this verified
input into an immutable, hashed release and never train directly from `runs/`
or unverified proposals.

## Recommended next task

Implement the dataset release builder as a small, deterministic CLI. It should:

1. register one or more verified Label Studio campaigns with metadata and
   checksums;
2. merge selected campaigns while deduplicating identical image hashes;
3. reject conflicting annotations instead of silently choosing one;
4. assign complete source videos to train/validation/test (never split nearby
   frames from one source across partitions);
5. write an immutable release manifest containing tool version, input hashes,
   category schema, split ratios, and output checksums.

After that, add the training/evaluation command and measure recall and top-N
precision on labeled reference footage. Do not pin a final detector checkpoint
before those measurements exist.

## Known operational issue

Label Studio must be able to reach the local image server. If it reports an
invalid URL for `http://localhost:8001/images/...`, start
`serve_label_studio` from the exported `serve/` directory and use
`--image-base-url` matching the reachable scheme and host. Keep that server
running during annotation.

## Validation baseline

Run the standard checks before handing off changes:

```bash
uv run python -m unittest discover -s tests -v
uv run ruff check .
```
