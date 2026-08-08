# Project status

Last updated: 2026-08-08

This file is the handoff note for continuing work in another chat. It records
what is actually in the repository; generated runs, datasets, and raw videos
are local and are not part of Git.

## Current state

- The production detector can inspect and analyze local videos and writes
  isolated, reproducible results below `runs/`.
- The product detector runtime now accepts only the versioned closed-set ONNX
  contract. PyTorch, Transformers, Grounding DINO, prompts, and tokenizers are
  absent from the product environment.
- No production ONNX artifact is committed yet. Detector invocation fails
  explicitly until an evaluated model release is promoted into the package.
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
- The independent `model-development/` project implements immutable dataset
  releases, a baseline Faster R-CNN training path, validation/test evaluation,
  and ONNX export with graph and PyTorch parity checks.
- Training has its own package and lockfile and contains no import of
  `lightning_extractor`. Its only product-facing output is an ONNX release
  bundle with manifest and checksums.

## End-to-end workflow

```text
raw video
  -> lightning analyze --output runs
  -> prepare_training_dataset runs --output dataset (blank human-annotation tasks)
  -> export_label_studio dataset --output label-studio-dataset
  -> review every task in Label Studio
  -> export full Label Studio JSON
  -> import_label_studio_dataset ... --output verified-dataset
  -> lightning-model release
  -> lightning-model train
  -> lightning-model evaluate
  -> lightning-model export-onnx
  -> deliberately promote the evaluated ONNX bundle to the product
```
`runs/` is analysis evidence, not a training dataset. `dataset/` contains
copied frames and unreviewed annotation tasks and is not trustworthy until reviewed.
`verified-dataset/` contains human-corrected annotations and deterministic
source-grouped splits. The independent release builder copies this verified
input into an immutable, hashed release; training must never read directly
from `runs/` or unverified proposals.

## Recommended next task

Create and review the first real verified dataset release, then run the new
baseline training and evaluation commands. The code path is present, but no
model should be promoted until recall and precision have been measured on
labeled footage and the candidate ONNX bundle has passed parity checks.

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
uv run --project model-development pytest
uv run --project model-development ruff check .
```
