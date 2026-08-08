# Project status

Last updated: 2026-08-08

This repository is the production video-analysis CLI. The model development
project is maintained separately in the `lightning-strike-model-lab` repository.

## Current state

- The CLI inspects and analyzes local videos and writes isolated reproducible
  runs below `runs/`.
- The CLI exports selected frames and neutral provenance manifests for optional
  downstream use.
- The product detector runtime accepts only a manifest-driven ONNX model bundle.
- No production ONNX artifact is currently bundled; the checked-in manifest is
  explicitly `unreleased`.
- Training, annotation, dataset releases, evaluation, and ONNX export are not
  part of this repository.

## Production workflow

```text
raw video
  -> lightning analyze
  -> review and select evidence
  -> neutral frame/provenance export
  -> (separate model-lab repository)
  -> evaluated ONNX release
  -> deliberate promotion into the CLI distribution
```

`runs/` is analysis evidence, not a training dataset. Frame handoffs contain
images and provenance only. The separate model-lab repository owns annotation,
dataset release construction, training, evaluation, and model promotion.

## Repository boundary

The CLI repository must not gain training-framework dependencies, annotation
format implementations, dataset splitters, model checkpoints, or training
commands. The model-lab repository must not import `lightning_extractor`.
Communication happens through versioned files and released ONNX bundles.

## Recommended next task

Run a full-length validation on the current footage and create the first neutral
frame handoff. In the separate model-lab repository, use that handoff to build
the first reviewed dataset release and evaluate candidate detector architectures.
No model should be promoted until recall and precision have been measured on
labeled footage and the candidate ONNX bundle has passed parity checks.

## Validation baseline

```bash
uv sync --extra dev
uv run python -m unittest discover -s tests -v
uv run ruff check .
```
