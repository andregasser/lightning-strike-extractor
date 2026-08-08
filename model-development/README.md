# Lightning Model Lab

This is an independent Python project for verified dataset releases, detector
training, evaluation, and ONNX export. It deliberately does not depend on or
import `lightning_extractor`. Its stable input is source-grouped, human-verified
COCO JSON; the video CLI is only one possible producer of suitable images.

## Environment

```bash
cd model-development
uv sync --extra train --extra dev
```

The product CLI does not need to be installed in this environment.

## Pipeline

```bash
uv run lightning-model release ../verified-dataset \
  --release-id lightning-2026.08.1 --output releases/lightning-2026.08.1

uv run lightning-model train releases/lightning-2026.08.1 \
  --output experiments/baseline --epochs 10

uv run lightning-model evaluate releases/lightning-2026.08.1 \
  experiments/baseline/checkpoint.pt \
  --output experiments/baseline/validation.json

uv run lightning-model export-onnx experiments/baseline/checkpoint.pt \
  --training-report experiments/baseline/training.json \
  --evaluation-report experiments/baseline/validation.json \
  --version 0.1.0 --output releases/model-0.1.0
```

Dataset releases reject unverified annotations, source leakage, conflicting
boxes for identical image hashes, unsafe paths, and overwrites. ONNX export
checks the graph and compares its outputs with PyTorch before publishing the
model bundle.

The generated `model.onnx`, `manifest.json`, and `checksums.json` form the only
contract consumed by the product runtime. Training checkpoints are never part
of the product distribution.
