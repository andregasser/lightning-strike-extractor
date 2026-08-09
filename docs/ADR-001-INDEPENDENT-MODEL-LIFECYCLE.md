# ADR 001: Independent model lifecycle and ONNX product runtime

Status: accepted, 2026-08-08

## Decision

Detector development is an independent repository with its own package
metadata, dependencies, lockfile, commands, and tests. It must not import
`lse`. Its inputs are versioned frame exports and annotation
artifacts. Therefore the product CLI may provide useful source data but is not
required to build, train, evaluate, or export a model.

The product executes only a closed-set ONNX detector. It does not contain
PyTorch, Transformers, Grounding DINO, prompts, tokenizers, training code, or
dataset-release logic. A released model bundle consists of `model.onnx`, a
versioned manifest, and checksums. The manifest fixes preprocessing, tensor
names, class schema, thresholds, provenance, and runtime compatibility.

## Dependency rule

```text
frame export -> model lab -> ONNX release -> product runtime
                         no Python imports across either boundary
```

Run exports are evidence and optional data sources, not a training API. Model
promotion is a deliberate copy/release action after dataset validation,
evaluation, ONNX graph validation, and PyTorch/ONNX parity checks.

## Consequences

- Training dependencies can evolve without changing the product environment.
- Product installation remains small and deterministic.
- An unreleased or missing ONNX artifact produces an explicit runtime error;
  the CLI never downloads an implicit research checkpoint.
- Model and application versions are independent and traceable through the
  model and dataset release manifests.
