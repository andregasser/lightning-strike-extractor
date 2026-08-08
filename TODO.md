# Project TODO

Public roadmap for Lightning Strike Extractor. Items are ordered roughly by
expected user value, but implementation order may change as real-world video
testing reveals new priorities.

## Validation on real footage

- [x] Support one or many files and directories in a single batch.
- [x] Add resumable, checkpointed analysis runs.
- [x] Run a five-minute calibration batch across the current footage set.
- [ ] Complete full-length analysis of the current reference footage.
- [ ] Review false positives caused by camera movement and exposure changes.
- [x] Create small labeled positive and negative video fixtures.
- [ ] Measure event recall and top-N candidate precision.
- [ ] Add camera- and scenario-specific configuration profiles.

## Review workflow

- [ ] Define a persistent review schema with accepted, rejected, and unreviewed states.
- [ ] Add tags such as `visible-channel`, `cloud-flash`, `ground-strike`, and `artifact`.
- [ ] Build a local HTML review interface for runs and batches.
- [ ] Add keyboard navigation, sorting, filtering, and progress tracking.
- [ ] Preserve review decisions across regenerated contact sheets and exports.

## Exports

- [ ] Export accepted full-resolution stills.
- [ ] Export configurable clips around accepted events.
- [ ] Render chronological highlight reels.
- [ ] Support batch-level exports across multiple source videos.
- [ ] Write an export manifest with source timestamps and review metadata.

## Reliability and performance

- [ ] Reduce repeated random seeks when ranking HEVC footage.
- [ ] Explore buffering event windows during sequential decoding.
- [ ] Benchmark `--jobs` values across SSD, CPU, and codec combinations.
- [ ] Improve handling and reporting of variable-frame-rate media.
- [ ] Add optional proxy generation for unusually expensive source codecs.
- [ ] Evaluate optional GPU acceleration only after CPU profiling.

## Project quality

- [ ] Add GitHub Actions for supported Python versions, tests, and Ruff.
- [ ] Add a clear open-source license.
- [ ] Add contribution and issue-reporting guidance.
- [ ] Add machine-readable result schema versions and compatibility tests.
- [ ] Keep README commands, output examples, and roadmap links current.

## Distribution

- [ ] Define a release and changelog process.
- [ ] Validate installation on macOS, Linux, and Windows.
- [ ] Support installation through `uv tool install` and `pipx`.
- [ ] Evaluate publishing stable releases to PyPI.

## Detector lifecycle

- [x] Separate model development from the product Python package.
- [x] Replace the product PyTorch/Transformers runtime with ONNX Runtime.
- [x] Add immutable verified dataset releases with conflict detection.
- [x] Add independent baseline training and evaluation commands.
- [x] Add ONNX graph validation and PyTorch parity checks.
- [ ] Build and review the first real verified dataset release.
- [ ] Measure detector recall and precision on labeled reference footage.
- [ ] Promote the first evaluated production ONNX bundle.
