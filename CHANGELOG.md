# Changelog

All notable changes are documented in this file. The project follows Semantic
Versioning and keeps release notes in GitHub Releases.

## 0.3.0 — 2026-09-24

First public release of Starling ML.

- Added the public `Engine(config)` API, editable standard configurations and
  the non-mutating configuration analyzer.
- Separated masks, importance weights and error costs across pointwise and
  overlap objectives.
- Added exact sufficient-statistics accumulation, full-state checkpoints and
  deterministic resume for the built-in batch source.
- Added fifteen executable synthetic recipes, including segmentation,
  language modeling, diffusion, GAN, detection and reinforcement-learning
  primitives.
- Added optional wrappers for SMP, MONAI, Transformers, Diffusers, nnU-Net and
  experiment trackers.
- Added CPU regression tests and explicit validation limits for DDP, CUDA and
  pretrained integrations.

The pre-release package was internally spelled `starlilng-ml`. Before the
first public publication it was renamed to distribution `starling-ml` and
Python package `starling_ml`.
