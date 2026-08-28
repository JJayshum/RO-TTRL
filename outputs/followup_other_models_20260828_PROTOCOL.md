# Phi-3.5-mini and Mistral-7B replication (2026-08-28)

Purpose: test whether the model/task heterogeneity observed for Gemma-2-2B persists in two other model families.

- Models: Phi-3.5-mini-Instruct and Mistral-7B-Instruct-v0.3 (cached checkpoints)
- Benchmarks: ARC-Challenge, OpenBookQA, MMLU
- Orbits: 64 per benchmark and model; views: 4; samples per view: 4
- Sample seed: 20260828; sample offset: 0 after explicit exclusion of each model's prior panel manifests
- Run seeds: Phi 102808/102909/103010; Mistral 112808/112909/113010
- Analysis: report each model and pooled model-by-benchmark results with paired-bootstrap intervals
- Stopping rule: complete all six benchmark-model runs regardless of interval outcome
