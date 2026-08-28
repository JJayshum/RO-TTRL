# Gemma-2-2B smaller-model replication (2026-08-28)

Purpose: test whether the relational-alignment effect is stronger for a model smaller than Qwen2.5-3B.

- Model: Gemma-2-2B-it (cached cross-family checkpoint)
- Benchmarks: ARC-Challenge, OpenBookQA, MMLU
- Orbits: 64 per benchmark; views: 4; samples per view: 4
- Sample seed: 20260828; sample offset: 0 after explicit exclusion of prior Gemma panel manifests
- Run seeds: 92808 (ARC), 92909 (OpenBookQA), 93010 (MMLU)
- Analysis: report all three completed benchmarks and pooled paired-bootstrap intervals
- Stopping rule: complete all benchmarks regardless of the interval outcome
