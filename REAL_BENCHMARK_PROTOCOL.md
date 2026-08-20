# Real-Benchmark Option-Permutation Gate

Protocol frozen: 16 August 2026, before any benchmark result was inspected.

## Question

Does exact option-space alignment improve matched-compute inference on established
four-option reasoning and knowledge benchmarks, rather than only on the synthetic
modular-arithmetic task?

## Benchmarks and sample

- ARC-Challenge validation, OpenBookQA validation, and MMLU test.
- Exactly 64 eligible four-option questions per benchmark.
- Selection is deterministic: normalized question content is hashed with sample
  seed 20260816, and the 64 lowest hashes are selected.
- Dataset fingerprints, source indices, and content hashes are saved before
  inference.
- Generation seeds are 11808 (ARC-Challenge), 11909 (OpenBookQA), and 12010
  (MMLU).

## Frozen inference configuration

- Qwen2.5-3B-Instruct in FP16 on one RTX 3090 24 GB.
- Identity plus three cyclic option rotations.
- Four rollouts per view; the root comparator receives 16 rollouts on the
  identity prompt.
- Temperature 0.8, top-p 0.95, maximum 192 new tokens.
- Exact inverse maps, full support including parse failure, smoothing alpha 1,
  and twenty randomized-gauge controls computed from the same generations.
- Labels are used only after generation for evaluation.

## Primary analysis and gate

- Primary endpoint: pooled paired mapped-minus-root top-1 accuracy over all 192
  questions, with a paired bootstrap interval.
- Secondary endpoints: per-benchmark effects and correct-map-minus-randomized.
- The transfer gate passes only if:
  - pooled mapped-minus-root is at least +0.03 and its 95% interval excludes zero;
  - at least two of three benchmark effects are nonnegative;
  - pooled correct-map-minus-randomized is at least +0.05 and its 95% interval
    excludes zero;
  - pooled truth support is at least 0.80 and parse failure is at most 0.10.
- Failure triggers code/data diagnostics and an easy synthetic reference check
  only if environment drift is plausible. It does not trigger prompt or threshold
  tuning on these questions.

## Decision

Passing justifies a second model or 7B confirmation and, only after that,
benchmark adaptation. Failing means the current option-permutation result does
not transfer sufficiently to support a CCF-B main-track paper.
