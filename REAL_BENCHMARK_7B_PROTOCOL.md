# Disjoint-Sample 7B Benchmark Confirmation

Protocol frozen: 16 August 2026, after the complete 3B transfer gate passed and
before Qwen2.5-7B-Instruct inference began.

## Question

Does the real-benchmark option-alignment effect replicate at a larger model
scale on questions not used in the 3B gate?

## Frozen sample and configuration

- Qwen2.5-7B-Instruct in FP16 on one RTX 3090 24 GB.
- ARC-Challenge validation, OpenBookQA validation, and MMLU test.
- The same content-hash ranking and sample seed 20260816 as the 3B gate, but
  rows 64-127 rather than rows 0-63: 64 questions per benchmark with zero
  overlap by construction.
- Generation seeds: 21808 (ARC-Challenge), 21909 (OpenBookQA), and 22010
  (MMLU).
- All prompts, four cyclic views, four samples per view, matched 16-sample root
  comparator, temperature, top-p, token limit, smoothing, parser, and twenty
  randomized gauges remain unchanged.

## Confirmation gate

The 7B confirmation passes only if:

- pooled mapped-minus-root is at least +0.03 and its paired 95% interval
  excludes zero;
- at least two of three per-benchmark effects are nonnegative;
- pooled correct-map-minus-randomized is at least +0.05 and its interval
  excludes zero;
- pooled truth support is at least 0.80 and parse failure is at most 0.10.

Passing establishes model-scale and disjoint-sample replication sufficient to
justify developing real-benchmark adaptation and reliability gating. Failure
leaves the transfer evidence specific to Qwen2.5-3B and blocks adaptation.
