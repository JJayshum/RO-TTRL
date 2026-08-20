# Second-Model-Family Confirmation

Protocol frozen: 16 August 2026, after Qwen2.5 3B/7B inference replication and
the real-benchmark adaptation diagnostic, before Llama inference began.

## Question

Does exact option-space alignment transfer beyond the Qwen2.5 model family?

## Frozen configuration

- Model: Llama-3.2-3B-Instruct in FP16 on one RTX 3090 24 GB.
- Benchmarks: ARC-Challenge validation, OpenBookQA validation, and MMLU test.
- Questions: content-hash-ranked rows 208-271 from each source, 64 per
  benchmark. These are disjoint from every previous inference, adaptation
  training, and adaptation evaluation block.
- Generation seeds: 31808, 31909, and 32010 respectively.
- Identity plus three cyclic option rotations; four samples per view; 16 root
  samples; temperature 0.8; top-p 0.95; maximum 192 new tokens.
- Exact inverse maps, full support including parse failure, smoothing alpha 1,
  and twenty randomized-gauge controls from the same transformed generations.

## Gate

The second-family result passes only if pooled mapped-minus-root is at least
+0.03 with a paired 95% interval excluding zero, at least two of three dataset
effects are nonnegative, pooled correct-map-minus-randomized is at least +0.05
with its interval excluding zero, truth support is at least 0.80, and parse
failure is at most 0.10.

Passing establishes cross-family replication and triggers paper-scale sample
expansion plus compute-scaling and reliability analyses. Failure blocks a claim
of model-family generality and triggers one code/prompt compatibility audit only.
