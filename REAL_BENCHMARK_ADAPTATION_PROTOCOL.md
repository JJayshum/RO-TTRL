# Real-Benchmark 3B Adaptation Diagnostic

Protocol frozen: 16 August 2026, after the disjoint-sample 7B inference
confirmation passed and before benchmark adaptation began.

## Question

Can mapped relational rewards improve a 3B policy on a mixture of established
benchmarks, compared with a matched-compute root-only TTRL update?

## Frozen train and evaluation data

- Qwen2.5-3B-Instruct in FP16 with LoRA rank 8.
- ARC-Challenge, OpenBookQA, and MMLU using the previously hashed source files.
- Training: content-hash-ranked rows 128-133 from each benchmark, six per
  benchmark and 18 unlabeled orbits total.
- Evaluation: rows 144-175, 32 per benchmark and 96 untouched orbits total.
- Training and evaluation are disjoint from each other and from the 3B and 7B
  inference blocks.

## Matched adaptation

- Mapped TTRL: four cyclic views and eight rollouts per view.
- Root TTRL: 32 rollouts on the identity prompt.
- Both methods receive 576 sampled training sequences.
- One on-policy LoRA pass, learning rate 1e-5, KL coefficient 0.02, full
  completion sequence log-probability, leave-one-out rewards, and within-view
  reward centering.
- Evaluation uses four rollouts per view, generation seed 31901, and the same
  prompt/parser configuration as the inference gates.

## Diagnostic gate

The single-seed diagnostic passes only if mapped TTRL exceeds root TTRL policy
pass@1 by at least +0.02, the paired 95% interval excludes zero, at least two of
three benchmark effects are nonnegative, evaluation parse failure is at most
0.10, and neither adapter shows numerical or KL collapse.

Passing justifies a preregistered multi-seed adaptation study. Failure blocks
additional adaptation spending; it does not invalidate the replicated
inference-time result.
