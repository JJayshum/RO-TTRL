# Frozen Secondary Analysis Protocol

Protocol frozen: 16 August 2026, after the one-question Llama memory smoke test
but before inspection of the 64-question Llama confirmation results.

## Purpose

Measure compute scaling, pooling ablations, and label-free reliability using the
same stored generations. These analyses do not trigger additional model calls.

## Compute scaling and pooling

- Total generation budgets are N=4, 8, and 16.
- Mapped inference uses the first 1, 2, or 4 stored samples from each of four
  views. Root self-consistency uses the first N stored root samples.
- Primary mapped aggregation is the preregistered alpha=1 log-opinion pool over
  the full support `{parse failure, A, B, C, D}`.
- The pooling ablation is a hard majority vote after exact canonical mapping.
- Ties use the existing deterministic lowest-answer-index rule.
- Report paired orbit bootstrap intervals for mapped-minus-root and
  log-pool-minus-hard-vote at every budget. The N=16 result remains primary;
  N=4 and N=8 are secondary.

## Label-free stability and selective risk

- For each orbit, recompute the N=16 mapped prediction four times, omitting one
  transformed view at a time.
- The primary stability score is the fraction of leave-one-view-out predictions
  equal to the full four-view prediction.
- The secondary score is the probability gap between the two best answer
  options under the full log-opinion pool.
- Rank examples lexicographically by decreasing stability, then decreasing
  margin, then content SHA-256 as a deterministic label-free tie break.
- Report accuracy and risk at fixed 25%, 50%, 75%, and 100% coverage. These are
  descriptive on the confirmation block; no threshold is selected from labels.

## Audit note

The memory smoke test used the first ARC item from the confirmation block and
revealed that all sampled predictions for that single item were wrong. No code,
prompt, parser, model setting, gate, or analysis definition was changed in
response. The full confirmation remains a frozen 64-question run, and this
disclosure is retained rather than silently moving the evaluation block.
