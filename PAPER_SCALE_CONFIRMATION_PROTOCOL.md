# Paper-Scale Shared-Block Confirmation

Protocol frozen: 16 August 2026, after the Llama-3.2-3B cross-family gate
failed and its single permitted compatibility audit found no implementation
incompatibility, before selecting or generating the shared-block results.

## Status and claim boundary

This is a new, disjoint estimation study, not a replacement for the failed
Llama gate. The failed gate remains reported. No result from this study may be
used to describe the original gate as passed, and no further confirmation block
will be started in response to this study's outcome.

The intended paper claim is that exact option-space alignment has replicated
benefit within Qwen2.5 at 3B and 7B, with cross-family behavior estimated on
Llama-3.2-3B. Model-family generality requires positive Llama evidence; a
non-significant Llama result must be reported as unresolved.

## Models and shared questions

- Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct, and Llama-3.2-3B-Instruct in FP16.
- ARC-Challenge validation, OpenBookQA validation, and MMLU test.
- Remove every content hash used by the earlier 3B, 7B, adaptation, and Llama
  blocks. Rank the remaining eligible questions by SHA-256 using sample seed
  20260817 and take exactly 64 per benchmark.
- All three models use the identical 192 questions.

## Frozen generation and analysis

- Identity plus three cyclic rotations, four samples per view, and sixteen root
  samples at matched generation count.
- Temperature 0.8, top-p 0.95, maximum 192 new tokens, smoothing alpha 1, and
  twenty randomized-gauge controls.
- Generation seeds are 41808/41909/42010 for Qwen2.5-3B,
  51808/51909/52010 for Qwen2.5-7B, and 61808/61909/62010 for Llama-3.2-3B.
- Preserve parsed samples, root samples, raw completions, exact permutations,
  manifests, environment versions, and token totals.
- Report N=4/8/16 root versus mapped scaling, log-opinion versus hard vote,
  unmapped and randomized-map controls, per-benchmark effects, repair/corrupt
  counts, leave-one-view-out stability, and risk at 25/50/75/100% coverage.

## Paper-level decision rule

The shared block supports the scoped paper claim only if:

1. the fixed-model average mapped-minus-root effect has a stratified paired 95%
   bootstrap interval excluding zero;
2. at least two of three model effects are nonnegative and no model has a
   mapped-minus-root interval wholly below zero;
3. the fixed-model correct-map-minus-randomized effect is at least +0.05 with
   its interval excluding zero;
4. truth support is at least 0.80 and parse failure is at most 0.10 for every
   model; and
5. all manifests, data-overlap checks, tests, and numerical audits pass.

Passing does not by itself establish universal model-family generality. A new
positive Llama interval would support cross-family replication; otherwise Llama
remains a transparent limitation.
