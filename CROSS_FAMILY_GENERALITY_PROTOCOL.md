# Cross-Family Generality Panel

Protocol frozen: 17 August 2026, after the Qwen2.5/Llama shared-block study
completed and before any generation from the candidate models was inspected.

## Question and claim boundary

Does exact option-space alignment replicate across three additional instruction
model families? A positive result supports broad cross-family replication; it
does not prove universal benefit for every model, task, or transformation.

All candidate models and decision rules are fixed in this document. Every
candidate result will be reported. Models, prompts, parsers, pooling, sampling
settings, or thresholds may not be changed after candidate generations are
inspected. A tokenizer or loading incompatibility may be repaired only if the
repair is structural, label-blind, documented, and followed by a fresh run.

## Fixed model panel

- Microsoft Phi-3.5-mini-instruct (`LLM-Research/Phi-3.5-mini-instruct`).
- Mistral-7B-Instruct-v0.3 (`LLM-Research/Mistral-7B-Instruct-v0.3`).
- Google Gemma-2-2B-it (`LLM-Research/gemma-2-2b-it`).

The ModelScope `master` snapshots available on 17 August 2026 are used. Exact
downloaded-file SHA-256 hashes are recorded before inference. Models are loaded
in FP16 on one RTX 3090 24 GB with their official tokenizer chat templates.

## Shared questions and generation

- ARC-Challenge validation, OpenBookQA validation, and MMLU test.
- The exact 192-question paper-scale shared block is reused: 64 questions per
  benchmark selected with sample seed 20260817 after excluding all pre-shared
  inference and adaptation manifests.
- Reuse is intentional: the new estimand is model-family transfer on matched
  questions. No candidate model output has previously been inspected.
- Identity plus three cyclic option rotations, four samples per view, and
  sixteen root samples at matched generation count.
- Temperature 0.8, top-p 0.95, maximum 192 new tokens, smoothing alpha 1, and
  twenty randomized-gauge controls from the same transformed generations.
- Generation seeds are 71808/71909/72010 for Phi, 81808/81909/82010 for
  Mistral, and 91808/91909/92010 for Gemma.
- Parsed samples, raw completions, permutations, manifests, package versions,
  completion-token totals, model hashes, and analysis outputs are retained.

## Frozen analysis

The independent unit is a question. Bootstrap resampling is paired across
models and stratified by benchmark. The three candidate checkpoints are fixed,
not treated as a random sample from all language models. Report each model's
mapped-minus-root effect and interval, the fixed-model average effect, exact-map
versus randomized-map control, repair/corruption counts, N=4/8/16 scaling,
hard-vote ablation, leave-one-view-out stability, and fixed risk coverage.

An individual model is a positive replication only if mapped-minus-root is at
least +0.03 and its paired 95% interval excludes zero.

The panel supports broad cross-family replication only if all criteria pass:

1. the fixed-new-family average mapped-minus-root interval excludes zero on
   the positive side;
2. at least two of three model effects are nonnegative;
3. no model has a mapped-minus-root interval wholly below zero;
4. at least one candidate is an individual positive replication;
5. the fixed-model correct-map-minus-randomized effect is at least +0.05 and
   its paired interval excludes zero;
6. truth support is at least 0.80 and parse failure is at most 0.10 for every
   candidate; and
7. shared-question identity, manifests, structural audits, tests, numerical
   checks, model hashes, and output checksums pass.

Failure of the panel or any individual model is retained as evidence about the
method's boundary. It does not trigger prompt tuning or replacement models on
this block.
