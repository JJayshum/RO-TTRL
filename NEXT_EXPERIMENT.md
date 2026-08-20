# Harder Option-Family Experiment

Protocol frozen: 16 August 2026, before any medium-difficulty result was inspected.

## Question

Does the corrected Relational-Orbit advantage survive a harder option-permutation
task distribution, where each modular-arithmetic problem has 2-3 operations and
uses moduli 11, 13, 17, or 19?

## Model and compute

- Qwen2.5-3B-Instruct, FP16, one RTX 3090 24 GB.
- Four cyclic option-permutation views.
- Four samples per view for mapped inference; 16 root samples for the
  matched-compute root comparator.
- Temperature 0.8, top-p 0.95, maximum 192 new tokens.
- Exact inverse answer maps, full answer support including parse failure, and
  the existing frozen parser and prompts.

## Stage A: inference gate

- Two independent seeds: 6808 and 6909.
- 64 fresh orbits per seed.
- Twenty randomized-gauge controls, computed from the same generations.
- Advance only if the pooled mapped-minus-root effect is at least +0.10, the
  pooled correct-map-minus-randomized effect is at least +0.10, truth-support
  coverage is at least 0.80, and parse failure is at most 0.10.
- A failed gate triggers diagnostics, not hyperparameter tuning on these seeds.

## Stage B: matched-compute adaptation

Run only if Stage A passes.

- Five training seeds: 6501, 7501, 8501, 9501, and 10501.
- Corresponding untouched evaluation seeds: 6601, 7601, 8601, 9601, and 10601.
- Per method and seed: 16 unlabeled training orbits, 512 sampled sequences,
  one on-policy LoRA pass, rank 8, learning rate 1e-5, and KL coefficient 0.02.
- Evaluation: 64 fresh orbits, four samples per view.
- Primary effect: mapped TTRL minus matched-compute root TTRL policy pass@1.
- Secondary effect: mapped TTRL minus the unadapted base.
- Aggregate uncertainty: hierarchical bootstrap resampling seeds and then
  paired evaluation orbits.

## Decision rule

The harder-task policy result is positive if the mapped-minus-root equal-seed
estimate is at least +0.03, at least four of five seeds are positive, and the
hierarchical 95% interval excludes zero. Otherwise, the current evidence remains
limited to the easy option family. Boolean and affine results remain reported as
negative controls regardless of this outcome.

## Post-gate diagnostic

Added after Stage A failed, so this diagnostic is not part of the confirmatory
medium-difficulty result and cannot change its gate decision.

- Run one 32-orbit easy reference at seed 7808 with the same code, model, and
  inference settings.
- Purpose: distinguish an environment/model regression from a genuine
  difficulty boundary.
- The environment-drift explanation is rejected if mapped-minus-root is at
  least +0.15 and correct-map-minus-randomized is at least +0.20, consistent
  with the earlier easy-family effect.
